import os
import re
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, CheckboxButtonGroup,
    DataTable, TableColumn, NumberFormatter, Div, HoverTool, Label
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from scipy.stats import spearmanr

###############################################################################
# 1) Configuration
###############################################################################
# Local data directory path
DATA_DIR = "summary_data"  # Directory containing the summary files

# We'll look for files matching the pattern: summary_testNNN.txt
FILE_PATTERN = r"^summary_test(\d{3})\.txt$"

# (Optional) Specify a default test to visualize on startup
DEFAULT_TEST = "test001"  # Change this to your preferred default test

###############################################################################
# 2) Helpers: Data Parsing
###############################################################################
def moving_average(arr, window_size=5):
    """
    Simple moving average on a float array (np.nan used for missing).
    Returns an equally sized array of floats (with np.nan where not enough data).
    """
    n = len(arr)
    out = np.full(n, np.nan)
    halfw = window_size // 2
    
    for i in range(n):
        if np.isnan(arr[i]):
            continue
        start = max(0, i - halfw)
        end = min(n, i + halfw + 1)
        window = arr[start:end]
        good = window[~np.isnan(window)]
        if len(good) > 0:
            out[i] = np.mean(good)
    return out

def parse_summary_file(local_path):
    """
    Reads a summary_test###.txt with columns:
      AlnIndex, Residue, B_Factor, ExpFrust, AFFrust, EvolFrust
    Returns (df_original, df_for_plot), plus a correlation dict for non-smoothed data.
    """
    if not os.path.isfile(local_path):
        return None, None, {}

    df = pd.read_csv(local_path, sep='\t')
    # Convert any 'n/a' to np.nan in numeric columns
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        df[col] = df[col].apply(lambda x: np.nan if str(x).lower()=='n/a' else float(x))

    # Non-smoothed for correlation (and for scatter plots)
    df_original = df.copy()

    # Smoothed for plotting
    df_for_plot = df.copy()
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr = df_for_plot[col].values
        df_for_plot[col] = moving_average(arr, window_size=5)

    # ---------------------------
    #  (1) Min–Max Normalization
    #      after smoothing
    # ---------------------------
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr = df_for_plot[col].values
        col_min = np.nanmin(arr)
        col_max = np.nanmax(arr)
        if col_max > col_min:  # Avoid divide-by-zero if all values are the same
            df_for_plot[col] = (arr - col_min) / (col_max - col_min)

    # Compute Spearman correlations on NON-smoothed data
    corrs = {}
    sub = df_original.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    if not sub.empty:
        combos = [
            ("B_Factor", "ExpFrust"),
            ("B_Factor", "AFFrust"),
            ("B_Factor", "EvolFrust"),
            ("ExpFrust", "AFFrust"),
            ("ExpFrust", "EvolFrust"),
            ("AFFrust",  "EvolFrust"),
        ]
        for (mA, mB) in combos:
            # Handle constant input warnings by checking variance
            if sub[mA].nunique() < 2 or sub[mB].nunique() < 2:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(sub[mA], sub[mB])
            corrs[(mA, mB)] = (rho, pval)
    return df_original, df_for_plot, corrs

###############################################################################
# 3) Load data from local directory
###############################################################################
data_by_test = {}
all_corr_rows = []

# List all files in the data directory
for filename in os.listdir(DATA_DIR):
    # Check if it matches summary_testNNN.txt
    match = re.match(FILE_PATTERN, filename)
    if not match:
        continue  # skip unrelated files

    # Extract test### from the file name
    test_number = match.group(1)  # e.g. "001"
    test_name = "test" + test_number  # e.g. "test001"

    # Get the full path to the file
    file_path = os.path.join(DATA_DIR, filename)

    # Parse the data
    df_orig, df_plot, corrs = parse_summary_file(file_path)
    if df_orig is None:
        continue

    data_by_test[test_name] = {
        "df_original": df_orig,
        "df_plot": df_plot,
        "corrs": corrs
    }

    # Accumulate correlation info for a master table
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo
        all_corr_rows.append([test_name, mA, mB, rho, pval])

# Build DataFrame of correlations
df_all_corr = pd.DataFrame(all_corr_rows, columns=["Test","MetricA","MetricB","Rho","Pval"])

###############################################################################
# 4) Bokeh Application
###############################################################################

# (A) ColumnDataSource for the main plot (smoothed + normalized)
source_plot = ColumnDataSource(data=dict(
    x=[],
    residue=[],
    b_factor=[],
    exp_frust=[],
    af_frust=[],
    evol_frust=[]
))

# (B) The main figure
p = figure(
    title="(No Data)",
    sizing_mode='stretch_width',
    height=600,
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom", active_scroll="wheel_zoom"
)

# Define separate HoverTools for each metric
hover_bf = HoverTool(
    renderers=[],
    tooltips=[
        ("Index", "@x"),
        ("Residue", "@residue"),
        ("B-Factor", "@b_factor")
    ],
    name="hover_b_factor"
)

hover_ef = HoverTool(
    renderers=[],
    tooltips=[
        ("Index", "@x"),
        ("Residue", "@residue"),
        ("ExpFrust", "@exp_frust")
    ],
    name="hover_exp_frust"
)

hover_af = HoverTool(
    renderers=[],
    tooltips=[
        ("Index", "@x"),
        ("Residue", "@residue"),
        ("AFFrust", "@af_frust")
    ],
    name="hover_af_frust"
)

hover_ev = HoverTool(
    renderers=[],
    tooltips=[
        ("Index", "@x"),
        ("Residue", "@residue"),
        ("EvolFrust", "@evol_frust")
    ],
    name="hover_evol_frust"
)

# Add HoverTools to the figure
p.add_tools(hover_bf, hover_ef, hover_af, hover_ev)

# Set axes labels
p.xaxis.axis_label = "Residue Index"
p.yaxis.axis_label = "Normalized Residue Flexibility / Frustration"

# Add lines for each metric and associate HoverTools
color_map = {
    "b_factor":  ("B-Factor", "#1f77b4"),
    "exp_frust": ("ExpFrust", "#2ca02c"),
    "af_frust":  ("AFFrust", "#ff7f0e"),
    "evol_frust":("EvolFrust","#d62728")
}
renderers = {}
for col_key, (label, col) in color_map.items():
    renderer = p.line(
        x="x", y=col_key, source=source_plot,
        line_width=2, alpha=0.7, color=col,
        legend_label=label
    )
    renderers[col_key] = renderer
    # Assign the corresponding HoverTool to this renderer
    if col_key == "b_factor":
        hover_bf.renderers.append(renderer)
    elif col_key == "exp_frust":
        hover_ef.renderers.append(renderer)
    elif col_key == "af_frust":
        hover_af.renderers.append(renderer)
    elif col_key == "evol_frust":
        hover_ev.renderers.append(renderer)

p.legend.location = "top_left"
p.legend.title = "Metrics"
p.legend.click_policy = "hide"

# --- Create three scatter plots (NON-NORMALIZED data) for B-Factor vs. each frustration metric. ---
p_scatter_exp = figure(
    width=300,
    height=300,
    title="",
    x_axis_label="B-Factor",
    y_axis_label="ExpFrust",
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom", active_scroll="wheel_zoom"
)
p_scatter_af = figure(
    width=300,
    height=300,
    title="",
    x_axis_label="B-Factor",
    y_axis_label="AFFrust",
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom", active_scroll="wheel_zoom"
)
p_scatter_evol = figure(
    width=300,
    height=300,
    title="",
    x_axis_label="B-Factor",
    y_axis_label="EvolFrust",
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom", active_scroll="wheel_zoom"
)

# ColumnDataSources for the scatter plots (use non-smoothed, non-normalized data)
source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[]))
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[]))
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[]))

# Scatter renders
r_scatter_exp = p_scatter_exp.scatter("x", "y", source=source_scatter_exp, color="#2ca02c", alpha=0.7)
r_scatter_af  = p_scatter_af.scatter("x", "y", source=source_scatter_af,  color="#ff7f0e", alpha=0.7)
r_scatter_evol= p_scatter_evol.scatter("x", "y", source=source_scatter_evol, color="#d62728", alpha=0.7)

# Helper function to add a regression line & label to a figure
def add_regression_line_and_label(fig, xvals, yvals, color="black"):
    """
    Adds a linear regression line (y = m*x + b) to a Bokeh figure,
    plus a Label showing slope, intercept, and R-value.
    """
    # Guard against empty or constant arrays
    if len(xvals) < 2 or np.all(xvals == xvals[0]):
        return
    
    # Remove nans
    not_nan = ~np.isnan(xvals) & ~np.isnan(yvals)
    if not any(not_nan):
        return
    
    xvals_clean = xvals[not_nan]
    yvals_clean = yvals[not_nan]
    if len(xvals_clean) < 2:
        return
    
    # Linear regression via polyfit
    m, b = np.polyfit(xvals_clean, yvals_clean, 1)
    # Compute correlation (Pearson's r) just for display
    # (Spearman doesn't make sense for "line of best fit" in the usual sense.)
    corr = np.corrcoef(xvals_clean, yvals_clean)[0,1]
    
    # Plot the line
    xline = np.linspace(xvals_clean.min(), xvals_clean.max(), 100)
    yline = m*xline + b
    fig.line(xline, yline, line_width=2, line_dash='dashed', color=color)
    
    # Put the regression equation on the plot
    label_text = f"y = {m:.2f}x + {b:.2f}\nr = {corr:.2f}"
    
    # Create a Label in data coordinates near the top-left
    label_obj = Label(
        x=xvals_clean.min(),
        y=yvals_clean.max(),
        text=label_text,
        text_color=color,
        text_font_size="10px",
        text_font_style="bold"
    )
    fig.add_layout(label_obj)

# (C) SELECT widget to pick the test### to show
test_options = sorted(data_by_test.keys())

# Determine the default test to select
if DEFAULT_TEST in test_options:
    initial_test = DEFAULT_TEST
elif test_options:
    initial_test = test_options[0]
else:
    initial_test = ""

select_test = Select(
    title="Select Protein (test###):",
    value=initial_test,
    options=test_options
)

def update_plot(attr, old, new):
    """
    Updates both:
      - The main (smoothed + normalized) line plot
      - The three scatter plots (non-smoothed + non-normalized)
    whenever a new test is selected.
    """
    td = select_test.value
    if td not in data_by_test:
        # Clear all data if invalid
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        source_scatter_exp.data = dict(x=[], y=[])
        source_scatter_af.data = dict(x=[], y=[])
        source_scatter_evol.data = dict(x=[], y=[])
        p.title.text = "(No Data)"
        p_scatter_exp.title.text = ""
        p_scatter_af.title.text = ""
        p_scatter_evol.title.text = ""
        return
    
    # --- Update the main line plot (smoothed + normalized) ---
    dfp = data_by_test[td]["df_plot"]
    sub_plot = dfp.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    if sub_plot.empty:
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        p.title.text = f"{td} (No valid rows)."
    else:
        new_data = dict(
            x=sub_plot["AlnIndex"].tolist(),
            residue=sub_plot["Residue"].tolist(),
            b_factor=sub_plot["B_Factor"].tolist(),
            exp_frust=sub_plot["ExpFrust"].tolist(),
            af_frust=sub_plot["AFFrust"].tolist(),
            evol_frust=sub_plot["EvolFrust"].tolist()
        )
        source_plot.data = new_data
        p.title.text = f"{td} (Smoothed + Normalized)"
    
    # --- Update the scatter plots (non-smoothed + non-normalized) ---
    df_orig = data_by_test[td]["df_original"]
    sub_orig = df_orig.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    # Clear old lines/labels if any
    p_scatter_exp.renderers = [r_scatter_exp]
    p_scatter_af.renderers = [r_scatter_af]
    p_scatter_evol.renderers = [r_scatter_evol]
    p_scatter_exp.annotations.clear()
    p_scatter_af.annotations.clear()
    p_scatter_evol.annotations.clear()
    
    if sub_orig.empty:
        source_scatter_exp.data = dict(x=[], y=[])
        source_scatter_af.data = dict(x=[], y=[])
        source_scatter_evol.data = dict(x=[], y=[])
        p_scatter_exp.title.text = f"{td} (No Data)"
        p_scatter_af.title.text  = f"{td} (No Data)"
        p_scatter_evol.title.text= f"{td} (No Data)"
    else:
        # ExpFrust
        x_exp = sub_orig["B_Factor"].values
        y_exp = sub_orig["ExpFrust"].values
        source_scatter_exp.data = dict(x=x_exp, y=y_exp)
        p_scatter_exp.title.text = f"{td} Experimental Frustration"
        add_regression_line_and_label(p_scatter_exp, x_exp, y_exp, color="#2ca02c")
        
        # AFFrust
        x_af = sub_orig["B_Factor"].values
        y_af = sub_orig["AFFrust"].values
        source_scatter_af.data = dict(x=x_af, y=y_af)
        p_scatter_af.title.text = f"{td} AF Frustration"
        add_regression_line_and_label(p_scatter_af, x_af, y_af, color="#ff7f0e")
        
        # EvolFrust
        x_evol = sub_orig["B_Factor"].values
        y_evol = sub_orig["EvolFrust"].values
        source_scatter_evol.data = dict(x=x_evol, y=y_evol)
        p_scatter_evol.title.text = f"{td} Evolutionary Frustration"
        add_regression_line_and_label(p_scatter_evol, x_evol, y_evol, color="#d62728")

select_test.on_change("value", update_plot)

# Trigger the callback to populate the plot with the initial selection
update_plot(None, None, initial_test)

# (D) CORRELATION TABLE
if df_all_corr.empty:
    columns = [
        TableColumn(field="Test", title="Test"),
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Rho"),
        TableColumn(field="Pval", title="p-value")
    ]
    source_corr = ColumnDataSource(dict(Test=[], MetricA=[], MetricB=[], Rho=[], Pval=[]))
    data_table = DataTable(columns=columns, source=source_corr, height=400, width=1200)
else:
    source_corr = ColumnDataSource(df_all_corr)
    columns = [
        TableColumn(field="Test", title="Test"),
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Spearman Rho", formatter=NumberFormatter(format="0.3f")),
        TableColumn(field="Pval", title="p-value", formatter=NumberFormatter(format="0.2e"))
    ]
    data_table = DataTable(columns=columns, source=source_corr, height=400, width=1200)

# (E) FILTERS for correlation table using CheckboxButtonGroup
tests_in_corr = sorted(df_all_corr["Test"].unique()) if not df_all_corr.empty else []
if not df_all_corr.empty:
    # Build a list of "MetricA vs MetricB" strings
    combo_options = sorted([
        f"{row['MetricA']} vs {row['MetricB']}" 
        for _, row in df_all_corr.iterrows()
    ])
    # Remove duplicates
    combo_options = sorted(list(set(combo_options)))
else:
    combo_options = []

cbg_tests = CheckboxButtonGroup(
    labels=tests_in_corr,
    active=[]  # No active selections by default
)

cbg_combos = CheckboxButtonGroup(
    labels=combo_options,
    active=[]  # No active selections by default
)

def update_corr_filter(attr, old, new):
    """Filter the correlation table based on selected tests and metric pairs."""
    if df_all_corr.empty:
        return
    selected_tests = [cbg_tests.labels[i] for i in cbg_tests.active]
    selected_combos = [cbg_combos.labels[i] for i in cbg_combos.active]
    
    # If no filters are selected, show all data
    if not selected_tests and not selected_combos:
        filtered = df_all_corr
    else:
        df_tmp = df_all_corr.copy()
        df_tmp["combo_str"] = df_tmp.apply(lambda r: f"{r['MetricA']} vs {r['MetricB']}", axis=1)
        
        # Apply filters
        if selected_tests and selected_combos:
            filtered = df_tmp[
                (df_tmp["Test"].isin(selected_tests)) &
                (df_tmp["combo_str"].isin(selected_combos))
            ].drop(columns=["combo_str"])
        elif selected_tests:
            filtered = df_tmp[df_tmp["Test"].isin(selected_tests)].drop(columns=["combo_str"])
        elif selected_combos:
            filtered = df_tmp[df_tmp["combo_str"].isin(selected_combos)].drop(columns=["combo_str"])
        else:
            filtered = df_all_corr

    source_corr.data = filtered.to_dict(orient="list")

cbg_tests.on_change("active", update_corr_filter)
cbg_combos.on_change("active", update_corr_filter)

# (G) Header Section
header = Div(text="""
    <h1>Evolutionary Frustration</h1>
    <p>
        Evolutionary frustration leverages multiple sequence alignment (MSA) derived coupling scores and statistical potentials to calculate the mutational frustration of various proteins without the need for protein structures. By benchmarking the evolutionary frustration metric against experimental data (B-Factor) and two structure-based frustration metrics, we seek to validate the efficacy of sequence-derived evolutionary constraints in representing protein flexibility.
    </p>
    <p>
        The data displayed compare the agreement of different frustration metrics with the experimental B-Factor, which measures the average positional uncertainty of each amino acid derived from crystallographic data. The metrics include:
    </p>
    <ul>
        <li><strong>Experimental Frustration</strong>: Uses the Frustratometer web tool with an experimentally derived protein structure.</li>
        <li><strong>AF Frustration</strong>: Utilizes a sequence-derived protein structure generated by AlphaFold as the Frustratometer input.</li>
        <li><strong>Evolutionary Frustration</strong>: Derived from evolutionary constraints represented by coupling scores without compressing them into a single structure.</li>
    </ul>
    <p>
        The correlation table below the graphs presents Spearman correlation coefficients and p-values between different metrics using <em>non-smoothed</em> data. 
        <br>Visualized curves in the main plot are smoothed with a moving average method (window size = 5) and <strong>min–max normalized</strong>. Note that normalization does not affect <em>rank-based</em> correlation metrics (Spearman). 
        <br><em>Min–max normalization is performed individually on each protein's data and is not suitable for comparisons across different proteins.</em>
    </p>
    <h3>Contributors</h3>
    <p>
        <strong>Adam Kuhn<sup>1,2,3,4</sup>, Vinícius Contessoto<sup>4</sup>, George N Phillips Jr.<sup>2,3</sup>, José Onuchic<sup>1,2,3,4</sup></strong><br>
        <sup>1</sup>Department of Physics, Rice University, 6100 Main St, Houston, TX 77005<br>
        <sup>2</sup>Department of Chemistry, Rice University, 6100 Main St, Houston, TX 77005<br>
        <sup>3</sup>Department of Biosciences, Rice University, 6100 Main St, Houston, TX 77005<br>
        <sup>4</sup>Center for Theoretical Biological Physics, Rice University, 6100 Main St, Houston, TX 77005
    </p>
""", sizing_mode='stretch_width', styles={'margin-bottom': '20px'})

# (G) Description of the Protein Visualizer
description_visualizer = Div(text="""
    <h2>Protein Visualizer Instructions</h2>
    <p>
        The protein visualizer allows you to interact with the protein structure using various controls and visual metrics:
    </p>
    <ul>
        <li><strong>Oscillation:</strong> Press <code>O</code> to make the protein ribbon oscillate. The amplitude and frequency of oscillation are mapped to the average B-factor.</li>
        <li><strong>Color Representation:</strong> The ribbon color represents the experimental frustration. Light blue residues are minimally frustrated; magenta residues are highly frustrated.</li>
        <li><strong>Luminosity:</strong> The luminosity of each residue represents evolutionary frustration.</li>
        <li><strong>Fragmentation:</strong> Press <code>B</code> to fragment the protein. Each fragment is plotted on a 3D plot representing the three frustration metrics.</li>
        <li><strong>Navigation Controls:</strong> <code>W/A/S/D/Shift/Space</code> for camera movement, <code>C</code> to zoom, <code>Ctrl</code> to accelerate, etc.</li>
        <li><strong>Folding Controls:</strong> <code>Q</code> to unfold, <code>E</code> to refold. In the unfolded state, <code>O</code> triggers oscillation or sets the height by B-factor.</li>
        <li><strong>Pause:</strong> Press <code>P</code> to pause the visualizer and select another protein.</li>
    </ul>
""", sizing_mode='stretch_width', styles={'margin-bottom': '20px'})

# (I) Unity iframe
unity_iframe = Div(
    text="""
    <div style="width: 100%; display: flex; justify-content: center; align-items: center; margin: 20px 0;">
        <iframe 
            src="https://igotintogradschool2025.site/unity/" 
            style="width: 95vw; height: 90vh; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
    </div>
    """,
    sizing_mode='stretch_width',
    styles={'margin-top': '20px'}
)
unity_iframe.visible = True  # Always visible

unity_container = column(
    description_visualizer,
    unity_iframe,
    sizing_mode='stretch_width'
)

# Row of scatter plots
scatter_row = row(
    p_scatter_exp,
    p_scatter_af,
    p_scatter_evol,
    sizing_mode='stretch_width'
)

# Visualization layout: main line plot + row of scatter + unity section
visualization_section = column(
    select_test,
    p,
    scatter_row,         # <<-- The three scatter plots go here
    unity_container,
    sizing_mode='stretch_width',
    css_classes=['visualization-section']
)

# (F) Controls for the correlation table
controls_section = column(
    Div(text="<b>Filter Correlation Table</b>", styles={'font-size': '16px', 'margin': '10px 0'}),
    row(
        Div(text="<i>Select Proteins:</i>", width=150),
        cbg_tests,
        sizing_mode='stretch_width'
    ),
    row(
        Div(text="<i>Select Metric Pairs:</i>", width=150),
        cbg_combos,
        sizing_mode='stretch_width'
    ),
    sizing_mode='stretch_width'
)

# Custom CSS
custom_styles = Div(text="""
    <style>
        .visualization-section {
            margin: 20px 0;
            width: 100%;
        }
        .controls-row {
            margin: 10px 0;
            gap: 10px;
        }
        .bk-root {
            width: 100% !important;
        }
    </style>
""")

# Main layout
main_layout = column(
    custom_styles,
    header,
    visualization_section,
    controls_section,
    data_table,
    sizing_mode='stretch_width'
)

curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration"