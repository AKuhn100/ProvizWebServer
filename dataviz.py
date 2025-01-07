import os
import re
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, CheckboxButtonGroup,
    DataTable, TableColumn, NumberFormatter, Div, HoverTool
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

def min_max_norm(series):
    """
    Returns the min-max normalized version of a pandas Series or 1D numpy array.
    If all values are NaN or constant, returns all zeros or NaNs accordingly.
    """
    arr = series.values.astype(float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan, dtype=np.float64)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if np.isclose(min_val, max_val):
        # All values the same (or effectively the same)
        # Return zeros (or we could choose np.nan)
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - min_val) / (max_val - min_val)

def parse_summary_file(local_path):
    """
    Reads a summary_test###.txt with columns:
      AlnIndex, Residue, B_Factor, ExpFrust, AFFrust, EvolFrust
    Returns (df_original, df_for_plot), plus a correlation dict for non-smoothed data.
    
    df_original  = raw data (non-smoothed)
    df_for_plot  = smoothed + min–max normalized data (for main line plot)
    """
    if not os.path.isfile(local_path):
        return None, None, {}

    df = pd.read_csv(local_path, sep='\t')
    # Convert any 'n/a' to np.nan in numeric columns
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        df[col] = df[col].apply(lambda x: np.nan if str(x).lower()=='n/a' else float(x))

    # Non-smoothed for correlation and raw scatter
    df_original = df.copy()

    # Add min–max normalized columns for the frustration metrics (raw, non-smoothed)
    # so we can use them in the scatter plots
    for col in ["ExpFrust", "AFFrust", "EvolFrust"]:
        df_original[f"{col}_norm"] = min_max_norm(df_original[col])

    # Smoothed for plotting, then min–max normalization
    df_for_plot = df.copy()
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr_smoothed = moving_average(df_for_plot[col].values, window_size=5)
        df_for_plot[col] = min_max_norm(pd.Series(arr_smoothed))  # <== min–max on smoothed

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

# (A) ColumnDataSource for the main plot
source_plot = ColumnDataSource(data=dict(
    x=[],
    residue=[],
    b_factor=[],
    exp_frust=[],
    af_frust=[],
    evol_frust=[]
))

# (A2) ColumnDataSource for the scatter plots
# We'll store the raw (non-normalized) B-factor plus the min–max normalized frustrations.
source_scatter = ColumnDataSource(data=dict(
    x_bfactor=[],
    exp_frust_norm=[],
    af_frust_norm=[],
    evol_frust_norm=[]
))

# (B) The main figure
p = figure(
    title="(No Data)",
    sizing_mode='stretch_width',  # Make the plot stretch to full width
    height=600,                   # Adjust height as needed
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom", active_scroll="wheel_zoom"
)

# Define separate HoverTools for each metric
hover_bf = HoverTool(
    renderers=[],
    tooltips=[
        ("Index", "@x"),
        ("Residue", "@residue"),
        ("B-Factor (min–max)", "@b_factor")
    ],
    name="hover_b_factor"
)

hover_ef = HoverTool(
    renderers=[],
    tooltips=[
        ("Index", "@x"),
        ("Residue", "@residue"),
        ("ExpFrust (min–max)", "@exp_frust")
    ],
    name="hover_exp_frust"
)

hover_af = HoverTool(
    renderers=[],
    tooltips=[
        ("Index", "@x"),
        ("Residue", "@residue"),
        ("AFFrust (min–max)", "@af_frust")
    ],
    name="hover_af_frust"
)

hover_ev = HoverTool(
    renderers=[],
    tooltips=[
        ("Index", "@x"),
        ("Residue", "@residue"),
        ("EvolFrust (min–max)", "@evol_frust")
    ],
    name="hover_evol_frust"
)

p.add_tools(hover_bf, hover_ef, hover_af, hover_ev)

# Set axes labels
p.xaxis.axis_label = "Residue Index"
p.yaxis.axis_label = "Normalized Residue Flexibility"

# Add lines for each metric
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

# (B2) Three scatter plots: B-Factor (non-normalized) vs. each frustration (min–max)
p_scatter_exp = figure(
    title="B-factor vs ExpFrust",
    width=350, height=300,
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    x_axis_label="B-Factor (raw)",
    y_axis_label="ExpFrust (min–max)",
    match_aspect=False
)
p_scatter_exp.scatter(
    x="x_bfactor",
    y="exp_frust_norm",
    source=source_scatter,
    color="#2ca02c",
    alpha=0.6,
    size=6
)

p_scatter_af = figure(
    title="B-factor vs AFFrust",
    width=350, height=300,
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    x_axis_label="B-Factor (raw)",
    y_axis_label="AFFrust (min–max)",
    match_aspect=False
)
p_scatter_af.scatter(
    x="x_bfactor",
    y="af_frust_norm",
    source=source_scatter,
    color="#ff7f0e",
    alpha=0.6,
    size=6
)

p_scatter_evol = figure(
    title="B-factor vs EvolFrust",
    width=350, height=300,
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    x_axis_label="B-Factor (raw)",
    y_axis_label="EvolFrust (min–max)",
    match_aspect=False
)
p_scatter_evol.scatter(
    x="x_bfactor",
    y="evol_frust_norm",
    source=source_scatter,
    color="#d62728",
    alpha=0.6,
    size=6
)

# (C) SELECT widget to pick the test### to show
test_options = sorted(data_by_test.keys())
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
    td = select_test.value
    if td not in data_by_test:
        # Clear the sources
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        source_scatter.data = dict(x_bfactor=[], exp_frust_norm=[], af_frust_norm=[], evol_frust_norm=[])
        p.title.text = "(No Data)"
        return
    
    dfp = data_by_test[td]["df_plot"]      # smoothed + min–max
    dfo = data_by_test[td]["df_original"]  # raw data + raw frustrations + norms for scatter

    # For the main plot, drop rows with any NaN in B_Factor, ExpFrust, AFFrust, EvolFrust
    sub_plot = dfp.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    if sub_plot.empty:
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        p.title.text = f"{td} (No valid rows)."
    else:
        new_data_plot = dict(
            x=sub_plot["AlnIndex"].tolist(),
            residue=sub_plot["Residue"].tolist(),
            b_factor=sub_plot["B_Factor"].tolist(),
            exp_frust=sub_plot["ExpFrust"].tolist(),
            af_frust=sub_plot["AFFrust"].tolist(),
            evol_frust=sub_plot["EvolFrust"].tolist()
        )
        source_plot.data = new_data_plot
        p.title.text = f"{td} (Smoothed + min–max)"

    # For the scatter plots, we use raw B-factor and min–max frustrations (non-smoothed)
    sub_scat = dfo.dropna(subset=["B_Factor","ExpFrust_norm","AFFrust_norm","EvolFrust_norm"])
    if sub_scat.empty:
        source_scatter.data = dict(x_bfactor=[], exp_frust_norm=[], af_frust_norm=[], evol_frust_norm=[])
    else:
        new_data_scat = dict(
            x_bfactor=sub_scat["B_Factor"].tolist(),
            exp_frust_norm=sub_scat["ExpFrust_norm"].tolist(),
            af_frust_norm=sub_scat["AFFrust_norm"].tolist(),
            evol_frust_norm=sub_scat["EvolFrust_norm"].tolist()
        )
        source_scatter.data = new_data_scat

select_test.on_change("value", update_plot)

# Trigger the callback to populate everything with the initial selection
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
    combo_options = sorted(list({
        f"{row['MetricA']} vs {row['MetricB']}" 
        for _, row in df_all_corr.iterrows()
    }))
else:
    combo_options = []

cbg_tests = CheckboxButtonGroup(labels=tests_in_corr, active=[])
cbg_combos = CheckboxButtonGroup(labels=combo_options, active=[])

def update_corr_filter(attr, old, new):
    """Filter the correlation table based on selected tests and metric pairs."""
    if df_all_corr.empty:
        return
    selected_tests = [cbg_tests.labels[i] for i in cbg_tests.active]
    selected_combos = [cbg_combos.labels[i] for i in cbg_combos.active]
    
    if not selected_tests and not selected_combos:
        filtered = df_all_corr
    else:
        df_tmp = df_all_corr.copy()
        df_tmp["combo_str"] = df_tmp.apply(
            lambda r: f"{r['MetricA']} vs {r['MetricB']}", axis=1
        )
        if selected_tests and selected_combos:
            filtered = df_tmp[
                (df_tmp["Test"].isin(selected_tests)) &
                (df_tmp["combo_str"].isin(selected_combos))
            ].drop(columns=["combo_str"])
        elif selected_tests:
            filtered = df_tmp[
                df_tmp["Test"].isin(selected_tests)
            ].drop(columns=["combo_str"])
        elif selected_combos:
            filtered = df_tmp[
                df_tmp["combo_str"].isin(selected_combos)
            ].drop(columns=["combo_str"])
        else:
            filtered = df_all_corr

    source_corr.data = filtered.to_dict(orient="list")

cbg_tests.on_change("active", update_corr_filter)
cbg_combos.on_change("active", update_corr_filter)

# (G) Header
header = Div(text="""
    <h1>Evolutionary Frustration</h1>
    <p>
        Evolutionary frustration leverages multiple sequence alignment (MSA) derived coupling scores 
        and statistical potentials to calculate the mutational frustration of various proteins without 
        the need for protein structures. By benchmarking the evolutionary frustration metric against 
        experimental data (B-Factor) and two structure-based frustration metrics, we seek to validate 
        the efficacy of sequence derived evolutionary constraints in representing protein flexibility.
    </p>
    <p>
        The data displayed compare the agreement of different frustration metrics with the experimental 
        B-Factor, which measures the average positional uncertainty of each amino acid derived from 
        crystallographic data. The metrics include:
    </p>
    <ul>
        <li><strong>Experimental Frustration</strong>: Uses the Frustratometer web tool with an experimentally derived protein structure.</li>
        <li><strong>AF Frustration</strong>: Utilizes a sequence-derived protein structure generated by AlphaFold as the Frustratometer input.</li>
        <li><strong>Evolutionary Frustration</strong>: Derived from evolutionary constraints represented by coupling scores without compressing them into a single structure.</li>
    </ul>
    <p>
        The correlation table below presents Spearman correlation coefficients and p-values 
        between different metrics (non-smoothed data). The main plot uses a 
        <em>smoothed (window size = 5)</em> version of each metric and then applies 
        <em>min–max normalization</em>.
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
        <li><strong>Oscillation:</strong> Pressing <code>O</code> causes the protein ribbon to oscillate. The amplitude and frequency of oscillation are mapped to the average B-factor associated with each residue of the protein.</li>
        <li><strong>Color Representation:</strong> The color of the protein represents the experimental frustration of each residue, calculated by the Frustratometer and subsequently compressed into a per-residue metric. Light blue residues are minimally frustrated, while magenta residues are highly frustrated.</li>
        <li><strong>Luminosity:</strong> The luminosity of each residue represents the evolutionary frustration of that residue.</li>
        <li><strong>Fragmentation:</strong> Pressing <code>B</code> fragments the protein, and each fragment is plotted on a 3D plot representing the three aforementioned metrics.</li>
        <li><strong>Navigation Controls:</strong>
            <ul>
                <li><code>W</code>, <code>A</code>, <code>S</code>, <code>D</code>: Move the camera.</li>
                <li><code>Shift</code>: Move the camera down.</li>
                <li><code>Space</code>: Move the camera up.</li>
                <li>Hold <code>C</code>: Zoom the camera in.</li>
                <li>Hold <code>Left Control</code>: Increase movement speed.</li>
            </ul>
        </li>
        <li><strong>Folding Controls:</strong>
            <ul>
                <li><code>Q</code>: When the protein is in its folded state, pressing <code>Q</code> will unfold the protein.</li>
                <li><code>E</code>: When the protein is in its unfolded state, pressing <code>E</code> will refold the protein back to its original structure.</li>
                <li>In the unfolded state, the protein can be static or oscillating (controlled by <code>O</code>). In the oscillating state, B-factor is mapped by frequency and amplitude of oscillation. In the static state, B-factor is mapped by the height of each residue.</li>
            </ul>
        </li>
        <li><strong>Pause:</strong> Pressing <code>P</code> pauses the visualizer and allows you to select another protein.</li>
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
unity_iframe.visible = True

unity_container = column(description_visualizer, unity_iframe, sizing_mode='stretch_width')

# Combine the main line plot, the scatter plots, and the Unity viewer
visualization_section = column(
    select_test,
    p,
    row(p_scatter_exp, p_scatter_af, p_scatter_evol),  # New scatter plots in one row
    unity_container,
    sizing_mode='stretch_width',
    css_classes=['visualization-section']
)

# (F) Controls section
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