import os
import pandas as pd
import numpy as np
import re

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, CheckboxButtonGroup,
    DataTable, TableColumn, NumberFormatter, Div, HoverTool, Label, GlyphRenderer, 
    Spacer, Panel, Tabs, Slider
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from scipy.stats import spearmanr

###############################################################################
# 1) Configuration
###############################################################################
# Local data directory path
DATA_DIR = "summary_data"  # Directory containing the summary files

# No filename constraints: process any *.txt file with the required columns
FILE_PATTERN = r"^summary_.+\.txt$"  # Optional: can be removed if you want all *.txt files

# (Optional) Specify a default file to visualize on startup
# This should match the exact filename (e.g., "summary_1E6X.txt")
# Or set to "" to automatically select the first available file
DEFAULT_FILE = "summary_test001.txt"  # Change this to your preferred default (or "")

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
    Reads a summary_XXXX.txt with columns:
      AlnIndex, Residue, B_Factor, ExpFrust, AFFrust, EvolFrust
    Returns (df_original, df_for_plot), plus a correlation dict for non-smoothed data.
    Returns (None, None, {}) if parsing fails or required columns are missing.
    """
    required_cols = ["AlnIndex", "Residue", "B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]
    
    if not os.path.isfile(local_path):
        print(f"File not found: {local_path}")
        return None, None, {}
    
    try:
        df = pd.read_csv(local_path, sep='\t')
    except Exception as e:
        print(f"Skipping {local_path}: failed to parse tab-separated data. Error: {e}")
        return None, None, {}
    
    # Check if all required columns exist
    if not set(required_cols).issubset(df.columns):
        print(f"Skipping {local_path}: missing one or more required columns {required_cols}")
        return None, None, {}
    
    # Convert any 'n/a' to np.nan in numeric columns
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        df[col] = df[col].apply(lambda x: np.nan if str(x).lower() == 'n/a' else float(x))
    
    # Non-smoothed for correlation & scatter plots
    df_original = df.copy()
    
    # Smoothed for plotting
    df_for_plot = df.copy()
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr = df_for_plot[col].values
        df_for_plot[col] = moving_average(arr, window_size=5)
    
    # Skip min–max normalization if column is all-NaNs
    # or if min == max (avoid divide-by-zero).
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr = df_for_plot[col].values
        valid_mask = ~np.isnan(arr)
        if not np.any(valid_mask):
            continue  # All values are NaN; skip
        col_min = np.nanmin(arr)
        col_max = np.nanmax(arr)
        if col_max > col_min:
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
data_by_file = {}
all_corr_rows = []

# List all files in the data directory
for filename in os.listdir(DATA_DIR):
    # Optionally, enforce the FILE_PATTERN
    if not re.match(FILE_PATTERN, filename):
        print(f"Skipping {filename}: does not match pattern {FILE_PATTERN}")
        continue  # skip unrelated files
    
    file_path = os.path.join(DATA_DIR, filename)
    df_orig, df_plot, corrs = parse_summary_file(file_path)
    if df_orig is None:
        continue  # Parsing failed or columns missing
    
    data_by_file[filename] = {
        "df_original": df_orig,
        "df_plot": df_plot,
        "corrs": corrs
    }
    
    # Accumulate correlation info for a master table
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo
        all_corr_rows.append([filename, mA, mB, rho, pval])

# Build DataFrame of correlations
df_all_corr = pd.DataFrame(all_corr_rows, columns=["Test","MetricA","MetricB","Rho","Pval"])

###############################################################################
# 4) Bokeh Application
###############################################################################
source_plot = ColumnDataSource(data=dict(
    x=[],
    residue=[],
    b_factor=[],
    exp_frust=[],
    af_frust=[],
    evol_frust=[]
))

p = figure(
    title="(No Data)",
    sizing_mode='stretch_width',
    height=600,
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom", 
    active_scroll=None
)

# Define separate HoverTools for each metric
hover_bf = HoverTool(
    renderers=[],
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("B-Factor", "@b_factor")],
    name="hover_b_factor"
)
hover_ef = HoverTool(
    renderers=[],
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("ExpFrust", "@exp_frust")],
    name="hover_exp_frust"
)
hover_af = HoverTool(
    renderers=[],
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("AFFrust", "@af_frust")],
    name="hover_af_frust"
)
hover_ev = HoverTool(
    renderers=[],
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("EvolFrust", "@evol_frust")],
    name="hover_evol_frust"
)

p.add_tools(hover_bf, hover_ef, hover_af, hover_ev)
p.xaxis.axis_label = "Residue Index"
p.yaxis.axis_label = "Normalized Residue Flexibility / Frustration"

# Add lines
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

# --- Three scatter plots (NON-NORMALIZED) ---
# Scatter plots configuration with disabled wheel zoom by default
p_scatter_exp = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="B-Factor",
    y_axis_label="ExpFrust",
    tools=["pan", "box_zoom", "wheel_zoom", "reset", "save"],
    active_drag="box_zoom",
    active_scroll=None  # Disable wheel zoom by default
)
p_scatter_af = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="B-Factor",
    y_axis_label="AFFrust",
    tools=["pan", "box_zoom", "wheel_zoom", "reset", "save"],
    active_drag="box_zoom",
    active_scroll=None  # Disable wheel zoom by default
)
p_scatter_evol = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="B-Factor",
    y_axis_label="EvolFrust",
    tools=["pan", "box_zoom", "wheel_zoom", "reset", "save"],
    active_drag="box_zoom",
    active_scroll=None  # Disable wheel zoom by default
)

source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[]))
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[]))
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[]))

# Create Div elements for regression info
regression_info_exp = Div(
    text="", 
    styles={
        'background-color': '#f8f9fa',
        'padding': '10px',
        'border': '1px solid #ddd',
        'border-radius': '4px',
        'margin-top': '10px',
        'font-size': '14px',
        'text-align': 'center',
        'width': '100%'
    },
    sizing_mode="stretch_width"
)
regression_info_af = Div(
    text="",
    styles={
        'background-color': '#f8f9fa',
        'padding': '10px',
        'border': '1px solid #ddd',
        'border-radius': '4px',
        'margin-top': '10px',
        'font-size': '14px',
        'text-align': 'center',
        'width': '100%'
    },
    sizing_mode="stretch_width"
)
regression_info_evol = Div(
    text="",
    styles={
        'background-color': '#f8f9fa',
        'padding': '10px',
        'border': '1px solid #ddd',
        'border-radius': '4px',
        'margin-top': '10px',
        'font-size': '14px',
        'text-align': 'center',
        'width': '100%'
    },
    sizing_mode="stretch_width"
)

p_scatter_exp.scatter("x", "y", source=source_scatter_exp, color="#2ca02c", alpha=0.7)
p_scatter_af.scatter("x", "y", source=source_scatter_af,  color="#ff7f0e", alpha=0.7)
p_scatter_evol.scatter("x", "y", source=source_scatter_evol, color="#d62728", alpha=0.7)

def add_regression_line_and_info(fig, xvals, yvals, color="black", info_div=None):
    """Adds a linear regression line and updates the regression info Div."""
    if len(xvals) < 2 or np.all(xvals == xvals[0]):
        if info_div:
            info_div.text = "Insufficient data for regression"
        return
    
    not_nan = ~np.isnan(xvals) & ~np.isnan(yvals)
    if not any(not_nan):
        if info_div:
            info_div.text = "No valid data points"
        return
    
    xvals_clean = xvals[not_nan]
    yvals_clean = yvals[not_nan]
    if len(xvals_clean) < 2:
        if info_div:
            info_div.text = "Insufficient data for regression"
        return
    
    # Linear regression
    m, b = np.polyfit(xvals_clean, yvals_clean, 1)
    # Pearson correlation for display
    corr = np.corrcoef(xvals_clean, yvals_clean)[0, 1]
    
    # Plot line with a unique name for easy removal later
    xline = np.linspace(xvals_clean.min(), xvals_clean.max(), 100)
    yline = m*xline + b
    line_renderer = fig.line(
        xline, yline,
        line_width=2, line_dash='dashed', color=color,
        name="regression_line"
    )
    
    # Update regression info div with equation format
    if info_div:
        info_div.text = f"""
        <div style='color: {color}'>
            <strong>y = {m:.3f}x + {b:.3f}</strong><br>
            <span style='font-size: 12px'>N = {len(xvals_clean)} points</span>
        </div>
        """

# Dropdown select
file_options = sorted(data_by_file.keys())
if DEFAULT_FILE and DEFAULT_FILE in file_options:
    initial_file = DEFAULT_FILE
elif file_options:
    initial_file = file_options[0]
else:
    initial_file = ""

select_file = Select(
    title="Select Protein (summary_XXXX.txt):",
    value=initial_file,
    options=file_options
)

# Add slider for moving average window size
window_slider = Slider(
    start=1, 
    end=21, 
    value=5, 
    step=2, 
    title="Moving Average Window Size",
    width=400
)

def update_moving_average(attr, old, new):
    """Update plot when slider value changes"""
    update_plot(None, None, select_file.value)

window_slider.on_change('value', update_moving_average)

def update_plot(attr, old, new):
    """
    Updates both the main plot and scatter plots when a new file is selected.
    """
    filename = select_file.value
    if filename not in data_by_file:
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        source_scatter_exp.data = dict(x=[], y=[])
        source_scatter_af.data = dict(x=[], y=[])
        source_scatter_evol.data = dict(x=[], y=[])
        p.title.text = "(No Data)"
        p_scatter_exp.title.text = ""
        p_scatter_af.title.text = ""
        p_scatter_evol.title.text = ""
        regression_info_exp.text = ""
        regression_info_af.text = ""
        regression_info_evol.text = ""
        return
    
    # Get window size from slider
    window_size = window_slider.value
    
    # Update main line plot with new window size
    df_orig = data_by_file[filename]["df_original"]
    df_plot = df_orig.copy()
    
    # Apply moving average with current window size
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr = df_plot[col].values
        df_plot[col] = moving_average(arr, window_size=window_size)
    
    # Normalize the smoothed data
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr = df_plot[col].values
        valid_mask = ~np.isnan(arr)
        if not np.any(valid_mask):
            continue
        col_min = np.nanmin(arr)
        col_max = np.nanmax(arr)
        if col_max > col_min:
            df_plot[col] = (arr - col_min) / (col_max - col_min)
    
    sub_plot = df_plot.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    if sub_plot.empty:
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        p.title.text = f"{filename} (No valid rows)."
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
        p.title.text = f"{filename} (Smoothed + Normalized)"
    
    # Update scatter plots (using NON-smoothed data)
    df_orig = data_by_file[filename]["df_original"]
    sub_orig = df_orig.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    
    # For each scatter figure, remove old regression lines
    for fig in [p_scatter_exp, p_scatter_af, p_scatter_evol]:
        old_lines = fig.select({'type': GlyphRenderer, 'name': 'regression_line'})
        for line in old_lines:
            if line in fig.renderers:
                fig.renderers.remove(line)

    if sub_orig.empty:
        source_scatter_exp.data = dict(x=[], y=[])
        source_scatter_af.data = dict(x=[], y=[])
        source_scatter_evol.data = dict(x=[], y=[])
        p_scatter_exp.title.text = f"{filename} (No Data)"
        p_scatter_af.title.text = f"{filename} (No Data)"
        p_scatter_evol.title.text = f"{filename} (No Data)"
        regression_info_exp.text = ""
        regression_info_af.text = ""
        regression_info_evol.text = ""
    else:
        # ExpFrust
        x_exp = sub_orig["B_Factor"].values
        y_exp = sub_orig["ExpFrust"].values
        source_scatter_exp.data = dict(x=x_exp, y=y_exp)
        p_scatter_exp.title.text = f"{filename} Experimental Frustration"
        add_regression_line_and_info(p_scatter_exp, x_exp, y_exp, color="#2ca02c", info_div=regression_info_exp)
        
        # AFFrust
        x_af = sub_orig["B_Factor"].values
        y_af = sub_orig["AFFrust"].values
        source_scatter_af.data = dict(x=x_af, y=y_af)
        p_scatter_af.title.text = f"{filename} AF Frustration"
        add_regression_line_and_info(p_scatter_af, x_af, y_af, color="#ff7f0e", info_div=regression_info_af)
        
        # EvolFrust
        x_evol = sub_orig["B_Factor"].values
        y_evol = sub_orig["EvolFrust"].values
        source_scatter_evol.data = dict(x=x_evol, y=y_evol)
        p_scatter_evol.title.text = f"{filename} Evolutionary Frustration"
        add_regression_line_and_info(p_scatter_evol, x_evol, y_evol, color="#d62728", info_div=regression_info_evol)

select_file.on_change("value", update_plot)
if initial_file:
    update_plot(None, None, initial_file)

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

# (E) FILTERS for correlation table
tests_in_corr = sorted(df_all_corr["Test"].unique()) if not df_all_corr.empty else []
if not df_all_corr.empty:
    combo_options = sorted({
        f"{row['MetricA']} vs {row['MetricB']}" 
        for _, row in df_all_corr.iterrows()
    })
else:
    combo_options = []

cbg_tests = CheckboxButtonGroup(
    labels=tests_in_corr,
    active=[]
)
cbg_combos = CheckboxButtonGroup(
    labels=combo_options,
    active=[]
)

def update_corr_filter(attr, old, new):
    """Filter correlation table based on selected tests and metric pairs."""
    if df_all_corr.empty:
        return
    selected_tests = [cbg_tests.labels[i] for i in cbg_tests.active]
    selected_combos = [cbg_combos.labels[i] for i in cbg_combos.active]
    
    if not selected_tests and not selected_combos:
        filtered = df_all_corr
    else:
        df_tmp = df_all_corr.copy()
        df_tmp["combo_str"] = df_tmp.apply(lambda r: f"{r['MetricA']} vs {r['MetricB']}", axis=1)
        
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

# Create columns for each scatter plot and its regression info with minimum width
scatter_col_exp = column(
    p_scatter_exp, 
    regression_info_exp, 
    sizing_mode="stretch_width",
    styles={'flex': '1 1 350px', 'min-width': '350px'}
)
scatter_col_af = column(
    p_scatter_af, 
    regression_info_af, 
    sizing_mode="stretch_width",
    styles={'flex': '1 1 350px', 'min-width': '350px'}
)
scatter_col_evol = column(
    p_scatter_evol, 
    regression_info_evol, 
    sizing_mode="stretch_width",
    styles={'flex': '1 1 350px', 'min-width': '350px'}
)

# Update scatter plots row with flex layout and minimum widths
scatter_row = row(
    scatter_col_exp,
    scatter_col_af,
    scatter_col_evol,
    sizing_mode="stretch_width",
    styles={
        'display': 'flex', 
        'justify-content': 'space-between', 
        'gap': '20px',
        'width': '100%',
        'margin': '0 auto',
        'flex-wrap': 'wrap'
    }
)

# Add header and description
header = Div(text="""
    <h1>Evolutionary Frustration</h1>
    <p>
        Evolutionary frustration leverages multiple sequence alignment (MSA) derived coupling scores 
        and statistical potentials to calculate the mutational frustration of various proteins without the need for protein structures. 
        By benchmarking the evolutionary frustration metric against experimental data (B-Factor) and two structure-based metrics, 
        we aim to validate sequence-derived evolutionary constraints in representing protein flexibility.
    </p>
    <ul>
        <li><strong>Experimental Frustration</strong>: Derived via the Frustratometer using a crystal structure.</li>
        <li><strong>AF Frustration</strong>: Derived via the Frustratometer using an AlphaFold structure.</li>
        <li><strong>Evolutionary Frustration</strong>: Derived directly from sequence alignment (no structure needed).</li>
    </ul>
    <p>
        The correlation table below shows Spearman correlation coefficients and p-values for <em>non-smoothed</em> data. 
        The curves in the main plot are <em>smoothed</em> with a simple moving average and 
        <strong>min–max normalized</strong> (per protein). Normalization does not affect Spearman correlations but be mindful 
        that min–max scaling is not suitable for comparing magnitudes <em>across</em> proteins.
    </p>
    <h3>Contributors</h3>
    <p>
        <strong>Adam Kuhn<sup>1,2,3,4</sup>, Vinícius Contessoto<sup>4</sup>, 
        George N Phillips Jr.<sup>2,3</sup>, José Onuchic<sup>1,2,3,4</sup></strong><br>
        <sup>1</sup>Department of Physics, Rice University<br>
        <sup>2</sup>Department of Chemistry, Rice University<br>
        <sup>3</sup>Department of Biosciences, Rice University<br>
        <sup>4</sup>Center for Theoretical Biological Physics, Rice University
    </p>
""", sizing_mode='stretch_width', styles={'margin-bottom': '20px'})

# Unity Container
description_visualizer = Div(text="""
    <h2>Protein Visualizer Instructions</h2>
    <p>
        The protein visualizer allows you to interact with the protein structure using various controls and visual metrics:
    </p>
    <ul>
        <li><strong>Oscillation (O):</strong> Ribbon oscillates with amplitude/frequency mapped to average B-factor.</li>
        <li><strong>Color (ExpFrust):</strong> Ribbon color indicates experimental frustration.</li>
        <li><strong>Luminosity (EvolFrust):</strong> Indicates evolutionary frustration.</li>
        <li><strong>Fragmentation (B):</strong> Splits the protein into fragments for 3D frustration plotting.</li>
        <li><strong>Navigation:</strong> <code>W/A/S/D</code>, <code>Shift</code>, <code>Space</code> to move the camera, <code>C</code> to zoom, etc.</li>
        <li><strong>Folding (Q/E):</strong> Unfold/fold the protein. <code>O</code> toggles oscillation or sets height by B-factor in the unfolded state.</li>
        <li><strong>Pause (P):</strong> Pauses the scene so you can select another protein.</li>
    </ul>
""", sizing_mode='stretch_width', styles={'margin-bottom': '20px'})

unity_iframe = Div(
    text="""
    <div style="width: 100%; display: flex; justify-content: center; align-items: center; margin: 20px 0;">
        <iframe 
            src="https://igotintogradschool2025.site/unity/"
            style="width: 95vw; height: 90vh; border: 2px solid #ddd; border-radius: 8px; 
                   box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
    </div>
    """,
    sizing_mode='stretch_width',
    styles={'margin-top': '20px'}
)
unity_iframe.visible = True

unity_container = column(
    description_visualizer,
    unity_iframe,
    sizing_mode='stretch_width'
)

# Controls section
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

# Custom styles
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

# Main layout with slider
visualization_section = column(
    select_file,
    window_slider,
    p,
    scatter_row,
    unity_container,
    sizing_mode='stretch_width',
    css_classes=['visualization-section']
)

main_layout = column(
    custom_styles,
    header,
    visualization_section,
    controls_section,
    data_table,
    sizing_mode='stretch_width'
)

# Set up document
curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration"

###############################################################################
# 5) CORRELATION TABLE AND FILTERS
###############################################################################

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

# (E) FILTERS for correlation table
tests_in_corr = sorted(df_all_corr["Test"].unique()) if not df_all_corr.empty else []
if not df_all_corr.empty:
    combo_options = sorted({
        f"{row['MetricA']} vs {row['MetricB']}" 
        for _, row in df_all_corr.iterrows()
    })
else:
    combo_options = []

cbg_tests = CheckboxButtonGroup(
    labels=tests_in_corr,
    active=[]
)
cbg_combos = CheckboxButtonGroup(
    labels=combo_options,
    active=[]
)

def update_corr_filter(attr, old, new):
    """Filter correlation table based on selected tests and metric pairs."""
    if df_all_corr.empty:
        return
    selected_tests = [cbg_tests.labels[i] for i in cbg_tests.active]
    selected_combos = [cbg_combos.labels[i] for i in cbg_combos.active]
    
    if not selected_tests and not selected_combos:
        filtered = df_all_corr
    else:
        df_tmp = df_all_corr.copy()
        df_tmp["combo_str"] = df_tmp.apply(lambda r: f"{r['MetricA']} vs {r['MetricB']}", axis=1)
        
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

###############################################################################
# End of Script
###############################################################################