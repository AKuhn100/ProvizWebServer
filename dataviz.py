# -*- coding: utf-8 -*-
"""
Bokeh application for visualizing protein flexibility (B-Factor) and various
frustration metrics (Experimental, AlphaFold-based, Evolutionary).

Simplified layout focusing on: Unity Visualizer, Line Plot per PDB ID,
Scatter Plots per PDB ID, and Correlation Table.

Color Scheme: B-Factor (Grey), ExpFrust (Red), AFFrust (Blue), EvolFrust (Green).
"""
import os
import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, linregress
# Removed unused math import (pi)

# Bokeh imports
from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, Div, Spacer, # Removed CheckboxGroup, Slider
    DataTable, TableColumn, NumberFormatter, HoverTool,
    GlyphRenderer, Range1d # Removed Whisker, Label
)
from bokeh.plotting import figure
from bokeh.layouts import column, row # Removed layout, jitter not needed
from bokeh.palettes import Category10 # Still used for initial colors, but overridden

###############################################################################
# 1) Configuration
###############################################################################
# Local data directory path
DATA_DIR = "summary_data"

# Updated FILE_PATTERN to capture the 4-char ID in group 1
FILE_PATTERN = r"^summary_([A-Za-z0-9]{4})\.txt$"

# Default PDB ID (4-char) - leave blank for automatic selection
DEFAULT_PDB_ID = ""

# Define FIXED moving average window size
FIXED_WINDOW_SIZE = 5

# Define New Color Scheme
COLOR_SCHEME = {
    "B_Factor": "#7f7f7f",   # Grey
    "ExpFrust": "#d62728",   # Red
    "AFFrust":  "#1f77b4",   # Blue
    "EvolFrust":"#2ca02c"    # Green
}

###############################################################################
# 2) Helpers: Data Parsing and Aggregation
###############################################################################

# Helper function to extract PDB ID
def extract_pdb_id(filename):
    """Extracts the 4-character PDB ID from the filename using FILE_PATTERN."""
    match = re.match(FILE_PATTERN, filename)
    if match:
        return match.group(1) # Return the captured group (the PDB ID)
    return None

def moving_average(arr, window_size=5):
    """
    Computes a simple moving average on a float array.
    Returns an equally sized array with np.nan where insufficient data exists.
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
    Parses a summary file and returns the original DataFrame and correlations.
    """
    required_cols = ["AlnIndex", "Residue", "B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]

    if not os.path.isfile(local_path):
        print(f"File not found: {local_path}")
        return None, {}

    try:
        df = pd.read_csv(local_path, sep='\t')
    except Exception as e:
        print(f"Skipping {local_path}: failed to parse data. Error: {e}")
        return None, {}

    if not set(required_cols).issubset(df.columns):
        print(f"Skipping {local_path}: missing required columns.")
        return None, {}

    process_cols = ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]
    for col in process_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_original = df.copy()

    # Compute Spearman correlations on original data
    corrs = {}
    sub = df_original.dropna(subset=process_cols)
    if not sub.empty:
        combos = [
            ("B_Factor", "ExpFrust"), ("B_Factor", "AFFrust"), ("B_Factor", "EvolFrust"),
            ("ExpFrust", "AFFrust"), ("ExpFrust", "EvolFrust"), ("AFFrust",  "EvolFrust"),
        ]
        for (mA, mB) in combos:
            if sub[mA].nunique() < 2 or sub[mB].nunique() < 2:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(sub[mA], sub[mB])
            corrs[(mA, mB)] = (rho, pval)

    return df_original, corrs

def remove_regression_renderers(fig):
    """
    Removes all renderers and associated hover tools whose names start with 'regression_'.
    """
    renderers_to_keep = []
    renderers_to_remove_names = set()
    for r in fig.renderers:
        renderer_name = getattr(r, 'name', None)
        if isinstance(renderer_name, str) and renderer_name.startswith('regression_'):
            renderers_to_remove_names.add(renderer_name)
        else:
            renderers_to_keep.append(r)

    tools_to_keep = []
    for tool in fig.tools:
        tool_name = getattr(tool, 'name', None)
        is_regression_tool = isinstance(tool_name, str) and tool_name.startswith('regression_')
        targets_removed_renderer = False
        if isinstance(tool, HoverTool) and tool.renderers:
            for r in tool.renderers:
                 renderer_name = getattr(r, 'name', None)
                 if renderer_name in renderers_to_remove_names:
                     targets_removed_renderer = True
                     break
        if targets_removed_renderer or is_regression_tool:
            continue
        tools_to_keep.append(tool)

    fig.renderers = renderers_to_keep
    fig.tools = tools_to_keep


###############################################################################
# 3) Load and Aggregate Data from Local Directory
###############################################################################
# Use PDB ID as the key
data_by_id = {}
all_corr_rows = []

# Aggregation lists - only needed for correlation table now
# pdb_ids = [] # Not strictly needed anymore, can get from data_by_id.keys()

# Iterate through files
found_files = 0
skipped_files = 0
print(f"Loading data from: {DATA_DIR}")
for filename in os.listdir(DATA_DIR):
    pdb_id = extract_pdb_id(filename)
    if not pdb_id:
        skipped_files += 1
        continue

    file_path = os.path.join(DATA_DIR, filename)
    df_orig, corrs = parse_summary_file(file_path)
    if df_orig is None:
        skipped_files += 1
        continue

    found_files += 1
    # Store data using PDB ID as the key
    data_by_id[pdb_id] = {
        "df_original": df_orig,
        "corrs": corrs
    }

    # Collect correlation data using PDB ID
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo
        all_corr_rows.append([pdb_id, mA, mB, rho, pval])

print(f"Found and processed {found_files} files matching pattern.")
if skipped_files > 0:
    print(f"Skipped {skipped_files} files (pattern mismatch or parsing error).")

# Correlation DataFrame - use PDB_ID column name
df_all_corr = pd.DataFrame(all_corr_rows, columns=["PDB_ID","MetricA","MetricB","Rho","Pval"])

# Removed aggregation logic for data_proviz and melted dataframes as plots are removed

###############################################################################
# 4) Bokeh Application Components
###############################################################################

# (A) Main Plot: Smoothed + Normalized Data
source_plot = ColumnDataSource(data=dict(
    x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[]
))

p = figure(
    title="(No PDB ID Selected)",
    sizing_mode='stretch_width', height=600,
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom", active_scroll="wheel_zoom"
)

# Define HoverTools
hover_bf = HoverTool(renderers=[], tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. B-Factor", "@b_factor{0.3f}")], name="hover_b_factor")
hover_ef = HoverTool(renderers=[], tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. ExpFrust", "@exp_frust{0.3f}")], name="hover_exp_frust")
hover_af = HoverTool(renderers=[], tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. AFFrust", "@af_frust{0.3f}")], name="hover_af_frust")
hover_ev = HoverTool(renderers=[], tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. EvolFrust", "@evol_frust{0.3f}")], name="hover_evol_frust")

p.add_tools(hover_bf, hover_ef, hover_af, hover_ev)
p.xaxis.axis_label = "Residue Index"
p.yaxis.axis_label = f"Smoothed (Window={FIXED_WINDOW_SIZE}) & Normalized Value" # Updated label

# Add lines using the new COLOR_SCHEME
# Map internal keys to display labels and colors
line_plot_config = {
    "b_factor":  ("B-Factor", COLOR_SCHEME["B_Factor"]),
    "exp_frust": ("ExpFrust", COLOR_SCHEME["ExpFrust"]),
    "af_frust":  ("AFFrust", COLOR_SCHEME["AFFrust"]),
    "evol_frust":("EvolFrust", COLOR_SCHEME["EvolFrust"])
}
renderers = {}
for col_key, (label, color) in line_plot_config.items():
    renderer = p.line(
        x="x", y=col_key, source=source_plot, line_width=2, alpha=0.8, color=color, legend_label=label
    )
    renderers[col_key] = renderer
    # Link hover tools
    if col_key == "b_factor": hover_bf.renderers.append(renderer)
    elif col_key == "exp_frust": hover_ef.renderers.append(renderer)
    elif col_key == "af_frust": hover_af.renderers.append(renderer)
    elif col_key == "evol_frust": hover_ev.renderers.append(renderer)

p.legend.location = "top_left"
p.legend.title = "Metrics"
p.legend.click_policy = "hide"

# (B) Scatter Plots (Experimental, AF, Evolutionary Frustration)
common_scatter_tools = ["pan", "box_zoom", "wheel_zoom", "reset", "save"]
# Use new colors in titles/labels if desired, or keep generic
p_scatter_exp = figure(
    sizing_mode="stretch_both", aspect_ratio=1, min_width=350, min_height=350,
    title="", x_axis_label="Normalized B-Factor", y_axis_label="Normalized ExpFrust",
    tools=common_scatter_tools, active_drag="box_zoom", active_scroll=None
)
p_scatter_af = figure(
    sizing_mode="stretch_both", aspect_ratio=1, min_width=350, min_height=350,
    title="", x_axis_label="Normalized B-Factor", y_axis_label="Normalized AFFrust",
    tools=common_scatter_tools, active_drag="box_zoom", active_scroll=None
)
p_scatter_evol = figure(
    sizing_mode="stretch_both", aspect_ratio=1, min_width=350, min_height=350,
    title="", x_axis_label="Normalized B-Factor", y_axis_label="Normalized EvolFrust",
    tools=common_scatter_tools, active_drag="box_zoom", active_scroll=None
)

# ColumnDataSources
source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[]))
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[]))
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[]))

# Regression info Divs
div_styles = {
    'background-color': '#f8f9fa', 'padding': '10px', 'border': '1px solid #ddd',
    'border-radius': '4px', 'margin-top': '10px', 'font-size': '14px',
    'text-align': 'center', 'width': '100%'
}
regression_info_exp = Div(text="", styles=div_styles, sizing_mode="stretch_width")
regression_info_af = Div(text="", styles=div_styles, sizing_mode="stretch_width")
regression_info_evol = Div(text="", styles=div_styles, sizing_mode="stretch_width")

# Scatter glyphs using new COLOR_SCHEME
scatter_exp_renderer = p_scatter_exp.scatter("x", "y", source=source_scatter_exp, color=COLOR_SCHEME["ExpFrust"], alpha=0.6, size=6)
scatter_af_renderer = p_scatter_af.scatter("x", "y", source=source_scatter_af,  color=COLOR_SCHEME["AFFrust"], alpha=0.6, size=6)
scatter_evol_renderer = p_scatter_evol.scatter("x", "y", source=source_scatter_evol, color=COLOR_SCHEME["EvolFrust"], alpha=0.6, size=6)

# Add HoverTools for scatter points
scatter_hover_tooltips = [
    ("Index", "@index"), ("Residue", "@residue"),
    ("Orig. B-Factor", "@x_orig{0.3f}"), ("Orig. Frust", "@y_orig{0.3f}"),
    ("Norm. B-Factor", "@x{0.3f}"), ("Norm. Frust", "@y{0.3f}"),
]
p_scatter_exp.add_tools(HoverTool(renderers=[scatter_exp_renderer], tooltips=scatter_hover_tooltips))
p_scatter_af.add_tools(HoverTool(renderers=[scatter_af_renderer], tooltips=scatter_hover_tooltips))
p_scatter_evol.add_tools(HoverTool(renderers=[scatter_evol_renderer], tooltips=scatter_hover_tooltips))


def add_regression_line_and_info(fig, xvals, yvals, color="black", info_div=None, plot_type=""):
    """
    Adds a linear regression line and updates the regression info Div.
    Uses the provided color.
    """
    not_nan_mask = ~np.isnan(xvals) & ~np.isnan(yvals)
    xvals_clean = xvals[not_nan_mask]
    yvals_clean = yvals[not_nan_mask]

    if len(xvals_clean) < 2 or np.all(xvals_clean == xvals_clean[0]) or np.all(yvals_clean == yvals_clean[0]):
        if info_div: info_div.text = "<i style='color: gray;'>Insufficient data or variance for regression</i>"
        return

    try:
        slope, intercept, r_value, p_value, std_err = linregress(xvals_clean, yvals_clean)
    except ValueError as e:
         if info_div: info_div.text = f"<i style='color: red;'>Regression Error: {e}</i>"
         return

    x_min, x_max = np.min(xvals_clean), np.max(xvals_clean)
    if x_min == x_max:
         if info_div: info_div.text = "<i style='color: gray;'>Insufficient variance for regression line</i>"
         return
    x_range = np.linspace(x_min, x_max, 100)
    y_range = slope * x_range + intercept
    regression_line_name = f'regression_line_{plot_type}'
    regression_line_renderer = fig.line(
        x_range, y_range, line_width=2, line_dash='dashed', color=color, name=regression_line_name
    )

    # Add Hover Tool for the Regression Line
    hover_regression = HoverTool(
        renderers=[regression_line_renderer],
        tooltips=[
            ("Regression Equation", f"y = {slope:.3f}x + {intercept:.3f}"),
            ("R-squared", f"{r_value**2:.3f}"),
            ("p-value", f"{p_value:.2e}")
        ],
        mode='mouse', name=f'regression_hover_tool_{plot_type}'
    )
    existing_tool_names = [getattr(t, 'name', None) for t in fig.tools]
    if hover_regression.name not in existing_tool_names:
        fig.add_tools(hover_regression)

    # Update regression info div - use the provided color for the text
    if info_div:
        info_div.text = f"""
        <div style='color: {color}'>
            <strong>y = {slope:.3f}x + {intercept:.3f}</strong><br>
            <span style='font-size: 12px'>R² = {r_value**2:.3f} | p = {p_value:.2e}</span>
        </div>
        """

# Dropdown select - use PDB IDs
pdb_id_options = sorted(data_by_id.keys())
if DEFAULT_PDB_ID and DEFAULT_PDB_ID in pdb_id_options:
    initial_pdb_id = DEFAULT_PDB_ID
elif pdb_id_options:
    initial_pdb_id = pdb_id_options[0]
else:
    initial_pdb_id = ""

select_pdb_id = Select(
    title="Select PDB ID:", value=initial_pdb_id, options=pdb_id_options, width=200 # Reduced width
)

# Removed Slider and its callback (update_moving_average)

def min_max_normalize(arr):
    """Applies min-max normalization, handles NaNs and division by zero."""
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if np.isnan(arr_min) or np.isnan(arr_max): return np.full_like(arr, np.nan)
    if arr_max > arr_min:
        norm_arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        norm_arr = np.zeros_like(arr)
        norm_arr[~np.isnan(arr)] = 0.5
    norm_arr[np.isnan(arr)] = np.nan
    return norm_arr

def update_plot(attr, old, new):
    """Updates plots when a new PDB ID is selected."""
    pdb_id = select_pdb_id.value
    # window_size is now fixed
    window_size = FIXED_WINDOW_SIZE

    print(f"Updating plots for PDB ID: {pdb_id}, Window Size: {window_size}")

    if not pdb_id or pdb_id not in data_by_id:
        # Clear plots
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        source_scatter_exp.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        source_scatter_af.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        source_scatter_evol.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        p.title.text = "(No PDB ID Selected)"
        p_scatter_exp.title.text = ""
        p_scatter_af.title.text = ""
        p_scatter_evol.title.text = ""
        regression_info_exp.text = "<i style='color: gray;'>No PDB ID selected</i>"
        regression_info_af.text = "<i style='color: gray;'>No PDB ID selected</i>"
        regression_info_evol.text = "<i style='color: gray;'>No PDB ID selected</i>"
        remove_regression_renderers(p_scatter_exp)
        remove_regression_renderers(p_scatter_af)
        remove_regression_renderers(p_scatter_evol)
        return

    # --- Update Main Line Plot ---
    df_orig = data_by_id[pdb_id]["df_original"]
    df_plot = df_orig.copy()
    metrics_to_process = ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]
    for col in metrics_to_process:
        if col in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[col]):
            df_plot[col] = moving_average(df_plot[col].values, window_size=window_size)
        else:
            df_plot[col] = np.nan
    for col in metrics_to_process:
        if col in df_plot.columns:
            df_plot[col] = min_max_normalize(df_plot[col].values)

    new_data = dict(
        x=df_plot["AlnIndex"].tolist(), residue=df_plot["Residue"].tolist(),
        b_factor=df_plot["B_Factor"].tolist(), exp_frust=df_plot["ExpFrust"].tolist(),
        af_frust=df_plot["AFFrust"].tolist(), evol_frust=df_plot["EvolFrust"].tolist()
    )
    source_plot.data = new_data
    p.title.text = f"PDB ID: {pdb_id} (Smoothed Window={window_size}, Normalized)"

    # --- Update Scatter Plots ---
    df_scatter_base = data_by_id[pdb_id]["df_original"].copy()
    remove_regression_renderers(p_scatter_exp)
    remove_regression_renderers(p_scatter_af)
    remove_regression_renderers(p_scatter_evol)

    x_norm = min_max_normalize(df_scatter_base["B_Factor"].values)
    y_exp_norm = min_max_normalize(df_scatter_base["ExpFrust"].values)
    y_af_norm = min_max_normalize(df_scatter_base["AFFrust"].values)
    y_evol_norm = min_max_normalize(df_scatter_base["EvolFrust"].values)

    indices = df_scatter_base["AlnIndex"].tolist()
    residues = df_scatter_base["Residue"].tolist()
    x_orig = df_scatter_base["B_Factor"].tolist()
    y_exp_orig = df_scatter_base["ExpFrust"].tolist()
    y_af_orig = df_scatter_base["AFFrust"].tolist()
    y_evol_orig = df_scatter_base["EvolFrust"].tolist()

    valid_exp = ~np.isnan(x_norm) & ~np.isnan(y_exp_norm)
    valid_af = ~np.isnan(x_norm) & ~np.isnan(y_af_norm)
    valid_evol = ~np.isnan(x_norm) & ~np.isnan(y_evol_norm)

    # Update ExpFrust Scatter
    if np.any(valid_exp):
        source_scatter_exp.data = dict(x=x_norm, y=y_exp_norm, x_orig=x_orig, y_orig=y_exp_orig, residue=residues, index=indices)
        p_scatter_exp.title.text = f"{pdb_id}: B-Factor vs ExpFrust"
        add_regression_line_and_info(
            fig=p_scatter_exp, xvals=x_norm, yvals=y_exp_norm,
            color=COLOR_SCHEME["ExpFrust"], info_div=regression_info_exp, plot_type="exp" # Use new color
        )
    else:
        source_scatter_exp.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        p_scatter_exp.title.text = f"{pdb_id}: B-Factor vs ExpFrust (No Valid Data)"
        regression_info_exp.text = "<i style='color: gray;'>No valid data points</i>"

    # Update AFFrust Scatter
    if np.any(valid_af):
        source_scatter_af.data = dict(x=x_norm, y=y_af_norm, x_orig=x_orig, y_orig=y_af_orig, residue=residues, index=indices)
        p_scatter_af.title.text = f"{pdb_id}: B-Factor vs AFFrust"
        add_regression_line_and_info(
            fig=p_scatter_af, xvals=x_norm, yvals=y_af_norm,
            color=COLOR_SCHEME["AFFrust"], info_div=regression_info_af, plot_type="af" # Use new color
        )
    else:
        source_scatter_af.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        p_scatter_af.title.text = f"{pdb_id}: B-Factor vs AFFrust (No Valid Data)"
        regression_info_af.text = "<i style='color: gray;'>No valid data points</i>"

    # Update EvolFrust Scatter
    if np.any(valid_evol):
        source_scatter_evol.data = dict(x=x_norm, y=y_evol_norm, x_orig=x_orig, y_orig=y_evol_orig, residue=residues, index=indices)
        p_scatter_evol.title.text = f"{pdb_id}: B-Factor vs EvolFrust"
        add_regression_line_and_info(
            fig=p_scatter_evol, xvals=x_norm, yvals=y_evol_norm,
            color=COLOR_SCHEME["EvolFrust"], info_div=regression_info_evol, plot_type="evol" # Use new color
        )
    else:
        source_scatter_evol.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        p_scatter_evol.title.text = f"{pdb_id}: B-Factor vs EvolFrust (No Valid Data)"
        regression_info_evol.text = "<i style='color: gray;'>No valid data points</i>"

select_pdb_id.on_change("value", update_plot)
if initial_pdb_id:
    update_plot(None, None, initial_pdb_id)

###############################################################################
# 5) CORRELATION TABLE
###############################################################################
# (D) CORRELATION TABLE (Filters Removed)
if df_all_corr.empty:
    columns = [
        TableColumn(field="PDB_ID", title="PDB ID"), TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"), TableColumn(field="Rho", title="Spearman Rho"),
        TableColumn(field="Pval", title="p-value")
    ]
    source_corr = ColumnDataSource(dict(PDB_ID=[], MetricA=[], MetricB=[], Rho=[], Pval=[]))
else:
    source_corr = ColumnDataSource(df_all_corr)
    columns = [
        TableColumn(field="PDB_ID", title="PDB ID"), TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Spearman Rho", formatter=NumberFormatter(format="0.3f")),
        TableColumn(field="Pval", title="p-value", formatter=NumberFormatter(format="0.2e"))
    ]
# Removed width=1200, rely on stretch_width
data_table = DataTable(columns=columns, source=source_corr, height=400, sizing_mode='stretch_width', index_position=None)

# (E) FILTERS REMOVED
# Removed CheckboxGroup creation, layout (controls_layout), and update_corr_filter function

###############################################################################
# 6) Additional Aggregated Plots REMOVED
###############################################################################
# Removed p_avg_plot, p_std_plot, p_corr_plot, bar_plot_instance creation and layout (additional_plots)
# Removed create_bar_plot_with_sd function

###############################################################################
# 7) User Interface Components and Final Layout
###############################################################################

# Consolidated Header Text
header = Div(text=f"""
    <h1>Evolutionary Frustration Analysis</h1>
    <p>
        This dashboard visualizes protein flexibility (B-Factor, <span style='color:{COLOR_SCHEME["B_Factor"]};'><b>Grey</b></span>)
        and compares it with three 'frustration' metrics:
        Experimental (<span style='color:{COLOR_SCHEME["ExpFrust"]};'><b>Red</b></span>),
        AlphaFold-based (<span style='color:{COLOR_SCHEME["AFFrust"]};'><b>Blue</b></span>), and
        Evolutionary (<span style='color:{COLOR_SCHEME["EvolFrust"]};'><b>Green</b></span>).
        Select a PDB ID from the dropdown menu to view its specific data.
    </p>
    <ul>
        <li>The <strong>3D Visualizer</strong> below provides an interactive view (external application).</li>
        <li>The <strong>Line Plot</strong> shows smoothed (window={FIXED_WINDOW_SIZE}) and normalized metrics along the sequence.</li>
        <li>The <strong>Scatter Plots</strong> show correlations between normalized B-Factor and each frustration metric (non-smoothed data points).</li>
        <li>The <strong>Correlation Table</strong> shows non-smoothed Spearman correlations for all loaded PDB IDs.</li>
    </ul>
    <hr>
    """, sizing_mode='stretch_width', styles={'margin-bottom': '15px'})

# Unity Container (Description slightly simplified)
description_visualizer = Div(text="""
    <h2>3D Protein Visualizer (External Unity App)</h2>
    <p>Interact with the protein structure. Controls may include: Oscillation (O), Color (ExpFrust), Luminosity (EvolFrust), Fragmentation (B), Navigation (WASD/QE/Shift/Space), Pause (P).</p>
""", sizing_mode='stretch_width', styles={'margin-bottom': '10px'})

unity_iframe = Div(
    text="""
    <div style="width: 100%; display: flex; justify-content: center; align-items: center; margin: 20px 0;">
        <iframe
            src="https://igotintogradschool2025.site/unity/"
            style="width: 95%; max-width: 1000px; height: 70vh; min-height: 500px; border: 1px solid #ccc; border-radius: 8px;
                   box-shadow: 0 2px 5px rgba(0,0,0,0.1);"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen title="External Unity Protein Visualizer">
        </iframe>
    </div>""",
    sizing_mode='stretch_width', styles={'margin-top': '15px'}
)
unity_container = column(description_visualizer, unity_iframe, sizing_mode='stretch_width')

# Removed Custom Styles Div as it wasn't doing much

# Scatter Plots Layout (using original structure but ensure it's defined before use)
scatter_col_exp = column(p_scatter_exp, regression_info_exp, sizing_mode="stretch_width", styles={'flex': '1 1 350px', 'min-width': '350px'})
scatter_col_af = column(p_scatter_af, regression_info_af, sizing_mode="stretch_width", styles={'flex': '1 1 350px', 'min-width': '350px'})
scatter_col_evol = column(p_scatter_evol, regression_info_evol, sizing_mode="stretch_width", styles={'flex': '1 1 350px', 'min-width': '350px'})

scatter_row = row(
    scatter_col_exp, scatter_col_af, scatter_col_evol,
    sizing_mode="stretch_width",
    styles={
        'display': 'flex', 'justify-content': 'space-around', 'gap': '20px', # Use space-around
        'width': '100%', 'margin': '20px auto 0 auto', 'flex-wrap': 'wrap'
    }
)

# Correlation Table Title
correlation_title = Div(text="<h2>Overall Spearman Correlations</h2>", styles={'margin-top': '30px'})

# Final Layout Assembly (New Order)
main_layout = column(
    header,             # Consolidated text
    select_pdb_id,      # Dropdown selector
    unity_container,    # Iframe + description
    p,                  # Main line plot
    scatter_row,        # Row of three scatter plots
    correlation_title,  # Title for the table
    data_table,         # Correlation table (no filters)
    sizing_mode='stretch_width'
)

# Set up document
curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration Dashboard"

print("Bokeh application layout created and added to document.")
# End of script
