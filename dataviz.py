# -*- coding: utf-8 -*-
"""
Bokeh application for visualizing protein flexibility (B-Factor) and various
frustration metrics (Experimental, AlphaFold-based, Evolutionary).

This script reads summary data files (summary_XXXX.txt), extracts the 4-character
PDB ID (XXXX), processes the data (smoothing, normalization), calculates
correlations, and presents the results through interactive Bokeh plots
and a data table, using the PDB ID for identification. It also includes an
embedded iframe for a Unity visualizer.
"""
import os
import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, linregress
from math import pi # For label rotation

# Bokeh imports
from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, CheckboxGroup, Div, Spacer,
    DataTable, TableColumn, NumberFormatter, HoverTool,
    GlyphRenderer, Slider, Whisker, Label, Range1d
)
from bokeh.plotting import figure
from bokeh.layouts import column, row, layout
from bokeh.palettes import Category10
from bokeh.transform import jitter # For dot plot

###############################################################################
# 1) Configuration
###############################################################################
# Local data directory path
DATA_DIR = "summary_data"  # Directory containing the summary files

# Updated FILE_PATTERN to capture the 4-char ID in group 1
FILE_PATTERN = r"^summary_([A-Za-z0-9]{4})\.txt$"

# Default PDB ID (4-char) - leave blank for automatic selection
DEFAULT_PDB_ID = "" # Changed from DEFAULT_FILE

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
    Smoothing and normalization are handled later in the update_plot function.
    """
    required_cols = ["AlnIndex", "Residue", "B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]

    if not os.path.isfile(local_path):
        print(f"File not found: {local_path}")
        return None, {} # Return None and empty dict

    try:
        df = pd.read_csv(local_path, sep='\t')
    except Exception as e:
        print(f"Skipping {local_path}: failed to parse data. Error: {e}")
        return None, {}

    # Check for required columns
    if not set(required_cols).issubset(df.columns):
        print(f"Skipping {local_path}: missing required columns.")
        return None, {}

    # Replace 'n/a' with NaN and convert to float
    process_cols = ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]
    for col in process_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_original = df.copy()
    # NOTE: Removed the smoothing/normalization from here, it's done in update_plot

    # Compute Spearman correlations on original data
    corrs = {}
    sub = df_original.dropna(subset=process_cols)
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
            if sub[mA].nunique() < 2 or sub[mB].nunique() < 2:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(sub[mA], sub[mB])
            corrs[(mA, mB)] = (rho, pval)

    # Return original df and correlations only
    return df_original, corrs

def remove_regression_renderers(fig):
    """
    Removes all renderers from the given figure whose names start with 'regression_'.
    Also removes associated HoverTools targeting those renderers.
    """
    renderers_to_keep = []
    renderers_to_remove_names = set()

    # Identify renderers to remove
    for r in fig.renderers:
        renderer_name = getattr(r, 'name', None)
        if isinstance(renderer_name, str) and renderer_name.startswith('regression_'):
            renderers_to_remove_names.add(renderer_name)
        else:
            renderers_to_keep.append(r)

    # Identify hover tools associated with removed renderers or named regression_...
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
            continue # Skip this tool
        tools_to_keep.append(tool)

    fig.renderers = renderers_to_keep
    fig.tools = tools_to_keep


###############################################################################
# 3) Load and Aggregate Data from Local Directory
###############################################################################
# Use PDB ID as the key
data_by_id = {} # Changed from data_by_file
all_corr_rows = []

# Aggregation lists - use PDB IDs
pdb_ids = [] # Changed from protein_names
avg_bfactors = []
std_bfactors = []
spearman_exp = []
spearman_af = []
spearman_evol = []

# Possible frustration columns
POSSIBLE_FRUST_COLUMNS = ['ExpFrust', 'AFFrust', 'EvolFrust']

# Color mapping for plots (using original Category10 assignments)
FRUSTRATION_COLORS = {
    "ExpFrust.": Category10[10][1],  # Blue
    "AFFrust.": Category10[10][2],   # Green
    "EvolFrust.": Category10[10][3]  # Purple (Adjusted from original for consistency)
}

# Iterate through files
found_files = 0
skipped_files = 0
print(f"Loading data from: {DATA_DIR}")
for filename in os.listdir(DATA_DIR):
    # Extract PDB ID from filename
    pdb_id = extract_pdb_id(filename)
    if not pdb_id:
        # print(f"Skipping {filename}: does not match pattern {FILE_PATTERN}")
        skipped_files += 1
        continue

    file_path = os.path.join(DATA_DIR, filename)
    # Parse file, get original df and correlations
    df_orig, corrs = parse_summary_file(file_path)
    if df_orig is None:
        skipped_files += 1
        continue

    found_files += 1
    # Store data using PDB ID as the key
    data_by_id[pdb_id] = {
        "df_original": df_orig,
        # "df_for_plot": df_plot, # Removed, plot df generated in update_plot
        "corrs": corrs
    }

    # Collect correlation data using PDB ID
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo
        all_corr_rows.append([pdb_id, mA, mB, rho, pval]) # Use pdb_id

    # Aggregate data for additional plots using PDB ID
    avg_b = df_orig['B_Factor'].mean()
    std_b = df_orig['B_Factor'].std()

    spearman_r_exp = corrs.get(("B_Factor", "ExpFrust"), (np.nan, np.nan))[0]
    spearman_r_af = corrs.get(("B_Factor", "AFFrust"), (np.nan, np.nan))[0]
    spearman_r_evol = corrs.get(("B_Factor", "EvolFrust"), (np.nan, np.nan))[0]

    pdb_ids.append(pdb_id) # Use pdb_id
    avg_bfactors.append(avg_b)
    std_bfactors.append(std_b)
    spearman_exp.append(spearman_r_exp)
    spearman_af.append(spearman_r_af)
    spearman_evol.append(spearman_r_evol)

print(f"Found and processed {found_files} files matching pattern.")
if skipped_files > 0:
    print(f"Skipped {skipped_files} files (pattern mismatch or parsing error).")


# Correlation DataFrame - use PDB_ID column name
df_all_corr = pd.DataFrame(all_corr_rows, columns=["PDB_ID","MetricA","MetricB","Rho","Pval"])

# Aggregated DataFrame for Additional Plots - use PDB_ID column name
data_proviz = pd.DataFrame({
    'PDB_ID': pdb_ids, # Use pdb_ids
    'Avg_B_Factor': avg_bfactors,
    'Std_B_Factor': std_bfactors,
    'Spearman_ExpFrust': spearman_exp,
    'Spearman_AFFrust': spearman_af,
    'Spearman_EvolFrust': spearman_evol
})

# Melt data for plotting - use PDB_ID
data_long_avg = data_proviz.melt(
    id_vars=['PDB_ID', 'Avg_B_Factor'], # Use PDB_ID
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)

data_long_std = data_proviz.melt(
    id_vars=['PDB_ID', 'Std_B_Factor'], # Use PDB_ID
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)

# Clean Frust_Type names
data_long_avg['Frust_Type'] = data_long_avg['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')
data_long_std['Frust_Type'] = data_long_std['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

# Remove rows with NaN correlations
data_long_avg.dropna(subset=['Spearman_Rho'], inplace=True)
data_long_std.dropna(subset=['Spearman_Rho'], inplace=True)

###############################################################################
# 4) Bokeh Application Components
###############################################################################

# (A) Main Plot: Smoothed + Normalized Data
source_plot = ColumnDataSource(data=dict(
    x=[],
    residue=[],
    b_factor=[],
    exp_frust=[],
    af_frust=[],
    evol_frust=[]
))

p = figure(
    title="(No PDB ID Selected)", # Updated default title
    sizing_mode='stretch_width',
    height=600,
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom",
    active_scroll="wheel_zoom" # Re-enabled wheel scroll based on original code
)

# Define separate HoverTools for each metric (using original tooltips)
hover_bf = HoverTool(
    renderers=[],
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. B-Factor", "@b_factor{0.3f}")], # Display normalized value
    name="hover_b_factor"
)
hover_ef = HoverTool(
    renderers=[],
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. ExpFrust", "@exp_frust{0.3f}")],
    name="hover_exp_frust"
)
hover_af = HoverTool(
    renderers=[],
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. AFFrust", "@af_frust{0.3f}")],
    name="hover_af_frust"
)
hover_ev = HoverTool(
    renderers=[],
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. EvolFrust", "@evol_frust{0.3f}")],
    name="hover_evol_frust"
)

p.add_tools(hover_bf, hover_ef, hover_af, hover_ev)
p.xaxis.axis_label = "Residue Index"
p.yaxis.axis_label = "Smoothed & Normalized Value" # Updated label

# Add lines (using original color map)
color_map = {
    "b_factor":  ("B-Factor", Category10[10][0]),
    "exp_frust": ("ExpFrust", Category10[10][1]),
    "af_frust":  ("AFFrust", Category10[10][2]),
    "evol_frust":("EvolFrust", Category10[10][3])
}
renderers = {}
for col_key, (label, col) in color_map.items():
    renderer = p.line(
        x="x", y=col_key, source=source_plot,
        line_width=2, alpha=0.7, color=col,
        legend_label=label
    )
    renderers[col_key] = renderer
    # Link hover tools
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

# (B) Scatter Plots (Experimental, AF, Evolutionary Frustration)
# Scatter plots configuration (reverted active_scroll to None as per original)
common_scatter_tools = ["pan", "box_zoom", "wheel_zoom", "reset", "save"]
p_scatter_exp = figure(
    sizing_mode="stretch_both", aspect_ratio=1, min_width=350, min_height=350,
    title="", x_axis_label="Normalized B-Factor", y_axis_label="Normalized ExpFrust", # Updated Y label
    tools=common_scatter_tools, active_drag="box_zoom", active_scroll=None
)
p_scatter_af = figure(
    sizing_mode="stretch_both", aspect_ratio=1, min_width=350, min_height=350,
    title="", x_axis_label="Normalized B-Factor", y_axis_label="Normalized AFFrust", # Updated Y label
    tools=common_scatter_tools, active_drag="box_zoom", active_scroll=None
)
p_scatter_evol = figure(
    sizing_mode="stretch_both", aspect_ratio=1, min_width=350, min_height=350,
    title="", x_axis_label="Normalized B-Factor", y_axis_label="Normalized EvolFrust", # Updated Y label
    tools=common_scatter_tools, active_drag="box_zoom", active_scroll=None
)

# ColumnDataSources include normalized ('x', 'y') and original ('x_orig', 'y_orig') plus residue/index
source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[]))
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[]))
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[]))

# Create Div elements for regression info (using original styles)
div_styles = {
    'background-color': '#f8f9fa', 'padding': '10px', 'border': '1px solid #ddd',
    'border-radius': '4px', 'margin-top': '10px', 'font-size': '14px',
    'text-align': 'center', 'width': '100%' # Changed from auto to 100% based on original
}
regression_info_exp = Div(text="", styles=div_styles, sizing_mode="stretch_width")
regression_info_af = Div(text="", styles=div_styles, sizing_mode="stretch_width")
regression_info_evol = Div(text="", styles=div_styles, sizing_mode="stretch_width")

# Initial scatter glyphs (using original colors)
scatter_exp_renderer = p_scatter_exp.scatter("x", "y", source=source_scatter_exp, color=Category10[10][1], alpha=0.7, size=6) # Use size=6
scatter_af_renderer = p_scatter_af.scatter("x", "y", source=source_scatter_af,  color=Category10[10][2], alpha=0.7, size=6)
scatter_evol_renderer = p_scatter_evol.scatter("x", "y", source=source_scatter_evol, color=Category10[10][3], alpha=0.7, size=6)

# Add HoverTools for scatter points (displaying original and normalized values)
scatter_hover_tooltips = [
    ("Index", "@index"),
    ("Residue", "@residue"),
    ("Orig. B-Factor", "@x_orig{0.3f}"),
    ("Orig. Frust", "@y_orig{0.3f}"),
    ("Norm. B-Factor", "@x{0.3f}"),
    ("Norm. Frust", "@y{0.3f}"),
]
p_scatter_exp.add_tools(HoverTool(renderers=[scatter_exp_renderer], tooltips=scatter_hover_tooltips))
p_scatter_af.add_tools(HoverTool(renderers=[scatter_af_renderer], tooltips=scatter_hover_tooltips))
p_scatter_evol.add_tools(HoverTool(renderers=[scatter_evol_renderer], tooltips=scatter_hover_tooltips))


def add_regression_line_and_info(fig, xvals, yvals, color="black", info_div=None, plot_type=""):
    """
    Adds a linear regression line and updates the regression info Div.
    Includes hover for the regression line itself.
    """
    # Ensure there are enough valid data points for regression
    not_nan_mask = ~np.isnan(xvals) & ~np.isnan(yvals)
    xvals_clean = xvals[not_nan_mask]
    yvals_clean = yvals[not_nan_mask]

    if len(xvals_clean) < 2 or np.all(xvals_clean == xvals_clean[0]) or np.all(yvals_clean == yvals_clean[0]):
        if info_div:
            info_div.text = "<i style='color: gray;'>Insufficient data or variance for regression</i>"
        return

    try:
        slope, intercept, r_value, p_value, std_err = linregress(xvals_clean, yvals_clean)
    except ValueError as e:
         if info_div:
            info_div.text = f"<i style='color: red;'>Regression Error: {e}</i>"
         return

    # --- Add Visible Regression Line ---
    x_min, x_max = np.min(xvals_clean), np.max(xvals_clean)
    if x_min == x_max: # Should be caught above, but safety check
         if info_div: info_div.text = "<i style='color: gray;'>Insufficient variance for regression line</i>"
         return
    x_range = np.linspace(x_min, x_max, 100)
    y_range = slope * x_range + intercept
    regression_line_name = f'regression_line_{plot_type}'
    regression_line_renderer = fig.line(
        x_range, y_range, line_width=2, line_dash='dashed', color=color, name=regression_line_name
    )

    # --- Add Hover Tool for the Regression Line ---
    # (Using the simpler approach from original code - invisible line + hover tool)
    # Create a separate data source for regression line hover
    # regression_source = ColumnDataSource(data=dict(
    #     x=x_range,
    #     y=y_range,
    #     equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
    # ))
    # Plot regression line again with this data source, invisible (for hover)
    # invisible_regression_name = f'regression_hover_{plot_type}'
    # invisible_regression = fig.line(
    #     'x', 'y',
    #     source=regression_source,
    #     line_width=10, # Make it wide for easier hover
    #     alpha=0, # Make it invisible
    #     name=invisible_regression_name
    # )
    # Add a separate HoverTool for the regression line targeting the invisible line
    hover_regression = HoverTool(
        renderers=[regression_line_renderer], # Target the VISIBLE line renderer
        tooltips=[
            ("Regression Equation", f"y = {slope:.3f}x + {intercept:.3f}"), # Static tooltip content
            ("R-squared", f"{r_value**2:.3f}"),
            ("p-value", f"{p_value:.2e}")
        ],
        mode='mouse',
        name=f'regression_hover_tool_{plot_type}' # Give tool a unique name
    )
    # Check if tool already exists before adding
    existing_tool_names = [getattr(t, 'name', None) for t in fig.tools]
    if hover_regression.name not in existing_tool_names:
        fig.add_tools(hover_regression)


    # Update regression info div with equation (using original formatting)
    if info_div:
        info_div.text = f"""
        <div style='color: {color}'>
            <strong>y = {slope:.3f}x + {intercept:.3f}</strong><br>
            <span style='font-size: 12px'>R² = {r_value**2:.3f} | p = {p_value:.2e}</span>
        </div>
        """ # Added p-value display

# Dropdown select - use PDB IDs
pdb_id_options = sorted(data_by_id.keys()) # Use keys from data_by_id
if DEFAULT_PDB_ID and DEFAULT_PDB_ID in pdb_id_options:
    initial_pdb_id = DEFAULT_PDB_ID
elif pdb_id_options:
    initial_pdb_id = pdb_id_options[0]
else:
    initial_pdb_id = ""

select_pdb_id = Select( # Renamed from select_file
    title="Select PDB ID:", # Updated title
    value=initial_pdb_id,
    options=pdb_id_options # Use PDB IDs
)

# Add slider for moving average window size (no changes needed here)
window_slider = Slider(
    start=1, end=21, value=5, step=2,
    title="Moving Average Window Size", width=400
)

def update_moving_average(attr, old, new):
    """Update plot when slider value changes"""
    # Pass None, None, None because update_plot reads current widget values
    update_plot(None, None, None)

window_slider.on_change('value', update_moving_average)

def min_max_normalize(arr):
    """
    Applies min-max normalization to a numpy array (scales [0, 1]).
    Handles NaNs and division by zero.
    """
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if np.isnan(arr_min) or np.isnan(arr_max): # Handle all-NaN case
        return np.full_like(arr, np.nan)
    if arr_max > arr_min:
        norm_arr = (arr - arr_min) / (arr_max - arr_min)
    else: # Handle case where all values are the same
        norm_arr = np.zeros_like(arr)
        norm_arr[~np.isnan(arr)] = 0.5 # Or 0.0, depending on desired output for constant values
    norm_arr[np.isnan(arr)] = np.nan # Preserve original NaNs
    return norm_arr


def update_plot(attr, old, new):
    """
    Updates both the main plot and scatter plots when a new PDB ID is selected or slider changes.
    """
    pdb_id = select_pdb_id.value # Get selected PDB ID
    window_size = window_slider.value # Get slider value

    print(f"Updating plots for PDB ID: {pdb_id}, Window Size: {window_size}")

    if not pdb_id or pdb_id not in data_by_id:
        # Clear plots if no valid PDB ID selected
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
        # Remove regression elements
        remove_regression_renderers(p_scatter_exp)
        remove_regression_renderers(p_scatter_af)
        remove_regression_renderers(p_scatter_evol)
        return

    # --- Update Main Line Plot ---
    # Get original data for the selected PDB ID
    df_orig = data_by_id[pdb_id]["df_original"]
    df_plot = df_orig.copy() # Create copy for processing

    # Apply moving average with current window size
    metrics_to_process = ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]
    for col in metrics_to_process:
        if col in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[col]):
            df_plot[col] = moving_average(df_plot[col].values, window_size=window_size)
        else:
            df_plot[col] = np.nan # Ensure column exists even if problematic

    # Normalize the smoothed data
    for col in metrics_to_process:
        if col in df_plot.columns:
            df_plot[col] = min_max_normalize(df_plot[col].values)

    # Update source_plot data (handles NaNs implicitly)
    new_data = dict(
        x=df_plot["AlnIndex"].tolist(),
        residue=df_plot["Residue"].tolist(),
        b_factor=df_plot["B_Factor"].tolist(),
        exp_frust=df_plot["ExpFrust"].tolist(),
        af_frust=df_plot["AFFrust"].tolist(),
        evol_frust=df_plot["EvolFrust"].tolist()
    )
    source_plot.data = new_data
    p.title.text = f"PDB ID: {pdb_id} (Smoothed Window={window_size}, Normalized)" # Use PDB ID

    # --- Update Scatter Plots (using ORIGINAL, non-smoothed data, then normalized) ---
    df_scatter_base = data_by_id[pdb_id]["df_original"].copy()

    # **Remove all existing regression renderers before adding new ones**
    remove_regression_renderers(p_scatter_exp)
    remove_regression_renderers(p_scatter_af)
    remove_regression_renderers(p_scatter_evol)

    # Prepare data for scatter plots: Normalize original values
    x_norm = min_max_normalize(df_scatter_base["B_Factor"].values)
    y_exp_norm = min_max_normalize(df_scatter_base["ExpFrust"].values)
    y_af_norm = min_max_normalize(df_scatter_base["AFFrust"].values)
    y_evol_norm = min_max_normalize(df_scatter_base["EvolFrust"].values)

    # Common data for tooltips
    indices = df_scatter_base["AlnIndex"].tolist()
    residues = df_scatter_base["Residue"].tolist()
    x_orig = df_scatter_base["B_Factor"].tolist()
    y_exp_orig = df_scatter_base["ExpFrust"].tolist()
    y_af_orig = df_scatter_base["AFFrust"].tolist()
    y_evol_orig = df_scatter_base["EvolFrust"].tolist()

    # Check if there's valid data to plot after normalization
    valid_exp = ~np.isnan(x_norm) & ~np.isnan(y_exp_norm)
    valid_af = ~np.isnan(x_norm) & ~np.isnan(y_af_norm)
    valid_evol = ~np.isnan(x_norm) & ~np.isnan(y_evol_norm)

    # Update ExpFrust Scatter
    if np.any(valid_exp):
        source_scatter_exp.data = dict(x=x_norm, y=y_exp_norm, x_orig=x_orig, y_orig=y_exp_orig, residue=residues, index=indices)
        p_scatter_exp.title.text = f"{pdb_id}: B-Factor vs ExpFrust" # Use PDB ID
        add_regression_line_and_info(
            fig=p_scatter_exp, xvals=x_norm, yvals=y_exp_norm,
            color=Category10[10][1], info_div=regression_info_exp, plot_type="exp"
        )
    else:
        source_scatter_exp.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        p_scatter_exp.title.text = f"{pdb_id}: B-Factor vs ExpFrust (No Valid Data)"
        regression_info_exp.text = "<i style='color: gray;'>No valid data points</i>"

    # Update AFFrust Scatter
    if np.any(valid_af):
        source_scatter_af.data = dict(x=x_norm, y=y_af_norm, x_orig=x_orig, y_orig=y_af_orig, residue=residues, index=indices)
        p_scatter_af.title.text = f"{pdb_id}: B-Factor vs AFFrust" # Use PDB ID
        add_regression_line_and_info(
            fig=p_scatter_af, xvals=x_norm, yvals=y_af_norm,
            color=Category10[10][2], info_div=regression_info_af, plot_type="af"
        )
    else:
        source_scatter_af.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        p_scatter_af.title.text = f"{pdb_id}: B-Factor vs AFFrust (No Valid Data)"
        regression_info_af.text = "<i style='color: gray;'>No valid data points</i>"

    # Update EvolFrust Scatter
    if np.any(valid_evol):
        source_scatter_evol.data = dict(x=x_norm, y=y_evol_norm, x_orig=x_orig, y_orig=y_evol_orig, residue=residues, index=indices)
        p_scatter_evol.title.text = f"{pdb_id}: B-Factor vs EvolFrust" # Use PDB ID
        add_regression_line_and_info(
            fig=p_scatter_evol, xvals=x_norm, yvals=y_evol_norm,
            color=Category10[10][3], info_div=regression_info_evol, plot_type="evol"
        )
    else:
        source_scatter_evol.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        p_scatter_evol.title.text = f"{pdb_id}: B-Factor vs EvolFrust (No Valid Data)"
        regression_info_evol.text = "<i style='color: gray;'>No valid data points</i>"


select_pdb_id.on_change("value", update_plot) # Use select_pdb_id
# Initial plot update if a default PDB ID is set and valid
if initial_pdb_id:
    update_plot(None, None, initial_pdb_id)


###############################################################################
# 5) CORRELATION TABLE AND FILTERS
###############################################################################

# (D) CORRELATION TABLE
if df_all_corr.empty:
    # Define columns even if empty, use PDB_ID
    columns = [
        TableColumn(field="PDB_ID", title="PDB ID"),
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Spearman Rho"),
        TableColumn(field="Pval", title="p-value")
    ]
    source_corr = ColumnDataSource(dict(PDB_ID=[], MetricA=[], MetricB=[], Rho=[], Pval=[]))
    data_table = DataTable(columns=columns, source=source_corr, height=400, width=1200, sizing_mode='stretch_width') # Use sizing_mode
else:
    source_corr = ColumnDataSource(df_all_corr)
    # Define columns with PDB_ID and formatters
    columns = [
        TableColumn(field="PDB_ID", title="PDB ID"), # Use PDB_ID
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Spearman Rho", formatter=NumberFormatter(format="0.3f")),
        TableColumn(field="Pval", title="p-value", formatter=NumberFormatter(format="0.2e"))
    ]
    data_table = DataTable(columns=columns, source=source_corr, height=400, width=1200, sizing_mode='stretch_width') # Use sizing_mode

# (E) FILTERS for correlation table

# Define helper function to split labels into columns
def split_labels(labels, num_columns):
    """
    Splits a list of labels into a list of lists, each sublist containing labels for one column.
    """
    if not labels or num_columns <= 0: # Handle empty labels or invalid columns
        return [labels]
    k, m = divmod(len(labels), num_columns)
    return [labels[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_columns)]

# Define the number of columns for better layout
NUM_COLUMNS = 3

# Get unique PDB IDs and combo options for filters
tests_in_corr = sorted(df_all_corr["PDB_ID"].unique()) if not df_all_corr.empty else [] # Use PDB_ID
if not df_all_corr.empty:
    combo_options = sorted(list(set( # Use list(set(...)) for unique options
        f"{row['MetricA']} vs {row['MetricB']}"
        for _, row in df_all_corr.iterrows()
    )))
else:
    combo_options = []

# Create CheckboxGroups for PDB IDs
if tests_in_corr:
    test_labels_split = split_labels(tests_in_corr, NUM_COLUMNS)
    checkbox_tests_columns = [
        CheckboxGroup(
            labels=col_labels, active=[], name=f'tests_column_{i+1}', height=150 # Added height limit
        ) for i, col_labels in enumerate(test_labels_split) if col_labels # Check if col_labels is not empty
    ]
else: # Handle no PDB IDs found
    checkbox_tests_columns = [CheckboxGroup(labels=["(No PDB IDs)"], active=[], disabled=True, name='tests_column_1')]

# Create CheckboxGroups for Metric Pairs
if combo_options:
    combo_labels_split = split_labels(combo_options, NUM_COLUMNS)
    checkbox_combos_columns = [
        CheckboxGroup(
            labels=col_labels, active=[], name=f'combos_column_{i+1}', height=150 # Added height limit
        ) for i, col_labels in enumerate(combo_labels_split) if col_labels # Check if col_labels is not empty
    ]
else: # Handle no combos found
    checkbox_combos_columns = [CheckboxGroup(labels=["(No Metric Pairs)"], active=[], disabled=True, name='combos_column_1')]


# Create Columns for Tests and Metric Pairs layout (using original widths)
tests_layout = row(*checkbox_tests_columns, sizing_mode='stretch_width') # Removed fixed width
combos_layout = row(*checkbox_combos_columns, sizing_mode='stretch_width') # Removed fixed width

# Add Titles Above Each CheckboxGroup
tests_title = Div(text="<b>Select PDB IDs:</b>", styles={'font-size': '14px', 'margin-bottom': '5px'}) # Updated title
combos_title = Div(text="<b>Select Metric Pairs:</b>", styles={'font-size': '14px', 'margin-bottom': '5px'})

# Combine Titles and CheckboxGroups into Columns
tests_column = column(tests_title, tests_layout, sizing_mode='stretch_width')
combos_column = column(combos_title, combos_layout, sizing_mode='stretch_width')

# Arrange Tests and Combos Side by Side with Spacer
controls_layout = row(
    tests_column,
    Spacer(width=50),
    combos_column,
    sizing_mode='stretch_width'
)

# Define helper function to get selected labels from multiple CheckboxGroups
def get_selected_labels(checkbox_columns):
    """
    Aggregates selected labels from multiple CheckboxGroup widgets.
    """
    selected = []
    for checkbox in checkbox_columns:
        if checkbox.labels and not checkbox.disabled: # Check if labels exist and not disabled
             selected.extend([checkbox.labels[i] for i in checkbox.active])
    return selected

def update_corr_filter(attr, old, new):
    """Filter correlation table based on selected PDB IDs and metric pairs."""
    if df_all_corr.empty:
        return

    # Aggregate selected PDB IDs and metric pairs from all CheckboxGroups
    selected_tests = get_selected_labels(checkbox_tests_columns) # These are PDB IDs
    selected_combos = get_selected_labels(checkbox_combos_columns)

    filtered_df = df_all_corr.copy() # Start with full data

    # Apply PDB ID filter
    if selected_tests:
        filtered_df = filtered_df[filtered_df["PDB_ID"].isin(selected_tests)] # Filter on PDB_ID column

    # Apply combo filter
    if selected_combos:
        # Need temporary column for filtering based on constructed string
        filtered_df["combo_str"] = filtered_df.apply(lambda r: f"{r['MetricA']} vs {r['MetricB']}", axis=1)
        filtered_df = filtered_df[filtered_df["combo_str"].isin(selected_combos)]
        # Drop the temporary column if it exists
        if "combo_str" in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=["combo_str"])

    # If no filters selected, filtered_df remains the full copy

    source_corr.data = filtered_df.to_dict(orient="list")
    print(f"Correlation table updated. Showing {len(filtered_df)} rows.")


# Attach callbacks to all CheckboxGroups
for checkbox in checkbox_tests_columns + checkbox_combos_columns:
    checkbox.on_change('active', update_corr_filter)

###############################################################################
# 6) Additional Aggregated Plots (Converted from Plotly to Bokeh)
###############################################################################

# (F) Spearman Rho vs Average B-Factor
source_avg_plot = ColumnDataSource(data_long_avg)

p_avg_plot = figure(
    title="Spearman Correlation vs Average B-Factor",
    x_axis_label="Average B-Factor (per PDB ID)", # Use PDB ID
    y_axis_label="Spearman Correlation (Frustration vs B-Factor)", # Simplified label
    sizing_mode='stretch_width', height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom", active_scroll="wheel_zoom" # Re-enabled scroll
)

# Define color palette for Frustration Types (using adjusted FRUSTRATION_COLORS)
frust_types_avg = data_long_avg['Frust_Type'].unique().tolist()
# color_map_frust_avg = {frust: FRUSTRATION_COLORS.get(frust, 'gray') for frust in frust_types_avg} # Use FRUSTRATION_COLORS

# Create a list to hold scatter renderers
scatter_renderers_avg = []

# Add scatter glyphs with named renderers and collect renderers
for frust in frust_types_avg:
    subset = data_long_avg[data_long_avg['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    color = FRUSTRATION_COLORS.get(frust, 'gray') # Get color from map
    scatter = p_avg_plot.scatter(
        'Avg_B_Factor', 'Spearman_Rho', source=source_subset,
        color=color, size=8, alpha=0.6, legend_label=frust, muted_alpha=0.1,
        name=f'scatter_avg_{frust}' # Unique name per type
    )
    scatter_renderers_avg.append(scatter)

    # Add regression lines with hover (using original logic)
    if len(subset) >= 2 and subset['Avg_B_Factor'].nunique() > 1:
        try:
            slope, intercept, r_value, p_value, std_err = linregress(subset['Avg_B_Factor'], subset['Spearman_Rho'])
            x_min, x_max = subset['Avg_B_Factor'].min(), subset['Avg_B_Factor'].max()
            if x_min == x_max: continue # Skip if no variance
            x_range = np.linspace(x_min, x_max, 100)
            y_range = slope * x_range + intercept

            # regression_source = ColumnDataSource(data=dict(
            #     x=x_range, y=y_range,
            #     equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
            # ))

            regression_line = p_avg_plot.line(
                x_range, y_range, # Use direct numpy arrays
                color=color, line_dash='dashed',
                name=f'regression_line_avg_{frust}' # Unique name
            )

            # Add HoverTool only to the regression_line
            hover_regression = HoverTool(
                renderers=[regression_line],
                tooltips=[
                    ("Regression Equation", f"y = {slope:.3f}x + {intercept:.3f}"), # Static tooltip
                    ("R-squared", f"{r_value**2:.3f}"),
                    ("p-value", f"{p_value:.2e}")
                ],
                mode='mouse',
                name=f'regression_hover_tool_avg_{frust}' # Unique name for tool
            )
            # Check if tool already exists before adding
            existing_tool_names = [getattr(t, 'name', None) for t in p_avg_plot.tools]
            if hover_regression.name not in existing_tool_names:
                 p_avg_plot.add_tools(hover_regression)
        except ValueError:
            print(f"Could not compute regression for {frust} in Avg B-Factor plot.")
            pass # Ignore if regression fails

# Create and add the standard HoverTool only for the scatter renderers
hover_scatter_avg = HoverTool(
    tooltips=[
        ("PDB ID", "@PDB_ID"), # Use PDB ID
        ("Frustration Type", "@Frust_Type"),
        ("Avg B-Factor", "@Avg_B_Factor{0.3f}"), # Added avg B-factor to tooltip
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    renderers=scatter_renderers_avg,  # Only attach to scatter renderers
    mode='mouse', name="scatter_hover_avg" # Give tool a name
)
# Check if tool already exists before adding
existing_tool_names = [getattr(t, 'name', None) for t in p_avg_plot.tools]
if hover_scatter_avg.name not in existing_tool_names:
    p_avg_plot.add_tools(hover_scatter_avg)


p_avg_plot.legend.location = "top_left"
p_avg_plot.legend.title = "Frustration Type"
p_avg_plot.legend.click_policy = "mute"

# (G) Spearman Rho vs Std Dev of B-Factor
source_std_plot = ColumnDataSource(data_long_std)

p_std_plot = figure(
    title="Spearman Correlation vs Std Dev of B-Factor",
    x_axis_label="Standard Deviation of B-Factor (per PDB ID)", # Use PDB ID
    y_axis_label="Spearman Correlation (Frustration vs B-Factor)", # Simplified label
    sizing_mode='stretch_width', height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom", active_scroll="wheel_zoom" # Re-enabled scroll
)

# Define color palette for Frustration Types
frust_types_std = data_long_std['Frust_Type'].unique().tolist()
# color_map_frust_std = {frust: FRUSTRATION_COLORS.get(frust, 'gray') for frust in frust_types_std} # Use FRUSTRATION_COLORS

# Create a list to hold scatter renderers
scatter_renderers_std = []

# Add scatter glyphs with named renderers and collect renderers
for frust in frust_types_std:
    subset = data_long_std[data_long_std['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    color = FRUSTRATION_COLORS.get(frust, 'gray') # Get color from map
    scatter = p_std_plot.scatter(
        'Std_B_Factor', 'Spearman_Rho', source=source_subset,
        color=color, size=8, alpha=0.6, legend_label=frust, muted_alpha=0.1,
        name=f'scatter_std_{frust}' # Unique name
    )
    scatter_renderers_std.append(scatter)

    # Add regression lines with hover (using original logic)
    if len(subset) >= 2 and subset['Std_B_Factor'].nunique() > 1:
        try:
            slope, intercept, r_value, p_value, std_err = linregress(subset['Std_B_Factor'], subset['Spearman_Rho'])
            x_min, x_max = subset['Std_B_Factor'].min(), subset['Std_B_Factor'].max()
            if x_min == x_max: continue
            x_range = np.linspace(x_min, x_max, 100)
            y_range = slope * x_range + intercept

            # regression_source = ColumnDataSource(data=dict(
            #     x=x_range, y=y_range,
            #     equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
            # ))

            regression_line = p_std_plot.line(
                x_range, y_range, # Use direct numpy arrays
                color=color, line_dash='dashed',
                name=f'regression_line_std_{frust}' # Unique name
            )

            # Add HoverTool only to the regression_line
            hover_regression = HoverTool(
                renderers=[regression_line],
                tooltips=[
                    ("Regression Equation", f"y = {slope:.3f}x + {intercept:.3f}"),
                    ("R-squared", f"{r_value**2:.3f}"),
                    ("p-value", f"{p_value:.2e}")
                ],
                mode='mouse',
                name=f'regression_hover_tool_std_{frust}' # Unique name for tool
            )
            # Check if tool already exists before adding
            existing_tool_names = [getattr(t, 'name', None) for t in p_std_plot.tools]
            if hover_regression.name not in existing_tool_names:
                p_std_plot.add_tools(hover_regression)
        except ValueError:
            print(f"Could not compute regression for {frust} in Std Dev B-Factor plot.")
            pass

# Create and add the standard HoverTool only for the scatter renderers
hover_scatter_std = HoverTool(
    tooltips=[
        ("PDB ID", "@PDB_ID"), # Use PDB ID
        ("Frustration Type", "@Frust_Type"),
        ("Std Dev B-Factor", "@Std_B_Factor{0.3f}"), # Added Std Dev to tooltip
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    renderers=scatter_renderers_std,  # Only attach to scatter renderers
    mode='mouse', name="scatter_hover_std" # Give tool a name
)
# Check if tool already exists before adding
existing_tool_names = [getattr(t, 'name', None) for t in p_std_plot.tools]
if hover_scatter_std.name not in existing_tool_names:
    p_std_plot.add_tools(hover_scatter_std)


p_std_plot.legend.location = "top_left"
p_std_plot.legend.title = "Frustration Type"
p_std_plot.legend.click_policy = "mute"

# (H) Spearman Rho per PDB ID and Frustration Metric (Dot Plot)
# Melt data_proviz for the third plot - use PDB_ID
data_long_corr = data_proviz.melt(
    id_vars=['PDB_ID'], # Use PDB_ID
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type', value_name='Spearman_Rho'
)

# Clean Frust_Type names
data_long_corr['Frust_Type'] = data_long_corr['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

# Remove rows with NaN correlations
data_long_corr.dropna(subset=['Spearman_Rho'], inplace=True)

# Get PDB IDs for x-axis range
pdb_id_x_range = sorted(data_proviz['PDB_ID'].unique().tolist()) # Use PDB_ID

p_corr_plot = figure(
    title="Spearman Correlation per PDB ID and Frustration Metric", # Use PDB ID
    x_axis_label="PDB ID", # Use PDB ID
    y_axis_label="Spearman Correlation (Frustration vs B-Factor)", # Simplified label
    x_range=pdb_id_x_range, # Use PDB IDs for range
    sizing_mode='stretch_width', height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom", active_scroll="wheel_zoom", # Re-enabled scroll
    toolbar_location="above"
)

# Define color palette for Frustration Types
frust_types_corr = data_long_corr['Frust_Type'].unique().tolist()
# color_map_corr = {frust: FRUSTRATION_COLORS.get(frust, 'gray') for frust in frust_types_corr} # Use FRUSTRATION_COLORS

# Add HoverTool (use PDB_ID)
hover_corr = HoverTool(
    tooltips=[
        ("PDB ID", "@PDB_ID"), # Use PDB_ID
        ("Frustration Metric", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    mode='mouse', name="dot_hover_corr" # Name the tool
)
# Check if tool already exists before adding
existing_tool_names = [getattr(t, 'name', None) for t in p_corr_plot.tools]
if hover_corr.name not in existing_tool_names:
    p_corr_plot.add_tools(hover_corr)


# Add horizontal line at y=0 (using original x range logic)
# p_corr_plot.line(
#     x=[-0.5, len(data_proviz['PDB_ID']) - 0.5], # Adjusted range slightly
#     y=[0, 0], line_width=1, line_dash='dashed', color='gray', name='y_zero_line'
# )
# Alternative: Use the categorical range directly
p_corr_plot.line(x=pdb_id_x_range, y=0, line_width=1, line_dash='dashed', color='gray', name='y_zero_line')


# Add scatter glyphs (using jitter)
dot_renderers = [] # To potentially attach hover tool specifically
for frust in frust_types_corr:
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    color = FRUSTRATION_COLORS.get(frust, 'gray') # Get color
    # Use jitter on PDB_ID
    dot = p_corr_plot.scatter(
        x=jitter('PDB_ID', width=0.6, range=p_corr_plot.x_range),
        y='Spearman_Rho', source=source_subset,
        color=color, size=8, alpha=0.6, legend_label=frust, muted_alpha=0.1,
        name=f'dot_{frust}' # Name the renderer
    )
    dot_renderers.append(dot)

# Attach hover tool to the dot renderers specifically (optional, already added globally)
# hover_corr.renderers = dot_renderers

# Add mean lines for each frustration type (using original logic)
mean_line_renderers = [] # To potentially attach hover tool specifically
for frust in frust_types_corr:
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    mean_value = subset['Spearman_Rho'].mean()
    if np.isnan(mean_value): continue # Skip if mean is NaN

    color = FRUSTRATION_COLORS.get(frust, 'gray') # Get color
    # Create source for the mean line with hover information
    # mean_source = ColumnDataSource(data=dict(
    #     x=[-0.5, len(data_proviz['PDB_ID']) - 0.5], # Adjusted range
    #     y=[mean_value, mean_value],
    #     mean_value=[f"{mean_value:.3f}"] * 2,
    #     frust_type=[frust] * 2
    # ))

    # Add mean line with hover
    mean_line = p_corr_plot.line(
        x=pdb_id_x_range, # Use categorical range
        y=mean_value,
        color=color, line_dash='dashed', line_width=2, # Make slightly thicker
        name=f'mean_line_{frust}'  # Unique name based on frust type
    )
    mean_line_renderers.append(mean_line)

    # Add hover tool for mean line (using original logic)
    mean_hover = HoverTool(
        renderers=[mean_line],
        tooltips=[
            ("Frustration Type", f"{frust}"), # Static tooltip
            ("Mean Correlation", f"{mean_value:.3f}")
        ],
        mode='mouse', name=f'mean_hover_{frust}' # Name the tool
    )
    # Check if tool already exists before adding
    existing_tool_names = [getattr(t, 'name', None) for t in p_corr_plot.tools]
    if mean_hover.name not in existing_tool_names:
        p_corr_plot.add_tools(mean_hover)


p_corr_plot.legend.location = "top_left"
p_corr_plot.legend.title = "Frustration Type"
p_corr_plot.legend.click_policy = "mute"

# Rotate x-axis labels to prevent overlapping
p_corr_plot.xaxis.major_label_orientation = pi / 3 # Use pi/3 for ~60 degrees rotation


###############################################################################
# 7) User Interface Components
###############################################################################

# Add header and description (updated to mention PDB ID)
header = Div(text="""
    <h1>Evolutionary Frustration Analysis</h1>
    <p>
        Evolutionary frustration leverages multiple sequence alignment (MSA) derived coupling scores
        and statistical potentials to calculate the mutational frustration of various proteins without the need for protein structures.
        By benchmarking the evolutionary frustration metric against experimental data (B-Factor) and two structure-based metrics,
        we aim to validate sequence-derived evolutionary constraints in representing protein flexibility.
        <strong>Select a PDB ID from the dropdown menu to view the data.</strong>
    </p>
    <ul>
        <li><strong>Experimental Frustration</strong>: Derived via the Frustratometer using a crystal structure.</li>
        <li><strong>AF Frustration</strong>: Derived via the Frustratometer using an AlphaFold structure.</li>
        <li><strong>Evolutionary Frustration</strong>: Derived directly from sequence alignment (no structure needed).</li>
    </ul>
    <p>
        The correlation table below shows Spearman correlation coefficients and p-values for <em>non-smoothed</em> data across all loaded PDB IDs.
        The curves in the main plot are <em>smoothed</em> with a simple moving average and
        <strong>min–max normalized</strong> (per PDB ID). Normalization does not affect Spearman correlations but be mindful
        that min–max scaling is not suitable for comparing magnitudes <em>across</em> PDB IDs.
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

# Unity Container (no changes needed here)
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
        <li><strong>Folding (Q/E):</strong> Unfold/fold the protein. O toggles oscillation or sets height by B-factor in the unfolded state.</li>
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
unity_iframe.visible = True # Keep visible as per original

unity_container = column(
    description_visualizer,
    unity_iframe,
    sizing_mode='stretch_width'
)

# Controls section title
controls_section = Div(text="<b>Filter Correlation Table</b>", styles={'font-size': '16px', 'margin': '10px 0'})

# Custom styles (no changes needed here)
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

# (G) Bar Plot with Mean, SD (using original function structure)
def create_bar_plot_with_sd(data_proviz):
    """
    Creates a bar chart displaying the mean Spearman correlation for each frustration metric,
    with error bars representing the standard deviation.
    Adjusts the y-axis range to ensure whiskers are fully visible.
    """
    if data_proviz.empty: # Handle empty input df
        print("Warning: Cannot create bar plot, data_proviz is empty.")
        p_bar = figure(title="Mean Spearman Correlation (No Data)", height=400)
        p_bar.text(x=0, y=0, text="No data available for bar plot.")
        return p_bar

    # Compute mean and standard deviation of Spearman Rho per metric
    spearman_columns = ['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust']
    # Ensure columns exist before aggregation
    valid_spearman_cols = [col for col in spearman_columns if col in data_proviz.columns]
    if not valid_spearman_cols:
        print("Warning: No Spearman columns found in data_proviz for bar plot.")
        p_bar = figure(title="Mean Spearman Correlation (No Data)", height=400)
        p_bar.text(x=0, y=0, text="No Spearman data found.")
        return p_bar

    stats_corrs = data_proviz[valid_spearman_cols].agg(['mean', 'std']).transpose().reset_index()
    stats_corrs.rename(columns={
        'index': 'Metric', 'mean': 'Mean_Spearman_Rho', 'std': 'Std_Spearman_Rho'
    }, inplace=True)

    # Clean Metric names
    stats_corrs['Metric'] = stats_corrs['Metric'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

    # Assign colors based on Metric using the predefined FRUSTRATION_COLORS dictionary
    stats_corrs['Color'] = stats_corrs['Metric'].map(FRUSTRATION_COLORS)
    stats_corrs['Color'].fillna('gray', inplace=True) # Default color if metric not in map

    # Calculate upper and lower bounds for error bars
    stats_corrs['upper'] = stats_corrs['Mean_Spearman_Rho'] + stats_corrs['Std_Spearman_Rho']
    stats_corrs['lower'] = stats_corrs['Mean_Spearman_Rho'] - stats_corrs['Std_Spearman_Rho']

    # Create ColumnDataSource for the bar plot
    source_bar = ColumnDataSource(stats_corrs)

    # ** Adjust y-axis range to include padding **
    # Determine the minimum and maximum values for the y-axis, ignoring NaNs
    min_lower = np.nanmin(source_bar.data['lower']) if 'lower' in source_bar.data else np.nan
    max_upper = np.nanmax(source_bar.data['upper']) if 'upper' in source_bar.data else np.nan

    y_range_with_padding = None # Default to auto-range
    if not (np.isnan(min_lower) or np.isnan(max_upper)):
        y_range_val = max_upper - min_lower
        padding = y_range_val * 0.1 if y_range_val > 0 else 0.5 # Add padding
        y_range_with_padding = Range1d(start=min_lower - padding, end=max_upper + padding)


    # Create figure
    p_bar = figure(
        title="Mean Spearman Correlation between B-Factor and Frustration Metrics",
        x_axis_label="Frustration Metric", y_axis_label="Mean Spearman Rho",
        x_range=stats_corrs['Metric'].tolist(),
        y_range=y_range_with_padding, # Apply calculated range
        sizing_mode='stretch_width', height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save", toolbar_location="above"
    )

    # Add vertical bars and capture the renderer
    vbar_renderer = p_bar.vbar(
        x='Metric', top='Mean_Spearman_Rho', width=0.6, source=source_bar,
        color='Color', legend_field='Metric', line_color="black" # Use legend_field
    )

    # Add error bars using Whisker
    whisker = Whisker(
        base='Metric', upper='upper', lower='lower', source=source_bar,
        level="overlay", line_color='black' # Ensure whiskers are visible
    )
    p_bar.add_layout(whisker)

    # Add horizontal line at y=0 for reference
    p_bar.line(x=stats_corrs['Metric'].tolist(), y=0, line_width=1, line_dash='dashed', color='gray')

    # Customize hover tool and correctly reference the vbar renderer
    hover_bar = HoverTool(
        tooltips=[
            ("Metric", "@Metric"),
            ("Mean Spearman Rho", "@Mean_Spearman_Rho{0.3f}"),
            ("Std Dev", "@Std_Spearman_Rho{0.3f}")
        ],
        renderers=[vbar_renderer],  # Correctly pass the renderer
        mode='mouse', name="bar_hover" # Name the tool
    )
    # Check if tool already exists before adding
    existing_tool_names = [getattr(t, 'name', None) for t in p_bar.tools]
    if hover_bar.name not in existing_tool_names:
        p_bar.add_tools(hover_bar)


    # Remove legend as it's redundant with colors/axis labels
    p_bar.legend.visible = False
    # Optional: remove vertical grid lines
    p_bar.xgrid.grid_line_color = None

    return p_bar

# Create the bar plot
bar_plot_instance = create_bar_plot_with_sd(data_proviz)

# (F) Layout for Additional Plots section
additional_plots = column(
    Div(text="<h2>Overall Comparisons Across PDB IDs</h2>", styles={'margin-top': '20px'}), # Updated title
    p_avg_plot,
    p_std_plot,
    p_corr_plot,
    bar_plot_instance,  # Use the created bar plot instance
    sizing_mode='stretch_width',
    spacing=20,
    name="additional_plots"
)

# (G) Scatter Plots Layout (using original structure)
scatter_col_exp = column(
    p_scatter_exp, regression_info_exp, sizing_mode="stretch_width",
    styles={'flex': '1 1 350px', 'min-width': '350px'} # Original styles
)
scatter_col_af = column(
    p_scatter_af, regression_info_af, sizing_mode="stretch_width",
    styles={'flex': '1 1 350px', 'min-width': '350px'}
)
scatter_col_evol = column(
    p_scatter_evol, regression_info_evol, sizing_mode="stretch_width",
    styles={'flex': '1 1 350px', 'min-width': '350px'}
)

# Update scatter plots row with flex layout and minimum widths (original styles)
scatter_row = row(
    scatter_col_exp, scatter_col_af, scatter_col_evol,
    sizing_mode="stretch_width",
    styles={
        'display': 'flex', 'justify-content': 'space-between', 'gap': '20px',
        'width': '100%', 'margin': '20px auto 0 auto', # Added margin-top
        'flex-wrap': 'wrap'
    }
)

# (I) Main visualization section layout
visualization_section = column(
    row(select_pdb_id, window_slider, styles={'gap': '20px'}), # Use select_pdb_id
    p,
    scatter_row,
    # unity_container, # Unity container moved lower in original final layout
    # additional_plots,  # Additional plots moved lower
    sizing_mode='stretch_width',
    css_classes=['visualization-section'],
    name="main_vis_section" # Added name for clarity
)

# Correlation Table Section Layout
correlation_section = column(
    controls_section, # Title "Filter Correlation Table"
    controls_layout,  # Checkbox filters
    Spacer(height=15), # Add space
    data_table,
    sizing_mode='stretch_width',
    name="correlation_table_section" # Added name
)


# Main layout assembly (closer to original structure)
main_layout = column(
    custom_styles,
    header,
    visualization_section, # Dropdown, slider, line plot, scatter plots
    unity_container, # Unity visualizer section
    additional_plots, # Aggregated plots section
    correlation_section, # Correlation table and filters section
    # Footer added here in original structure, but seems better at the end
    sizing_mode='stretch_width'
)

# Add footer at the very end
# footer = Div(...) # Define footer if needed, or assume it's part of header Div

# Set up document
curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration Dashboard" # Updated title

print("Bokeh application layout created and added to document.")
# End of script
