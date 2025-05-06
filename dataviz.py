import os
import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, linregress

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, CheckboxGroup, Div, Spacer,
    DataTable, TableColumn, NumberFormatter, HoverTool,
    GlyphRenderer, Slider, Whisker, Label, Range1d
)
from bokeh.plotting import figure
from bokeh.layouts import column, row, layout
from bokeh.palettes import Category10
from math import pi # Import pi for label rotation

###############################################################################
# 1) Configuration
###############################################################################
# Local data directory path
DATA_DIR = "summary_data"  # Directory containing the summary files

# Filename pattern to match summary_XXXX.txt where XXXX is a 4-char alphanumeric PDB ID
FILE_PATTERN = r"^summary_[a-zA-Z0-9]{4}\.txt$" # MODIFIED: Pattern changed

# Default file to visualize on startup (Example PDB ID)
DEFAULT_FILE = "summary_1ABC.txt"  # MODIFIED: Changed to example PDB format or set to "" if no default needed

###############################################################################
# 2) Helpers: Data Parsing and Aggregation
###############################################################################

def get_pdb_id(filename):
    """
    Extracts the 4-character PDB ID from a filename matching summary_XXXX.txt.
    Returns the original filename if the pattern doesn't match.
    """
    # ADDED: Helper function to extract PDB ID
    match = re.match(r"^summary_([a-zA-Z0-9]{4})\.txt$", filename)
    if match:
        return match.group(1)
    return filename # Fallback

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
    Parses a summary file and returns original and processed DataFrames along with correlations.
    """
    required_cols = ["AlnIndex", "Residue", "B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]

    if not os.path.isfile(local_path):
        print(f"File not found: {local_path}")
        return None, None, {}

    try:
        df = pd.read_csv(local_path, sep='\t')
    except Exception as e:
        print(f"Skipping {local_path}: failed to parse data. Error: {e}")
        return None, None, {}

    # Check for required columns
    if not set(required_cols).issubset(df.columns):
        print(f"Skipping {local_path}: missing required columns.")
        return None, None, {}

    # Replace 'n/a' with NaN and convert to float
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_original = df.copy()
    df_for_plot = df.copy()

    # Apply moving average (done later in update_plot based on slider)
    # Min-Max normalization (done later in update_plot)

    # Compute Spearman correlations on original data
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
            if sub[mA].nunique() < 2 or sub[mB].nunique() < 2:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(sub[mA], sub[mB])
            corrs[(mA, mB)] = (rho, pval)

    # Return only original DF now, processing happens in update_plot
    return df_original, corrs # MODIFIED: Removed processed DF return here

def remove_regression_renderers(fig):
    """
    Removes all renderers from the given figure whose names start with 'regression_'.
    Safely handles cases where the renderer or its name might be None.
    """
    new_renderers = []
    renderers_to_remove = set() # Track names to avoid issues during iteration

    # First pass: Identify renderers to remove by name
    for r in fig.renderers:
        if r is not None:
            name = getattr(r, 'name', '')
            if isinstance(name, str) and name.startswith('regression_'):
                 renderers_to_remove.add(name)

    # Second pass: Build the new list excluding the identified renderers
    for r in fig.renderers:
        if r is not None:
            name = getattr(r, 'name', '')
            if isinstance(name, str) and name in renderers_to_remove:
                continue # Skip renderers related to regression
        new_renderers.append(r)

    fig.renderers = new_renderers

    # Also remove associated HoverTools for regression lines
    new_tools = [tool for tool in fig.tools if not (isinstance(tool, HoverTool) and tool.renderers and getattr(tool.renderers[0], 'name', '').startswith('regression_'))]
    fig.tools = new_tools


###############################################################################
# 3) Load and Aggregate Data from Local Directory
###############################################################################
data_by_file = {}
all_corr_rows = []

# Aggregation lists
pdb_ids = [] # MODIFIED: Changed from protein_names to pdb_ids
avg_bfactors = []
std_bfactors = []
spearman_exp = []
spearman_af = []
spearman_evol = []

# Possible frustration columns
POSSIBLE_FRUST_COLUMNS = ['ExpFrust', 'AFFrust', 'EvolFrust']

# Color mapping for plots
FRUSTRATION_COLORS = {
    "ExpFrust.": Category10[10][0],  # Red
    "AFFrust.": Category10[10][1],   # Blue
    "EvolFrust.": Category10[10][2]  # Green
}

# Iterate through files
for filename in os.listdir(DATA_DIR):
    if not re.match(FILE_PATTERN, filename):
        print(f"Skipping {filename}: does not match pattern {FILE_PATTERN}")
        continue

    file_path = os.path.join(DATA_DIR, filename)
    # MODIFIED: parse_summary_file now only returns df_orig, corrs
    df_orig, corrs = parse_summary_file(file_path)
    if df_orig is None:
        continue

    pdb_id = get_pdb_id(filename) # ADDED: Extract PDB ID

    # Store original data by full filename for easy lookup
    data_by_file[filename] = {
        "df_original": df_orig,
        "corrs": corrs,
        "pdb_id": pdb_id # ADDED: Store PDB ID here too if needed elsewhere
    }

    # Collect correlation data using PDB ID
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo
        # MODIFIED: Use pdb_id instead of filename in correlation rows
        all_corr_rows.append([pdb_id, mA, mB, rho, pval])

    # Aggregate data for additional plots using PDB ID
    avg_b = df_orig['B_Factor'].mean()
    std_b = df_orig['B_Factor'].std()

    spearman_r_exp = corrs.get(("B_Factor", "ExpFrust"), (np.nan, np.nan))[0]
    spearman_r_af = corrs.get(("B_Factor", "AFFrust"), (np.nan, np.nan))[0]
    spearman_r_evol = corrs.get(("B_Factor", "EvolFrust"), (np.nan, np.nan))[0]

    pdb_ids.append(pdb_id) # MODIFIED: Append pdb_id
    avg_bfactors.append(avg_b)
    std_bfactors.append(std_b)
    spearman_exp.append(spearman_r_exp)
    spearman_af.append(spearman_r_af)
    spearman_evol.append(spearman_r_evol)

# Correlation DataFrame (Now uses PDB ID in 'Test' column)
# MODIFIED: Renamed 'Test' column to 'PDB_ID'
df_all_corr = pd.DataFrame(all_corr_rows, columns=["PDB_ID","MetricA","MetricB","Rho","Pval"])

# Aggregated DataFrame for Additional Plots (Now uses PDB ID in 'PDB_ID' column)
# MODIFIED: Renamed 'Protein' column to 'PDB_ID'
data_proviz = pd.DataFrame({
    'PDB_ID': pdb_ids, # MODIFIED: Use pdb_ids list
    'Avg_B_Factor': avg_bfactors,
    'Std_B_Factor': std_bfactors,
    'Spearman_ExpFrust': spearman_exp,
    'Spearman_AFFrust': spearman_af,
    'Spearman_EvolFrust': spearman_evol
})

# Melt data for plotting
data_long_avg = data_proviz.melt(
    id_vars=['PDB_ID', 'Avg_B_Factor'], # MODIFIED: Use 'PDB_ID'
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)

data_long_std = data_proviz.melt(
    id_vars=['PDB_ID', 'Std_B_Factor'], # MODIFIED: Use 'PDB_ID'
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
        legend_label=label, name=f"line_{col_key}" # Add name for potential future use
    )
    renderers[col_key] = renderer
    # Assign renderers to hover tools AFTER they are created
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
# Scatter plots configuration with disabled wheel zoom by default
p_scatter_exp = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="Normalized B-Factor",
    y_axis_label="Normalized ExpFrust", # MODIFIED: Specific y-axis label
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"], # Removed default hover
    active_drag="box_zoom",
    active_scroll=None
)
p_scatter_af = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="Normalized B-Factor",
    y_axis_label="Normalized AFFrust", # MODIFIED: Specific y-axis label
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"], # Removed default hover
    active_drag="box_zoom",
    active_scroll=None
)
p_scatter_evol = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="Normalized B-Factor",
    y_axis_label="Normalized EvolFrust", # MODIFIED: Specific y-axis label
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"], # Removed default hover
    active_drag="box_zoom",
    active_scroll=None
)

# ColumnDataSources will now include normalized data and original data for tooltips
source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[])) # Added residue
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[])) # Added residue
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[])) # Added residue


# Create Div elements for regression info (No changes needed here)
# ... (regression_info_exp, _af, _evol definitions remain the same) ...
regression_info_exp = Div(
    text="",
    styles={
        'background-color': '#f8f9fa', 'padding': '10px', 'border': '1px solid #ddd',
        'border-radius': '4px', 'margin-top': '10px', 'font-size': '14px',
        'text-align': 'center', 'width': '100%'
    },
    sizing_mode="stretch_width"
)
regression_info_af = Div(
    text="",
    styles={
        'background-color': '#f8f9fa', 'padding': '10px', 'border': '1px solid #ddd',
        'border-radius': '4px', 'margin-top': '10px', 'font-size': '14px',
        'text-align': 'center', 'width': '100%'
    },
    sizing_mode="stretch_width"
)
regression_info_evol = Div(
    text="",
    styles={
        'background-color': '#f8f9fa', 'padding': '10px', 'border': '1px solid #ddd',
        'border-radius': '4px', 'margin-top': '10px', 'font-size': '14px',
        'text-align': 'center', 'width': '100%'
    },
    sizing_mode="stretch_width"
)


# Initial scatter glyphs, assign names AND CAPTURE RENDERERS
# MODIFICATION START
scatter_renderer_exp = p_scatter_exp.scatter(
    "x", "y", source=source_scatter_exp, color=Category10[10][1], alpha=0.7, name='scatter_exp'
)
scatter_renderer_af = p_scatter_af.scatter(
    "x", "y", source=source_scatter_af,  color=Category10[10][2], alpha=0.7, name='scatter_af'
)
scatter_renderer_evol = p_scatter_evol.scatter(
    "x", "y", source=source_scatter_evol, color=Category10[10][3], alpha=0.7, name='scatter_evol'
)

# Define and Add HoverTool AFTER creating the scatter renderers
# Using the `renderers` attribute correctly
hover_scatter_exp = HoverTool(
    renderers=[scatter_renderer_exp], # Use the captured renderer object
    tooltips=[("Residue", "@residue"), ("Orig B-Factor", "@x_orig{0.0f}"), ("Orig ExpFrust", "@y_orig{0.3f}")]
    # Removed incorrect 'names' attribute
)
hover_scatter_af = HoverTool(
    renderers=[scatter_renderer_af], # Use the captured renderer object
    tooltips=[("Residue", "@residue"), ("Orig B-Factor", "@x_orig{0.0f}"), ("Orig AFFrust", "@y_orig{0.3f}")]
    # Removed incorrect 'names' attribute
)
hover_scatter_evol = HoverTool(
    renderers=[scatter_renderer_evol], # Use the captured renderer object
    tooltips=[("Residue", "@residue"), ("Orig B-Factor", "@x_orig{0.0f}"), ("Orig EvolFrust", "@y_orig{0.3f}")]
    # Removed incorrect 'names' attribute
)

p_scatter_exp.add_tools(hover_scatter_exp)
p_scatter_af.add_tools(hover_scatter_af)
p_scatter_evol.add_tools(hover_scatter_evol)

def add_regression_line_and_info(fig, xvals, yvals, color="black", info_div=None, plot_type=""):
    """
    Adds a linear regression line and updates the regression info Div.
    Uses NORMALIZED data for regression fitting and line plotting.
    """
    if info_div: info_div.text = "" # Clear previous info
    if len(xvals) < 2 or np.all(np.isnan(xvals)) or np.all(np.isnan(yvals)):
        if info_div: info_div.text = "Insufficient data for regression"
        return

    not_nan = ~np.isnan(xvals) & ~np.isnan(yvals)
    if not any(not_nan) or sum(not_nan) < 2:
        if info_div: info_div.text = "Insufficient valid data points for regression"
        return

    xvals_clean = xvals[not_nan]
    yvals_clean = yvals[not_nan]
    if len(xvals_clean) < 2 or np.all(xvals_clean == xvals_clean[0]): # Check for variance in x
        if info_div: info_div.text = "Insufficient variance in data for regression"
        return

    # Linear regression on NORMALIZED data
    try:
        slope, intercept, r_value, p_value, std_err = linregress(xvals_clean, yvals_clean)
    except ValueError as e:
        print(f"Regression failed for {plot_type}: {e}")
        if info_div: info_div.text = "Regression calculation failed"
        return

    # Plot regression line based on normalized range
    x_min, x_max = np.nanmin(xvals_clean), np.nanmax(xvals_clean)
    if x_min == x_max: x_max += 1e-6 # Avoid zero range
    x_range = np.linspace(x_min, x_max, 100)
    y_range = slope * x_range + intercept

    regression_line_name = f'regression_line_{plot_type}'
    regression_hover_name = f'regression_hover_{plot_type}' # For the hover source

    # Create a source for the regression line data and hover info
    regression_source = ColumnDataSource(data=dict(
        x=x_range,
        y=y_range,
        equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
    ))

    # Plot the visible regression line
    regression_line = fig.line(
        'x', 'y',
        source=regression_source,
        line_width=2, line_dash='dashed', color=color,
        name=regression_line_name # Name the visible line
    )

    # Add a separate HoverTool specifically for this regression line
    hover_regression = HoverTool(
        renderers=[regression_line], # Attach only to the visible line
        tooltips=[
            ("Regression Equation", "@equation"),
            ("R²", f"{r_value**2:.3f}") # Include R-squared in tooltip
        ],
        mode='mouse',
        name=f'regression_hovertool_{plot_type}' # Give the tool a name
    )
    if not any(isinstance(tool, HoverTool) and tool.name == hover_regression.name for tool in fig.tools):
         fig.add_tools(hover_regression)

    # Update regression info div with equation and R²
    if info_div:
        info_div.text = f"""
        <div style='color: {color}'>
            <strong>y = {slope:.3f}x + {intercept:.3f}</strong><br>
            <span style='font-size: 12px'>R² = {r_value**2:.3f}</span>
        </div>
        """

# Dropdown select
# MODIFIED: Options remain full filenames, but the title is changed
# User sees PDB IDs, but the value passed internally is the filename
file_options = sorted(data_by_file.keys())
# Generate display options (PDB IDs) corresponding to file_options
# Bokeh Select doesn't directly support (value, display) tuples easily.
# We will keep options as filenames and extract PDB ID for display in titles etc.
if DEFAULT_FILE and DEFAULT_FILE in file_options:
    initial_file = DEFAULT_FILE
elif file_options:
    initial_file = file_options[0]
else:
    initial_file = ""

select_file = Select(
    title="Select Protein PDB ID:", # MODIFIED: Title changed
    value=initial_file,
    options=file_options # Keep full filenames as options for lookup
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

def min_max_normalize(arr):
    """
    Applies min-max normalization to a numpy array. Handles NaN and division by zero.
    """
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return arr # Return original if all NaN

    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)

    if arr_max > arr_min:
        normalized_arr = (arr - arr_min) / (arr_max - arr_min)
        return normalized_arr
    elif arr_max == arr_min:
         # If all valid values are the same, normalize to 0.5 (or 0 or 1)
         # Keep NaNs as NaNs
         out = np.full_like(arr, 0.5, dtype=np.float64)
         out[np.isnan(arr)] = np.nan
         return out
    else: # Should not happen if valid_mask is true, but for safety
        return arr


def update_plot(attr, old, new):
    """
    Updates both the main plot and scatter plots when a new file is selected or slider changes.
    """
    filename = select_file.value
    if not filename or filename not in data_by_file:
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        source_scatter_exp.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[])
        source_scatter_af.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[])
        source_scatter_evol.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[])
        p.title.text = "(No Data Selected)"
        p_scatter_exp.title.text = ""
        p_scatter_af.title.text = ""
        p_scatter_evol.title.text = ""
        regression_info_exp.text = ""
        regression_info_af.text = ""
        regression_info_evol.text = ""
        # Remove any lingering regression lines/tools if no data
        remove_regression_renderers(p_scatter_exp)
        remove_regression_renderers(p_scatter_af)
        remove_regression_renderers(p_scatter_evol)
        return

    # Get PDB ID for display
    pdb_id = data_by_file[filename]["pdb_id"] # Fetch stored PDB ID

    # Get window size from slider
    window_size = window_slider.value

    # --- Update main line plot ---
    df_orig = data_by_file[filename]["df_original"]
    df_plot = df_orig.copy() # Start with fresh copy for processing

    metrics_to_process = ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]

    # Apply moving average with current window size
    for col in metrics_to_process:
        arr = df_plot[col].values.astype(float) # Ensure float type
        df_plot[col] = moving_average(arr, window_size=window_size)

    # Normalize the smoothed data for the line plot
    for col in metrics_to_process:
        df_plot[col] = min_max_normalize(df_plot[col].values)

    # Prepare data for Bokeh ColumnDataSource (handle potential NaNs after processing)
    plot_data = {
        "x": df_plot["AlnIndex"],
        "residue": df_plot["Residue"],
        "b_factor": df_plot["B_Factor"],
        "exp_frust": df_plot["ExpFrust"],
        "af_frust": df_plot["AFFrust"],
        "evol_frust": df_plot["EvolFrust"]
    }
    source_plot.data = plot_data
    # MODIFIED: Use pdb_id in title
    p.title.text = f"{pdb_id}: Smoothed & Normalized Flexibility/Frustration"

    # --- Update scatter plots (using ORIGINAL, NON-smoothed data) ---
    df_scatter_orig = data_by_file[filename]["df_original"].copy()

    # **Remove all existing regression renderers before adding new ones**
    remove_regression_renderers(p_scatter_exp)
    remove_regression_renderers(p_scatter_af)
    remove_regression_renderers(p_scatter_evol)

    # Reset data sources
    source_scatter_exp.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[])
    source_scatter_af.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[])
    source_scatter_evol.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[])

    # Reset regression info divs
    regression_info_exp.text = ""
    regression_info_af.text = ""
    regression_info_evol.text = ""

    # Prepare data for scatter plots (normalize original B-Factor vs original Frustration)
    b_factor_orig = df_scatter_orig["B_Factor"].values.astype(float)
    exp_frust_orig = df_scatter_orig["ExpFrust"].values.astype(float)
    af_frust_orig = df_scatter_orig["AFFrust"].values.astype(float)
    evol_frust_orig = df_scatter_orig["EvolFrust"].values.astype(float)
    residues = df_scatter_orig["Residue"].tolist()

    # Normalize for plotting axes [0, 1]
    b_factor_norm = min_max_normalize(b_factor_orig)
    exp_frust_norm = min_max_normalize(exp_frust_orig)
    af_frust_norm = min_max_normalize(af_frust_orig)
    evol_frust_norm = min_max_normalize(evol_frust_orig)

    # Check if there's valid data to plot *after* normalization attempts
    valid_exp = ~np.isnan(b_factor_norm) & ~np.isnan(exp_frust_norm)
    valid_af = ~np.isnan(b_factor_norm) & ~np.isnan(af_frust_norm)
    valid_evol = ~np.isnan(b_factor_norm) & ~np.isnan(evol_frust_norm)

    # Update ExpFrust Scatter
    if np.any(valid_exp):
        source_scatter_exp.data = dict(
            x=b_factor_norm, y=exp_frust_norm,
            x_orig=b_factor_orig, y_orig=exp_frust_orig, residue=residues
        )
        p_scatter_exp.title.text = f"{pdb_id}: B-Factor vs ExpFrust (Normalized)"
        add_regression_line_and_info(
            fig=p_scatter_exp,
            xvals=b_factor_norm, # Use normalized data for regression fit/plot
            yvals=exp_frust_norm,
            color=Category10[10][1],
            info_div=regression_info_exp,
            plot_type="exp"
        )
    else:
        p_scatter_exp.title.text = f"{pdb_id}: B-Factor vs ExpFrust (No Valid Data)"

    # Update AFFrust Scatter
    if np.any(valid_af):
        source_scatter_af.data = dict(
            x=b_factor_norm, y=af_frust_norm,
            x_orig=b_factor_orig, y_orig=af_frust_orig, residue=residues
        )
        p_scatter_af.title.text = f"{pdb_id}: B-Factor vs AFFrust (Normalized)"
        add_regression_line_and_info(
            fig=p_scatter_af,
            xvals=b_factor_norm,
            yvals=af_frust_norm,
            color=Category10[10][2],
            info_div=regression_info_af,
            plot_type="af"
        )
    else:
         p_scatter_af.title.text = f"{pdb_id}: B-Factor vs AFFrust (No Valid Data)"

    # Update EvolFrust Scatter
    if np.any(valid_evol):
        source_scatter_evol.data = dict(
            x=b_factor_norm, y=evol_frust_norm,
            x_orig=b_factor_orig, y_orig=evol_frust_orig, residue=residues
        )
        p_scatter_evol.title.text = f"{pdb_id}: B-Factor vs EvolFrust (Normalized)"
        add_regression_line_and_info(
            fig=p_scatter_evol,
            xvals=b_factor_norm,
            yvals=evol_frust_norm,
            color=Category10[10][3],
            info_div=regression_info_evol,
            plot_type="evol"
        )
    else:
        p_scatter_evol.title.text = f"{pdb_id}: B-Factor vs EvolFrust (No Valid Data)"


select_file.on_change("value", update_plot)
# Initialize plot with default file if available
if initial_file:
    update_plot(None, None, initial_file)


###############################################################################
# 5) CORRELATION TABLE AND FILTERS
###############################################################################

# (D) CORRELATION TABLE
# MODIFIED: Changed 'Test' column to 'PDB_ID' and updated title
if df_all_corr.empty:
    columns = [
        TableColumn(field="PDB_ID", title="PDB ID"), # MODIFIED title
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Rho"),
        TableColumn(field="Pval", title="p-value")
    ]
    source_corr = ColumnDataSource(dict(PDB_ID=[], MetricA=[], MetricB=[], Rho=[], Pval=[])) # MODIFIED field name
    data_table = DataTable(columns=columns, source=source_corr, height=400, width=1200, sizing_mode="stretch_width")
else:
    source_corr = ColumnDataSource(df_all_corr) # df_all_corr now uses 'PDB_ID'
    columns = [
        TableColumn(field="PDB_ID", title="PDB ID"), # MODIFIED title
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Spearman Rho", formatter=NumberFormatter(format="0.3f")),
        TableColumn(field="Pval", title="p-value", formatter=NumberFormatter(format="0.2e"))
    ]
    data_table = DataTable(columns=columns, source=source_corr, height=400, width=1200, sizing_mode="stretch_width")

# (E) FILTERS for correlation table

# Define helper function to split labels into columns
def split_labels(labels, num_columns):
    if not labels or num_columns <= 0: return [labels]
    k, m = divmod(len(labels), num_columns)
    return [labels[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_columns)]

# Define the number of columns for better layout
NUM_COLUMNS = 3 # Adjust as needed based on expected number of PDB IDs

# MODIFIED: Use 'PDB_ID' column from df_all_corr
tests_in_corr = sorted(df_all_corr["PDB_ID"].unique()) if not df_all_corr.empty else []
if not df_all_corr.empty:
    combo_options = sorted({
        f"{row['MetricA']} vs {row['MetricB']}"
        for _, row in df_all_corr.iterrows()
    })
else:
    combo_options = []

if tests_in_corr: # MODIFIED: Check if list is not empty
    # Split PDB ID labels into columns
    test_labels_split = split_labels(tests_in_corr, NUM_COLUMNS) # Now contains PDB IDs
    combo_labels_split = split_labels(combo_options, NUM_COLUMNS)

    # Create CheckboxGroups for PDB IDs
    checkbox_tests_columns = [
        CheckboxGroup(
            labels=col_labels, # Labels are PDB IDs
            active=[], name=f'tests_column_{i+1}'
        ) for i, col_labels in enumerate(test_labels_split)
    ]

    # Create CheckboxGroups for Metric Pairs
    checkbox_combos_columns = [
        CheckboxGroup(
            labels=col_labels,
            active=[], name=f'combos_column_{i+1}'
        ) for i, col_labels in enumerate(combo_labels_split)
    ]
else: # Handle case with no data
    checkbox_tests_columns = [CheckboxGroup(labels=[], active=[], name='tests_column_1')]
    checkbox_combos_columns = [CheckboxGroup(labels=[], active=[], name='combos_column_1')]

# Create Columns for Tests and Metric Pairs
tests_layout = row(*checkbox_tests_columns, sizing_mode='stretch_width') # Removed fixed width
combos_layout = row(*checkbox_combos_columns, sizing_mode='stretch_width') # Removed fixed width

# Add Titles Above Each CheckboxGroup
tests_title = Div(text="<b>Select PDB IDs:</b>", styles={'font-size': '14px', 'margin-bottom': '5px'}) # MODIFIED Title
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
    selected = []
    for checkbox in checkbox_columns:
        if checkbox.labels: # Check if labels exist
            selected.extend([checkbox.labels[i] for i in checkbox.active])
    return selected

def update_corr_filter(attr, old, new):
    """Filter correlation table based on selected PDB IDs and metric pairs."""
    if df_all_corr.empty: return

    # Aggregate selected PDB IDs and metric pairs
    selected_tests = get_selected_labels(checkbox_tests_columns) # Contains selected PDB IDs
    selected_combos = get_selected_labels(checkbox_combos_columns)

    filtered = df_all_corr.copy() # Start with full data

    # Filter by selected PDB IDs if any are selected
    if selected_tests:
        # MODIFIED: Filter using 'PDB_ID' column
        filtered = filtered[filtered["PDB_ID"].isin(selected_tests)]

    # Filter by selected metric combinations if any are selected
    if selected_combos:
        # Create temporary combo string column for filtering
        filtered["combo_str"] = filtered.apply(lambda r: f"{r['MetricA']} vs {r['MetricB']}", axis=1)
        filtered = filtered[filtered["combo_str"].isin(selected_combos)]
        # Drop the temporary column
        if "combo_str" in filtered.columns:
            filtered = filtered.drop(columns=["combo_str"])

    source_corr.data = filtered.to_dict(orient="list")


# Attach callbacks to all CheckboxGroups
for checkbox in checkbox_tests_columns + checkbox_combos_columns:
    checkbox.on_change('active', update_corr_filter)

###############################################################################
# 6) Additional Aggregated Plots (Converted from Plotly to Bokeh)
###############################################################################
# Note: These plots now use the `data_proviz` DataFrame which has a 'PDB_ID' column

# (F) Spearman Rho vs Average B-Factor
source_avg_plot = ColumnDataSource(data_long_avg) # data_long_avg now contains 'PDB_ID'

p_avg_plot = figure(
    title="Spearman Correlation vs Average B-Factor",
    x_axis_label="Average B-Factor",
    y_axis_label="Spearman Correlation (Frustration vs B-Factor)", # Clarified label
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save,hover", # Added hover tool here
    active_drag="box_zoom",
    active_scroll=None,
    tooltips=[ # Define tooltips directly here for the scatter points
        ("PDB ID", "@PDB_ID"), # MODIFIED: Use PDB_ID
        ("Frustration Type", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}"),
        ("Avg B-Factor", "@Avg_B_Factor{0.1f}")
    ]
)

# Define color palette for Frustration Types
frust_types_avg = data_long_avg['Frust_Type'].unique().tolist()
palette_avg = Category10[max(3, len(frust_types_avg))]
color_map_frust_avg = {frust: palette_avg[i] for i, frust in enumerate(frust_types_avg)}

# Add scatter glyphs with legend grouping
for frust in frust_types_avg:
    subset = data_long_avg[data_long_avg['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    p_avg_plot.scatter(
        'Avg_B_Factor', 'Spearman_Rho',
        source=source_subset,
        color=color_map_frust_avg[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1
    )

    # Add regression lines for each frustration type
    if len(subset) >= 2 and subset['Avg_B_Factor'].nunique() > 1: # Need at least 2 points with different x values
        x_vals = subset['Avg_B_Factor'].values
        y_vals = subset['Spearman_Rho'].values
        # Filter out NaNs just for regression
        valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        if valid.sum() >= 2:
            x_clean, y_clean = x_vals[valid], y_vals[valid]
            if np.std(x_clean) > 1e-6: # Ensure variance
                 try:
                    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
                    x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_range = slope * x_range + intercept
                    p_avg_plot.line(x_range, y_range, color=color_map_frust_avg[frust], line_dash='dashed', line_width=2, alpha=0.8)
                 except ValueError as e:
                    print(f"Regression failed for AVG plot, {frust}: {e}")


p_avg_plot.legend.location = "top_left"
p_avg_plot.legend.title = "Frustration Type"
p_avg_plot.legend.click_policy = "mute"


# (G) Spearman Rho vs Std Dev of B-Factor
source_std_plot = ColumnDataSource(data_long_std) # data_long_std now contains 'PDB_ID'

p_std_plot = figure(
    title="Spearman Correlation vs Std Dev of B-Factor",
    x_axis_label="Standard Deviation of B-Factor",
    y_axis_label="Spearman Correlation (Frustration vs B-Factor)", # Clarified label
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save,hover", # Added hover tool
    active_drag="box_zoom",
    active_scroll=None,
    tooltips=[ # Define tooltips for scatter points
        ("PDB ID", "@PDB_ID"), # MODIFIED: Use PDB_ID
        ("Frustration Type", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}"),
        ("Std Dev B-Factor", "@Std_B_Factor{0.1f}")
    ]
)

# Define color palette for Frustration Types
frust_types_std = data_long_std['Frust_Type'].unique().tolist()
palette_std = Category10[max(3, len(frust_types_std))]
color_map_frust_std = {frust: palette_std[i] for i, frust in enumerate(frust_types_std)}

# Add scatter glyphs with legend grouping
for frust in frust_types_std:
    subset = data_long_std[data_long_std['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    p_std_plot.scatter(
        'Std_B_Factor', 'Spearman_Rho',
        source=source_subset,
        color=color_map_frust_std[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1
    )

    # Add regression lines for each frustration type
    if len(subset) >= 2 and subset['Std_B_Factor'].nunique() > 1:
        x_vals = subset['Std_B_Factor'].values
        y_vals = subset['Spearman_Rho'].values
        valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        if valid.sum() >= 2:
             x_clean, y_clean = x_vals[valid], y_vals[valid]
             if np.std(x_clean) > 1e-6: # Ensure variance
                 try:
                    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
                    x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_range = slope * x_range + intercept
                    p_std_plot.line(x_range, y_range, color=color_map_frust_std[frust], line_dash='dashed', line_width=2, alpha=0.8)
                 except ValueError as e:
                     print(f"Regression failed for STD plot, {frust}: {e}")

p_std_plot.legend.location = "top_left"
p_std_plot.legend.title = "Frustration Type"
p_std_plot.legend.click_policy = "mute"

# (H) Spearman Rho per Protein and Frustration Metric
# Melt data_proviz for the third plot
# MODIFIED: Use PDB_ID as id_var
data_long_corr = data_proviz.melt(
    id_vars=['PDB_ID'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)

# Clean Frust_Type names
data_long_corr['Frust_Type'] = data_long_corr['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

# Remove rows with NaN correlations
data_long_corr.dropna(subset=['Spearman_Rho'], inplace=True)

# Use PDB IDs for x-range if data exists
x_range_corr = sorted(data_proviz['PDB_ID'].unique().tolist()) if not data_proviz.empty else []

p_corr_plot = figure(
    title="Spearman Correlation per PDB ID and Frustration Metric", # MODIFIED title
    x_axis_label="PDB ID", # MODIFIED label
    y_axis_label="Spearman Correlation (Frustration vs B-Factor)", # Clarified label
    x_range=x_range_corr, # Use PDB IDs for x-axis categories
    sizing_mode='stretch_width',
    height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save,hover", # Added hover tool
    active_drag="box_zoom",
    active_scroll=None,
    toolbar_location="above",
    tooltips=[ # Define tooltips for scatter points
        ("PDB ID", "@PDB_ID"), # MODIFIED: Use PDB_ID
        ("Frustration Metric", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ]
)

# Define color palette for Frustration Types
if not data_long_corr.empty:
    frust_types_corr = data_long_corr['Frust_Type'].unique().tolist()
    palette_corr = Category10[max(3, len(frust_types_corr))]
    color_map_corr = {frust: palette_corr[i] for i, frust in enumerate(frust_types_corr)}
else:
    frust_types_corr = []
    color_map_corr = {}

# Add horizontal line at y=0
if x_range_corr: # Only add if there's an x-range
    p_corr_plot.line(
        x=[x_range_corr[0], x_range_corr[-1]], # Use actual PDB IDs for range if possible
        y=[0, 0],
        line_width=1,
        line_dash='dashed',
        color='gray',
        name='y_zero_line'
    )

# Add scatter glyphs (points colored by frustration type)
for frust in frust_types_corr:
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    p_corr_plot.scatter(
        'PDB_ID', 'Spearman_Rho', # Use PDB_ID on x-axis
        source=source_subset,
        color=color_map_corr[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1
    )

# Add mean lines for each frustration type across all proteins
if x_range_corr: # Only add if there's an x-range
    for frust in frust_types_corr:
        subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
        if not subset.empty:
            mean_value = subset['Spearman_Rho'].mean()
            p_corr_plot.line(
                x=[x_range_corr[0], x_range_corr[-1]], # Span across all proteins
                y=[mean_value, mean_value],
                color=color_map_corr[frust],
                line_dash='dotted', # Use dotted for mean lines
                line_width=2,
                legend_label=f"{frust} (Mean: {mean_value:.3f})" # Add mean value to legend
            )


p_corr_plot.legend.location = "top_left"
p_corr_plot.legend.title = "Frustration Type"
p_corr_plot.legend.click_policy = "mute"
p_corr_plot.legend.label_text_font_size = "8pt" # Make legend text smaller if needed

# Rotate x-axis labels to prevent overlapping
p_corr_plot.xaxis.major_label_orientation = pi / 4  # 45 degrees
p_corr_plot.xaxis.major_label_text_font_size = "8pt" # Make labels smaller

###############################################################################
# 7) User Interface Components
###############################################################################

# Add header and description (No changes needed here unless text requires update)
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
        The correlation table below shows Spearman correlation coefficients and p-values for <em>original (non-smoothed)</em> data.
        The curves in the main plot above are <em>smoothed</em> with a simple moving average (window size adjustable via slider) and
        <strong>min–max normalized</strong> per protein to the [0, 1] range for visualization. The scatter plots below the main plot show
        <em>original (non-smoothed)</em> B-factors vs. Frustration metrics, also min-max normalized for plotting axes, with regression lines fit to this normalized data.
        Normalization aids visualization but doesn't affect Spearman correlations. Min–max scaling is not suitable for comparing magnitudes <em>across</em> proteins.
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

# Unity Container (No changes needed here)
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
unity_iframe.visible = True # Set to False if you want to hide it initially

unity_container = column(
    description_visualizer,
    unity_iframe,
    sizing_mode='stretch_width'
)

# Controls section for Table Filter (Title)
controls_section = Div(text="<b>Filter Correlation Table</b>", styles={'font-size': '16px', 'margin': '10px 0'})

# Custom styles (No changes needed)
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
        /* Optional: Style for regression info divs */
        .bk-clearfix { margin-top: 5px; }
    </style>
""")

# (G) Bar Plot with Mean, SD (Function `create_bar_plot_with_sd`)
def create_bar_plot_with_sd(data_proviz_df):
    """
    Creates a bar chart displaying the mean Spearman correlation for each frustration metric,
    with error bars representing the standard deviation.
    """
    # Compute mean and standard deviation of Spearman Rho per metric
    spearman_columns = ['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust']
    # Check if columns exist and data is not empty
    valid_columns = [col for col in spearman_columns if col in data_proviz_df.columns]
    if not valid_columns or data_proviz_df.empty:
        print("Insufficient data for bar plot.")
        # Return an empty placeholder or message
        return Div(text="Bar plot cannot be generated: Insufficient data.")

    stats_corrs = data_proviz_df[valid_columns].agg(['mean', 'std']).transpose().reset_index()
    stats_corrs.rename(columns={
        'index': 'Metric',
        'mean': 'Mean_Spearman_Rho',
        'std': 'Std_Spearman_Rho'
    }, inplace=True)

    # Clean Metric names
    stats_corrs['Metric'] = stats_corrs['Metric'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

    # Assign colors based on Metric using the predefined FRUSTRATION_COLORS dictionary
    stats_corrs['Color'] = stats_corrs['Metric'].map(FRUSTRATION_COLORS).fillna('gray') # Use gray for unmapped

    # Create ColumnDataSource for the bar plot
    source_bar = ColumnDataSource(stats_corrs)

    # Create figure
    p_bar = figure(
        title="Mean Spearman Correlation between B-Factor and Frustration Metrics (All PDBs)",
        x_axis_label="Frustration Metric",
        y_axis_label="Mean Spearman Rho",
        x_range=stats_corrs['Metric'].tolist(),
        sizing_mode='stretch_width',
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save", # Removed hover from default tools
        toolbar_location="above"
    )

    # Add vertical bars and capture the renderer
    vbar_renderer = p_bar.vbar(
        x='Metric',
        top='Mean_Spearman_Rho',
        width=0.6,
        source=source_bar,
        color='Color',
        # legend_label="Frustration Metric", # Removed - redundant with x-axis
        line_color="black"
    )

    # Calculate upper and lower bounds for error bars (handle NaN std dev)
    source_bar.data['upper'] = source_bar.data['Mean_Spearman_Rho'] + source_bar.data['Std_Spearman_Rho'].fillna(0)
    source_bar.data['lower'] = source_bar.data['Mean_Spearman_Rho'] - source_bar.data['Std_Spearman_Rho'].fillna(0)

    # Add error bars using Whisker
    whisker = Whisker(
        base='Metric',
        upper='upper',
        lower='lower',
        source=source_bar,
        level="overlay",
        line_color='black' # Style the whiskers
    )
    p_bar.add_layout(whisker)


    # Adjust y-axis range to include padding
    min_val = source_bar.data['lower'].min() if source_bar.data['lower'].size > 0 else 0
    max_val = source_bar.data['upper'].max() if source_bar.data['upper'].size > 0 else 1
    y_range_span = max_val - min_val if (max_val - min_val) > 1e-6 else 1.0
    y_padding = y_range_span * 0.1

    p_bar.y_range = Range1d(start=min_val - y_padding, end=max_val + y_padding)

    # Add horizontal line at y=0 for reference
    p_bar.line(x=stats_corrs['Metric'].tolist(), y=0, line_width=1, line_dash='dashed', color='gray')

    # Add hover tool specifically for the bars
    hover_bar = HoverTool(
        tooltips=[
            ("Metric", "@Metric"),
            ("Mean Spearman Rho", "@Mean_Spearman_Rho{0.3f}"),
            ("Std Dev", "@Std_Spearman_Rho{0.3f}")
        ],
        renderers=[vbar_renderer], # Attach only to the bars
        mode='mouse'
    )
    p_bar.add_tools(hover_bar)

    p_bar.xgrid.grid_line_color = None # Cleaner look
    p_bar.xaxis.axis_label_text_font_style = "normal"
    p_bar.yaxis.axis_label_text_font_style = "normal"

    return p_bar

# (F) Layout for Additional Plots
# Create the bar plot using the aggregated data (data_proviz)
bar_plot_component = create_bar_plot_with_sd(data_proviz)

additional_plots = column(
    p_avg_plot,
    p_std_plot,
    p_corr_plot,
    bar_plot_component, # Integrated Bar Plot
    sizing_mode='stretch_width',
    spacing=20,
    name="additional_plots"
)

# (G) Scatter Plots Layout (using flex for better responsiveness)
scatter_col_exp = column(
    p_scatter_exp,
    regression_info_exp,
    sizing_mode="stretch_width",
    styles={'min-width': '300px'} # Minimum width for wrapping
)
scatter_col_af = column(
    p_scatter_af,
    regression_info_af,
    sizing_mode="stretch_width",
    styles={'min-width': '300px'}
)
scatter_col_evol = column(
    p_scatter_evol,
    regression_info_evol,
    sizing_mode="stretch_width",
    styles={'min-width': '300px'}
)

scatter_row = row(
    scatter_col_exp,
    scatter_col_af,
    scatter_col_evol,
    sizing_mode="stretch_width",
    styles={'flex-wrap': 'wrap', 'gap': '15px'} # Allow wrapping and add gap
)

# (I) Main layout section including selection, plots, and unity
visualization_section = column(
    row(select_file, window_slider, styles={'gap': '20px'}), # Put select and slider in a row
    p, # Main line plot
    scatter_row, # Row of 3 scatter plots
    unity_container, # Unity iframe section
    additional_plots,  # Aggregated plots section
    sizing_mode='stretch_width',
    css_classes=['visualization-section']
)

# Main layout assembly
main_layout = column(
    custom_styles,
    header,
    visualization_section, # Combined plots/unity section
    controls_section, # Title for table filters
    controls_layout,  # Filters for the table
    data_table, # Correlation table
    sizing_mode='stretch_width'
)

# Set up document
curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration Analysis" # Updated title