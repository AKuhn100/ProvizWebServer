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
# Define the directory where the summary data files are located.
# IMPORTANT: This directory must exist and contain the data files for the script to run.
DATA_DIR = "summary_data"  # Directory containing the summary files

# Define the regular expression pattern to match the summary filenames
# and capture the 4-character ID.
FILE_PATTERN = r"^summary_([A-Za-z0-9]{4})\.txt$" # Captures the ID in group 1

# Define a default PDB ID to load initially.
# If left blank (""), the script will load the first PDB ID found.
# This should be the 4-character ID, not the full filename.
DEFAULT_PDB_ID = ""

###############################################################################
# 2) Helpers: Data Parsing and Aggregation
###############################################################################

def extract_pdb_id(filename):
    """Extracts the 4-character PDB ID from the filename."""
    match = re.match(FILE_PATTERN, filename)
    if match:
        return match.group(1) # Return the captured group
    return None # Return None if pattern doesn't match

def moving_average(arr, window_size=5):
    """
    Computes a simple moving average on a 1D numpy array containing floats.

    Args:
        arr (np.array): The input array of numbers.
        window_size (int): The size of the moving average window (should be odd).

    Returns:
        np.array: An array of the same size as input, containing the moving average.
                  Edges where the window cannot be centered will have np.nan.
    """
    n = len(arr)
    out = np.full(n, np.nan)  # Initialize output array with NaNs
    halfw = window_size // 2 # Integer division to find half window size

    for i in range(n):
        # Skip calculation if the current point is NaN
        if np.isnan(arr[i]):
            continue
        # Define window boundaries, handling edges
        start = max(0, i - halfw)
        end = min(n, i + halfw + 1)
        window = arr[start:end]
        # Filter out NaNs within the window
        good = window[~np.isnan(window)]
        # Calculate mean if there are valid numbers in the window
        if len(good) > 0:
            out[i] = np.mean(good)
    return out

def parse_summary_file(local_path):
    """
    Parses a single summary file (.txt, tab-separated).

    Reads the file, checks for required columns, handles missing values ('n/a'),
    and computes Spearman correlations between B-Factor and frustration metrics
    on the original data. Smoothing and normalization happen later in update_plot.

    Args:
        local_path (str): The full path to the summary file.

    Returns:
        tuple: Contains:
            - pd.DataFrame: Original data read from the file.
            - dict: Spearman correlations {(metricA, metricB): (rho, pval)}.
            Returns (None, {}) if the file is invalid or parsing fails.
    """
    # List of columns required in the input file
    required_cols = ["AlnIndex", "Residue", "B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]

    # Check if the file exists
    if not os.path.isfile(local_path):
        print(f"File not found: {local_path}")
        return None, {}

    # Try reading the tab-separated file
    try:
        df = pd.read_csv(local_path, sep='\t')
    except Exception as e:
        print(f"Skipping {local_path}: failed to parse data. Error: {e}")
        return None, {}

    # Validate required columns
    if not set(required_cols).issubset(df.columns):
        print(f"Skipping {local_path}: missing required columns ({required_cols}). Found: {df.columns.tolist()}")
        return None, {}

    # --- Data Cleaning and Preparation ---
    # Columns to process (convert 'n/a' to NaN)
    process_cols = ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]

    # Replace 'n/a' strings with actual NaN values and convert columns to numeric
    for col in process_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # 'coerce' turns parsing errors into NaN

    # Keep a copy of the original data
    df_original = df.copy()

    # --- Spearman Correlation Calculation (on Original Data) ---
    corrs = {}
    # Drop rows where any of the key metrics are NaN for correlation calculation
    sub = df_original.dropna(subset=process_cols)

    if not sub.empty:
        # Define pairs of metrics for correlation analysis
        combos = [
            ("B_Factor", "ExpFrust"),
            ("B_Factor", "AFFrust"),
            ("B_Factor", "EvolFrust"),
            ("ExpFrust", "AFFrust"),
            ("ExpFrust", "EvolFrust"),
            ("AFFrust",  "EvolFrust"),
        ]
        for (mA, mB) in combos:
            # Check if there's enough variance in both columns to calculate correlation
            if sub[mA].nunique() < 2 or sub[mB].nunique() < 2:
                rho, pval = np.nan, np.nan # Not enough data variance
            else:
                # Calculate Spearman's rank correlation coefficient (rho) and p-value
                rho, pval = spearmanr(sub[mA], sub[mB])
            corrs[(mA, mB)] = (rho, pval) # Store results

    # Return only original df and correlations. Plot df is generated in update_plot.
    return df_original, corrs

def remove_regression_renderers(fig):
    """
    Removes all renderers (like lines, hover tools) from a Bokeh figure
    whose 'name' attribute starts with 'regression_'. Also removes associated
    hover tools.

    Args:
        fig (bokeh.plotting.figure): The figure to modify.
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

    # Identify hover tools associated with removed renderers
    tools_to_keep = []
    for tool in fig.tools:
        if isinstance(tool, HoverTool):
            targets_removed_renderer = False
            if tool.renderers: # Check if renderers list is not empty/None
                for r in tool.renderers:
                     renderer_name = getattr(r, 'name', None)
                     if renderer_name in renderers_to_remove_names:
                         targets_removed_renderer = True
                         break
            # Also check if the tool itself has a regression-related name
            tool_name = getattr(tool, 'name', None)
            if targets_removed_renderer or (isinstance(tool_name, str) and tool_name.startswith('regression_')):
                continue # Skip this HoverTool
        tools_to_keep.append(tool)

    fig.renderers = renderers_to_keep
    fig.tools = tools_to_keep


###############################################################################
# 3) Load and Aggregate Data from Local Directory
###############################################################################
print(f"Loading data from: {DATA_DIR}")
# Use PDB ID as the key for the dictionary
data_by_id = {}         # Dictionary keyed by PDB ID
all_corr_rows = []      # List to build the global correlation dataframe

# Lists for aggregating summary statistics across all proteins
pdb_ids = []            # List of PDB IDs
avg_bfactors = []       # List of mean B-Factors per protein
std_bfactors = []       # List of standard deviation of B-Factors per protein
spearman_exp = []       # List of Spearman Rho (B-Factor vs ExpFrust) per protein
spearman_af = []        # List of Spearman Rho (B-Factor vs AFFrust) per protein
spearman_evol = []      # List of Spearman Rho (B-Factor vs EvolFrust) per protein

# Possible frustration column names (used later if needed)
POSSIBLE_FRUST_COLUMNS = ['ExpFrust', 'AFFrust', 'EvolFrust']

# Color mapping for frustration types used in aggregated plots
# Using Category10 palette for distinct colors
FRUSTRATION_COLORS = {
    "ExpFrust.": Category10[10][1],   # Blue
    "AFFrust.": Category10[10][2],    # Green
    "EvolFrust.": Category10[10][3]   # Purple
}

# Iterate through files in the specified data directory
found_files = 0
skipped_files = 0
for filename in os.listdir(DATA_DIR):
    # Extract PDB ID using the helper function
    pdb_id = extract_pdb_id(filename)
    if not pdb_id:
        # print(f"Skipping {filename}: does not match pattern {FILE_PATTERN}") # Optional: reduce console noise
        skipped_files += 1
        continue

    file_path = os.path.join(DATA_DIR, filename)
    # Parse the file using the helper function
    df_orig, corrs = parse_summary_file(file_path)

    # If parsing failed or file was invalid, skip to the next file
    if df_orig is None:
        skipped_files += 1
        continue

    found_files += 1
    # Store the parsed data in the dictionary, keyed by PDB ID
    data_by_id[pdb_id] = {
        "df_original": df_orig,
        "corrs": corrs
    }

    # --- Collect data for the global correlation table ---
    # Use PDB ID instead of filename
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo # Unpack the metric pair
        all_corr_rows.append([pdb_id, mA, mB, rho, pval]) # Use pdb_id here

    # --- Aggregate summary statistics for additional plots ---
    # Calculate mean and std dev of B-Factor (use original data)
    avg_b = df_orig['B_Factor'].mean()
    std_b = df_orig['B_Factor'].std()

    # Extract specific Spearman correlations (B-Factor vs Frustration metrics)
    spearman_r_exp = corrs.get(("B_Factor", "ExpFrust"), (np.nan, np.nan))[0] # Get rho
    spearman_r_af = corrs.get(("B_Factor", "AFFrust"), (np.nan, np.nan))[0]   # Get rho
    spearman_r_evol = corrs.get(("B_Factor", "EvolFrust"), (np.nan, np.nan))[0] # Get rho

    # Append aggregated data to lists, using PDB ID
    pdb_ids.append(pdb_id) # Use pdb_id here
    avg_bfactors.append(avg_b)
    std_bfactors.append(std_b)
    spearman_exp.append(spearman_r_exp)
    spearman_af.append(spearman_r_af)
    spearman_evol.append(spearman_r_evol)

print(f"Found and processed {found_files} files matching pattern.")
if skipped_files > 0:
    print(f"Skipped {skipped_files} files (pattern mismatch or parsing error).")

# Create the DataFrame for the correlation table
# Rename 'Test' column to 'PDB_ID'
df_all_corr = pd.DataFrame(all_corr_rows, columns=["PDB_ID","MetricA","MetricB","Rho","Pval"])

# Create the DataFrame containing aggregated stats per protein
# Rename 'Protein' column to 'PDB_ID'
data_proviz = pd.DataFrame({
    'PDB_ID': pdb_ids, # Use pdb_ids here
    'Avg_B_Factor': avg_bfactors,
    'Std_B_Factor': std_bfactors,
    'Spearman_ExpFrust': spearman_exp,
    'Spearman_AFFrust': spearman_af,
    'Spearman_EvolFrust': spearman_evol
})

# --- Prepare data for aggregated scatter plots using pd.melt ---
# Melt for Avg B-Factor vs Spearman Rho plot
data_long_avg = data_proviz.melt(
    id_vars=['PDB_ID', 'Avg_B_Factor'], # Keep these columns (use PDB_ID)
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'], # Columns to unpivot
    var_name='Frust_Type',      # New column for the metric name
    value_name='Spearman_Rho'    # New column for the correlation value
)

# Melt for Std Dev B-Factor vs Spearman Rho plot
data_long_std = data_proviz.melt(
    id_vars=['PDB_ID', 'Std_B_Factor'], # Use PDB_ID
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)

# Clean up the 'Frust_Type' names for better legend labels
data_long_avg['Frust_Type'] = data_long_avg['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')
data_long_std['Frust_Type'] = data_long_std['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

# Remove rows where Spearman Rho is NaN (cannot be plotted)
data_long_avg.dropna(subset=['Spearman_Rho'], inplace=True)
data_long_std.dropna(subset=['Spearman_Rho'], inplace=True)

###############################################################################
# 4) Bokeh Application Components
###############################################################################

# --- (A) Main Plot: Smoothed + Normalized Data per Protein ---
# ColumnDataSource: Holds the data for the main line plot. Updated dynamically.
source_plot = ColumnDataSource(data=dict(
    x=[],          # AlnIndex (Residue Index)
    residue=[],    # Residue name (e.g., 'A', 'L')
    b_factor=[],   # Smoothed, normalized B-Factor
    exp_frust=[],  # Smoothed, normalized Experimental Frustration
    af_frust=[],   # Smoothed, normalized AlphaFold Frustration
    evol_frust=[]  # Smoothed, normalized Evolutionary Frustration
))

# Create the main figure object
p = figure(
    title="(No PDB ID Selected)", # Title updated dynamically
    sizing_mode='stretch_width', # Plot width adjusts to container
    height=600,                 # Fixed height
    tools=["pan","box_zoom","wheel_zoom","reset","save"], # Available tools
    active_drag="box_zoom",     # Default drag tool
    active_scroll="wheel_zoom"  # Enable wheel zoom
)

# Define separate HoverTools for each line for clarity
# Each hover tool is initially associated with no renderers; this is set later.
hover_bf = HoverTool(
    renderers=[], # Will be linked to the B-Factor line renderer
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. B-Factor", "@b_factor{0.3f}")], # Tooltip content
    name="hover_b_factor" # Unique name for the tool
)
hover_ef = HoverTool(
    renderers=[], # Will be linked to the ExpFrust line renderer
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. ExpFrust", "@exp_frust{0.3f}")],
    name="hover_exp_frust"
)
hover_af = HoverTool(
    renderers=[], # Will be linked to the AFFrust line renderer
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. AFFrust", "@af_frust{0.3f}")],
    name="hover_af_frust"
)
hover_ev = HoverTool(
    renderers=[], # Will be linked to the EvolFrust line renderer
    tooltips=[("Index", "@x"), ("Residue", "@residue"), ("Norm. EvolFrust", "@evol_frust{0.3f}")],
    name="hover_evol_frust"
)

# Add the hover tools to the plot
p.add_tools(hover_bf, hover_ef, hover_af, hover_ev)

# Set axis labels
p.xaxis.axis_label = "Residue Index (AlnIndex)"
p.yaxis.axis_label = "Smoothed & Normalized Value"

# Define colors and labels for the lines
color_map = {
    "b_factor":  ("B-Factor", Category10[10][0]), # Grey/Black often used for B-Factor
    "exp_frust": ("ExpFrust", Category10[10][1]), # Blue
    "af_frust":  ("AFFrust", Category10[10][2]),  # Green
    "evol_frust":("EvolFrust", Category10[10][3]) # Purple
}

# Add line renderers to the plot and link hover tools
renderers = {} # Dictionary to store the renderer objects
for col_key, (label, col) in color_map.items():
    renderer = p.line(
        x="x", y=col_key, source=source_plot, # Data source and columns
        line_width=2, alpha=0.8, color=col,   # Styling
        legend_label=label                   # Label for the legend
    )
    renderers[col_key] = renderer # Store the renderer

    # Link the appropriate hover tool to this specific line renderer
    if col_key == "b_factor":
        hover_bf.renderers.append(renderer)
    elif col_key == "exp_frust":
        hover_ef.renderers.append(renderer)
    elif col_key == "af_frust":
        hover_af.renderers.append(renderer)
    elif col_key == "evol_frust":
        hover_ev.renderers.append(renderer)

# Configure the legend
p.legend.location = "top_left"
p.legend.title = "Metrics"
p.legend.click_policy = "hide" # Clicking legend item hides the line

# --- (B) Scatter Plots: B-Factor vs Frustration (Original, Normalized Data) ---
# Create figures for the three scatter plots
common_scatter_tools = ["pan", "box_zoom", "wheel_zoom", "reset", "save"]
p_scatter_exp = figure(
    sizing_mode="stretch_both", # Adjusts to container size
    aspect_ratio=1,             # Makes the plot square
    min_width=300,              # Minimum dimensions
    min_height=300,
    title="",                   # Title set dynamically
    x_axis_label="Normalized B-Factor",
    y_axis_label="Normalized ExpFrust",
    tools=common_scatter_tools,
    active_drag="box_zoom",
    active_scroll="wheel_zoom"
)
p_scatter_af = figure(
    sizing_mode="stretch_both", aspect_ratio=1, min_width=300, min_height=300,
    title="", x_axis_label="Normalized B-Factor", y_axis_label="Normalized AFFrust",
    tools=common_scatter_tools, active_drag="box_zoom", active_scroll="wheel_zoom"
)
p_scatter_evol = figure(
    sizing_mode="stretch_both", aspect_ratio=1, min_width=300, min_height=300,
    title="", x_axis_label="Normalized B-Factor", y_axis_label="Normalized EvolFrust",
    tools=common_scatter_tools, active_drag="box_zoom", active_scroll="wheel_zoom"
)

# ColumnDataSources for scatter plots (hold normalized and original values)
# 'x', 'y' are normalized for plotting; 'x_orig', 'y_orig' for potential tooltips
source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[]))
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[]))
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[]))

# Create Div elements to display regression line information below each scatter plot
div_styles = { # Common styles for the info boxes
    'background-color': '#f8f9fa', 'padding': '10px', 'border': '1px solid #ddd',
    'border-radius': '4px', 'margin-top': '10px', 'font-size': '13px',
    'text-align': 'center', 'width': 'auto', 'min-height': '40px' # Ensure space even if empty
}
regression_info_exp = Div(text="<i>Regression info appears here</i>", styles=div_styles, sizing_mode="stretch_width")
regression_info_af = Div(text="<i>Regression info appears here</i>", styles=div_styles, sizing_mode="stretch_width")
regression_info_evol = Div(text="<i>Regression info appears here</i>", styles=div_styles, sizing_mode="stretch_width")

# Add scatter glyphs to the plots
scatter_exp_renderer = p_scatter_exp.scatter("x", "y", source=source_scatter_exp, color=color_map["exp_frust"][1], alpha=0.6, size=6)
scatter_af_renderer = p_scatter_af.scatter("x", "y", source=source_scatter_af,  color=color_map["af_frust"][1], alpha=0.6, size=6)
scatter_evol_renderer = p_scatter_evol.scatter("x", "y", source=source_scatter_evol, color=color_map["evol_frust"][1], alpha=0.6, size=6)

# Add HoverTools for the scatter points
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
    Calculates and adds a linear regression line to a scatter plot and updates
    an associated Div element with the regression equation and R-squared value.

    Args:
        fig (bokeh.plotting.figure): The figure to add the line to.
        xvals (np.array): X-values for regression (should be normalized).
        yvals (np.array): Y-values for regression (should be normalized).
        color (str): Color for the regression line.
        info_div (bokeh.models.Div): The Div element to update with info.
        plot_type (str): A unique identifier string (e.g., "exp", "af") to ensure
                         renderer names are unique across plots.
    """
    # Ensure there are enough valid data points for regression
    not_nan_mask = ~np.isnan(xvals) & ~np.isnan(yvals)
    xvals_clean = xvals[not_nan_mask]
    yvals_clean = yvals[not_nan_mask]

    # Check for sufficient data points and variance
    if len(xvals_clean) < 2 or np.all(xvals_clean == xvals_clean[0]) or np.all(yvals_clean == yvals_clean[0]):
        if info_div:
            info_div.text = "<i style='color: gray;'>Insufficient data or variance for regression</i>"
        return # Exit if regression is not possible

    try:
        # Perform linear regression using scipy.stats.linregress
        slope, intercept, r_value, p_value, std_err = linregress(xvals_clean, yvals_clean)
    except ValueError as e:
         if info_div:
            info_div.text = f"<i style='color: red;'>Regression Error: {e}</i>"
         return # Exit on error


    # --- Add Visible Regression Line ---
    # Generate points for the line spanning the data range
    x_min, x_max = np.min(xvals_clean), np.max(xvals_clean)
    # Handle case where min == max after cleaning (should be caught earlier, but safety check)
    if x_min == x_max:
         if info_div:
            info_div.text = "<i style='color: gray;'>Insufficient variance for regression line</i>"
         return
    x_range = np.linspace(x_min, x_max, 100) # 100 points for a smooth line
    y_range = slope * x_range + intercept
    regression_line_name = f'regression_line_{plot_type}' # Unique name

    # Add the line glyph to the figure
    regression_line_renderer = fig.line(
        x_range, y_range,
        line_width=2, line_dash='dashed', color=color,
        name=regression_line_name # Assign the unique name
    )

    # --- Add Hover Tool for the Regression Line ---
    # Create the HoverTool targeting the *visible* regression line
    hover_regression = HoverTool(
        renderers=[regression_line_renderer], # Target the visible line
        tooltips=[
            ("Regression", f"y = {slope:.3f}x + {intercept:.3f}"), # Display equation
            ("R-squared", f"{r_value**2:.3f}"), # Display R-squared
            ("p-value", f"{p_value:.2e}") # Display p-value
        ],
        mode='mouse', # Activate on mouse hover
        name=f'regression_hover_tool_{plot_type}' # Unique name for the tool
    )
    # Check if a similar tool already exists before adding
    existing_tool_names = [getattr(t, 'name', None) for t in fig.tools]
    if hover_regression.name not in existing_tool_names:
        fig.add_tools(hover_regression)


    # --- Update the Information Div ---
    if info_div:
        # Format the output string with HTML for styling
        info_div.text = f"""
        <div style='color: {color};'>
            <strong>Regression: y = {slope:.3f}x + {intercept:.3f}</strong><br>
            <span style='font-size: 12px;'>R² = {r_value**2:.3f} | p = {p_value:.2e}</span>
        </div>
        """

# --- (C) Controls: File Selection Dropdown and Slider ---
# Dropdown for selecting the PDB ID
pdb_id_options = sorted(data_by_id.keys()) # Get list of loaded PDB IDs
if not pdb_id_options:
    print("WARNING: No data files found or loaded. The application might not work correctly.")
    initial_pdb_id = ""
elif DEFAULT_PDB_ID and DEFAULT_PDB_ID in pdb_id_options:
    initial_pdb_id = DEFAULT_PDB_ID # Use hardcoded default if valid
elif pdb_id_options:
    initial_pdb_id = pdb_id_options[0] # Otherwise, use the first PDB ID found
else:
    initial_pdb_id = "" # Fallback

select_pdb_id = Select(
    title="Select PDB ID:", # Updated title
    value=initial_pdb_id,
    options=pdb_id_options, # Use PDB IDs as options
    width=400 # Fixed width for the dropdown
)

# Slider for controlling the moving average window size
window_slider = Slider(
    start=1,       # Minimum window size (1 means no smoothing)
    end=21,        # Maximum window size
    value=5,       # Initial window size
    step=2,        # Step size (ensures odd numbers for centered window)
    title="Moving Average Window Size (for Line Plot)",
    width=400      # Fixed width for the slider
)

def update_moving_average(attr, old, new):
    """Callback function triggered when the slider value changes."""
    # Simply re-call the main update function to recalculate and redraw plots
    print(f"Slider changed: {old} -> {new}. Updating plot...") # Debug print
    update_plot(None, None, None) # Pass None for attr, old, new as they are not needed directly

# Attach the callback to the slider's 'value' property
window_slider.on_change('value', update_moving_average)

def min_max_normalize(arr):
    """
    Applies min-max normalization to a numpy array (scales to [0, 1]).
    Handles NaNs and cases where max equals min.

    Args:
        arr (np.array): Input array.

    Returns:
        np.array: Normalized array, or array of zeros if max == min.
    """
    # Use nanmin/nanmax to ignore NaN values if present
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)

    # Check if min/max are NaN (happens if array is all NaN)
    if np.isnan(arr_min) or np.isnan(arr_max):
        return np.full_like(arr, np.nan) # Return array of NaNs

    # Avoid division by zero if all valid values are the same
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    else:
        # If max == min, return array of zeros (or 0.5, or handle as needed)
        # Returning zeros assumes a baseline state when there's no variation.
        out = np.zeros_like(arr)
        # Preserve NaNs in the output
        out[np.isnan(arr)] = np.nan
        return out


def update_plot(attr, old, new):
    """
    Core function to update all plots when the selected PDB ID or slider changes.

    Reads data for the selected PDB ID, applies the current moving average window,
    normalizes data, updates ColumnDataSources for line and scatter plots,
    and recalculates/redraws regression lines.

    Args:
        attr (str): Name of the attribute that changed (e.g., 'value'). Not used directly.
        old: Old value of the attribute. Not used directly.
        new: New value of the attribute. Not used directly (reads current widget values).
    """
    pdb_id = select_pdb_id.value # Get current PDB ID from dropdown
    window_size = window_slider.value # Get current window size from slider
    print(f"Updating plots for PDB ID: {pdb_id} with window size: {window_size}") # Debug print

    # --- Handle case where no PDB ID is selected or data is missing ---
    if not pdb_id or pdb_id not in data_by_id:
        print(f"No data for {pdb_id}, clearing plots.") # Debug print
        # Clear main plot data
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        # Clear scatter plot data
        source_scatter_exp.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        source_scatter_af.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        source_scatter_evol.data = dict(x=[], y=[], x_orig=[], y_orig=[], residue=[], index=[])
        # Reset titles and regression info
        p.title.text = "(No PDB ID Selected)"
        p_scatter_exp.title.text = ""
        p_scatter_af.title.text = ""
        p_scatter_evol.title.text = ""
        regression_info_exp.text = "<i style='color: gray;'>No PDB ID selected</i>"
        regression_info_af.text = "<i style='color: gray;'>No PDB ID selected</i>"
        regression_info_evol.text = "<i style='color: gray;'>No PDB ID selected</i>"
        # Remove any leftover regression lines
        remove_regression_renderers(p_scatter_exp)
        remove_regression_renderers(p_scatter_af)
        remove_regression_renderers(p_scatter_evol)
        return # Stop execution for this update

    # --- Process Data for the Selected PDB ID ---
    # Get the original DataFrame for the selected PDB ID
    df_orig = data_by_id[pdb_id]["df_original"]
    df_plot = df_orig.copy() # Create a copy to modify for the line plot

    # Columns to smooth and normalize for the line plot
    metrics_to_process = ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]

    # 1. Apply Moving Average (using current slider value)
    for col in metrics_to_process:
        # Ensure column exists and is numeric before applying moving average
        if col in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[col]):
            arr = df_plot[col].values
            df_plot[col] = moving_average(arr, window_size=window_size)
        else:
            print(f"Warning: Column '{col}' not found or not numeric in {pdb_id}. Skipping smoothing.")
            df_plot[col] = np.nan # Fill with NaN if problematic

    # 2. Normalize the *smoothed* data for the line plot
    for col in metrics_to_process:
         if col in df_plot.columns:
             df_plot[col] = min_max_normalize(df_plot[col].values)
         else:
             # This case should ideally not happen if handled above, but as a fallback:
             df_plot[col] = np.nan


    # --- Update Main Line Plot ---
    # Prepare data dictionary for ColumnDataSource, handling potential NaNs from processing
    # Using df_plot which now contains smoothed and normalized data
    plot_data_dict = {
        "x": df_plot["AlnIndex"].tolist(),
        "residue": df_plot["Residue"].tolist(),
        "b_factor": df_plot["B_Factor"].tolist(),
        "exp_frust": df_plot["ExpFrust"].tolist(),
        "af_frust": df_plot["AFFrust"].tolist(),
        "evol_frust": df_plot["EvolFrust"].tolist()
    }
    source_plot.data = plot_data_dict # Update the data source, triggers plot redraw
    p.title.text = f"PDB ID: {pdb_id} (Smoothed Window={window_size}, Normalized)" # Update plot title with PDB ID

    # --- Update Scatter Plots (using ORIGINAL, non-smoothed data, but normalized) ---
    # Get the original data again
    df_scatter_base = data_by_id[pdb_id]["df_original"].copy()

    # Normalize the original data for scatter plots
    scatter_x_norm = min_max_normalize(df_scatter_base["B_Factor"].values)
    scatter_y_exp_norm = min_max_normalize(df_scatter_base["ExpFrust"].values)
    scatter_y_af_norm = min_max_normalize(df_scatter_base["AFFrust"].values)
    scatter_y_evol_norm = min_max_normalize(df_scatter_base["EvolFrust"].values)

    # Prepare common data for tooltips
    scatter_residues = df_scatter_base["Residue"].tolist()
    scatter_indices = df_scatter_base["AlnIndex"].tolist()
    scatter_x_orig = df_scatter_base["B_Factor"].tolist() # Keep original for tooltip


    # **Remove existing regression renderers before adding new ones**
    remove_regression_renderers(p_scatter_exp)
    remove_regression_renderers(p_scatter_af)
    remove_regression_renderers(p_scatter_evol)

    # Update ExpFrust Scatter Plot
    source_scatter_exp.data = dict(
        x=scatter_x_norm, y=scatter_y_exp_norm,
        x_orig=scatter_x_orig, y_orig=df_scatter_base["ExpFrust"].tolist(), # Orig for tooltip
        residue=scatter_residues, index=scatter_indices
    )
    p_scatter_exp.title.text = f"{pdb_id}: B-Factor vs ExpFrust" # Use PDB ID in title
    add_regression_line_and_info(
        fig=p_scatter_exp,
        xvals=scatter_x_norm, # Use normalized data for regression
        yvals=scatter_y_exp_norm,
        color=color_map["exp_frust"][1], # Use consistent color
        info_div=regression_info_exp,
        plot_type="exp" # Unique identifier
    )

    # Update AFFrust Scatter Plot
    source_scatter_af.data = dict(
        x=scatter_x_norm, y=scatter_y_af_norm,
        x_orig=scatter_x_orig, y_orig=df_scatter_base["AFFrust"].tolist(),
        residue=scatter_residues, index=scatter_indices
    )
    p_scatter_af.title.text = f"{pdb_id}: B-Factor vs AFFrust" # Use PDB ID in title
    add_regression_line_and_info(
        fig=p_scatter_af,
        xvals=scatter_x_norm,
        yvals=scatter_y_af_norm,
        color=color_map["af_frust"][1],
        info_div=regression_info_af,
        plot_type="af"
    )

    # Update EvolFrust Scatter Plot
    source_scatter_evol.data = dict(
        x=scatter_x_norm, y=scatter_y_evol_norm,
        x_orig=scatter_x_orig, y_orig=df_scatter_base["EvolFrust"].tolist(),
        residue=scatter_residues, index=scatter_indices
    )
    p_scatter_evol.title.text = f"{pdb_id}: B-Factor vs EvolFrust" # Use PDB ID in title
    add_regression_line_and_info(
        fig=p_scatter_evol,
        xvals=scatter_x_norm,
        yvals=scatter_y_evol_norm,
        color=color_map["evol_frust"][1],
        info_div=regression_info_evol,
        plot_type="evol"
    )

# Attach the update_plot callback to the PDB ID selection dropdown
select_pdb_id.on_change("value", update_plot)

# Trigger the initial plot update when the script starts, if a PDB ID is selected
if initial_pdb_id:
    print(f"Initial plot load for PDB ID: {initial_pdb_id}")
    update_plot(None, None, initial_pdb_id)
else:
    print("No initial PDB ID selected, plots will be empty until selection.")


###############################################################################
# 5) CORRELATION TABLE AND FILTERS
###############################################################################

# --- (D) Correlation Data Table ---
# Define table columns with formatters for numbers
# Updated title for the first column to "PDB ID"
columns = [
    TableColumn(field="PDB_ID", title="PDB ID"), # Use PDB_ID field and title
    TableColumn(field="MetricA", title="Metric A"),
    TableColumn(field="MetricB", title="Metric B"),
    TableColumn(field="Rho", title="Spearman Rho", formatter=NumberFormatter(format="0.000")), # 3 decimal places
    TableColumn(field="Pval", title="p-value", formatter=NumberFormatter(format="0.00e+0")) # Scientific notation
]

# Create ColumnDataSource for the table
if df_all_corr.empty:
    # If no correlation data was loaded, create an empty source
    print("WARNING: Correlation DataFrame is empty. Table will be empty.")
    source_corr = ColumnDataSource(dict(PDB_ID=[], MetricA=[], MetricB=[], Rho=[], Pval=[])) # Use PDB_ID
else:
    # Otherwise, use the populated DataFrame
    source_corr = ColumnDataSource(df_all_corr)

# Create the DataTable widget
data_table = DataTable(
    columns=columns,
    source=source_corr,
    height=400, # Fixed height
    # width=1200, # Let sizing_mode handle width
    sizing_mode='stretch_width', # Allow width to adjust
    selectable=True, # Allow row selection
    editable=False,  # Data is not editable in the table
    index_position=None # Hide the default index column
)

# --- (E) Filters for Correlation Table ---

# Helper function to distribute labels into columns for CheckboxGroups
def split_labels(labels, num_columns):
    """Splits a list of labels into N sublists for columnar layout."""
    if num_columns <= 0 or not labels:
        return [labels] # Return single list if invalid columns or no labels
    k, m = divmod(len(labels), num_columns) # Calculate items per column
    # Create sublists, distributing remainder 'm'
    return [labels[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_columns)]

# Number of columns for filter checkboxes (adjust for layout)
NUM_FILTER_COLUMNS = 4

# Get unique values for filters from the correlation DataFrame
if not df_all_corr.empty:
    # Use PDB_ID column for test filters
    tests_in_corr = sorted(df_all_corr["PDB_ID"].unique()) # Get unique PDB IDs
    # Create unique combo strings like "MetricA vs MetricB" for filtering
    combo_options = sorted(list(set(
        f"{row['MetricA']} vs {row['MetricB']}"
        for _, row in df_all_corr.iterrows()
    )))
else:
    # Handle case where no correlation data exists
    tests_in_corr = []
    combo_options = []

# Create CheckboxGroups for PDB IDs
if tests_in_corr:
    test_labels_split = split_labels(tests_in_corr, NUM_FILTER_COLUMNS)
    checkbox_tests_columns = [
        CheckboxGroup(
            labels=col_labels,
            active=[],  # Initially no selection
            name=f'tests_column_{i+1}', # Unique name for each group
            height=150 # Limit height to encourage scrolling if needed
        ) for i, col_labels in enumerate(test_labels_split) if col_labels # Only create if labels exist
    ]
else:
    checkbox_tests_columns = [CheckboxGroup(labels=["(No PDB IDs Found)"], active=[], disabled=True)]

# Create CheckboxGroups for Metric Pairs
if combo_options:
    combo_labels_split = split_labels(combo_options, NUM_FILTER_COLUMNS)
    checkbox_combos_columns = [
        CheckboxGroup(
            labels=col_labels,
            active=[],
            name=f'combos_column_{i+1}',
            height=150
        ) for i, col_labels in enumerate(combo_labels_split) if col_labels
    ]
else:
     checkbox_combos_columns = [CheckboxGroup(labels=["(No Metric Pairs Found)"], active=[], disabled=True)]

# Layout for the filter controls using rows of CheckboxGroups
tests_layout = row(*checkbox_tests_columns, sizing_mode='stretch_width')
combos_layout = row(*checkbox_combos_columns, sizing_mode='stretch_width')

# Titles for the filter sections
tests_title = Div(text="<b>Filter by PDB ID:</b>", styles={'font-size': '14px', 'margin-bottom': '5px'}) # Updated title
combos_title = Div(text="<b>Filter by Metric Pair:</b>", styles={'font-size': '14px', 'margin-bottom': '5px'})

# Combine titles and checkbox groups into columns for vertical arrangement
tests_filter_section = column(tests_title, tests_layout, sizing_mode='stretch_width')
combos_filter_section = column(combos_title, combos_layout, sizing_mode='stretch_width')

# Arrange the two filter sections side-by-side
controls_layout = row(
    tests_filter_section,
    Spacer(width=30), # Add space between filter sections
    combos_filter_section,
    sizing_mode='stretch_width'
)

# Helper function to get all selected labels from a list of CheckboxGroups
def get_selected_labels(checkbox_columns):
    """Aggregates active labels from a list of CheckboxGroup widgets."""
    selected = []
    for checkbox_group in checkbox_columns:
        # Map active indices back to labels
        selected.extend([checkbox_group.labels[i] for i in checkbox_group.active])
    return selected

def update_corr_filter(attr, old, new):
    """Callback to filter the correlation table based on checkbox selections."""
    if df_all_corr.empty:
        return # Do nothing if there's no data

    # Get currently selected items from all relevant checkbox groups
    selected_tests = get_selected_labels(checkbox_tests_columns) # These are PDB IDs now
    selected_combos = get_selected_labels(checkbox_combos_columns)

    # Start with the full dataset
    filtered_df = df_all_corr.copy()

    # Apply PDB ID filter if any are selected
    if selected_tests:
        filtered_df = filtered_df[filtered_df["PDB_ID"].isin(selected_tests)] # Filter by PDB_ID column

    # Apply combo filter if any combos are selected
    if selected_combos:
        # Recreate the "MetricA vs MetricB" string for filtering
        filtered_df["combo_str"] = filtered_df.apply(lambda r: f"{r['MetricA']} vs {r['MetricB']}", axis=1)
        filtered_df = filtered_df[filtered_df["combo_str"].isin(selected_combos)]
        # Drop the temporary column
        if "combo_str" in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=["combo_str"])

    # Update the table's data source
    source_corr.data = filtered_df.to_dict(orient="list")
    print(f"Table filtered. Showing {len(filtered_df)} rows.") # Debug print

# Attach the callback function to the 'active' property of each CheckboxGroup
for checkbox_col in checkbox_tests_columns:
    checkbox_col.on_change('active', update_corr_filter)
for checkbox_col in checkbox_combos_columns:
    checkbox_col.on_change('active', update_corr_filter)

###############################################################################
# 6) Additional Aggregated Plots (Converted from Plotly to Bokeh)
###############################################################################
# These plots show relationships across all loaded proteins.

# --- (F) Spearman Rho vs Average B-Factor ---
source_avg_plot = ColumnDataSource(data_long_avg) # Use the melted data

p_avg_plot = figure(
    title="Overall: Spearman Correlation vs. Average B-Factor",
    x_axis_label="Average B-Factor (per PDB ID)", # Updated label
    y_axis_label="Spearman Rho (B-Factor vs. Frustration)",
    sizing_mode='stretch_width',
    height=450,
    tools="pan,wheel_zoom,box_zoom,reset,save,hover", # Add hover tool directly
    tooltips=[ # Define tooltips for scatter points - use PDB_ID
        ("PDB ID", "@PDB_ID"),
        ("Frustration Type", "@Frust_Type"),
        ("Avg B-Factor", "@Avg_B_Factor{0.3f}"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    active_drag="box_zoom"
)

# Get unique frustration types and assign colors using the predefined map
frust_types_avg = data_long_avg['Frust_Type'].unique().tolist()
# FRUSTRATION_COLORS defined earlier in section 3

# Add scatter glyphs, one for each frustration type for legend grouping
scatter_renderers_avg = []
for frust in frust_types_avg:
    subset = data_long_avg[data_long_avg['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    color = FRUSTRATION_COLORS.get(frust, 'gray') # Use predefined color, default to gray
    scatter = p_avg_plot.scatter(
        'Avg_B_Factor', 'Spearman_Rho',
        source=source_subset,
        color=color,
        size=8, alpha=0.7,
        legend_label=frust,
        muted_alpha=0.1, # Appearance when muted via legend click
        name=f'scatter_avg_{frust}' # Unique name
    )
    scatter_renderers_avg.append(scatter)

p_avg_plot.legend.location = "top_left"
p_avg_plot.legend.title = "Frustration Type"
p_avg_plot.legend.click_policy = "mute" # Mute series on legend click

# --- (G) Spearman Rho vs Std Dev of B-Factor ---
source_std_plot = ColumnDataSource(data_long_std) # Use the other melted data

p_std_plot = figure(
    title="Overall: Spearman Correlation vs. Std Dev of B-Factor",
    x_axis_label="Standard Deviation of B-Factor (per PDB ID)", # Updated label
    y_axis_label="Spearman Rho (B-Factor vs. Frustration)",
    sizing_mode='stretch_width',
    height=450,
    tools="pan,wheel_zoom,box_zoom,reset,save,hover",
    tooltips=[ # Use PDB_ID in tooltip
        ("PDB ID", "@PDB_ID"),
        ("Frustration Type", "@Frust_Type"),
        ("Std Dev B-Factor", "@Std_B_Factor{0.3f}"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    active_drag="box_zoom"
)

frust_types_std = data_long_std['Frust_Type'].unique().tolist()
# FRUSTRATION_COLORS defined earlier

scatter_renderers_std = []
for frust in frust_types_std:
    subset = data_long_std[data_long_std['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    color = FRUSTRATION_COLORS.get(frust, 'gray')
    scatter = p_std_plot.scatter(
        'Std_B_Factor', 'Spearman_Rho',
        source=source_subset,
        color=color,
        size=8, alpha=0.7,
        legend_label=frust,
        muted_alpha=0.1,
        name=f'scatter_std_{frust}'
    )
    scatter_renderers_std.append(scatter)

p_std_plot.legend.location = "top_left"
p_std_plot.legend.title = "Frustration Type"
p_std_plot.legend.click_policy = "mute"


# --- (H) Spearman Rho per PDB ID and Frustration Metric (Dot Plot) ---
# Melt data_proviz again, this time keeping PDB_ID as ID
data_long_corr = data_proviz.melt(
    id_vars=['PDB_ID'], # Use PDB_ID
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)
# Clean names and drop NaNs
data_long_corr['Frust_Type'] = data_long_corr['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')
data_long_corr.dropna(subset=['Spearman_Rho'], inplace=True)

# Get PDB IDs for x-axis range (categorical)
pdb_id_x_range = sorted(data_proviz['PDB_ID'].unique().tolist()) # Use PDB_ID

p_corr_plot = figure(
    title="Spearman Correlation (B-Factor vs. Frustration) per PDB ID", # Updated title
    x_axis_label="PDB ID", # Updated label
    y_axis_label="Spearman Rho",
    x_range=pdb_id_x_range, # Use list of PDB IDs for categorical axis
    sizing_mode='stretch_width',
    height=500,
    tools="pan,wheel_zoom,box_zoom,reset,save,hover",
    tooltips=[ # Tooltips for the dots - use PDB_ID
        ("PDB ID", "@PDB_ID"),
        ("Frustration Metric", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    toolbar_location="above"
)

# Use the same color map
frust_types_corr = data_long_corr['Frust_Type'].unique().tolist()

# Add jittered scatter points for each frustration type
dot_renderers = []
for i, frust in enumerate(frust_types_corr):
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    color = FRUSTRATION_COLORS.get(frust, 'gray')
    # Apply jitter to x-axis to prevent overlap within each PDB ID category
    dot = p_corr_plot.scatter(
        x=jitter('PDB_ID', width=0.6, range=p_corr_plot.x_range), # Jitter x-position based on PDB_ID
        y='Spearman_Rho',
        source=source_subset,
        color=color,
        size=9, alpha=0.7,
        legend_label=frust,
        muted_alpha=0.1,
        name=f'dot_{frust}'
    )
    dot_renderers.append(dot)

# Add horizontal line at y=0 for reference
p_corr_plot.line(x=pdb_id_x_range, y=0, line_width=1, line_dash='dashed', color='gray')

# Add mean lines (optional)
# mean_line_renderers = []
# for frust in frust_types_corr:
#     mean_value = data_long_corr[data_long_corr['Frust_Type'] == frust]['Spearman_Rho'].mean()
#     if not np.isnan(mean_value):
#         color = FRUSTRATION_COLORS.get(frust, 'gray')
#         mean_line = p_corr_plot.line(
#             x=pdb_id_x_range, y=mean_value,
#             color=color, line_dash='dotted', line_width=2,
#             name=f'mean_line_{frust}'
#         )
#         mean_line_renderers.append(mean_line)

# Configure legend and axis
p_corr_plot.legend.location = "top_left"
p_corr_plot.legend.title = "Frustration Type"
p_corr_plot.legend.click_policy = "mute"
p_corr_plot.xaxis.major_label_orientation = pi / 3 # Rotate labels more if many PDB IDs

# --- (I) Bar Plot: Mean Spearman Rho per Metric with Std Dev Error Bars ---
def create_bar_plot_with_sd(data_proviz):
    """Creates a bar chart of mean correlations with std dev whiskers."""
    if data_proviz.empty:
        print("WARNING: data_proviz is empty, cannot create bar plot.")
        p_bar = figure(title="Mean Spearman Correlation (No Data)", height=300)
        p_bar.text(x=0, y=0, text="No data available for this plot.")
        return p_bar

    spearman_columns = ['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust']
    stats_corrs = data_proviz[spearman_columns].agg(['mean', 'std']).transpose().reset_index()
    stats_corrs.rename(columns={
        'index': 'Metric', 'mean': 'Mean_Spearman_Rho', 'std': 'Std_Spearman_Rho'
    }, inplace=True)
    stats_corrs['Metric'] = stats_corrs['Metric'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')
    stats_corrs['Color'] = stats_corrs['Metric'].map(FRUSTRATION_COLORS)
    stats_corrs['Color'].fillna('gray', inplace=True)
    stats_corrs['upper'] = stats_corrs['Mean_Spearman_Rho'] + stats_corrs['Std_Spearman_Rho']
    stats_corrs['lower'] = stats_corrs['Mean_Spearman_Rho'] - stats_corrs['Std_Spearman_Rho']
    source_bar = ColumnDataSource(stats_corrs)

    min_val = stats_corrs['lower'].min()
    max_val = stats_corrs['upper'].max()
    if np.isnan(min_val) or np.isnan(max_val):
        y_range_with_padding = None
    else:
        padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.5
        y_range_with_padding = Range1d(start=min_val - padding, end=max_val + padding)

    p_bar = figure(
        title="Mean Spearman Correlation (B-Factor vs. Frustration) with Std Dev",
        x_axis_label="Frustration Metric", y_axis_label="Mean Spearman Rho",
        x_range=stats_corrs['Metric'].tolist(), y_range=y_range_with_padding,
        height=400, sizing_mode='stretch_width',
        tools="pan,wheel_zoom,box_zoom,reset,save", toolbar_location="above"
    )
    vbar_renderer = p_bar.vbar(
        x='Metric', top='Mean_Spearman_Rho', width=0.6, source=source_bar,
        color='Color', legend_field='Metric', line_color="black"
    )
    whisker = Whisker(
        base='Metric', upper='upper', lower='lower', source=source_bar,
        level="overlay", line_color='black'
    )
    p_bar.add_layout(whisker)
    hover_bar = HoverTool(
        tooltips=[("Metric", "@Metric"), ("Mean Rho", "@Mean_Spearman_Rho{0.3f}"), ("Std Dev", "@Std_Spearman_Rho{0.3f}")],
        renderers=[vbar_renderer], mode='mouse'
    )
    p_bar.add_tools(hover_bar)
    p_bar.line(x=stats_corrs['Metric'].tolist(), y=0, line_width=1, line_dash='dashed', color='gray')
    p_bar.legend.location = "top_right"
    p_bar.legend.title = "Frustration Type"
    p_bar.xgrid.grid_line_color = None
    p_bar.xaxis.major_label_orientation = 0 # No rotation needed for few categories

    return p_bar

# Create the bar plot instance
bar_plot = create_bar_plot_with_sd(data_proviz)

# --- Layout for Additional Plots Section ---
additional_plots_column = column(
    Div(text="<h2>Overall Protein Comparisons</h2>", styles={'margin-top': '20px'}),
    p_avg_plot,
    p_std_plot,
    p_corr_plot,
    bar_plot, # Add the bar plot
    sizing_mode='stretch_width',
    spacing=30, # Add vertical space between plots
    name="additional_plots"
)


###############################################################################
# 7) User Interface Components and Layout
###############################################################################

# --- Header and Introductory Text ---
header = Div(text="""
    <h1>Evolutionary Frustration Analysis Dashboard</h1>
    <p>
        This dashboard visualizes protein flexibility (measured by B-Factor) and compares it
        with three 'frustration' metrics calculated using different methods. Frustration
        indicates regions of a protein structure with energetically unfavorable interactions.
        Select a PDB ID from the dropdown to view its specific data.
    </p>
    <ul>
        <li><strong>B-Factor:</strong> Experimental measure of atomic displacement/flexibility from crystal structures.</li>
        <li><strong>Experimental Frustration (ExpFrust):</strong> Calculated using the <a href='http://frustratometer.qb.fcen.uba.ar/' target='_blank'>Frustratometer</a> web server with an experimental protein structure (e.g., PDB).</li>
        <li><strong>AF Frustration (AFFrust):</strong> Calculated using the Frustratometer with a structure predicted by AlphaFold.</li>
        <li><strong>Evolutionary Frustration (EvolFrust):</strong> Calculated based on evolutionary sequence conservation patterns (Multiple Sequence Alignment - MSA) and statistical potentials, <em>without requiring a 3D structure</em>.</li>
    </ul>
     <p>
        <strong>Goal:</strong> To assess how well sequence-derived Evolutionary Frustration correlates with experimental flexibility (B-Factor) and structure-based frustration metrics.
    </p>
    <hr>
    """, sizing_mode='stretch_width', styles={'margin-bottom': '15px'})

plot_description = Div(text="""
    <h2>Interactive Protein Analysis</h2>
    <p>
        Use the dropdown menu below to select a PDB ID. The plots will update automatically.
    </p>
    <ul>
        <li><strong>Main Line Plot:</strong> Shows B-Factor and frustration metrics along the protein sequence for the selected PDB ID. Data is <em>smoothed</em> using a moving average (adjust window size with the slider) and <em>min-max normalized</em> to [0, 1] for visual comparison within the selected protein. Hover over lines for details.</li>
        <li><strong>Scatter Plots:</strong> Show the relationship between normalized B-Factor and each normalized frustration metric for the selected PDB ID (using <em>original, non-smoothed</em> data points before normalization). Regression lines and R² values are shown. Hover over points for residue details and original values.</li>
        <li><strong>Correlation Table:</strong> Displays pre-calculated Spearman rank correlations (Rho) and p-values between pairs of metrics for <em>all</em> loaded PDB IDs (using <em>original, non-smoothed, non-normalized</em> data). Use the checkboxes below the table to filter the results by PDB ID and metric pair.</li>
        <li><strong>Overall Comparison Plots:</strong> Show aggregated views across all loaded PDB IDs, comparing correlations with average/standard deviation of B-Factors, and mean correlations per metric type.</li>
    </ul>
    <p><i><strong>Note:</strong> Min-max normalization is applied per PDB ID for visualization and does not affect Spearman correlations. Be cautious when comparing absolute normalized values across different proteins.</i></p>
    """, sizing_mode='stretch_width', styles={'margin-bottom': '20px'})


# --- Unity Visualizer Section (Optional, kept from original) ---
unity_description = Div(text="""
    <h2>3D Protein Visualizer (External Unity App)</h2>
    <p>
        The embedded view below links to an external Unity application for 3D visualization
        (developed separately). It may offer interactive features related to the data.
        <i>(Note: Functionality depends on the external application at the provided URL).</i>
    </p>
""", sizing_mode='stretch_width', styles={'margin-bottom': '10px'})

unity_iframe = Div(
    text="""
    <div style="width: 100%; display: flex; justify-content: center; align-items: center; margin: 20px 0;">
        <iframe
            src="https://igotintogradschool2025.site/unity/"
            style="width: 95%; max-width: 1000px; height: 70vh; min-height: 500px; border: 1px solid #ccc; border-radius: 8px;
                   box-shadow: 0 2px 5px rgba(0,0,0,0.1);"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
            title="External Unity Protein Visualizer">
            Your browser does not support iframes. You can access the visualizer directly at
            <a href='https://igotintogradschool2025.site/unity/' target='_blank'>https://igotintogradschool2025.site/unity/</a>.
        </iframe>
    </div>
    """,
    sizing_mode='stretch_width',
    styles={'margin-top': '15px'}
)
unity_container = column(
    unity_description,
    unity_iframe,
    sizing_mode='stretch_width'
)

# --- Footer / Credits ---
footer = Div(text="""
    <hr style='margin-top: 30px;'>
    <h3>Contributors</h3>
    <p style='font-size: 11pt;'>
        <strong>Adam Kuhn<sup>1,2,3,4</sup>, Vinícius Contessoto<sup>4</sup>,
        George N Phillips Jr.<sup>2,3</sup>, José Onuchic<sup>1,2,3,4</sup></strong><br>
        <small>
            <sup>1</sup>Department of Physics, Rice University |
            <sup>2</sup>Department of Chemistry, Rice University |
            <sup>3</sup>Department of Biosciences, Rice University |
            <sup>4</sup>Center for Theoretical Biological Physics, Rice University
        </small>
    </p>
    """, sizing_mode='stretch_width', styles={'margin-top': '20px'})


# --- Layout Assembly ---

# Arrange the individual scatter plots and their info divs vertically
scatter_col_exp = column(p_scatter_exp, regression_info_exp, sizing_mode="stretch_width", styles={'min-width': '300px'})
scatter_col_af = column(p_scatter_af, regression_info_af, sizing_mode="stretch_width", styles={'min-width': '300px'})
scatter_col_evol = column(p_scatter_evol, regression_info_evol, sizing_mode="stretch_width", styles={'min-width': '300px'})

# Arrange the scatter plot columns horizontally in a row
scatter_row = row(
    scatter_col_exp,
    scatter_col_af,
    scatter_col_evol,
    sizing_mode="stretch_width",
    styles={ # Basic flexbox for responsiveness
        'display': 'flex', 'flex-wrap': 'wrap', 'gap': '20px',
        'justify-content': 'space-around', 'margin-top': '20px'
    }
)

# Main visualization section (controls, line plot, scatter plots)
visualization_section = column(
    # Use select_pdb_id instead of select_file
    row(select_pdb_id, window_slider, styles={'gap': '20px'}), # Controls side-by-side
    p, # Main line plot
    scatter_row, # Row of scatter plots
    sizing_mode='stretch_width',
    name="visualization_section"
)

# Correlation table section (filters and table)
correlation_section = column(
    Div(text="<h2>Correlation Analysis Across All PDB IDs</h2>", styles={'margin-top': '30px'}), # Updated title
    controls_layout, # Checkbox filters
    Spacer(height=15),
    data_table, # The correlation table itself
    sizing_mode='stretch_width',
    name="correlation_section"
)


# Assemble the final layout
main_layout = column(
    header,
    plot_description,
    visualization_section,
    unity_container, # Include the Unity section
    additional_plots_column, # Include the aggregated plots
    correlation_section, # Include the correlation table and filters
    footer,
    sizing_mode='stretch_width' # Make the overall layout stretch to width
)

# Add the final layout to the current document
curdoc().add_root(main_layout)
# Set the title of the browser tab
curdoc().title = "Evolutionary Frustration Dashboard"

print("Bokeh application layout created and added to document.")
# End of script
