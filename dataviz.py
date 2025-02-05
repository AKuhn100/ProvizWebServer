import os
import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, linregress

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, CheckboxGroup, Div, Spacer,
    DataTable, TableColumn, NumberFormatter, HoverTool, 
    GlyphRenderer, Slider, Whisker, Label, Range1d, Legend
)
from bokeh.plotting import figure
from bokeh.layouts import column, row, layout
from bokeh.palettes import Category10

###############################################################################
# SECTION 1: Core Configuration and Imports
# 
# This section contains:
# - All required imports
# - Basic configuration settings
# - Core helper functions for data processing
#
# Dependencies: numpy, pandas, scipy, bokeh
# No dependencies on other sections
###############################################################################
# Local data directory path
DATA_DIR = "summary_data_20R"  # Directory containing the summary files

# Filename pattern to include only relevant files
FILE_PATTERN = r"^summary_.+\.txt$"  # Adjust or remove as needed

# Default file to visualize on startup
DEFAULT_FILE = "summary_test001.txt"  # Change to your preferred default or set to ""

###############################################################################
# 2) Helpers: Data Parsing and Aggregation
###############################################################################
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

    # Apply moving average
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        df_for_plot[col] = moving_average(df_for_plot[col].values, window_size=5)

    # Min-Max normalization
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        valid = ~df_for_plot[col].isna()
        if valid.any():
            col_min = df_for_plot.loc[valid, col].min()
            col_max = df_for_plot.loc[valid, col].max()
            if col_max > col_min:
                df_for_plot[col] = (df_for_plot[col] - col_min) / (col_max - col_min)

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

    return df_original, df_for_plot, corrs

def remove_regression_renderers(fig):
    """
    Removes all renderers from the given figure whose names start with 'regression_'.
    Safely handles cases where the renderer or its name might be None.
    """
    new_renderers = []
    for r in fig.renderers:
        # Check if the renderer is not None
        if r is not None:
            # Safely get the name attribute; default to empty string if not present
            name = getattr(r, 'name', '')
            # Ensure name is a string before calling startswith
            if isinstance(name, str) and name.startswith('regression_'):
                continue  # Skip renderers related to regression
        new_renderers.append(r)  # Retain all other renderers
    fig.renderers = new_renderers
    
###############################################################################
# SECTION 2: Data Loading and Processing
# 
# This section contains:
# - Data loading from local directory
# - Initial data processing and aggregation
# - Creation of data structures for visualization
#
# Dependencies: Section 1 (bokeh-core)
# Required before: All visualization sections
###############################################################################
data_by_file = {}
all_corr_rows = []

# Aggregation lists
protein_names = []
avg_bfactors = []
std_bfactors = []
spearman_exp = []
spearman_af = []
spearman_evol = []

# Possible frustration columns
POSSIBLE_FRUST_COLUMNS = ['ExpFrust', 'AFFrust', 'EvolFrust']

# Color mapping for plots - Consistent colors across all visualizations
FRUSTRATION_COLORS = {
    "ExpFrust.": "#E41A1C",    # Red
    "AFFrust.": "#377EB8",     # Blue
    "EvolFrust.": "#4DAF4A",   # Green
    "Spearman_Diff": "#E6AB02" # Gold
}

# B-factor color (orange) will be set in the main plot configuration

# Iterate through files
for filename in os.listdir(DATA_DIR):
    if not re.match(FILE_PATTERN, filename):
        print(f"Skipping {filename}: does not match pattern {FILE_PATTERN}")
        continue

    file_path = os.path.join(DATA_DIR, filename)
    df_orig, df_plot, corrs = parse_summary_file(file_path)
    if df_orig is None:
        continue

    data_by_file[filename] = {
        "df_original": df_orig,
        "df_for_plot": df_plot,
        "corrs": corrs
    }

    # Collect correlation data
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo
        all_corr_rows.append([filename, mA, mB, rho, pval])

    # Aggregate data for additional plots
    avg_b = df_orig['B_Factor'].mean()
    std_b = df_orig['B_Factor'].std()

    spearman_r_exp = corrs.get(("B_Factor", "ExpFrust"), (np.nan, np.nan))[0]
    spearman_r_af = corrs.get(("B_Factor", "AFFrust"), (np.nan, np.nan))[0]
    spearman_r_evol = corrs.get(("B_Factor", "EvolFrust"), (np.nan, np.nan))[0]

    protein_names.append(filename)
    avg_bfactors.append(avg_b)
    std_bfactors.append(std_b)
    spearman_exp.append(spearman_r_exp)
    spearman_af.append(spearman_r_af)
    spearman_evol.append(spearman_r_evol)

# Correlation DataFrame
df_all_corr = pd.DataFrame(all_corr_rows, columns=["Test","MetricA","MetricB","Rho","Pval"])

# Aggregated DataFrame for Additional Plots
data_proviz = pd.DataFrame({
    'Protein': protein_names,
    'Avg_B_Factor': avg_bfactors,
    'Std_B_Factor': std_bfactors,
    'Spearman_ExpFrust': spearman_exp,
    'Spearman_AFFrust': spearman_af,
    'Spearman_EvolFrust': spearman_evol
})

# Calculate Spearman difference
data_proviz['Spearman_Diff'] = (
    data_proviz['Spearman_EvolFrust'] - 
    data_proviz['Spearman_ExpFrust']
)

# Sort protein names based on Spearman difference
protein_order = data_proviz.sort_values('Spearman_Diff')['Protein'].tolist()

# Update data_long_avg and data_long_std to include ordered Protein categories
data_long_avg = data_proviz.melt(
    id_vars=['Protein', 'Avg_B_Factor'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)

data_long_std = data_proviz.melt(
    id_vars=['Protein', 'Std_B_Factor'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)

# Clean Frust_Type names for avg and std
data_long_avg['Frust_Type'] = data_long_avg['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')
data_long_std['Frust_Type'] = data_long_std['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

# Remove rows with NaN correlations
data_long_avg.dropna(subset=['Spearman_Rho'], inplace=True)
data_long_std.dropna(subset=['Spearman_Rho'], inplace=True)

# Create data_long_corr from the original correlation DataFrame first
data_long_corr = df_all_corr.copy()

# Clean and format correlation data
data_long_corr['Frust_Type'] = ''  # Empty Frust_Type for original correlations

# Create and add Spearman data for the visualization
spearman_viz_data = data_proviz.melt(
    id_vars=['Protein'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Rho'  # Changed from 'Spearman_Rho' to 'Rho' to match original data
)

# Clean Frust_Type names for visualization data
spearman_viz_data['Frust_Type'] = spearman_viz_data['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

# Add difference data
spearman_diff_data = pd.DataFrame({
    'Protein': data_proviz['Protein'],
    'Rho': data_proviz['Spearman_Diff'],  # Changed from 'Spearman_Rho' to 'Rho'
    'Frust_Type': 'Spearman_Diff',
    'Test': None,
    'MetricA': None,
    'MetricB': None,
    'Pval': None
})

# Combine all data
spearman_viz_data = pd.concat([
    spearman_viz_data, 
    spearman_diff_data[spearman_viz_data.columns]
], ignore_index=True)

# Ensure all required columns exist
required_cols = ['Test', 'MetricA', 'MetricB', 'Pval', 'Protein', 'Rho', 'Frust_Type']
for col in required_cols:
    if col not in spearman_viz_data.columns:
        spearman_viz_data[col] = None

# Combine table data with visualization data
data_long_corr = pd.concat([
    df_all_corr,
    spearman_viz_data[required_cols]
], ignore_index=True)

# Remove rows with NaN correlations and ensure proper types
data_long_corr = data_long_corr.dropna(subset=['Rho'])
data_long_corr['Frust_Type'] = data_long_corr['Frust_Type'].fillna('')

# Make Protein categorical with ordered categories based on Spearman_Diff
if 'Protein' in data_long_corr.columns and protein_order:
    data_long_corr['Protein'] = pd.Categorical(
        data_long_corr['Protein'],
        categories=protein_order,
        ordered=True
    )

# Update the FRUSTRATION_COLORS dictionary to include Spearman_Diff
FRUSTRATION_COLORS["Spearman_Diff"] = Category10[10][4]  # Orange color for difference

# Create ColumnDataSource for the correlation plot
source_corr_plot = ColumnDataSource(data_long_corr)

###############################################################################
# SECTION 3: Main Visualization Components
# 
# This section contains:
# A) Main line plot for normalized data
# B) Scatter plots for rank correlations
# C) Hover tools and info displays
#
# Dependencies: Sections 1-2
# Required before: Callbacks and layout sections
###############################################################################

# Initialize Data Sources
source_plot = ColumnDataSource(data=dict(
    x=[],
    residue=[],
    b_factor=[],
    exp_frust=[],
    af_frust=[],
    evol_frust=[]
))

source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], rank_x=[], rank_y=[]))
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], rank_x=[], rank_y=[]))
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[], rank_x=[], rank_y=[]))

# A) Main Line Plot
p = figure(
    title="(No Data)",
    sizing_mode='stretch_width',
    height=600,
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom", 
    active_scroll=None
)

# Main Plot Hover Tools
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

# Color definitions and line renderers
color_map = {
    "b_factor":  ("B-Factor", "#FF7F00"),      # Orange
    "exp_frust": ("ExpFrust", "#E41A1C"),      # Red
    "af_frust":  ("AFFrust", "#377EB8"),       # Blue
    "evol_frust":("EvolFrust", "#4DAF4A")      # Green
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

# B) Scatter Plots
hover_scatter = HoverTool(
    tooltips=[
        ("B-Factor", "@x_orig{0.3f}"),
        ("B-Factor Rank", "@rank_x{0.0f}"),
        ("Frustration", "@y_orig{0.3f}"),
        ("Frustration Rank", "@rank_y{0.0f}")
    ],
    mode='mouse'
)

p_scatter_exp = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="B-Factor Rank",
    y_axis_label="Experimental Frustration Rank",
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"],
    active_drag="box_zoom",
    active_scroll=None
)

p_scatter_af = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="B-Factor Rank",
    y_axis_label="AlphaFold Frustration Rank",
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"],
    active_drag="box_zoom",
    active_scroll=None
)

p_scatter_evol = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="B-Factor Rank",
    y_axis_label="Evolutionary Frustration Rank",
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"],
    active_drag="box_zoom",
    active_scroll=None
)

# Add scatter glyphs and hover tools
p_scatter_exp.scatter("x", "y", source=source_scatter_exp, color="#E41A1C", alpha=0.7)   # Red
p_scatter_af.scatter("x", "y", source=source_scatter_af,  color="#377EB8", alpha=0.7)    # Blue
p_scatter_evol.scatter("x", "y", source=source_scatter_evol, color="#4DAF4A", alpha=0.7) # Green

p_scatter_exp.add_tools(hover_scatter)
p_scatter_af.add_tools(hover_scatter)
p_scatter_evol.add_tools(hover_scatter)

# C) Info Displays - with centered alignment and 50% width
regression_info_exp = Div(
    text="", 
    styles={
        'background-color': '#f8f9fa',
        'padding': '10px',
        'border': '1px solid #ddd',
        'border-radius': '4px',
        'margin': '10px auto',  # Changed to auto margins for centering
        'font-size': '14px',
        'text-align': 'center',
        'width': '50%',         # Changed to 50% width
        'display': 'block'      # Ensures block-level display
    }
)

regression_info_af = Div(
    text="",
    styles={
        'background-color': '#f8f9fa',
        'padding': '10px',
        'border': '1px solid #ddd',
        'border-radius': '4px',
        'margin': '10px auto',  # Changed to auto margins for centering
        'font-size': '14px',
        'text-align': 'center',
        'width': '50%',         # Changed to 50% width
        'display': 'block'      # Ensures block-level display
    }
)

regression_info_evol = Div(
    text="",
    styles={
        'background-color': '#f8f9fa',
        'padding': '10px',
        'border': '1px solid #ddd',
        'border-radius': '4px',
        'margin': '10px auto',  # Changed to auto margins for centering
        'font-size': '14px',
        'text-align': 'center',
        'width': '50%',         # Changed to 50% width
        'display': 'block'      # Ensures block-level display
    }
)

###############################################################################
# SECTION 4: Callback Functions and Event Handlers
# 
# This section contains:
# - Plot update callbacks
# - Data normalization functions
# - Event handling for widget interactions
#
# Dependencies: Sections 1-3
# Required before: All visualization and layout sections
###############################################################################

# Initialize widgets
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

window_slider = Slider(
    start=1, 
    end=21, 
    value=5, 
    step=2, 
    title="Moving Average Window Size",
    width=400
)

def min_max_normalize(arr):
    """
    Applies min-max normalization to a numpy array.
    Returns an array normalized to [0, 1]. Handles division by zero.
    """
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    else:
        return np.zeros_like(arr)


def add_regression_line_and_info(fig, xvals, yvals, color="black", info_div=None, plot_type="", 
                               use_spearman=True, x_orig=None, y_orig=None):
    """
    Adds a regression line and updates the info Div.
    For Spearman correlation, uses ranks and displays Spearman's rho.
    """
    if len(xvals) < 2 or np.all(xvals == xvals[0]):
        if info_div:
            info_div.text = "Insufficient data for correlation"
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
            info_div.text = "Insufficient data for correlation"
        return

    if use_spearman and x_orig is not None and y_orig is not None:
        # Calculate Spearman correlation from original values
        rho, pval = spearmanr(x_orig, y_orig)
        
        # Use linear regression on ranks for the line
        slope, intercept, _, _, _ = linregress(xvals_clean, yvals_clean)
    else:
        # Fall back to Pearson for line fitting
        slope, intercept, r_value, pval, _ = linregress(xvals_clean, yvals_clean)
        rho = r_value

    # Plot regression line
    x_range = np.linspace(0, 1, 100)  # Use [0,1] for normalized ranks
    y_range = slope * x_range + intercept
    
    regression_line = fig.line(
        x_range, y_range, 
        line_width=2, line_dash='dashed', color=color, 
        name=f'regression_line_{plot_type}'
    )

    # Create hover source
    regression_source = ColumnDataSource(data=dict(
        x=x_range,
        y=y_range,
        equation=[f"ρ = {rho:.3f}"] * len(x_range)
    ))

    invisible_regression = fig.line(
        'x', 'y', 
        source=regression_source, 
        line_width=10, 
        alpha=0, 
        name=f'regression_hover_{plot_type}'
    )

    # Update regression info div
    if info_div:
        info_div.text = f"""
        <div style='color: {color}'>
            <strong>Spearman ρ = {rho:.3f}</strong><br>
            <span style='font-size: 12px'>p = {pval:.2e}</span>
        </div>
        """


def update_plot(attr, old, new):
    """
    Updates both the main plot and scatter plots when a new file is selected.
    """
    filename = select_file.value
    if filename not in data_by_file:
        # Reset all data sources if file not found
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        source_scatter_exp.data = dict(x=[], y=[], x_orig=[], y_orig=[], rank_x=[], rank_y=[])
        source_scatter_af.data = dict(x=[], y=[], x_orig=[], y_orig=[], rank_x=[], rank_y=[])
        source_scatter_evol.data = dict(x=[], y=[], x_orig=[], y_orig=[], rank_x=[], rank_y=[])
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

    # Update main line plot with normalized data
    df_orig = data_by_file[filename]["df_original"]
    df_plot = df_orig.copy()

    # Apply moving average
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr = df_plot[col].values
        df_plot[col] = moving_average(arr, window_size=window_size)
        df_plot[col] = min_max_normalize(df_plot[col])

    # Update main plot
    sub_plot = df_plot.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    if sub_plot.empty:
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        p.title.text = f"{filename} (No valid rows)."
    else:
        source_plot.data = {
            'x': sub_plot["AlnIndex"].tolist(),
            'residue': sub_plot["Residue"].tolist(),
            'b_factor': sub_plot["B_Factor"].tolist(),
            'exp_frust': sub_plot["ExpFrust"].tolist(),
            'af_frust': sub_plot["AFFrust"].tolist(),
            'evol_frust': sub_plot["EvolFrust"].tolist()
        }
        p.title.text = f"{filename} (Smoothed + Normalized)"

    # Update scatter plots
    df_orig = data_by_file[filename]["df_original"]
    sub_orig = df_orig.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])

    # Reset regression renderers and data sources
    remove_regression_renderers(p_scatter_exp)
    remove_regression_renderers(p_scatter_af)
    remove_regression_renderers(p_scatter_evol)

    if sub_orig.empty:
        p_scatter_exp.title.text = f"{filename} (No Data)"
        p_scatter_af.title.text = f"{filename} (No Data)"
        p_scatter_evol.title.text = f"{filename} (No Data)"
    else:
        # Update scatter plots with rank-based data
        for metric, source, plot, info, color in [
            ("ExpFrust", source_scatter_exp, p_scatter_exp, regression_info_exp, "#E41A1C"),
            ("AFFrust", source_scatter_af, p_scatter_af, regression_info_af, "#377EB8"),
            ("EvolFrust", source_scatter_evol, p_scatter_evol, regression_info_evol, "#4DAF4A")
        ]:
            x_orig = sub_orig["B_Factor"].values
            y_orig = sub_orig[metric].values
            
            # Calculate ranks
            rank_x = pd.Series(x_orig).rank()
            rank_y = pd.Series(y_orig).rank()
            
            # Normalize ranks to [0, 1]
            rank_x_norm = (rank_x - rank_x.min()) / (rank_x.max() - rank_x.min())
            rank_y_norm = (rank_y - rank_y.min()) / (rank_y.max() - rank_y.min())
            
            source.data = {
                'x': rank_x_norm,
                'y': rank_y_norm,
                'x_orig': x_orig,
                'y_orig': y_orig,
                'rank_x': rank_x,
                'rank_y': rank_y
            }
            
            plot.title.text = f"{filename} {metric}"
            
            add_regression_line_and_info(
                fig=plot, 
                xvals=rank_x_norm,
                yvals=rank_y_norm, 
                color=color, 
                info_div=info,
                plot_type=metric.lower(),
                use_spearman=True,
                x_orig=x_orig,
                y_orig=y_orig
            )


def update_moving_average(attr, old, new):
    """Update plot when slider value changes"""
    update_plot(None, None, select_file.value)


# Attach callbacks
window_slider.on_change('value', update_moving_average)
select_file.on_change("value", update_plot)

# Initialize plot with default file
if initial_file:
    update_plot(None, None, initial_file)

###############################################################################
# SECTION 5: Correlation Analysis Components
# 
# This section contains:
# - Correlation table setup
# - Table columns and formatting
# - Data source initialization
#
# Dependencies: Sections 1-2
# Required before: Filter controls and layout sections
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
    # Updated to include 'Frust_Type' from the integrated changes
    source_corr = ColumnDataSource(data_long_corr)
    columns = [
        TableColumn(field="Test", title="Test"),
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Spearman Rho", formatter=NumberFormatter(format="0.3f")),
        TableColumn(field="Pval", title="p-value", formatter=NumberFormatter(format="0.2e")),
        TableColumn(field="Frust_Type", title="Frustration Type")
    ]
    data_table = DataTable(columns=columns, source=source_corr, height=400, width=1200)

###############################################################################
# SECTION 6: Filter Controls and Callbacks
# 
# This section contains:
# - Filter UI components
# - Filter logic and callbacks
# - Layout for filter controls
#
# Dependencies: Sections 1-4
# Required before: Layout assembly
###############################################################################

# Define helper function to split labels into columns
def split_labels(labels, num_columns):
    """
    Splits a list of labels into a list of lists, each sublist containing labels for one column.
    """
    if num_columns <= 0:
        return [labels]
    k, m = divmod(len(labels), num_columns)
    return [labels[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_columns)]

# Define the number of columns for better layout
NUM_COLUMNS = 3

tests_in_corr = sorted(df_all_corr["Test"].unique()) if not df_all_corr.empty else []
if not df_all_corr.empty:
    combo_options = sorted({
        f"{row['MetricA']} vs {row['MetricB']}" 
        for _, row in df_all_corr.iterrows()
    })
else:
    combo_options = []

if not df_all_corr.empty:
    # Split labels into columns
    test_labels_split = split_labels(tests_in_corr, NUM_COLUMNS)
    combo_labels_split = split_labels(combo_options, NUM_COLUMNS)
    
    # Create CheckboxGroups for Tests
    checkbox_tests_columns = [
        CheckboxGroup(
            labels=col_labels,
            active=[],  # Initially no selection
            name=f'tests_column_{i+1}'
        ) for i, col_labels in enumerate(test_labels_split)
    ]
    
    # Create CheckboxGroups for Metric Pairs
    checkbox_combos_columns = [
        CheckboxGroup(
            labels=col_labels,
            active=[],  # Initially no selection
            name=f'combos_column_{i+1}'
        ) for i, col_labels in enumerate(combo_labels_split)
    ]
else:
    checkbox_tests_columns = [CheckboxGroup(labels=[], active=[], name='tests_column_1')]
    checkbox_combos_columns = [CheckboxGroup(labels=[], active=[], name='combos_column_1')]

# Create Columns for Tests and Metric Pairs
tests_layout = row(*checkbox_tests_columns, sizing_mode='stretch_width', width=300)
combos_layout = row(*checkbox_combos_columns, sizing_mode='stretch_width', width=300)

# Add Titles Above Each CheckboxGroup
tests_title = Div(text="<b>Select Tests:</b>", styles={'font-size': '14px', 'margin-bottom': '5px'})
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
        selected.extend([checkbox.labels[i] for i in checkbox.active])
    return selected

def update_corr_filter(attr, old, new):
    """Filter correlation table based on selected tests and metric pairs."""
    if data_long_corr.empty:
        return
    
    # Aggregate selected tests and metric pairs from all CheckboxGroups
    selected_tests = get_selected_labels(checkbox_tests_columns)
    selected_combos = get_selected_labels(checkbox_combos_columns)

    if not selected_tests and not selected_combos:
        filtered = data_long_corr
    else:
        df_tmp = data_long_corr.copy()
        df_tmp["combo_str"] = df_tmp.apply(lambda r: f"{r['MetricA']} vs {r['MetricB']}" if pd.notna(r['MetricA']) and pd.notna(r['MetricB']) else '', axis=1)

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
            filtered = data_long_corr

    source_corr.data = filtered.to_dict(orient="list")

# Attach callbacks to all CheckboxGroups
for checkbox in checkbox_tests_columns + checkbox_combos_columns:
    checkbox.on_change('active', update_corr_filter)

###############################################################################
# SECTION 7: Additional Visualization Components
# 
# This section contains:
# - Correlation plots with sorting functionality
# - Summary statistics
#
# Dependencies: Sections 1-3
# Required before: Layout assembly
###############################################################################

from math import pi  # Ensure pi is imported
from scipy import stats
import numpy as np
from bokeh.plotting import figure
from bokeh.models import Label, ColumnDataSource

def create_violin_plot():
    """Create a violin plot showing the distribution of correlations for each frustration type with embedded box plots."""
    # Prepare the data
    violin_data = []
    labels = {
        'ExpFrust.': 'Experimental',
        'AFFrust.': 'AlphaFold',
        'EvolFrust.': 'Evolutionary'
    }
    
    # Collect data for each frustration type
    for frust_type in ['ExpFrust.', 'AFFrust.', 'EvolFrust.']:
        data = data_long_corr[data_long_corr['Frust_Type'] == frust_type]['Rho']
        
        if not data.empty:
            # Calculate violin curve using kernel density estimation
            kernel = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            y_range = kernel(x_range)
            
            # Mirror the density curve for the violin plot
            violin_data.append({
                'x': np.concatenate([x_range, x_range[::-1]]),
                'y': np.concatenate([y_range, -y_range[::-1]]),
                'frust_type': frust_type,
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'q1': data.quantile(0.25),
                'q3': data.quantile(0.75),
                'min': data.min(),
                'max': data.max(),
                'label': labels[frust_type]
            })

    # Create violin plot
    p_violin = figure(
        title="Distribution of Spearman Correlations by Frustration Type",
        x_axis_label="Spearman Correlation Between Frustration and B-factor",
        y_axis_label="Frustration Metric",
        height=600,
        sizing_mode="stretch_width",
        toolbar_location=None
    )
    
    # Define box plot parameters
    box_height = 0.03  # Height of the box plot
    cap_size = 0.00   # Size of the whisker caps

    # Plot violins and box plots
    for i, data in enumerate(violin_data):
        # Create source for violin curve
        source = ColumnDataSource({
            'x': data['x'],
            'y': [i + y/10 for y in data['y']],  # Scale and shift the density curve
        })
        
        # Plot violin
        color = FRUSTRATION_COLORS[data['frust_type']]
        p_violin.patch('x', 'y', source=source, color=color, alpha=0.6, line_color='black')
        
        # Add mean line
        p_violin.line([data['mean'], data['mean']], [i - 0.2, i + 0.2], 
                     line_color='black', line_width=2)
        
        # Add text annotations with centered vertical alignment and right offset
        mean_label = Label(
            x=data['mean'], 
            y=i,
            text=f"μ = {data['mean']:.3f}\n\nσ = {data['std']:.3f}",
            text_font_size='10pt',
            text_align='left',
            text_baseline='middle',  # Vertically center the text
            x_offset=5,              # Offset to the right by 5 pixels
            y_offset=0
        )
        p_violin.add_layout(mean_label)
        
        # Add box plot components
        # 1. Draw the box (Q1 to Q3)
        p_violin.rect(
            x=(data['q1'] + data['q3']) / 2,  # Center of the box
            y=i,                               # y-position
            width=(data['q3'] - data['q1']),   # Width of the box (IQR)
            height=box_height,                 # Height of the box
            fill_color='black',
            line_color='black'
        )
        
        # 2. Draw the median line
        p_violin.segment(
            x0=data['median'], 
            y0=i - box_height / 2, 
            x1=data['median'], 
            y1=i + box_height / 2, 
            line_color='white', 
            line_width=2
        )
        
        # 3. Draw the whiskers (from Q1 to Min and Q3 to Max)
        # Whisker from Q1 to Min
        p_violin.segment(
            x0=data['min'], 
            y0=i, 
            x1=data['q1'], 
            y1=i, 
            line_color='black', 
            line_width=1
        )
        
        # Whisker from Q3 to Max
        p_violin.segment(
            x0=data['q3'], 
            y0=i, 
            x1=data['max'], 
            y1=i, 
            line_color='black', 
            line_width=1
        )
        
        # 4. Draw whisker caps
        # Left cap at Min
        p_violin.segment(
            x0=data['min'], 
            y0=i - cap_size, 
            x1=data['min'], 
            y1=i + cap_size, 
            line_color='black', 
            line_width=1
        )
        
        # Right cap at Max
        p_violin.segment(
            x0=data['max'], 
            y0=i - cap_size, 
            x1=data['max'], 
            y1=i + cap_size, 
            line_color='black', 
            line_width=1
        )

    # Customize plot
    p_violin.yaxis.ticker = list(range(len(violin_data)))
    p_violin.yaxis.major_label_overrides = {i: data['label'] for i, data in enumerate(violin_data)}
    p_violin.grid.grid_line_color = None
    p_violin.background_fill_color = "#f8f9fa"
    
    return p_violin

# Initialize data source
source_corr_plot = ColumnDataSource(data_long_corr)

# Create violin plot
p_violin = create_violin_plot()

# Initialize data source
source_corr_plot = ColumnDataSource(data_long_corr)

# Ensure protein_order is defined
if 'protein_order' not in globals():
    protein_order = []

# Create sorting button and callback
def update_sort_order(attr, old, new):
    """Update plot order when sort selection changes"""
    selected_metric = sort_select.value
    print(f"Sorting by: {selected_metric}")  # Debug print

    if selected_metric == "Spearman Diff":
        # Sort by Spearman difference
        sorted_data = data_proviz.sort_values('Spearman_Diff')
        new_order = sorted_data['Protein'].tolist()
    else:
        # Sort by the selected metric's correlation values
        metric_data = data_long_corr[data_long_corr['Frust_Type'] == selected_metric]
        sorted_data = metric_data.sort_values('Rho')
        new_order = sorted_data['Protein'].tolist()
    
    print(f"New order: {new_order}")  # Debug print
    
    # Update the plot's x-range
    p_corr_plot.x_range.factors = new_order

    # Update each renderer's data source
    for frust in frust_types_corr:
        if frust != "":
            subset = data_long_corr[data_long_corr['Frust_Type'] == frust].copy()
            if not subset.empty:
                # Reset the categorical order
                subset['Protein'] = pd.Categorical(subset['Protein'], 
                                                categories=new_order, 
                                                ordered=True)
                subset = subset.sort_values('Protein')
                renderer = next(r for r in p_corr_plot.renderers 
                             if isinstance(r, GlyphRenderer) and 
                             r.name == f'scatter_{frust}')
                renderer.data_source.data.update({
                    'Protein': subset['Protein'].tolist(),
                    'Rho': subset['Rho'].tolist(),
                    'Frust_Type': subset['Frust_Type'].tolist()
                })

sort_select = Select(
    title="Sort Proteins By:",
    value="Spearman Diff",
    options=["Spearman Diff", "ExpFrust.", "AFFrust.", "EvolFrust."],
    width=200
)

# Attach the callback
sort_select.on_change('value', update_sort_order)

# Spearman Rho per Protein and Frustration Metric
p_corr_plot = figure(
    title="Spearman Correlation per Protein and Frustration Metric",
    x_axis_label="Protein (ordered by selected metric)",
    y_axis_label="Spearman Correlation Between Frustration and B-Factor",
    x_range=protein_order,
    sizing_mode='stretch_width',
    height=600,
    tools="pan,box_zoom,wheel_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None,
    toolbar_location="above"
)

# Define color palette for Frustration Types
frust_types_corr = [ft for ft in data_long_corr['Frust_Type'].unique() if ft != ""]
color_map_corr = {frust: FRUSTRATION_COLORS.get(frust, Category10[10][i]) 
                 for i, frust in enumerate(frust_types_corr)}

# Add HoverTool
hover_corr = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Metric", "@Frust_Type"),
        ("Spearman Rho", "@Rho{0.3f}")
    ],
    mode='mouse'
)
p_corr_plot.add_tools(hover_corr)

# Add horizontal line at y=0
p_corr_plot.line(
    x=[-0.5, len(protein_order) - 0.5], 
    y=[0, 0], 
    line_width=1, 
    line_dash='dashed', 
    color='gray', 
    name='y_zero_line'
)

# Add scatter glyphs
legend_items = []

for frust in frust_types_corr:
    if frust != "":  # Skip empty Frust_Type
        subset = data_long_corr[data_long_corr['Frust_Type'] == frust].copy()
        
        if not subset.empty and 'Protein' in subset.columns and 'Rho' in subset.columns:
            # Ensure Protein is categorical with proper ordering
            subset['Protein'] = pd.Categorical(subset['Protein'], 
                                            categories=protein_order, 
                                            ordered=True)
            # Sort by Protein to maintain order
            subset = subset.sort_values('Protein')
            source_subset = ColumnDataSource(subset)
            
            renderer = p_corr_plot.scatter(
                x='Protein',
                y='Rho',
                source=source_subset,
                color=color_map_corr[frust],
                size=8,
                alpha=0.6,
                name=f'scatter_{frust}'
            )
            legend_items.append((frust, [renderer]))

if legend_items:
    legend = Legend(items=legend_items, 
                   location="top_left", 
                   title="Frustration Type", 
                   click_policy="mute")
    p_corr_plot.add_layout(legend)

# Rotate x-axis labels
p_corr_plot.xaxis.major_label_orientation = pi / 4  # 45 degrees

# Create layout for this section
plot_controls = row(
    sort_select,
    sizing_mode="stretch_width",
    name="plot_controls"
)

correlation_layout = column(
    plot_controls,
    p_corr_plot,
    sizing_mode="stretch_width",
    name="correlation_layout"
)

###############################################################################
# SECTION 7A: 20F Data Processing and Multi-Figure Layout
# 
# Copy/paste this entire snippet to replace your existing build_frustration_comparison_20F 
# function. Make sure it appears *after* read_frustration_file_20F and lowess_smoothing
# but *before* final layout references to "build_frustration_comparison_20F".
###############################################################################

def build_frustration_comparison_20F(filepath):
    """
    Constructs and returns a Bokeh layout of multiple plots 
    that approximates the original seaborn multi-subplot figure for 20F data.

    Color scheme:
      - REP1 ExpFrust: Dark red (#8B0000)
      - REP2 ExpFrust: Light red (#FF4444)
      - EvolFrust:     Green (#4DAF4A)
    
    Contains:
      1) Main line plot (full-width) with LOWESS smoothed lines
      2) Six rank-based scatter subplots, arranged in a 2-column-by-3-rows grid:
         (a) REP2 B-Factor vs. REP2 ExpFrust
         (b) REP2 B-Factor vs. REP1 ExpFrust
         (c) REP2 B-Factor vs. EvolFrust
         (d) REP1 B-Factor vs. REP2 ExpFrust
         (e) REP1 B-Factor vs. REP1 ExpFrust
         (f) REP1 B-Factor vs. EvolFrust
      Each scatter plot has a rank-based regression line and Spearman correlation text.
    """

    # 1) Read the 20F data
    rep1_df, rep2_df, evol_frust = read_frustration_file_20F(filepath)

    # 2) Merge on AlnIndex
    merged = rep1_df.merge(rep2_df, on='AlnIndex', suffixes=('_REP1','_REP2'))
    merged['EvolFrust'] = evol_frust

    # 3) Filter out rows missing any required columns
    req_cols = ['ExpFrust_REP1','ExpFrust_REP2','EvolFrust',
                'B_Factor_REP1','B_Factor_REP2']
    valid_mask = ~merged[req_cols].isna().any(axis=1)
    filtered = merged[valid_mask].copy()
    if filtered.empty:
        return column(figure(height=200, sizing_mode='stretch_width',
                             title="No valid data in file."))

    # 4) Apply LOWESS smoothing to each frustration/b-factor
    (x_rep1_exp, y_rep1_exp) = lowess_smoothing(filtered['AlnIndex'], filtered['ExpFrust_REP1'])
    (x_rep2_exp, y_rep2_exp) = lowess_smoothing(filtered['AlnIndex'], filtered['ExpFrust_REP2'])
    (x_evol,   y_evol)       = lowess_smoothing(filtered['AlnIndex'], filtered['EvolFrust'])
    (x_rep1_bf, y_rep1_bf)   = lowess_smoothing(filtered['AlnIndex'], filtered['B_Factor_REP1'])
    (x_rep2_bf, y_rep2_bf)   = lowess_smoothing(filtered['AlnIndex'], filtered['B_Factor_REP2'])

    # 5) Main line plot (full width)
    p_main = figure(
        sizing_mode='stretch_width',
        height=400,
        title="20F: REP1 vs REP2 Frustration (LOWESS Smoothed)",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        active_drag="box_zoom"
    )
    p_main.xaxis.axis_label = "Residue Number"
    p_main.yaxis.axis_label = "Frustration"

    # Create ColumnDataSource
    source_main = ColumnDataSource(data=dict(
        x_rep1_exp = x_rep1_exp,  y_rep1_exp = y_rep1_exp,
        x_rep2_exp = x_rep2_exp,  y_rep2_exp = y_rep2_exp,
        x_evol     = x_evol,      y_evol     = y_evol
    ))
    
    # Colors
    color_rep1 = "#8B0000"  # dark red
    color_rep2 = "#FF4444"  # light red
    color_evol = "#4DAF4A"  # green

    # REP1 ExpFrust line
    p_main.line('x_rep1_exp', 'y_rep1_exp', source=source_main,
                line_color=color_rep1, line_width=3,
                legend_label="REP1 ExpFrust")

    # REP2 ExpFrust line
    p_main.line('x_rep2_exp', 'y_rep2_exp', source=source_main,
                line_color=color_rep2, line_width=3, line_dash='dashed',
                legend_label="REP2 ExpFrust")

    # EvolFrust line
    p_main.line('x_evol', 'y_evol', source=source_main,
                line_color=color_evol, line_width=2, line_dash='dotdash',
                legend_label="EvolFrust")

    p_main.legend.location = "top_left"
    p_main.legend.click_policy = "hide"

    # 6) Create a helper to build rank-based scatter subplots
    def create_scatter_rank(xvals, yvals, title, color):
        """
        Build a single rank-based scatter figure with a regression line and Spearman text.
        """
        from bokeh.plotting import figure
        fig = figure(width=350, height=300, title=title,
                     tools="pan,box_zoom,reset,save",
                     active_drag="box_zoom")
        fig.xaxis.visible = False
        fig.yaxis.visible = False

        df_local = pd.DataFrame({'x': xvals, 'y': yvals}).dropna()
        if len(df_local) < 2:
            fig.title.text += "\n(Insufficient data)"
            return fig

        # Rank transform
        df_local['rank_x'] = df_local['x'].rank()
        df_local['rank_y'] = df_local['y'].rank()
        # 0-1 normalize ranks
        rx_min, rx_max = df_local['rank_x'].min(), df_local['rank_x'].max()
        ry_min, ry_max = df_local['rank_y'].min(), df_local['rank_y'].max()
        df_local['nx'] = (df_local['rank_x'] - rx_min) / (rx_max - rx_min + 1e-12)
        df_local['ny'] = (df_local['rank_y'] - ry_min) / (ry_max - ry_min + 1e-12)

        source_scat = ColumnDataSource(df_local)
        fig.scatter('nx', 'ny', source=source_scat, size=6, color=color, alpha=0.6)

        # Spearman correlation on original x,y
        rho, pval = spearmanr(df_local['x'], df_local['y'])

        # Regression line on the rank_x, rank_y
        slope, intercept, _, _, _ = linregress(df_local['nx'], df_local['ny'])
        x_line = np.linspace(0, 1, 50)
        y_line = slope * x_line + intercept
        fig.line(x_line, y_line, line_color='gray', line_dash='dashed')

        # Add correlation to the title
        fig.title.text += f"\nρ={rho:.3f}, p={pval:.1e}"

        return fig

    # 7) Build the six scatter subplots
    #    1) REP2 B-factor vs REP2 ExpFrust
    #    2) REP2 B-factor vs REP1 ExpFrust
    #    3) REP2 B-factor vs EvolFrust
    #    4) REP1 B-factor vs REP2 ExpFrust
    #    5) REP1 B-factor vs REP1 ExpFrust
    #    6) REP1 B-factor vs EvolFrust

    # For color consistency, use:
    #   - "#FF4444" for subplots related to REP2 experimental,
    #   - "#8B0000" for subplots related to REP1 experimental,
    #   - "#4DAF4A" for EvolFrust
    #   - B-factor: no color is needed since x-axis is B-factor rank

    p_s11 = create_scatter_rank(
        xvals=filtered['B_Factor_REP2'],
        yvals=filtered['ExpFrust_REP2'],
        title="REP2 B-Factor vs. REP2 ExpFrust",
        color=color_rep2
    )
    p_s12 = create_scatter_rank(
        xvals=filtered['B_Factor_REP2'],
        yvals=filtered['ExpFrust_REP1'],
        title="REP2 B-Factor vs. REP1 ExpFrust",
        color=color_rep1
    )
    p_s13 = create_scatter_rank(
        xvals=filtered['B_Factor_REP2'],
        yvals=filtered['EvolFrust'],
        title="REP2 B-Factor vs. EvolFrust",
        color=color_evol
    )
    p_s21 = create_scatter_rank(
        xvals=filtered['B_Factor_REP1'],
        yvals=filtered['ExpFrust_REP2'],
        title="REP1 B-Factor vs. REP2 ExpFrust",
        color=color_rep2
    )
    p_s22 = create_scatter_rank(
        xvals=filtered['B_Factor_REP1'],
        yvals=filtered['ExpFrust_REP1'],
        title="REP1 B-Factor vs. REP1 ExpFrust",
        color=color_rep1
    )
    p_s23 = create_scatter_rank(
        xvals=filtered['B_Factor_REP1'],
        yvals=filtered['EvolFrust'],
        title="REP1 B-Factor vs. EvolFrust",
        color=color_evol
    )

    # Arrange them in two columns, three rows
    scatter_grid_left = column(p_s11, p_s12, p_s13, sizing_mode='stretch_width')
    scatter_grid_right = column(p_s21, p_s22, p_s23, sizing_mode='stretch_width')
    scatter_layout = row(scatter_grid_left, scatter_grid_right, sizing_mode='stretch_width')

    # 8) Return the final layout
    return column(
        p_main,          # main line plot
        scatter_layout,  # 6 scatter plots
        sizing_mode='stretch_width'
    )

###############################################################################
# SECTION 8: UI Components and Static Content
# 
# This section contains:
# - Header and description components
# - Unity container and instructions
# - Static UI elements and styles
#
# Dependencies: Sections 1-2, plus data structures from data-processing
# Required before: Layout assembly
###############################################################################

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
    <div style="width: 95vw; display: flex; justify-content: center; align-items: center; margin: 20px auto; max-width: 1200px;">
        <iframe 
            src="https://igotintogradschool2025.site/unity/"
            style="width: 100%; height: 90vh; border: 2px solid #ddd; border-radius: 8px; 
                   box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
    </div>
    """,
    sizing_mode='stretch_width',
    styles={
        'margin-top': '20px',
        'display': 'flex',
        'justify-content': 'center'
    }
)
unity_iframe.visible = True

unity_container = column(
    description_visualizer,
    unity_iframe,
    sizing_mode='stretch_width'
)

# Note: File selection and window slider widgets are defined in Section 4 (Callbacks)

# Controls section
controls_section = Div(text="<b>Filter Correlation Table</b>", styles={'font-size': '16px', 'margin': '10px 0'})

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

###############################################################################
# SECTION 9: Final Layout Assembly
# 
# This section contains:
# - Final layout configuration
# - Component assembly
# - Document setup
#
# Dependencies: All previous sections must be fully loaded
# IMPORTANT: All widgets (select_file, window_slider, etc.) must be defined before this section
###############################################################################

# Scatter Plots Layout with centered regression info
scatter_col_exp = column(
    p_scatter_exp, 
    regression_info_exp, 
    sizing_mode="stretch_width",
    styles={
        'flex': '1 1 350px', 
        'min-width': '350px',
        'align-items': 'center',    # Center children horizontally
        'display': 'flex',          # Use flexbox
        'flex-direction': 'column'  # Stack children vertically
    }
)
scatter_col_af = column(
    p_scatter_af, 
    regression_info_af, 
    sizing_mode="stretch_width",
    styles={
        'flex': '1 1 350px', 
        'min-width': '350px',
        'align-items': 'center',    # Center children horizontally
        'display': 'flex',          # Use flexbox
        'flex-direction': 'column'  # Stack children vertically
    }
)
scatter_col_evol = column(
    p_scatter_evol, 
    regression_info_evol, 
    sizing_mode="stretch_width",
    styles={
        'flex': '1 1 350px', 
        'min-width': '350px',
        'align-items': 'center',    # Center children horizontally
        'display': 'flex',          # Use flexbox
        'flex-direction': 'column'  # Stack children vertically
    }
)

# Scatter plots row with consistent spacing
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
        'margin': '20px auto',
        'flex-wrap': 'wrap'
    }
)

# Create a sub-column for the 20F UI
visualization_section_20F = column(
    Div(text="<h2>20F Frustration Comparison</h2>"),
    select_file_20F,
    layout_20F_display,  # The column we update when select_file_20F changes
    sizing_mode='stretch_width'
)

# Define a spacer with desired height (e.g., 30 pixels)
spacer = Div(height=30)

# Main visualization section
visualization_section = column(
    unity_container,  # Unity iframe moved to the top
    select_file,
    window_slider,
    p,
    scatter_row,     # Add scatter plots back
    # correlation_layout,
    visualization_section_20F,
    spacer,          # Insert spacer here
    p_violin,
    controls_section,
    controls_layout,
    data_table,
    sizing_mode='stretch_width',
    css_classes=['visualization-section']
)

# Main layout assembly
main_layout = column(
    custom_styles,
    header,
    visualization_section,
    sizing_mode='stretch_width'
)

# Set up document
curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration"