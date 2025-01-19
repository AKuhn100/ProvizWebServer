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
DATA_DIR = "summary_data"  # Directory containing the summary files

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
# - Main line plot setup
# - Scatter plot configurations
# - Common visualization elements (hover tools, legends, etc.)
#
# Dependencies: Sections 1-2
# Required before: Callbacks and layout sections
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

# (B) Scatter Plots (Experimental, AF, Evolutionary Frustration)
# Scatter plots configuration with disabled wheel zoom by default
p_scatter_exp = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="Normalized B-Factor",
    y_axis_label="Normalized Experimental Frustration",
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"],
    active_drag="box_zoom",
    active_scroll=None  # Disable wheel zoom by default
)
p_scatter_af = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="Normalized B-Factor",
    y_axis_label="Normalized AlphaFold Frustration",
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"],
    active_drag="box_zoom",
    active_scroll=None  # Disable wheel zoom by default
)
p_scatter_evol = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="Normalized B-Factor",
    y_axis_label="Normalized Evolutionary Frustration",
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"],
    active_drag="box_zoom",
    active_scroll=None  # Disable wheel zoom by default
)

# ColumnDataSources will now include normalized data
source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[]))
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[]))
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[]))

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

# Initial scatter glyphs (empty)
p_scatter_exp.scatter("x", "y", source=source_scatter_exp, color=Category10[10][1], alpha=0.7)
p_scatter_af.scatter("x", "y", source=source_scatter_af,  color=Category10[10][2], alpha=0.7)
p_scatter_evol.scatter("x", "y", source=source_scatter_evol, color=Category10[10][3], alpha=0.7)

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


def add_regression_line_and_info(fig, xvals, yvals, color="black", info_div=None, plot_type=""):
    """
    Adds a linear regression line and updates the regression info Div.
    The plot_type parameter helps in uniquely naming the regression renderers.
    """
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
    slope, intercept, r_value, p_value, std_err = linregress(xvals_clean, yvals_clean)

    # Plot regression line visibly
    x_range = np.linspace(xvals_clean.min(), xvals_clean.max(), 100)
    y_range = slope * x_range + intercept
    regression_line_name = f'regression_line_{plot_type}'
    regression_line = fig.line(
        x_range, y_range, 
        line_width=2, line_dash='dashed', color=color, 
        name=regression_line_name
    )

    # Create a separate data source for regression line hover
    regression_source = ColumnDataSource(data=dict(
        x=x_range,
        y=y_range,
        equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
    ))

    # Plot regression line again with this data source, invisible (for hover)
    invisible_regression_name = f'regression_hover_{plot_type}'
    invisible_regression = fig.line(
        'x', 'y', 
        source=regression_source, 
        line_width=10, 
        alpha=0, 
        name=invisible_regression_name
    )

    # Update regression info div with equation
    if info_div:
        info_div.text = f"""
        <div style='color: {color}'>
            <strong>y = {slope:.3f}x + {intercept:.3f}</strong><br>
            <span style='font-size: 12px'>R² = {r_value**2:.3f}</span>
        </div>
        """


def update_plot(attr, old, new):
    """
    Updates both the main plot and scatter plots when a new file is selected.
    """
    filename = select_file.value
    if filename not in data_by_file:
        source_plot.data = dict(x=[], residue=[], b_factor=[], exp_frust=[], af_frust=[], evol_frust=[])
        source_scatter_exp.data = dict(x=[], y=[], x_orig=[], y_orig=[])
        source_scatter_af.data = dict(x=[], y=[], x_orig=[], y_orig=[])
        source_scatter_evol.data = dict(x=[], y=[], x_orig=[], y_orig=[])
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
        df_plot[col] = min_max_normalize(arr)

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

    # Reset regression renderers and data sources
    remove_regression_renderers(p_scatter_exp)
    remove_regression_renderers(p_scatter_af)
    remove_regression_renderers(p_scatter_evol)

    # Reset data sources
    source_scatter_exp.data = dict(x=[], y=[], x_orig=[], y_orig=[])
    source_scatter_af.data = dict(x=[], y=[], x_orig=[], y_orig=[])
    source_scatter_evol.data = dict(x=[], y=[], x_orig=[], y_orig=[])

    regression_info_exp.text = ""
    regression_info_af.text = ""
    regression_info_evol.text = ""

    if sub_orig.empty:
        p_scatter_exp.title.text = f"{filename} (No Data)"
        p_scatter_af.title.text = f"{filename} (No Data)"
        p_scatter_evol.title.text = f"{filename} (No Data)"
    else:
        # Update scatter plots with normalized data
        for metric, source, plot, info, color in [
            ("ExpFrust", source_scatter_exp, p_scatter_exp, regression_info_exp, Category10[10][1]),
            ("AFFrust", source_scatter_af, p_scatter_af, regression_info_af, Category10[10][2]),
            ("EvolFrust", source_scatter_evol, p_scatter_evol, regression_info_evol, Category10[10][3])
        ]:
            x_orig = sub_orig["B_Factor"].values
            y_orig = sub_orig[metric].values
            x_norm = min_max_normalize(x_orig)
            y_norm = min_max_normalize(y_orig)
            
            source.data = dict(x=x_norm, y=y_norm, x_orig=x_orig, y_orig=y_orig)
            plot.title.text = f"{filename} {metric}"
            
            add_regression_line_and_info(
                fig=plot, 
                xvals=x_norm,
                yvals=y_norm, 
                color=color, 
                info_div=info,
                plot_type=metric.lower()
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
# - Additional statistical plots
# - Aggregate visualizations
# - Correlation plots
#
# Dependencies: Sections 1-3
# Required before: Layout assembly
###############################################################################

# Initialize data sources with empty data if necessary
if data_long_avg.empty:
    data_long_avg = pd.DataFrame(columns=['Protein', 'Avg_B_Factor', 'Frust_Type', 'Spearman_Rho'])
    data_long_std = pd.DataFrame(columns=['Protein', 'Std_B_Factor', 'Frust_Type', 'Spearman_Rho'])
    data_long_corr = pd.DataFrame(columns=['Test', 'MetricA', 'MetricB', 'Rho', 'Pval', 'Frust_Type', 'Protein'])

# Create data sources
source_avg_plot = ColumnDataSource(data_long_avg)
source_std_plot = ColumnDataSource(data_long_std)
source_corr_plot = ColumnDataSource(data_long_corr)

# Ensure protein_order is defined
if 'protein_order' not in globals():
    protein_order = []

p_avg_plot = figure(
    title="Spearman Correlation vs Average B-Factor",
    x_axis_label="Average B-Factor",
    y_axis_label="Spearman Correlation Between Frustration and B-Factor",
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None
)

# Define color palette for Frustration Types
frust_types_avg = data_long_avg['Frust_Type'].unique().tolist()
palette_avg = Category10[max(3, len(frust_types_avg))]
color_map_frust_avg = {frust: palette_avg[i] for i, frust in enumerate(frust_types_avg)}

# Create a list to hold scatter renderers
scatter_renderers_avg = []

# Add scatter glyphs with named renderers and collect renderers
for frust in frust_types_avg:
    subset = data_long_avg[data_long_avg['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    scatter = p_avg_plot.scatter(
        'Avg_B_Factor', 'Spearman_Rho',
        source=source_subset,
        color=color_map_frust_avg[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1,
        name=f'scatter_{frust}'
    )
    scatter_renderers_avg.append(scatter)

    # Add regression lines with hover
    if len(subset) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(subset['Avg_B_Factor'], subset['Spearman_Rho'])
        x_range = np.linspace(subset['Avg_B_Factor'].min(), subset['Avg_B_Factor'].max(), 100)
        y_range = slope * x_range + intercept

        regression_source = ColumnDataSource(data=dict(
            x=x_range,
            y=y_range,
            equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
        ))

        regression_line = p_avg_plot.line(
            'x', 'y', 
            source=regression_source, 
            color=color_map_frust_avg[frust], 
            line_dash='dashed',
            name=f'regression_line_{frust}'
        )

        hover_regression = HoverTool(
            renderers=[regression_line],
            tooltips=[
                ("Regression Equation", "@equation")
            ],
            mode='mouse'
        )
        p_avg_plot.add_tools(hover_regression)

# Create and add the standard HoverTool
hover_scatter_avg = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Type", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    renderers=scatter_renderers_avg,
    mode='mouse'
)
p_avg_plot.add_tools(hover_scatter_avg)

p_avg_plot.legend.location = "top_left"
p_avg_plot.legend.title = "Frustration Type"
p_avg_plot.legend.click_policy = "mute"

# (G) Spearman Rho vs Std Dev of B-Factor
source_std_plot = ColumnDataSource(data_long_std)

p_std_plot = figure(
    title="Spearman Correlation vs Std Dev of B-Factor",
    x_axis_label="Standard Deviation of B-Factor",
    y_axis_label="Spearman Correlation Between Frustration and B-Factor",
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None
)

# Define color palette for Frustration Types
frust_types_std = data_long_std['Frust_Type'].unique().tolist()
palette_std = Category10[max(3, len(frust_types_std))]
color_map_frust_std = {frust: palette_std[i] for i, frust in enumerate(frust_types_std)}

scatter_renderers_std = []

for frust in frust_types_std:
    subset = data_long_std[data_long_std['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    scatter = p_std_plot.scatter(
        'Std_B_Factor', 'Spearman_Rho',
        source=source_subset,
        color=color_map_frust_std[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1,
        name=f'scatter_{frust}'
    )
    scatter_renderers_std.append(scatter)

    if len(subset) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(subset['Std_B_Factor'], subset['Spearman_Rho'])
        x_range = np.linspace(subset['Std_B_Factor'].min(), subset['Std_B_Factor'].max(), 100)
        y_range = slope * x_range + intercept

        regression_source = ColumnDataSource(data=dict(
            x=x_range,
            y=y_range,
            equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
        ))

        regression_line = p_std_plot.line(
            'x', 'y', 
            source=regression_source, 
            color=color_map_frust_std[frust], 
            line_dash='dashed',
            name=f'regression_line_{frust}'
        )

        hover_regression = HoverTool(
            renderers=[regression_line],
            tooltips=[
                ("Regression Equation", "@equation")
            ],
            mode='mouse'
        )
        p_std_plot.add_tools(hover_regression)

hover_scatter_std = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Type", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    renderers=scatter_renderers_std,
    mode='mouse'
)
p_std_plot.add_tools(hover_scatter_std)

p_std_plot.legend.location = "top_left"
p_std_plot.legend.title = "Frustration Type"
p_std_plot.legend.click_policy = "mute"

# (H) Spearman Rho per Protein and Frustration Metric
p_corr_plot = figure(
    title="Spearman Correlation per Protein and Frustration Metric",
    x_axis_label="Protein (Ordered by EvolFrust-ExpFrust)",
    y_axis_label="Spearman Correlation Between Frustration and B-Factor",
    x_range=protein_order,
    sizing_mode='stretch_width',
    height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None,
    toolbar_location="above"
)

# Define color palette for Frustration Types
frust_types_corr = [ft for ft in data_long_corr['Frust_Type'].unique() if ft != ""]
palette_corr = Category10[max(3, len(frust_types_corr))]
color_map_corr = {frust: FRUSTRATION_COLORS.get(frust, Category10[10][i]) for i, frust in enumerate(frust_types_corr)}

# Add HoverTool
hover_corr = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Metric", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
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
print(f"Available frustration types: {frust_types_corr}")
print(f"Data shape: {data_long_corr.shape}")
print(f"Columns: {data_long_corr.columns}")

for frust in frust_types_corr:
    if frust != "":  # Skip empty Frust_Type
        subset = data_long_corr[data_long_corr['Frust_Type'] == frust].copy()
        print(f"Subset for {frust}: {len(subset)} rows")
        
        if not subset.empty and 'Protein' in subset.columns and 'Rho' in subset.columns:
            print(f"Processing {frust} with {len(subset)} rows")
            print(f"Sample data for {frust}:")
            print(subset[['Protein', 'Rho', 'Frust_Type']].head())
            
            # Ensure Protein is categorical with proper ordering
            subset['Protein'] = pd.Categorical(subset['Protein'], categories=protein_order, ordered=True)
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
    legend = Legend(items=legend_items, location="top_left", title="Frustration Type", click_policy="mute")
    p_corr_plot.add_layout(legend)

# Add mean lines for each frustration type
for frust in frust_types_corr:
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    if frust == "":
        continue
    mean_value = subset['Rho'].mean()

    mean_source = ColumnDataSource(data=dict(
        x=[-0.5, len(protein_order) - 0.5],
        y=[mean_value, mean_value],
        mean_value=[f"{mean_value:.3f}"] * 2,
        frust_type=[frust] * 2
    ))

    mean_line = p_corr_plot.line(
        'x', 'y', 
        source=mean_source, 
        color=color_map_corr[frust], 
        line_dash='dashed',
        name=f'mean_line_{frust}'
    )

    mean_hover = HoverTool(
        renderers=[mean_line],
        tooltips=[
            ("Frustration Type", "@frust_type"),
            ("Mean Correlation", "@mean_value")
        ],
        mode='mouse'
    )
    p_corr_plot.add_tools(mean_hover)

p_corr_plot.legend.location = "top_left"
p_corr_plot.legend.title = "Frustration Type"
p_corr_plot.legend.click_policy = "mute"

# Rotate x-axis labels to prevent overlapping
from math import pi
p_corr_plot.xaxis.major_label_orientation = pi / 4  # 45 degrees

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
# Dependencies: All previous sections (1-7) must be fully loaded
# IMPORTANT: All widgets (select_file, window_slider, etc.) must be defined before this section
# This should be the last section in your script
###############################################################################
def create_bar_plot_with_sd(data_proviz):
    """
    Creates a bar chart displaying the mean Spearman correlation for each frustration metric,
    with error bars representing the standard deviation.
    """
    # Compute mean and standard deviation of Spearman Rho per metric
    spearman_columns = ['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust']
    stats_corrs = data_proviz[spearman_columns].agg(['mean', 'std']).transpose().reset_index()
    stats_corrs.rename(columns={
        'index': 'Metric',
        'mean': 'Mean_Spearman_Rho',
        'std': 'Std_Spearman_Rho'
    }, inplace=True)

    # Clean Metric names
    stats_corrs['Metric'] = stats_corrs['Metric'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')
    stats_corrs['Color'] = stats_corrs['Metric'].map(FRUSTRATION_COLORS)

    # Create ColumnDataSource for the bar plot
    source_bar = ColumnDataSource(stats_corrs)

    # Create figure
    p_bar = figure(
        title="Mean Spearman Correlation between B-Factor and Frustration Metrics",
        x_axis_label="Frustration Metric",
        y_axis_label="Mean Spearman Rho",
        x_range=stats_corrs['Metric'].tolist(),
        sizing_mode='stretch_width',
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above"
    )

    # Add vertical bars
    vbar_renderer = p_bar.vbar(
        x='Metric',
        top='Mean_Spearman_Rho',
        width=0.6,
        source=source_bar,
        color='Color',
        legend_label="Frustration Metric",
        line_color="black"
    )

    # Add error bars
    whisker = Whisker(
        base='Metric',
        upper='upper',
        lower='lower',
        source=source_bar,
        level="overlay"
    )
    p_bar.add_layout(whisker)

    # Calculate upper and lower bounds for error bars
    source_bar.data['upper'] = source_bar.data['Mean_Spearman_Rho'] + source_bar.data['Std_Spearman_Rho']
    source_bar.data['lower'] = source_bar.data['Mean_Spearman_Rho'] - source_bar.data['Std_Spearman_Rho']

    # Adjust y-axis range
    min_lower = source_bar.data['lower'].min()
    max_upper = source_bar.data['upper'].max()
    y_padding = (max_upper - min_lower) * 0.1 if (max_upper - min_lower) != 0 else 1
    p_bar.y_range = Range1d(start=min_lower - y_padding, end=max_upper + y_padding)

    # Add horizontal reference line
    p_bar.line(
        x=[-0.5, len(stats_corrs) - 0.5], 
        y=[0, 0], 
        line_width=1, 
        line_dash='dashed', 
        color='gray'
    )

    # Add hover tool
    hover_bar = HoverTool(
        tooltips=[
            ("Metric", "@Metric"),
            ("Mean Spearman Rho", "@Mean_Spearman_Rho{0.3f}"),
            ("Std Dev", "@Std_Spearman_Rho{0.3f}")
        ],
        renderers=[vbar_renderer],
        mode='mouse'
    )
    p_bar.add_tools(hover_bar)
    p_bar.legend.visible = False

    return p_bar

# Layout components
additional_plots = column(
    p_avg_plot,
    p_std_plot,
    p_corr_plot,
    create_bar_plot_with_sd(data_proviz),
    sizing_mode='stretch_width',
    spacing=20,
    name="additional_plots"
)

# Scatter Plots Layout
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

# Scatter plots row
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

# Main visualization section
visualization_section = column(
    unity_container,  # Unity iframe moved to the top
    select_file,
    window_slider,
    p,
    scatter_row,
    additional_plots,
    sizing_mode='stretch_width',
    css_classes=['visualization-section']
)

# Main layout assembly
main_layout = column(
    custom_styles,
    header,
    visualization_section,
    controls_section,
    controls_layout,
    data_table,
    sizing_mode='stretch_width'
)

# Set up document
curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration"