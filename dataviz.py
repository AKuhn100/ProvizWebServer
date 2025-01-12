import os
import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, linregress

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, CheckboxGroup,  # Replaced MultiSelect with CheckboxGroup
    DataTable, TableColumn, NumberFormatter, Div, HoverTool, GlyphRenderer, Slider
)
from bokeh.plotting import figure
from bokeh.layouts import column, row, layout
from bokeh.palettes import Category10

###############################################################################
# 1) Configuration
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


###############################################################################
# 3) Load and Aggregate Data from Local Directory
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

# Melt data for plotting Spearman Rho vs Average B-Factor
data_long_avg = data_proviz.melt(
    id_vars=['Protein', 'Avg_B_Factor'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)

# Melt data for plotting Spearman Rho vs Std Dev of B-Factor
data_long_std = data_proviz.melt(
    id_vars=['Protein', 'Std_B_Factor'],
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
    x_axis_label="B-Factor",
    y_axis_label="Spearman Correlation Between Frustration and B-Factor",  # Updated y-axis label
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
    x_axis_label="B-Factor",
    y_axis_label="Spearman Correlation Between Frustration and B-Factor",  # Updated y-axis label
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
    x_axis_label="B-Factor",
    y_axis_label="Spearman Correlation Between Frustration and B-Factor",  # Updated y-axis label
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"],
    active_drag="box_zoom",
    active_scroll=None  # Disable wheel zoom by default
)

source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[]))
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[]))
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[]))

# Create Div elements for regression info with empty text and hidden by default
regression_info_exp = Div(
    text="", 
    visible=False,  # Initially hidden
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
    visible=False,  # Initially hidden
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
    visible=False,  # Initially hidden
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

p_scatter_exp.scatter("x", "y", source=source_scatter_exp, color=Category10[10][1], alpha=0.7)
p_scatter_af.scatter("x", "y", source=source_scatter_af,  color=Category10[10][2], alpha=0.7)
p_scatter_evol.scatter("x", "y", source=source_scatter_evol, color=Category10[10][3], alpha=0.7)

def add_regression_line_and_info(fig, xvals, yvals, color="black", info_div=None):
    """Adds a linear regression line and updates the regression info Div."""
    if len(xvals) < 2 or np.all(xvals == xvals[0]):
        if info_div:
            info_div.text = "Insufficient data for regression"
            info_div.visible = True
        return
    
    not_nan = ~np.isnan(xvals) & ~np.isnan(yvals)
    if not any(not_nan):
        if info_div:
            info_div.text = "No valid data points"
            info_div.visible = True
        return
    
    xvals_clean = xvals[not_nan]
    yvals_clean = yvals[not_nan]
    if len(xvals_clean) < 2:
        if info_div:
            info_div.text = "Insufficient data for regression"
            info_div.visible = True
        return
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(xvals_clean, yvals_clean)
    
    # Plot regression line visibly
    x_range = np.linspace(xvals_clean.min(), xvals_clean.max(), 100)
    y_range = slope * x_range + intercept
    fig.line(x_range, y_range, line_width=2, line_dash='dashed', color=color, name='regression_line')
    
    # Create a separate data source for regression line hover
    regression_source = ColumnDataSource(data=dict(
        x=x_range,
        y=y_range,
        equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
    ))
    
    # Plot regression line again with this data source, invisible (for hover)
    invisible_regression = fig.line('x', 'y', source=regression_source, line_width=10, alpha=0, name='regression_hover')  # Increased line_width for better hover area
    
    # Add a separate HoverTool for the regression line
    hover_regression = HoverTool(
        renderers=[invisible_regression],
        tooltips=[
            ("Regression Equation", "@equation")
        ],
        mode='mouse'
    )
    fig.add_tools(hover_regression)
    
    # Update regression info div with equation and make it visible
    if info_div:
        info_div.text = f"""
        <div style='color: {color}'>
            <strong>y = {slope:.3f}x + {intercept:.3f}</strong><br>
            <span style='font-size: 12px'>R² = {r_value**2:.3f}</span>
        </div>
        """
        info_div.visible = True

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
        
        # Hide regression info Divs
        regression_info_exp.text = ""
        regression_info_exp.visible = False
        regression_info_af.text = ""
        regression_info_af.visible = False
        regression_info_evol.text = ""
        regression_info_evol.visible = False
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
    
    # For each scatter figure, remove old regression lines by filtering out renderers named 'regression_hover'
    p_scatter_exp.renderers = [r for r in p_scatter_exp.renderers if getattr(r, 'name', '') != 'regression_hover']
    p_scatter_af.renderers = [r for r in p_scatter_af.renderers if getattr(r, 'name', '') != 'regression_hover']
    p_scatter_evol.renderers = [r for r in p_scatter_evol.renderers if getattr(r, 'name', '') != 'regression_hover']
    
    # Reset data sources
    source_scatter_exp.data = dict(x=[], y=[])
    source_scatter_af.data = dict(x=[], y=[])
    source_scatter_evol.data = dict(x=[], y=[])
    
    # Hide regression info Divs initially
    regression_info_exp.text = ""
    regression_info_exp.visible = False
    regression_info_af.text = ""
    regression_info_af.visible = False
    regression_info_evol.text = ""
    regression_info_evol.visible = False
    
    if sub_orig.empty:
        p_scatter_exp.title.text = f"{filename} (No Data)"
        p_scatter_af.title.text = f"{filename} (No Data)"
        p_scatter_evol.title.text = f"{filename} (No Data)"
    else:
        # ExpFrust
        x_exp = sub_orig["B_Factor"].values
        y_exp = sub_orig["ExpFrust"].values
        source_scatter_exp.data = dict(x=x_exp, y=y_exp)
        p_scatter_exp.title.text = f"{filename} Experimental Frustration"
        add_regression_line_and_info(p_scatter_exp, x_exp, y_exp, color=Category10[10][1], info_div=regression_info_exp)
        
        # AFFrust
        x_af = sub_orig["B_Factor"].values
        y_af = sub_orig["AFFrust"].values
        source_scatter_af.data = dict(x=x_af, y=y_af)
        p_scatter_af.title.text = f"{filename} AF Frustration"
        add_regression_line_and_info(p_scatter_af, x_af, y_af, color=Category10[10][2], info_div=regression_info_af)
        
        # EvolFrust
        x_evol = sub_orig["B_Factor"].values
        y_evol = sub_orig["EvolFrust"].values
        source_scatter_evol.data = dict(x=x_evol, y=y_evol)
        p_scatter_evol.title.text = f"{filename} Evolutionary Frustration"
        add_regression_line_and_info(p_scatter_evol, x_evol, y_evol, color=Category10[10][3], info_div=regression_info_evol)

select_file.on_change("value", update_plot)
if initial_file:
    update_plot(None, None, initial_file)


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

# (E) FILTERS for correlation table using CheckboxGroup

# Prepare options
tests_in_corr = sorted(df_all_corr["Test"].unique()) if not df_all_corr.empty else []
if not df_all_corr.empty:
    combo_options = sorted({
        f"{row['MetricA']} vs {row['MetricB']}" 
        for _, row in df_all_corr.iterrows()
    })
else:
    combo_options = []

# Define CheckboxGroup for Tests
checkbox_tests = CheckboxGroup(
    labels=tests_in_corr,
    active=[],  # List of indices of active checkboxes
    width=300
)

# Define CheckboxGroup for Metric Pairs
checkbox_combos = CheckboxGroup(
    labels=combo_options,
    active=[],  # List of indices of active checkboxes
    width=300
)

# Function to map active indices to selected labels
def get_selected_labels(checkbox_group):
    return [checkbox_group.labels[i] for i in checkbox_group.active]

def update_corr_filter_checkbox(attr, old, new):
    """Filter correlation table based on selected tests and metric pairs using CheckboxGroup."""
    if df_all_corr.empty:
        return
    selected_tests = get_selected_labels(checkbox_tests)
    selected_combos = get_selected_labels(checkbox_combos)
    
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

# Attach callbacks
checkbox_tests.on_change("active", update_corr_filter_checkbox)
checkbox_combos.on_change("active", update_corr_filter_checkbox)

# Add header for filters
filters_header = Div(text="<b>Filter Correlation Table</b>", styles={'font-size': '16px', 'margin': '10px 0'})

# Create a scrollable Div for "Select Tests"
tests_scroll = Div(
    children=[checkbox_tests],
    styles={'height': '150px', 'overflow-y': 'auto', 'border': '1px solid #ddd', 'padding': '5px', 'border-radius': '4px'}
)

# Create a scrollable Div for "Select Metric Pairs"
combos_scroll = Div(
    children=[checkbox_combos],
    styles={'height': '150px', 'overflow-y': 'auto', 'border': '1px solid #ddd', 'padding': '5px', 'border-radius': '4px'}
)

# Arrange CheckboxGroups in a column
controls_section_layout = column(
    filters_header,
    Div(text="<b>Select Tests:</b>", styles={'margin-top': '10px'}),
    tests_scroll,
    Div(text="<b>Select Metric Pairs:</b>", styles={'margin-top': '20px'}),
    combos_scroll,
    sizing_mode='stretch_width'
)


###############################################################################
# 6) Additional Aggregated Plots (Converted from Plotly to Bokeh)
###############################################################################

# (F) Spearman Rho vs Average B-Factor
source_avg = ColumnDataSource(data_long_avg)

p_avg = figure(
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
frust_types = data_long_avg['Frust_Type'].unique().tolist()
palette = Category10[max(3, len(frust_types))]  # Ensure enough colors
color_map_frust = {frust: palette[i] for i, frust in enumerate(frust_types)}

# Add HoverTool
hover_avg = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Type", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    mode='mouse'
)
p_avg.add_tools(hover_avg)

# Add scatter glyphs
for frust in frust_types:
    subset = data_long_avg[data_long_avg['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    p_avg.scatter(
        'Avg_B_Factor', 'Spearman_Rho',
        source=source_subset,
        color=color_map_frust[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1
    )
    
    # Add regression lines with hover
    if len(subset) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(subset['Avg_B_Factor'], subset['Spearman_Rho'])
        x_range = np.linspace(subset['Avg_B_Factor'].min(), subset['Avg_B_Factor'].max(), 100)
        y_range = slope * x_range + intercept
        p_avg.line(x_range, y_range, color=color_map_frust[frust], line_dash='dashed')
        
        # Add regression equation hover
        regression_source = ColumnDataSource(data=dict(
            x=x_range,
            y=y_range,
            equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
        ))
        invisible_regression = p_avg.line('x', 'y', source=regression_source, line_width=10, alpha=0, name=f'regression_hover_{frust}')
        
        hover_regression = HoverTool(
            renderers=[invisible_regression],
            tooltips=[
                ("Regression Equation", "@equation")
            ],
            mode='mouse'
        )
        p_avg.add_tools(hover_regression)

p_avg.legend.location = "top_left"
p_avg.legend.title = "Frustration Type"
p_avg.legend.click_policy = "mute"

# (G) Spearman Rho vs Std Dev of B-Factor
source_std = ColumnDataSource(data_long_std)

p_std = figure(
    title="Spearman Correlation vs Std Dev of B-Factor",
    x_axis_label="Standard Deviation of B-Factor",
    y_axis_label="Spearman Correlation Between Frustration and B-Factor",
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None
)

# Add HoverTool
hover_std = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Type", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    mode='mouse'
)
p_std.add_tools(hover_std)

# Add scatter glyphs
for frust in frust_types:
    subset = data_long_std[data_long_std['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    p_std.scatter(
        'Std_B_Factor', 'Spearman_Rho',
        source=source_subset,
        color=color_map_frust[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1
    )
    
    # Add regression lines with hover
    if len(subset) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(subset['Std_B_Factor'], subset['Spearman_Rho'])
        x_range = np.linspace(subset['Std_B_Factor'].min(), subset['Std_B_Factor'].max(), 100)
        y_range = slope * x_range + intercept
        p_std.line(x_range, y_range, color=color_map_frust[frust], line_dash='dashed')
        
        # Add regression equation hover
        regression_source = ColumnDataSource(data=dict(
            x=x_range,
            y=y_range,
            equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
        ))
        invisible_regression = p_std.line('x', 'y', source=regression_source, line_width=10, alpha=0, name=f'regression_hover_{frust}')
        
        hover_regression = HoverTool(
            renderers=[invisible_regression],
            tooltips=[
                ("Regression Equation", "@equation")
            ],
            mode='mouse'
        )
        p_std.add_tools(hover_regression)

p_std.legend.location = "top_left"
p_std.legend.title = "Frustration Type"
p_std.legend.click_policy = "mute"

# (H) Spearman Rho per Protein and Frustration Metric
# Melt data_proviz for the third plot
data_long_corr = data_proviz.melt(
    id_vars=['Protein'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)

# Clean Frust_Type names
data_long_corr['Frust_Type'] = data_long_corr['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

# Remove rows with NaN correlations
data_long_corr.dropna(subset=['Spearman_Rho'], inplace=True)

source_corr_plot = ColumnDataSource(data_long_corr)

p_corr = figure(
    title="Spearman Correlation per Protein and Frustration Metric",
    x_axis_label="Protein",
    y_axis_label="Spearman Correlation Between Frustration and B-Factor",
    x_range=data_proviz['Protein'].tolist(),
    sizing_mode='stretch_width',
    height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None,
    toolbar_location="above"
)

# Define color palette for Frustration Types
frust_types_corr = data_long_corr['Frust_Type'].unique().tolist()
palette_corr = Category10[max(3, len(frust_types_corr))]  # Ensure enough colors
color_map_corr = {frust: palette_corr[i] for i, frust in enumerate(frust_types_corr)}

# Add HoverTool
hover_corr = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Metric", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    mode='mouse'
)
p_corr.add_tools(hover_corr)

# Add horizontal line at y=0
p_corr.line(x=[-0.5, len(data_proviz['Protein']) - 0.5], y=[0, 0], line_width=1, line_dash='dashed', color='gray', name='y_zero_line')

# Add scatter glyphs
for frust in frust_types_corr:
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    p_corr.scatter(
        'Protein', 'Spearman_Rho',
        source=source_subset,
        color=color_map_corr[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1
    )
    
    # Add regression lines with hover
    if len(subset) >= 2:
        # Assuming you want to regress Spearman_Rho against another variable, but currently it's Spearman_Rho vs itself
        # This is likely a mistake. For demonstration, let's assume you want to regress Spearman_Rho against Protein index
        # First, convert Protein to numerical indices
        subset_sorted = subset.sort_values('Protein')
        x_vals = np.arange(len(subset_sorted))
        y_vals = subset_sorted['Spearman_Rho'].values
        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_range = slope * x_range + intercept
        p_corr.line(x_range, y_range, color=color_map_corr[frust], line_dash='dashed')
        
        # Add regression equation hover
        regression_source = ColumnDataSource(data=dict(
            x=x_range,
            y=y_range,
            equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
        ))
        invisible_regression = p_corr.line('x', 'y', source=regression_source, line_width=10, alpha=0, name=f'regression_hover_{frust}')
        
        hover_regression = HoverTool(
            renderers=[invisible_regression],
            tooltips=[
                ("Regression Equation", "@equation")
            ],
            mode='mouse'
        )
        p_corr.add_tools(hover_regression)

p_corr.legend.location = "top_left"
p_corr.legend.title = "Frustration Type"
p_corr.legend.click_policy = "mute"

# Rotate x-axis labels to prevent overlapping
from math import pi
p_corr.xaxis.major_label_orientation = pi / 4  # 45 degrees


###############################################################################
# 7) User Interface Components
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

# Controls section
controls_section = Div(text="<b>Filter Correlation Table</b>", styles={'font-size': '16px', 'margin': '10px 0'})

# Arrange CheckboxGroups in a column (already handled in controls_section_layout)

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

# (F) Layout for Additional Plots
additional_plots = column(
    p_avg,
    p_std,
    p_corr,
    sizing_mode='stretch_width',
    spacing=20,
    name="additional_plots"
)

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

# Main layout with slider and additional plots
visualization_section = column(
    select_file,
    window_slider,
    p,
    scatter_row,
    unity_container,
    additional_plots,  # Integrated Additional Plots
    sizing_mode='stretch_width',
    css_classes=['visualization-section']
)

# Main layout assembly
main_layout = column(
    custom_styles,
    header,
    visualization_section,
    controls_section_layout,  # Updated to include CheckboxGroups
    data_table,
    sizing_mode='stretch_width'
)

# Set up document
curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration"