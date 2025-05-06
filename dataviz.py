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

###############################################################################
# 1) Configuration
###############################################################################
# Local data directory path
DATA_DIR = "summary_data"  # Directory containing the summary files

FILE_PATTERN = r"^summary_[A-Za-z0-9]{4}\.txt$"   # only summary_XXXX.txt, 4-char alphanumeric
+
# If you don’t want a hard-coded default, leave this blank so
# the first match in the directory is used automatically:
DEFAULT_FILE = ""

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

# Melt data for plotting
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
    x_axis_label="Normalized B-Factor",
    y_axis_label="Normalized Spearman Correlation Between Frustration and B-Factor",
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
    y_axis_label="Normalized Spearman Correlation Between Frustration and B-Factor",
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
    y_axis_label="Normalized Spearman Correlation Between Frustration and B-Factor",
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
        name=invisible_regression_name  # Unique name
    )

    # Add a separate HoverTool for the regression line
    hover_regression = HoverTool(
        renderers=[regression_line],
        tooltips=[
            ("Regression Equation", "@equation")
        ],
        mode='mouse'
    )
    fig.add_tools(hover_regression)

    # Update regression info div with equation
    if info_div:
        info_div.text = f"""
        <div style='color: {color}'>
            <strong>y = {slope:.3f}x + {intercept:.3f}</strong><br>
            <span style='font-size: 12px'>R² = {r_value**2:.3f}</span>
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
        return np.zeros_like(arr)  # If all values are the same, return zeros

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

    # **Remove all existing regression renderers**
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
        # ExpFrust
        x_exp_orig = sub_orig["B_Factor"].values
        y_exp_orig = sub_orig["ExpFrust"].values
        x_exp_norm = min_max_normalize(x_exp_orig)
        y_exp_norm = min_max_normalize(y_exp_orig)
        source_scatter_exp.data = dict(x=x_exp_norm, y=y_exp_norm, x_orig=x_exp_orig, y_orig=y_exp_orig)
        p_scatter_exp.title.text = f"{filename} Experimental Frustration"
        add_regression_line_and_info(
            fig=p_scatter_exp, 
            xvals=x_exp_norm,  # Use normalized data
            yvals=y_exp_norm, 
            color=Category10[10][1], 
            info_div=regression_info_exp,
            plot_type="exp"
        )

        # AFFrust
        x_af_orig = sub_orig["B_Factor"].values
        y_af_orig = sub_orig["AFFrust"].values
        x_af_norm = min_max_normalize(x_af_orig)
        y_af_norm = min_max_normalize(y_af_orig)
        source_scatter_af.data = dict(x=x_af_norm, y=y_af_norm, x_orig=x_af_orig, y_orig=y_af_orig)
        p_scatter_af.title.text = f"{filename} AF Frustration"
        add_regression_line_and_info(
            fig=p_scatter_af, 
            xvals=x_af_norm, 
            yvals=y_af_norm, 
            color=Category10[10][2], 
            info_div=regression_info_af,
            plot_type="af"
        )

        # EvolFrust
        x_evol_orig = sub_orig["B_Factor"].values
        y_evol_orig = sub_orig["EvolFrust"].values
        x_evol_norm = min_max_normalize(x_evol_orig)
        y_evol_norm = min_max_normalize(y_evol_orig)
        source_scatter_evol.data = dict(x=x_evol_norm, y=y_evol_norm, x_orig=x_evol_orig, y_orig=y_evol_orig)
        p_scatter_evol.title.text = f"{filename} Evolutionary Frustration"
        add_regression_line_and_info(
            fig=p_scatter_evol, 
            xvals=x_evol_norm, 
            yvals=y_evol_norm, 
            color=Category10[10][3], 
            info_div=regression_info_evol,
            plot_type="evol"
        )

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

# (E) FILTERS for correlation table

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
    if df_all_corr.empty:
        return
    
    # Aggregate selected tests and metric pairs from all CheckboxGroups
    selected_tests = get_selected_labels(checkbox_tests_columns)
    selected_combos = get_selected_labels(checkbox_combos_columns)

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
palette_avg = Category10[max(3, len(frust_types_avg))]  # Ensure enough colors
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
        name=f'scatter_{frust}'  # Add name to the renderer
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
            name=f'regression_line_{frust}'  # Unique name based on frust type
        )

        # Add HoverTool only to the regression_line
        hover_regression = HoverTool(
            renderers=[regression_line],
            tooltips=[
                ("Regression Equation", "@equation")
            ],
            mode='mouse'
        )
        p_avg_plot.add_tools(hover_regression)

# Create and add the standard HoverTool only for the scatter renderers
hover_scatter_avg = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Type", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    renderers=scatter_renderers_avg,  # Only attach to scatter renderers
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
palette_std = Category10[max(3, len(frust_types_std))]  # Ensure enough colors
color_map_frust_std = {frust: palette_std[i] for i, frust in enumerate(frust_types_std)}

# Create a list to hold scatter renderers
scatter_renderers_std = []

# Add scatter glyphs with named renderers and collect renderers
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
        name=f'scatter_{frust}'  # Add name to the renderer
    )
    scatter_renderers_std.append(scatter)

    # Add regression lines with hover
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
            name=f'regression_line_{frust}'  # Unique name based on frust type
        )

        # Add HoverTool only to the regression_line
        hover_regression = HoverTool(
            renderers=[regression_line],
            tooltips=[
                ("Regression Equation", "@equation")
            ],
            mode='mouse'
        )
        p_std_plot.add_tools(hover_regression)

# Create and add the standard HoverTool only for the scatter renderers
hover_scatter_std = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Type", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    renderers=scatter_renderers_std,  # Only attach to scatter renderers
    mode='mouse'
)
p_std_plot.add_tools(hover_scatter_std)

p_std_plot.legend.location = "top_left"
p_std_plot.legend.title = "Frustration Type"
p_std_plot.legend.click_policy = "mute"

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

p_corr_plot = figure(
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
p_corr_plot.add_tools(hover_corr)

# Add horizontal line at y=0
p_corr_plot.line(
    x=[-0.5, len(data_proviz['Protein']) - 0.5], 
    y=[0, 0], 
    line_width=1, 
    line_dash='dashed', 
    color='gray', 
    name='y_zero_line'
)

# Add scatter glyphs
for frust in frust_types_corr:
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    source_subset = ColumnDataSource(subset)
    p_corr_plot.scatter(
        'Protein', 'Spearman_Rho',
        source=source_subset,
        color=color_map_corr[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1
    )

# Add mean lines for each frustration type
for frust in frust_types_corr:
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    mean_value = subset['Spearman_Rho'].mean()

    # Create source for the mean line with hover information
    mean_source = ColumnDataSource(data=dict(
        x=[-0.5, len(data_proviz['Protein']) - 0.5],
        y=[mean_value, mean_value],
        mean_value=[f"{mean_value:.3f}"] * 2,
        frust_type=[frust] * 2
    ))

    # Add mean line with hover
    mean_line = p_corr_plot.line(
        'x', 'y',
        source=mean_source,
        color=color_map_corr[frust],
        line_dash='dashed',
        name=f'mean_line_{frust}'  # Unique name based on frust type
    )

    # Add hover tool for mean line
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

# Arrange CheckboxGroup widgets in the controls_layout already defined above
# Removed the old MultiSelect widgets and replaced with CheckboxGroups arranged in columns

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

# (G) NEW: Bar Plot with Mean, SD, and without T-Test Results
def create_bar_plot_with_sd(data_proviz):
    """
    Creates a bar chart displaying the mean Spearman correlation for each frustration metric,
    with error bars representing the standard deviation.
    Adjusts the y-axis range to ensure whiskers are fully visible.
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

    # Assign colors based on Metric using the predefined FRUSTRATION_COLORS dictionary
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

    # Add vertical bars and capture the renderer
    vbar_renderer = p_bar.vbar(
        x='Metric',
        top='Mean_Spearman_Rho',
        width=0.6,
        source=source_bar,
        color='Color',  # Reference the 'Color' column in the data source
        legend_label="Frustration Metric",
        line_color="black"
    )

    # Add error bars using Whisker
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

    # **New Addition**: Adjust y-axis range to include padding
    # Determine the minimum and maximum values for the y-axis
    min_lower = source_bar.data['lower'].min()
    max_upper = source_bar.data['upper'].max()

    # Calculate padding (10% of the range)
    y_padding = (max_upper - min_lower) * 0.1 if (max_upper - min_lower) != 0 else 1

    # Set the y_range with padding
    p_bar.y_range = Range1d(start=min_lower - y_padding, end=max_upper + y_padding)

    # Add horizontal line at y=0 for reference
    p_bar.line(x=[-0.5, len(stats_corrs) - 0.5], y=[0, 0], line_width=1, line_dash='dashed', color='gray')

    # Customize hover tool and correctly reference the vbar renderer
    hover_bar = HoverTool(
        tooltips=[
            ("Metric", "@Metric"),
            ("Mean Spearman Rho", "@Mean_Spearman_Rho{0.3f}"),
            ("Std Dev", "@Std_Spearman_Rho{0.3f}")
        ],
        renderers=[vbar_renderer],  # Correctly pass the renderer
        mode='mouse'
    )
    p_bar.add_tools(hover_bar)

    # Remove legend as it's redundant with colors
    p_bar.legend.visible = False

    return p_bar

# (F) Layout for Additional Plots
additional_plots = column(
    p_avg_plot,
    p_std_plot,
    p_corr_plot,
    create_bar_plot_with_sd(data_proviz),  # Integrated Bar Plot without T-Tests
    sizing_mode='stretch_width',
    spacing=20,
    name="additional_plots"
)

# (G) Scatter Plots Layout
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

# (I) Main layout with slider and additional plots
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
    controls_section,
    controls_layout,  # Updated controls layout with CheckboxGroups
    data_table,
    sizing_mode='stretch_width'
)

# Set up document
curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration"