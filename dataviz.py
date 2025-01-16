import os
import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, linregress

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, CheckboxGroup, Div, Spacer,
    DataTable, TableColumn, NumberFormatter, HoverTool, 
    GlyphRenderer, Slider, Whisker, Label, Range1d,
    RadioGroup
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from math import pi

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
    Parses a summary file and returns original and processed DataFrames along with 
    *B-Factor-based* correlations (stored in 'corrs'), for backward compatibility.
    If RMSF is present, it will be parsed but not included in the aggregated correlation 'corrs'.
    """
    required_cols = ["AlnIndex", "Residue", "B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]
    # RMSF is optional

    if not os.path.isfile(local_path):
        print(f"File not found: {local_path}")
        return None, None, {}

    try:
        df = pd.read_csv(local_path, sep='\t')
    except Exception as e:
        print(f"Skipping {local_path}: failed to parse data. Error: {e}")
        return None, None, {}

    # Ensure required columns exist
    if not set(required_cols).issubset(df.columns):
        print(f"Skipping {local_path}: missing required columns.")
        return None, None, {}

    # Replace 'n/a' with NaN and convert to float
    numeric_cols = ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]
    if "RMSF" in df.columns:
        numeric_cols.append("RMSF")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_original = df.copy()
    df_for_plot = df.copy()

    # Apply moving average to B_Factor, ExpFrust, AFFrust, EvolFrust (and RMSF if present)
    cols_for_moving_avg = ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]
    if "RMSF" in df_for_plot.columns:
        cols_for_moving_avg.append("RMSF")

    for col in cols_for_moving_avg:
        df_for_plot[col] = moving_average(df_for_plot[col].values, window_size=5)

    # Min-Max normalization for the same columns
    for col in cols_for_moving_avg:
        valid = ~df_for_plot[col].isna()
        if valid.any():
            col_min = df_for_plot.loc[valid, col].min()
            col_max = df_for_plot.loc[valid, col].max()
            if col_max > col_min:
                df_for_plot[col] = (df_for_plot[col] - col_min) / (col_max - col_min)

    # Compute Spearman correlations on original data *only for B-Factor*
    # (kept for the aggregated correlation table logic)
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
        if r is not None:
            name = getattr(r, 'name', '')
            if isinstance(name, str) and name.startswith('regression_'):
                continue
        new_renderers.append(r)
    fig.renderers = new_renderers

def has_rmsf(df):
    """Return True if a DataFrame has a valid RMSF column with at least some non-NaN data."""
    if "RMSF" not in df.columns:
        return False
    return not df["RMSF"].dropna().empty

###############################################################################
# 3) Load and Aggregate Data from Local Directory (B-Factor Only)
###############################################################################
data_by_file = {}
all_corr_rows = []

# Aggregation lists (only for B-Factor, as before)
protein_names = []
avg_bfactors = []
std_bfactors = []
spearman_exp = []
spearman_af = []
spearman_evol = []

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
        "corrs": corrs  # correlations specifically for B-Factor
    }

    # Collect correlation data (B-Factor only)
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo
        all_corr_rows.append([filename, mA, mB, rho, pval])

    # Aggregate data for additional B-Factor-based plots
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

# Correlation DataFrame (again, B-Factor-based)
df_all_corr = pd.DataFrame(all_corr_rows, columns=["Test","MetricA","MetricB","Rho","Pval"])

# Aggregated DataFrame for Additional Plots (B-Factor-based)
data_proviz = pd.DataFrame({
    'Protein': protein_names,
    'Avg_B_Factor': avg_bfactors,
    'Std_B_Factor': std_bfactors,
    'Spearman_ExpFrust': spearman_exp,
    'Spearman_AFFrust': spearman_af,
    'Spearman_EvolFrust': spearman_evol
})

# Melt data for plotting (Avg)
data_long_avg = data_proviz.melt(
    id_vars=['Protein', 'Avg_B_Factor'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)
data_long_avg['Frust_Type'] = data_long_avg['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')
data_long_avg.dropna(subset=['Spearman_Rho'], inplace=True)

# Melt data for plotting (Std)
data_long_std = data_proviz.melt(
    id_vars=['Protein', 'Std_B_Factor'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)
data_long_std['Frust_Type'] = data_long_std['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')
data_long_std.dropna(subset=['Spearman_Rho'], inplace=True)

###############################################################################
# 4) Bokeh Application Components
###############################################################################

# (A) Main Plot: Smoothed + Normalized Data
source_plot = ColumnDataSource(data=dict(
    x=[],
    residue=[],
    flex_value=[],  # Will hold either B-Factor or RMSF
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

# Define separate HoverTools 
hover_flex = HoverTool(
    renderers=[],
    tooltips=[
        ("Index", "@x"), 
        ("Residue", "@residue"), 
        ("Flexibility", "@flex_value")  # e.g. B-Factor or RMSF
    ],
    name="hover_flex"
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

p.add_tools(hover_flex, hover_ef, hover_af, hover_ev)
p.xaxis.axis_label = "Residue Index"
p.yaxis.axis_label = "Normalized Flexibility / Frustration"

# Add lines
color_map = {
    "flex_value": ("Flexibility", Category10[10][0]),
    "exp_frust":  ("ExpFrust",    Category10[10][1]),
    "af_frust":   ("AFFrust",     Category10[10][2]),
    "evol_frust": ("EvolFrust",   Category10[10][3])
}
renderers = {}
for col_key, (label, col) in color_map.items():
    renderer = p.line(
        x="x", y=col_key, source=source_plot,
        line_width=2, alpha=0.7, color=col,
        legend_label=label
    )
    renderers[col_key] = renderer
    if col_key == "flex_value":
        hover_flex.renderers.append(renderer)
    elif col_key == "exp_frust":
        hover_ef.renderers.append(renderer)
    elif col_key == "af_frust":
        hover_af.renderers.append(renderer)
    elif col_key == "evol_frust":
        hover_ev.renderers.append(renderer)

p.legend.location = "top_left"
p.legend.title = "Metrics"
p.legend.click_policy = "hide"

# (B) Scatter Plots
p_scatter_exp = figure(
    sizing_mode="stretch_both",
    aspect_ratio=1,
    min_width=350,
    min_height=350,
    title="",
    x_axis_label="Normalized Flexibility",
    y_axis_label="Normalized Spearman Correlation (Flex vs ExpFrust)",
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
    x_axis_label="Normalized Flexibility",
    y_axis_label="Normalized Spearman Correlation (Flex vs AFFrust)",
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
    x_axis_label="Normalized Flexibility",
    y_axis_label="Normalized Spearman Correlation (Flex vs EvolFrust)",
    tools=["pan", "box_zoom", "wheel_zoom", "reset","save"],
    active_drag="box_zoom",
    active_scroll=None
)

source_scatter_exp = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[]))
source_scatter_af = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[]))
source_scatter_evol = ColumnDataSource(data=dict(x=[], y=[], x_orig=[], y_orig=[]))

regression_info_exp = Div(text="", sizing_mode="stretch_width")
regression_info_af = Div(text="", sizing_mode="stretch_width")
regression_info_evol = Div(text="", sizing_mode="stretch_width")

p_scatter_exp.scatter("x", "y", source=source_scatter_exp, color=Category10[10][1], alpha=0.7)
p_scatter_af.scatter("x", "y", source=source_scatter_af,  color=Category10[10][2], alpha=0.7)
p_scatter_evol.scatter("x", "y", source=source_scatter_evol, color=Category10[10][3], alpha=0.7)

def add_regression_line_and_info(fig, xvals, yvals, color="black", info_div=None, plot_type=""):
    """
    Adds a linear regression line and updates the regression info Div.
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

    slope, intercept, r_value, p_value, std_err = linregress(xvals_clean, yvals_clean)
    x_range = np.linspace(xvals_clean.min(), xvals_clean.max(), 100)
    y_range = slope * x_range + intercept
    regression_line_name = f'regression_line_{plot_type}'
    regression_line = fig.line(
        x_range, y_range, 
        line_width=2, line_dash='dashed', color=color, 
        name=regression_line_name
    )

    # For hover
    regression_source = ColumnDataSource(data=dict(
        x=x_range,
        y=y_range,
        equation=[f"y = {slope:.3f}x + {intercept:.3f}"] * len(x_range)
    ))
    invisible_regression_name = f'regression_hover_{plot_type}'
    _ = fig.line(
        'x', 'y', 
        source=regression_source, 
        line_width=10, 
        alpha=0, 
        name=invisible_regression_name
    )

    hover_regression = HoverTool(
        renderers=[regression_line],
        tooltips=[
            ("Regression Equation", "@equation")
        ],
        mode='mouse'
    )
    fig.add_tools(hover_regression)

    if info_div:
        info_div.text = f"""
        <div style='color: {color}'>
            <strong>y = {slope:.3f}x + {intercept:.3f}</strong><br>
            <span style='font-size: 12px'>R² = {r_value**2:.3f}</span>
        </div>
        """

# (C) Global controls: Protein Selector + Window Slider + Flexibility Metric Radio
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

# RadioGroup to switch between B-Factor and RMSF
flex_radio = RadioGroup(
    labels=["Use B-Factor", "Use RMSF"],
    active=0,
    width=200
)

def current_flex_col(df):
    """
    Returns 'RMSF' if:
     - the radio button is set to RMSF, AND
     - the selected file has valid RMSF data
    Otherwise returns 'B_Factor'.
    """
    if flex_radio.active == 1 and has_rmsf(df):
        return "RMSF"
    return "B_Factor"

def min_max_normalize(arr):
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    else:
        return np.zeros_like(arr)

def update_plot(attr, old, new):
    """
    Updates both the main plot and scatter plots when:
      - a new file is selected,
      - the user changes the moving average window, or
      - the user changes the 'use RMSF' vs 'use B-Factor' radio button.
    """
    filename = select_file.value
    if filename not in data_by_file:
        source_plot.data = dict(x=[], residue=[], flex_value=[], exp_frust=[], af_frust=[], evol_frust=[])
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

    df_orig = data_by_file[filename]["df_original"].copy()
    window_size = window_slider.value

    # Determine which column we'll use for 'flex_value'
    flex_col = current_flex_col(df_orig)

    # Create a copy and apply moving average to all relevant columns
    df_plot = df_orig.copy()
    cols_for_mavg = [flex_col, "ExpFrust", "AFFrust", "EvolFrust"]
    for col in cols_for_mavg:
        if col in df_plot.columns:
            df_plot[col] = moving_average(df_plot[col].values, window_size=window_size)

    # Normalize the columns
    for col in cols_for_mavg:
        if col in df_plot.columns:
            arr = df_plot[col].values
            valid_mask = ~np.isnan(arr)
            if valid_mask.any():
                df_plot[col] = min_max_normalize(arr)

    # Drop rows that have no data for the chosen flex_col or frust metrics
    sub_plot = df_plot.dropna(subset=[flex_col, "ExpFrust", "AFFrust", "EvolFrust"])
    if sub_plot.empty:
        source_plot.data = dict(x=[], residue=[], flex_value=[], exp_frust=[], af_frust=[], evol_frust=[])
        p.title.text = f"{filename} (No valid rows)."
    else:
        new_data = dict(
            x=sub_plot["AlnIndex"].tolist(),
            residue=sub_plot["Residue"].tolist(),
            flex_value=sub_plot[flex_col].tolist(),
            exp_frust=sub_plot["ExpFrust"].tolist(),
            af_frust=sub_plot["AFFrust"].tolist(),
            evol_frust=sub_plot["EvolFrust"].tolist()
        )
        source_plot.data = new_data
        p.title.text = f"{filename} (Smoothed + Normalized) - Using {flex_col}"

    # ============ Scatter Plots (Compute correlation on the fly for the selected file) ============
    sub_orig = df_orig.dropna(subset=[flex_col, "ExpFrust", "AFFrust", "EvolFrust"])
    remove_regression_renderers(p_scatter_exp)
    remove_regression_renderers(p_scatter_af)
    remove_regression_renderers(p_scatter_evol)

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
        return

    # For each frustration column, we do min-max on flex_col and on that frustration
    # Then run linear regression, etc.
    # 1) ExpFrust
    x_exp_orig = sub_orig[flex_col].values
    y_exp_orig = sub_orig["ExpFrust"].values
    x_exp_norm = min_max_normalize(x_exp_orig)
    y_exp_norm = min_max_normalize(y_exp_orig)
    source_scatter_exp.data = dict(x=x_exp_norm, y=y_exp_norm, x_orig=x_exp_orig, y_orig=y_exp_orig)
    p_scatter_exp.title.text = f"{filename} - {flex_col} vs ExpFrust"
    add_regression_line_and_info(
        fig=p_scatter_exp, 
        xvals=x_exp_norm,  
        yvals=y_exp_norm, 
        color=Category10[10][1], 
        info_div=regression_info_exp,
        plot_type="exp"
    )

    # 2) AFFrust
    x_af_orig = sub_orig[flex_col].values
    y_af_orig = sub_orig["AFFrust"].values
    x_af_norm = min_max_normalize(x_af_orig)
    y_af_norm = min_max_normalize(y_af_orig)
    source_scatter_af.data = dict(x=x_af_norm, y=y_af_norm, x_orig=x_af_orig, y_orig=y_af_orig)
    p_scatter_af.title.text = f"{filename} - {flex_col} vs AFFrust"
    add_regression_line_and_info(
        fig=p_scatter_af, 
        xvals=x_af_norm, 
        yvals=y_af_norm, 
        color=Category10[10][2], 
        info_div=regression_info_af,
        plot_type="af"
    )

    # 3) EvolFrust
    x_evol_orig = sub_orig[flex_col].values
    y_evol_orig = sub_orig["EvolFrust"].values
    x_evol_norm = min_max_normalize(x_evol_orig)
    y_evol_norm = min_max_normalize(y_evol_orig)
    source_scatter_evol.data = dict(x=x_evol_norm, y=y_evol_norm, x_orig=x_evol_orig, y_orig=y_evol_orig)
    p_scatter_evol.title.text = f"{filename} - {flex_col} vs EvolFrust"
    add_regression_line_and_info(
        fig=p_scatter_evol, 
        xvals=x_evol_norm, 
        yvals=y_evol_norm, 
        color=Category10[10][3], 
        info_div=regression_info_evol,
        plot_type="evol"
    )

# Callbacks
select_file.on_change("value", update_plot)
window_slider.on_change('value', update_plot)
flex_radio.on_change('active', update_plot)

if initial_file:
    update_plot(None, None, initial_file)

###############################################################################
# 5) CORRELATION TABLE (B-Factor-based, aggregated) AND FILTERS
###############################################################################

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

# Simple checkbox-based filters for correlation table
def split_labels(labels, num_columns):
    if num_columns <= 0:
        return [labels]
    k, m = divmod(len(labels), num_columns)
    return [labels[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_columns)]

NUM_COLUMNS = 3
tests_in_corr = sorted(df_all_corr["Test"].unique()) if not df_all_corr.empty else []
if not df_all_corr.empty:
    combo_options = sorted({
        f"{row['MetricA']} vs {row['MetricB']}" 
        for _, row in df_all_corr.iterrows()
    })
else:
    combo_options = []

def get_selected_labels(checkbox_columns):
    selected = []
    for checkbox in checkbox_columns:
        selected.extend([checkbox.labels[i] for i in checkbox.active])
    return selected

def update_corr_filter(attr, old, new):
    """Filter correlation table based on selected tests and metric pairs (B-Factor only)."""
    if df_all_corr.empty:
        return
    
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

if not df_all_corr.empty:
    test_labels_split = split_labels(tests_in_corr, NUM_COLUMNS)
    combo_labels_split = split_labels(combo_options, NUM_COLUMNS)
    
    checkbox_tests_columns = [
        CheckboxGroup(labels=col_labels, active=[], name=f'tests_column_{i+1}')
        for i, col_labels in enumerate(test_labels_split)
    ]
    checkbox_combos_columns = [
        CheckboxGroup(labels=col_labels, active=[], name=f'combos_column_{i+1}')
        for i, col_labels in enumerate(combo_labels_split)
    ]
else:
    checkbox_tests_columns = [CheckboxGroup(labels=[], active=[], name='tests_column_1')]
    checkbox_combos_columns = [CheckboxGroup(labels=[], active=[], name='combos_column_1')]

for checkbox in checkbox_tests_columns + checkbox_combos_columns:
    checkbox.on_change('active', update_corr_filter)

tests_title = Div(text="<b>Select Tests:</b>", styles={'font-size': '14px', 'margin-bottom': '5px'})
combos_title = Div(text="<b>Select Metric Pairs:</b>", styles={'font-size': '14px', 'margin-bottom': '5px'})

tests_layout = row(*checkbox_tests_columns, sizing_mode='stretch_width', width=300)
combos_layout = row(*checkbox_combos_columns, sizing_mode='stretch_width', width=300)

tests_column = column(tests_title, tests_layout, sizing_mode='stretch_width')
combos_column = column(combos_title, combos_layout, sizing_mode='stretch_width')

controls_layout = row(
    tests_column,
    Spacer(width=50),
    combos_column,
    sizing_mode='stretch_width'
)

###############################################################################
# 6) Additional Aggregated Plots (B-Factor-based only)
###############################################################################

# (A) Spearman Rho vs Average B-Factor
source_avg_plot = ColumnDataSource(data_long_avg)
p_avg_plot = figure(
    title="Spearman Correlation vs Average B-Factor (B-Factor Only)",
    x_axis_label="Average B-Factor",
    y_axis_label="Spearman Correlation (B-Factor vs Frustration)",
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None
)
frust_types_avg = data_long_avg['Frust_Type'].unique().tolist()
palette_avg = Category10[max(3, len(frust_types_avg))]
color_map_frust_avg = {frust: palette_avg[i] for i, frust in enumerate(frust_types_avg)}

scatter_renderers_avg = []
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
            tooltips=[("Regression Equation", "@equation")],
            mode='mouse'
        )
        p_avg_plot.add_tools(hover_regression)

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

# (B) Spearman Rho vs Std Dev B-Factor
source_std_plot = ColumnDataSource(data_long_std)
p_std_plot = figure(
    title="Spearman Correlation vs Std Dev of B-Factor (B-Factor Only)",
    x_axis_label="Standard Deviation of B-Factor",
    y_axis_label="Spearman Correlation (B-Factor vs Frustration)",
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None
)
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
            tooltips=[("Regression Equation", "@equation")],
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

# (C) Spearman Rho per Protein (B-Factor-only)
data_long_corr = data_proviz.melt(
    id_vars=['Protein'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust'],
    var_name='Frust_Type',
    value_name='Spearman_Rho'
)
data_long_corr['Frust_Type'] = data_long_corr['Frust_Type'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')
data_long_corr.dropna(subset=['Spearman_Rho'], inplace=True)

source_corr_plot = ColumnDataSource(data_long_corr)
p_corr_plot = figure(
    title="Spearman Correlation per Protein (B-Factor vs Frustration)",
    x_axis_label="Protein",
    y_axis_label="Spearman Correlation (B-Factor vs Frustration)",
    x_range=data_proviz['Protein'].tolist(),
    sizing_mode='stretch_width',
    height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None,
    toolbar_location="above"
)
hover_corr = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Metric", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    mode='mouse'
)
p_corr_plot.add_tools(hover_corr)
p_corr_plot.line(
    x=[-0.5, len(data_proviz['Protein']) - 0.5], 
    y=[0, 0], 
    line_width=1, 
    line_dash='dashed', 
    color='gray', 
    name='y_zero_line'
)

frust_types_corr = data_long_corr['Frust_Type'].unique().tolist()
palette_corr = Category10[max(3, len(frust_types_corr))]
color_map_corr = {frust: palette_corr[i] for i, frust in enumerate(frust_types_corr)}

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
    mean_value = subset['Spearman_Rho'].mean()
    mean_source = ColumnDataSource(data=dict(
        x=[-0.5, len(data_proviz['Protein']) - 0.5],
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
p_corr_plot.xaxis.major_label_orientation = pi / 4

###############################################################################
# 7) Bar Plot for Mean +/- SD (B-Factor-only)
###############################################################################
def create_bar_plot_with_sd(data_proviz):
    """
    Creates a bar chart displaying the mean Spearman correlation for each frustration metric
    (B-Factor vs. frustration) with error bars representing the standard deviation.
    """
    spearman_columns = ['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust']
    stats_corrs = data_proviz[spearman_columns].agg(['mean', 'std']).transpose().reset_index()
    stats_corrs.rename(columns={
        'index': 'Metric',
        'mean': 'Mean_Spearman_Rho',
        'std': 'Std_Spearman_Rho'
    }, inplace=True)
    stats_corrs['Metric'] = stats_corrs['Metric'].str.replace('Spearman_', '').str.replace('Frust', 'Frust.')

    # Simple color map
    # (adjust color if you want to differentiate among ExpFrust, AFFrust, EvolFrust)
    color_map = {
        "ExpFrust.": Category10[10][0],
        "AFFrust.": Category10[10][1],
        "EvolFrust.": Category10[10][2],
    }
    stats_corrs['Color'] = stats_corrs['Metric'].map(color_map)

    source_bar = ColumnDataSource(stats_corrs)
    p_bar = figure(
        title="Mean Spearman Correlation (B-Factor vs Frustration)",
        x_axis_label="Frustration Metric",
        y_axis_label="Mean Spearman Rho",
        x_range=stats_corrs['Metric'].tolist(),
        sizing_mode='stretch_width',
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above"
    )
    vbar_renderer = p_bar.vbar(
        x='Metric',
        top='Mean_Spearman_Rho',
        width=0.6,
        source=source_bar,
        color='Color',
        legend_label="Frustration Metric",
        line_color="black"
    )
    whisker = Whisker(
        base='Metric',
        upper='upper',
        lower='lower',
        source=source_bar,
        level="overlay"
    )
    p_bar.add_layout(whisker)
    source_bar.data['upper'] = source_bar.data['Mean_Spearman_Rho'] + source_bar.data['Std_Spearman_Rho']
    source_bar.data['lower'] = source_bar.data['Mean_Spearman_Rho'] - source_bar.data['Std_Spearman_Rho']

    min_lower = source_bar.data['lower'].min()
    max_upper = source_bar.data['upper'].max()
    y_padding = (max_upper - min_lower) * 0.1 if (max_upper - min_lower) != 0 else 1
    p_bar.y_range = Range1d(start=min_lower - y_padding, end=max_upper + y_padding)
    p_bar.line(x=[-0.5, len(stats_corrs) - 0.5], y=[0, 0], line_width=1, line_dash='dashed', color='gray')

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

###############################################################################
# 8) Layout Assembly
###############################################################################
p_bar_plot = create_bar_plot_with_sd(data_proviz)

# Unity placeholder (or remove if not needed)
description_visualizer = Div(text="""
    <h2>Protein Visualizer Instructions</h2>
    <p>(Optional Unity or other 3D viewer details...)</p>
""", sizing_mode='stretch_width')

unity_iframe = Div(
    text="""
    <div style="width: 100%; text-align: center; margin: 20px 0;">
        <iframe 
            src="https://example.com/unity/"
            style="width: 95vw; height: 90vh; border: 2px solid #ddd; border-radius: 8px;"
            allowfullscreen>
        </iframe>
    </div>
    """,
    sizing_mode='stretch_width',
)
unity_iframe.visible = True

unity_container = column(description_visualizer, unity_iframe, sizing_mode='stretch_width')

header = Div(text="""
    <h1>Evolutionary Frustration</h1>
    <p>
        Demonstration of optional RMSF usage. Select "Use RMSF" above to see if RMSF is available.
        If RMSF is not present in a file, the plots will revert to B-Factor.
    </p>
""", sizing_mode='stretch_width')

controls_section = Div(text="<b>Filter Correlation Table (B-Factor-based)</b>", styles={'font-size': '16px', 'margin': '10px 0'})

visualization_section = column(
    row(select_file, window_slider, flex_radio, sizing_mode="stretch_width", css_classes=["controls-row"]),
    p,
    row(
        column(p_scatter_exp, regression_info_exp, sizing_mode="stretch_width"),
        column(p_scatter_af, regression_info_af, sizing_mode="stretch_width"),
        column(p_scatter_evol, regression_info_evol, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        css_classes=["controls-row"]
    ),
    unity_container,
    p_avg_plot,
    p_std_plot,
    p_corr_plot,
    p_bar_plot,
    sizing_mode='stretch_width'
)

main_layout = column(
    header,
    visualization_section,
    controls_section,
    controls_layout,
    data_table,
    sizing_mode='stretch_width'
)

curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration (RMSF Support)"