import os
import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, linregress

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, CheckboxGroup, Div, Spacer,
    DataTable, TableColumn, NumberFormatter, HoverTool, 
    Slider, Whisker, Range1d
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
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
    "ExpFrust.": Category10[10][0],        # Red
    "AFFrust.": Category10[10][1],         # Blue
    "EvolFrust.": Category10[10][2],       # Green
    "Spearman_Diff": Category10[10][3]     # Orange
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

# Compute Spearman_Diff and sort proteins
data_proviz['Spearman_Diff'] = data_proviz['Spearman_EvolFrust'] - data_proviz['Spearman_ExpFrust']

# Sort data_proviz based on Spearman_Diff from smallest to largest
data_proviz = data_proviz.sort_values('Spearman_Diff')

# Set 'Protein' as a categorical variable with ordered categories based on Spearman_Diff
data_proviz['Protein'] = pd.Categorical(
    data_proviz['Protein'],
    categories=data_proviz['Protein'],
    ordered=True
)

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

# (A) Widgets
# Define file selection widget
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

# Define window slider for moving average
window_slider = Slider(
    start=1,
    end=20,
    value=5,
    step=1,
    title="Moving Average Window Size"
)

# (B) Main Plot: Smoothed + Normalized Data
source_plot = ColumnDataSource(data=dict(
    x=[],
    residue=[],
    B_Factor=[],
    ExpFrust=[],
    AFFrust=[],
    EvolFrust=[]
))

p = figure(
    title="Normalized and Smoothed Metrics",
    sizing_mode='stretch_width',
    height=600,
    tools=["pan","box_zoom","wheel_zoom","reset","save"],
    active_drag="box_zoom", 
    active_scroll=None
)

# Define separate HoverTools for each metric
hover_bf = HoverTool(
    renderers=[],
    tooltips=[("Residue Index", "@x"), ("Residue", "@residue"), ("B-Factor", "@B_Factor")],
    name="hover_b_factor"
)
hover_exp = HoverTool(
    renderers=[],
    tooltips=[("Residue Index", "@x"), ("Residue", "@residue"), ("ExpFrust", "@ExpFrust")],
    name="hover_exp_frust"
)
hover_af = HoverTool(
    renderers=[],
    tooltips=[("Residue Index", "@x"), ("Residue", "@residue"), ("AFFrust", "@AFFrust")],
    name="hover_af_frust"
)
hover_evol = HoverTool(
    renderers=[],
    tooltips=[("Residue Index", "@x"), ("Residue", "@residue"), ("EvolFrust", "@EvolFrust")],
    name="hover_evol_frust"
)

p.add_tools(hover_bf, hover_exp, hover_af, hover_evol)
p.xaxis.axis_label = "Residue Index"
p.yaxis.axis_label = "Normalized Metrics"

# Add lines for each metric
color_map = {
    "B_Factor":  Category10[10][0],        # Red
    "ExpFrust": Category10[10][1],         # Blue
    "AFFrust":  Category10[10][2],         # Green
    "EvolFrust": Category10[10][3]         # Orange
}

for metric, color in color_map.items():
    renderer = p.line(
        x="x", y=metric, source=source_plot,
        line_width=2, alpha=0.7, color=color,
        legend_label=metric
    )
    # Attach hover tools
    if metric == "B_Factor":
        hover_bf.renderers.append(renderer)
    elif metric == "ExpFrust":
        hover_exp.renderers.append(renderer)
    elif metric == "AFFrust":
        hover_af.renderers.append(renderer)
    elif metric == "EvolFrust":
        hover_evol.renderers.append(renderer)

p.legend.location = "top_left"
p.legend.title = "Metrics"
p.legend.click_policy = "hide"

# (C) Scatter Plots
# Spearman Rho vs Average B-Factor
source_avg_plot = ColumnDataSource(data_long_avg)

p_avg_plot = figure(
    title="Spearman Correlation vs Average B-Factor",
    x_axis_label="Average B-Factor",
    y_axis_label="Spearman Rho",
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None
)

frust_types_avg = data_long_avg['Frust_Type'].unique().tolist()
palette_avg = Category10[max(4, len(frust_types_avg))]  # Ensure enough colors
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
        muted_alpha=0.1
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
            legend_label=f"{frust} Regression"
        )

        hover_regression = HoverTool(
            renderers=[regression_line],
            tooltips=[
                ("Regression Equation", "@equation")
            ],
            mode='mouse'
        )
        p_avg_plot.add_tools(hover_regression)

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

# Spearman Rho vs Std Dev of B-Factor
source_std_plot = ColumnDataSource(data_long_std)

p_std_plot = figure(
    title="Spearman Correlation vs Std Dev of B-Factor",
    x_axis_label="Std Dev of B-Factor",
    y_axis_label="Spearman Rho",
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None
)

frust_types_std = data_long_std['Frust_Type'].unique().tolist()
palette_std = Category10[max(4, len(frust_types_std))]  # Ensure enough colors
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
        muted_alpha=0.1
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
            legend_label=f"{frust} Regression"
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
    renderers=scatter_renderers_std,  # Only attach to scatter renderers
    mode='mouse'
)
p_std_plot.add_tools(hover_scatter_std)

p_std_plot.legend.location = "top_left"
p_std_plot.legend.title = "Frustration Type"
p_std_plot.legend.click_policy = "mute"

# Spearman Rho per Protein and Frustration Metric (Plot Three)
# Melt data_proviz for the third plot
data_long_corr = data_proviz.melt(
    id_vars=['Protein'],
    value_vars=['Spearman_ExpFrust', 'Spearman_AFFrust', 'Spearman_EvolFrust', 'Spearman_Diff'],  # Included 'Spearman_Diff'
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
    y_axis_label="Spearman Rho",
    x_range=data_proviz['Protein'].tolist(),  # Ordered based on Spearman_Diff
    sizing_mode='stretch_width',
    height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None,
    toolbar_location="above"
)

# Define color palette for Frustration Types, including 'Spearman_Diff'
frust_types_corr = data_long_corr['Frust_Type'].unique().tolist()
palette_corr = Category10[max(4, len(frust_types_corr))]  # Ensure enough colors
color_map_corr = {frust: FRUSTRATION_COLORS.get(frust, Category10[10][0]) for frust in frust_types_corr}

# Add horizontal line at y=0
p_corr_plot.line(
    x=[-0.5, len(data_proviz['Protein']) - 0.5], 
    y=[0, 0], 
    line_width=1, 
    line_dash='dashed', 
    color='gray'
)

# Add scatter glyphs
for frust in frust_types_corr:
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    p_corr_plot.scatter(
        'Protein', 'Spearman_Rho',
        source=ColumnDataSource(subset),
        color=color_map_corr[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1
    )

# Customize hover tool
hover_corr = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Metric", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    mode='mouse'
)
p_corr_plot.add_tools(hover_corr)

# Rotate x-axis labels to prevent overlapping
from math import pi
p_corr_plot.xaxis.major_label_orientation = pi / 4  # 45 degrees

# Update legend
p_corr_plot.legend.location = "top_left"
p_corr_plot.legend.title = "Frustration Type"
p_corr_plot.legend.click_policy = "mute"

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
# 6) Additional Aggregated Plots
###############################################################################

# (A) Spearman Rho vs Average B-Factor
source_avg_plot = ColumnDataSource(data_long_avg)

p_avg_plot = figure(
    title="Spearman Correlation vs Average B-Factor",
    x_axis_label="Average B-Factor",
    y_axis_label="Spearman Rho",
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None
)

# Define color palette for Frustration Types
frust_types_avg = data_long_avg['Frust_Type'].unique().tolist()
palette_avg = Category10[max(4, len(frust_types_avg))]  # Ensure enough colors
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
        muted_alpha=0.1
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
            legend_label=f"{frust} Regression"
        )

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

# (B) Spearman Rho vs Std Dev of B-Factor
source_std_plot = ColumnDataSource(data_long_std)

p_std_plot = figure(
    title="Spearman Correlation vs Std Dev of B-Factor",
    x_axis_label="Std Dev of B-Factor",
    y_axis_label="Spearman Rho",
    sizing_mode='stretch_width',
    height=400,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None
)

# Define color palette for Frustration Types
frust_types_std = data_long_std['Frust_Type'].unique().tolist()
palette_std = Category10[max(4, len(frust_types_std))]  # Ensure enough colors
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
        muted_alpha=0.1
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
            legend_label=f"{frust} Regression"
        )

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

# (C) Spearman Rho per Protein and Frustration Metric (Plot Three)
source_corr_plot = ColumnDataSource(data_long_corr)

p_corr_plot = figure(
    title="Spearman Correlation per Protein and Frustration Metric",
    x_axis_label="Protein",
    y_axis_label="Spearman Rho",
    x_range=data_proviz['Protein'].tolist(),  # Ordered based on Spearman_Diff
    sizing_mode='stretch_width',
    height=600,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_drag="box_zoom",
    active_scroll=None,
    toolbar_location="above"
)

# Define color palette for Frustration Types, including 'Spearman_Diff'
frust_types_corr = data_long_corr['Frust_Type'].unique().tolist()
palette_corr = Category10[max(4, len(frust_types_corr))]  # Ensure enough colors
color_map_corr = {frust: FRUSTRATION_COLORS.get(frust, Category10[10][0]) for frust in frust_types_corr}

# Add horizontal line at y=0
p_corr_plot.line(
    x=[-0.5, len(data_proviz['Protein']) - 0.5], 
    y=[0, 0], 
    line_width=1, 
    line_dash='dashed', 
    color='gray'
)

# Add scatter glyphs
for frust in frust_types_corr:
    subset = data_long_corr[data_long_corr['Frust_Type'] == frust]
    p_corr_plot.scatter(
        'Protein', 'Spearman_Rho',
        source=ColumnDataSource(subset),
        color=color_map_corr[frust],
        size=8,
        alpha=0.6,
        legend_label=frust,
        muted_alpha=0.1
    )

# Customize hover tool
hover_corr = HoverTool(
    tooltips=[
        ("Protein", "@Protein"),
        ("Frustration Metric", "@Frust_Type"),
        ("Spearman Rho", "@Spearman_Rho{0.3f}")
    ],
    mode='mouse'
)
p_corr_plot.add_tools(hover_corr)

# Rotate x-axis labels to prevent overlapping
p_corr_plot.xaxis.major_label_orientation = pi / 4  # 45 degrees

# Update legend
p_corr_plot.legend.location = "top_left"
p_corr_plot.legend.title = "Frustration Type"
p_corr_plot.legend.click_policy = "mute"

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
# 8) Layout Assembly
###############################################################################

# (A) Scatter Plots Layout
scatter_col_exp = column(
    p_avg_plot,
    sizing_mode="stretch_width"
)
scatter_col_af = column(
    p_std_plot,
    sizing_mode="stretch_width"
)
scatter_col_evol = column(
    p_corr_plot,
    sizing_mode="stretch_width"
)

# Arrange scatter plots side by side with spacing
scatter_row = row(
    scatter_col_exp,
    scatter_col_af,
    scatter_col_evol,
    sizing_mode="stretch_width",
    spacing=20
)

# (B) Main Visualization Section
visualization_section = column(
    select_file,
    window_slider,
    p,
    scatter_row,
    unity_container,
    sizing_mode='stretch_width',
    spacing=20,
    name="visualization_section"
)

# (C) Main Layout Assembly
main_layout = column(
    custom_styles,
    header,
    visualization_section,
    controls_section,
    controls_layout,  # Updated controls layout with CheckboxGroups
    data_table,
    sizing_mode='stretch_width'
)

# Add main layout to the current document
curdoc().add_root(main_layout)
curdoc().title = "Evolutionary Frustration"

###############################################################################
# 9) Callbacks
###############################################################################

def update_plots(attr, old, new):
    """
    Callback to update plots based on selected file and window slider.
    """
    selected_file = select_file.value
    window_size = window_slider.value

    if selected_file not in data_by_file:
        print(f"Selected file {selected_file} not found in data.")
        return

    df_orig = data_by_file[selected_file]["df_original"]
    df_plot = data_by_file[selected_file]["df_for_plot"]

    # Apply moving average with the selected window size
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        df_plot[col] = moving_average(df_plot[col].values, window_size=window_size)

    # Min-Max normalization
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        valid = ~df_plot[col].isna()
        if valid.any():
            col_min = df_plot.loc[valid, col].min()
            col_max = df_plot.loc[valid, col].max()
            if col_max > col_min:
                df_plot[col] = (df_plot[col] - col_min) / (col_max - col_min)

    # Update main plot data source
    source_plot.data = {
        'x': df_plot['AlnIndex'],
        'residue': df_plot['Residue'],
        'B_Factor': df_plot['B_Factor'],
        'ExpFrust': df_plot['ExpFrust'],
        'AFFrust': df_plot['AFFrust'],
        'EvolFrust': df_plot['EvolFrust']
    }

# Attach callbacks
select_file.on_change('value', update_plots)
window_slider.on_change('value', update_plots)

# Initialize plots with the default file
update_plots(None, None, None)