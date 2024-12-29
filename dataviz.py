import os
import re
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, MultiSelect, Toggle,
    DataTable, TableColumn, NumberFormatter, 
    Div
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from scipy.stats import spearmanr

###############################################################################
# 1) HELPER FUNCTIONS
###############################################################################

def get_test_directories(root_path):
    """Return sorted list of directories named test### in root_path."""
    dirs = []
    for entry in os.listdir(root_path):
        fullp = os.path.join(root_path, entry)
        if os.path.isdir(fullp) and re.match(r'test\d{3}', entry):
            dirs.append(entry)
    return sorted(dirs)

def moving_average(arr, window_size=5):
    """
    Simple moving average on a float array (np.nan used for missing).
    Returns an equally sized array of floats (with np.nan where not enough data).
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

def parse_summary_file(file_path):
    """
    Reads a 'summary.txt' with columns:
      AlnIndex, Residue, B_Factor, ExpFrust, AFFrust, EvolFrust
    Returns (df_original, df_for_plot), plus a correlation dict for the non-smoothed data.
      - df_original: numeric columns with np.nan for missing
      - df_for_plot: smoothed numeric columns for aesthetic plotting
      - corrs: dictionary of Spearman correlations among the 4 metrics (non-smoothed)
    """
    if not os.path.isfile(file_path):
        return None, None, {}
    
    df = pd.read_csv(file_path, sep='\t')
    # Convert any 'n/a' to np.nan in numeric columns
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        df[col] = df[col].apply(lambda x: np.nan if str(x).lower()=='n/a' else float(x))

    # Make a copy for correlation calculations (non-smoothed)
    df_original = df.copy()

    # Smooth the numeric columns for plotting
    df_for_plot = df.copy()
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr = df_for_plot[col].values
        df_for_plot[col] = moving_average(arr, window_size=5)

    # Compute Spearman correlations on NON-smoothed data, 
    # ignoring rows with any np.nan in the 4 metrics
    sub = df_original.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    corrs = {}
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
            rho, pval = spearmanr(sub[mA], sub[mB])
            corrs[(mA, mB)] = (rho, pval)

    return df_original, df_for_plot, corrs


###############################################################################
# 2) LOAD DATA FOR ALL test###
###############################################################################

root_dir = "/Users/adamkuhn/Desktop/proviz2.0/"
test_dirs = get_test_directories(root_dir)

# For each test###, parse the data
data_by_test = {}
all_corr_rows = []  # for building correlation table across all tests

for td in test_dirs:
    summary_path = os.path.join(root_dir, td, "summary.txt")
    df_orig, df_plot, corrs = parse_summary_file(summary_path)
    if df_orig is None:
        continue
    data_by_test[td] = {
        "df_original": df_orig,
        "df_plot": df_plot,
        "corrs": corrs
    }
    # We'll also accumulate correlation info for a master table
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo
        all_corr_rows.append([td, mA, mB, rho, pval])

df_all_corr = pd.DataFrame(all_corr_rows, columns=["Test","MetricA","MetricB","Rho","Pval"])


###############################################################################
# 3) BOKEH APPLICATION
###############################################################################

# 3a) A ColumnDataSource for the main plot lines (we update it based on the selected test###).
source_plot = ColumnDataSource(data=dict(
    x=[],
    residue=[],
    b_factor=[],
    exp_frust=[],
    af_frust=[],
    evol_frust=[]
))

# 3b) Figure that stretches to fill available space
p = figure(
    title="(No Data)",
    sizing_mode='stretch_both',  # <--- fill available space
    tools=["pan","box_zoom","wheel_zoom","reset","save","hover"],
    active_drag="box_zoom", active_scroll="wheel_zoom"
)
p.hover.tooltips = [
    ("Index", "@x"),
    ("Residue", "@residue"),
    ("Value", "@y")
]
p.xaxis.axis_label = "Residue Index"

# Add lines with a color legend
color_map = {
    "b_factor":  ("B-Factor", "#1f77b4"),
    "exp_frust": ("ExpFrust", "#2ca02c"),
    "af_frust":  ("AFFrust",  "#ff7f0e"),
    "evol_frust":("EvolFrust","#d62728")
}
for col_key, (label, col) in color_map.items():
    p.line(
        x="x", y=col_key, source=source_plot,
        line_width=2, alpha=0.7, color=col,
        legend_label=label
    )
p.legend.location = "top_left"
p.legend.title = "Metrics"
p.legend.click_policy = "hide"


###############################################################################
# 4) WIDGETS FOR PLOTTING
###############################################################################

# 4a) SELECT which test### to show
test_options = list(data_by_test.keys())
select_test = Select(
    title="Select Protein (test###):",
    value=test_options[0] if test_options else "",
    options=test_options
)

def update_plot(attr, old, new):
    """Callback when user selects a different test###."""
    td = select_test.value
    if td not in data_by_test:
        source_plot.data = {}
        p.title.text = "(No Data)"
        return
    dfp = data_by_test[td]["df_plot"]

    # Filter out rows that have np.nan in any metric => only fully-defined rows
    sub = dfp.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    if sub.empty:
        source_plot.data = {}
        p.title.text = f"{td} (No valid rows after filtering)."
        return

    # build new data
    new_data = dict(
        x=sub["AlnIndex"].tolist(),
        residue=sub["Residue"].tolist(),
        b_factor=sub["B_Factor"].tolist(),
        exp_frust=sub["ExpFrust"].tolist(),
        af_frust=sub["AFFrust"].tolist(),
        evol_frust=sub["EvolFrust"].tolist()
    )
    source_plot.data = new_data
    p.title.text = f"{td} (Smoothed)"

select_test.on_change("value", update_plot)

# Initialize the plot if we have at least one test
if test_options:
    select_test.value = test_options[0]


###############################################################################
# 5) BUILD A CORRELATION TABLE THAT CAN BE FILTERED & TOGGLED
###############################################################################

# If df_all_corr is empty, build an empty structure
if df_all_corr.empty:
    from bokeh.models import DataTable, TableColumn
    columns = [
        TableColumn(field="Test", title="Test"),
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Spearman Rho"),
        TableColumn(field="Pval", title="p-value"),
    ]
    source_corr = ColumnDataSource(dict(Test=[], MetricA=[], MetricB=[], Rho=[], Pval=[]))
    data_table = DataTable(columns=columns, source=source_corr, height=300)
else:
    # Convert to ColumnDataSource
    from bokeh.models import DataTable, TableColumn, NumberFormatter
    source_corr = ColumnDataSource(df_all_corr)
    columns = [
        TableColumn(field="Test", title="Test"),
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Spearman Rho", formatter=NumberFormatter(format="0.3f")),
        TableColumn(field="Pval", title="p-value", formatter=NumberFormatter(format="0.2e")),
    ]
    data_table = DataTable(columns=columns, source=source_corr, height=300)

# A toggle to show/hide the correlation table
toggle_table = Toggle(label="Show/Hide Correlation Table", button_type="primary")
data_table.visible = False

def toggle_table_callback(event):
    data_table.visible = not data_table.visible

toggle_table.on_click(toggle_table_callback)

# We also add multi-select widgets for filtering by (Test) and by (Metric pair).
tests_in_corr = sorted(df_all_corr["Test"].unique()) if not df_all_corr.empty else []
combo_in_corr = []
if not df_all_corr.empty:
    # Build a list of "MetricA vs MetricB" strings
    def combo_str(r):
        return f"{r['MetricA']} vs {r['MetricB']}"
    combos = df_all_corr.apply(combo_str, axis=1).unique()
    combo_in_corr = sorted(combos)

ms_test_filter = MultiSelect(
    title="Filter by Protein (test###):",
    value=tests_in_corr,  # default select all
    options=tests_in_corr
)
ms_combo_filter = MultiSelect(
    title="Filter by Metric Pair:",
    value=combo_in_corr,  # default select all
    options=combo_in_corr
)

def update_corr_filter(attr, old, new):
    """Filter the correlation table based on selected tests and combos."""
    if df_all_corr.empty:
        return
    # Build mask
    selected_tests = set(ms_test_filter.value)
    selected_combos = set(ms_combo_filter.value)
    
    # We create a column of "combo_str = MetricA vs MetricB"
    df_tmp = df_all_corr.copy()
    df_tmp["combo_str"] = df_tmp.apply(lambda r: f"{r['MetricA']} vs {r['MetricB']}", axis=1)
    
    filtered = df_tmp[
        (df_tmp["Test"].isin(selected_tests)) &
        (df_tmp["combo_str"].isin(selected_combos))
    ].drop(columns=["combo_str"])
    
    source_corr.data = filtered.to_dict(orient="list")

ms_test_filter.on_change("value", update_corr_filter)
ms_combo_filter.on_change("value", update_corr_filter)

# trigger initial filter
if not df_all_corr.empty:
    ms_test_filter.value = tests_in_corr
    ms_combo_filter.value = combo_in_corr
    update_corr_filter(None, None, None)


###############################################################################
# 6) LAYOUT
###############################################################################

# Put everything in a vertical layout that stretches to fill the page
# We'll put the plot in its own row (with stretch_both),
# plus the table & filters in another row or column.

p_section = column(select_test, p, sizing_mode='stretch_both')

# The table is not sized by 'stretch_both' by default. We'll keep a fixed height.
filters_and_table = column(
    row(ms_test_filter, ms_combo_filter),
    toggle_table,
    data_table,
    sizing_mode='fixed'
)

layout = row(
    p_section,
    filters_and_table,
    sizing_mode='stretch_both'
)

curdoc().add_root(layout)
curdoc().title = "Interactive Protein Visualization"