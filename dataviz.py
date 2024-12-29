import os
import re
import boto3
import tempfile
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, MultiSelect, Toggle,
    DataTable, TableColumn, NumberFormatter
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from scipy.stats import spearmanr

###############################################################################
# 1) Configuration
###############################################################################
# S3 bucket name
S3_BUCKET = "proteindata0"
# We'll look for files matching the pattern: summary_testNNN.txt
FILE_PATTERN = r"^summary_test(\d{3})\.txt$"

# Name of the data subfolder inside your bucket
S3_PREFIX = "summary_data/"  # Updated to reflect the proper data path

###############################################################################
# 2) Helpers: S3 + Data Parsing
###############################################################################
def list_s3_objects(bucket_name, prefix=""):
    """
    List all objects under a given S3 prefix.
    Returns a list of object keys.
    """
    s3 = boto3.client("s3")
    keys = []
    continuation_token = None

    while True:
        if continuation_token:
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                ContinuationToken=continuation_token
            )
        else:
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )

        if "Contents" in response:
            for obj in response["Contents"]:
                keys.append(obj["Key"])

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    return keys

def download_s3_file(bucket_name, key, local_path):
    """
    Download the S3 object <key> from <bucket_name> to <local_path>.
    """
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, key, local_path)

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

def parse_summary_file(local_path):
    """
    Reads a summary_test###.txt with columns:
      AlnIndex, Residue, B_Factor, ExpFrust, AFFrust, EvolFrust
    Returns (df_original, df_for_plot), plus a correlation dict for non-smoothed data.
    """
    if not os.path.isfile(local_path):
        return None, None, {}

    df = pd.read_csv(local_path, sep='\t')
    # Convert any 'n/a' to np.nan in numeric columns
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        df[col] = df[col].apply(lambda x: np.nan if str(x).lower()=='n/a' else float(x))

    # Non-smoothed for correlation
    df_original = df.copy()

    # Smoothed for plotting
    df_for_plot = df.copy()
    for col in ["B_Factor", "ExpFrust", "AFFrust", "EvolFrust"]:
        arr = df_for_plot[col].values
        df_for_plot[col] = moving_average(arr, window_size=5)

    # Compute Spearman correlations on NON-smoothed data
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
# 3) Load data from S3
###############################################################################
s3_keys = list_s3_objects(S3_BUCKET, prefix=S3_PREFIX)

data_by_test = {}
all_corr_rows = []

for k in s3_keys:
    # e.g. k might be "summary_data/summary_test001.txt"
    filename = os.path.basename(k)  # summary_test001.txt

    # Check if it matches summary_testNNN.txt
    match = re.match(FILE_PATTERN, filename)
    if not match:
        continue  # skip unrelated files

    # Extract test### from the file name
    test_number = match.group(1)  # e.g. "001"
    test_name = "test" + test_number  # e.g. "test001"

    # Download the file to a temp location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_name = tmp.name
    download_s3_file(S3_BUCKET, k, tmp_name)

    # Parse the data
    df_orig, df_plot, corrs = parse_summary_file(tmp_name)
    os.remove(tmp_name)  # clean up the local file

    if df_orig is None:
        continue

    data_by_test[test_name] = {
        "df_original": df_orig,
        "df_plot": df_plot,
        "corrs": corrs
    }

    # Accumulate correlation info for a master table
    for combo, (rho, pval) in corrs.items():
        mA, mB = combo
        all_corr_rows.append([test_name, mA, mB, rho, pval])

# Build DataFrame of correlations
df_all_corr = pd.DataFrame(all_corr_rows, columns=["Test","MetricA","MetricB","Rho","Pval"])

###############################################################################
# 4) Bokeh Application
###############################################################################

from bokeh.models import ColumnDataSource, Select, MultiSelect, Toggle, DataTable, TableColumn, NumberFormatter
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.io import curdoc

# (A) ColumnDataSource for the main plot
source_plot = ColumnDataSource(data=dict(
    x=[],
    residue=[],
    b_factor=[],
    exp_frust=[],
    af_frust=[],
    evol_frust=[]
))

# (B) The figure (fills available space)
p = figure(
    title="(No Data)",
    sizing_mode='stretch_both',
    tools=["pan","box_zoom","wheel_zoom","reset","save","hover"],
    active_drag="box_zoom", active_scroll="wheel_zoom"
)
p.hover.tooltips = [
    ("Index", "@x"),
    ("Residue", "@residue"),
    ("Value", "@y")
]
p.xaxis.axis_label = "Residue Index"

# Add lines for each metric
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

# (C) SELECT widget to pick the test### to show
test_options = sorted(data_by_test.keys())
select_test = Select(
    title="Select Protein (test###):",
    value=test_options[0] if test_options else "",
    options=test_options
)

def update_plot(attr, old, new):
    td = select_test.value
    if td not in data_by_test:
        source_plot.data = {}
        p.title.text = "(No Data)"
        return
    dfp = data_by_test[td]["df_plot"]

    # Filter rows that have no missing data
    sub = dfp.dropna(subset=["B_Factor","ExpFrust","AFFrust","EvolFrust"])
    if sub.empty:
        source_plot.data = {}
        p.title.text = f"{td} (No valid rows)."
        return

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
if test_options:
    select_test.value = test_options[0]  # triggers callback once

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
    data_table = DataTable(columns=columns, source=source_corr, height=300)
else:
    source_corr = ColumnDataSource(df_all_corr)
    columns = [
        TableColumn(field="Test", title="Test"),
        TableColumn(field="MetricA", title="MetricA"),
        TableColumn(field="MetricB", title="MetricB"),
        TableColumn(field="Rho", title="Spearman Rho", formatter=NumberFormatter(format="0.3f")),
        TableColumn(field="Pval", title="p-value", formatter=NumberFormatter(format="0.2e"))
    ]
    data_table = DataTable(columns=columns, source=source_corr, height=300)

toggle_table = Toggle(label="Show/Hide Correlation Table", button_type="primary")
data_table.visible = False

def toggle_table_callback(event):
    data_table.visible = not data_table.visible
toggle_table.on_click(toggle_table_callback)

# (E) FILTERS for correlation table
# Implement MultiSelect widgets for filtering by Test and Metric Pair

# Prepare options for MultiSelect
tests_in_corr = sorted(df_all_corr["Test"].unique()) if not df_all_corr.empty else []
if not df_all_corr.empty:
    # Build a list of "MetricA vs MetricB" strings
    combo_options = sorted([
        f"{row['MetricA']} vs {row['MetricB']}" 
        for _, row in df_all_corr.iterrows()
    ])
    # Remove duplicates
    combo_options = sorted(list(set(combo_options)))
else:
    combo_options = []

ms_test_filter = MultiSelect(
    title="Filter by Protein (test###):",
    value=tests_in_corr,  # default select all
    options=tests_in_corr
)
ms_combo_filter = MultiSelect(
    title="Filter by Metric Pair:",
    value=combo_options,  # default select all
    options=combo_options
)

def update_corr_filter(attr, old, new):
    """Filter the correlation table based on selected tests and metric pairs."""
    if df_all_corr.empty:
        return
    # Build mask
    selected_tests = set(ms_test_filter.value)
    selected_combos = set(ms_combo_filter.value)
    
    # Create a "combo_str" column for easy filtering
    df_tmp = df_all_corr.copy()
    df_tmp["combo_str"] = df_tmp.apply(lambda r: f"{r['MetricA']} vs {r['MetricB']}", axis=1)
    
    # Apply filters
    filtered = df_tmp[
        (df_tmp["Test"].isin(selected_tests)) &
        (df_tmp["combo_str"].isin(selected_combos))
    ].drop(columns=["combo_str"])
    
    source_corr.data = filtered.to_dict(orient="list")

ms_test_filter.on_change("value", update_corr_filter)
ms_combo_filter.on_change("value", update_corr_filter)

# Trigger initial filter
if not df_all_corr.empty:
    ms_test_filter.value = tests_in_corr
    ms_combo_filter.value = combo_options
    update_corr_filter(None, None, None)

# (F) Layout
# Arrange widgets and plot to fill the page

# Top row: Selection widgets
selection_widgets = row(select_test, sizing_mode='stretch_width')

# Middle row: Plot
plot_section = column(selection_widgets, p, sizing_mode='stretch_both')

# Bottom row: Toggle button and correlation table with filters
filters = column(
    row(ms_test_filter, ms_combo_filter, sizing_mode='stretch_width'),
    toggle_table,
    data_table,
    sizing_mode='stretch_width'
)

# Final layout: Plot on the left, filters and table on the right
layout = row(
    plot_section,
    filters,
    sizing_mode='stretch_both'
)

curdoc().add_root(layout)
curdoc().title = "Protein Visualization (S3 version)"