"""
Streamlit-ready conversion of the INRIX timeseries dashboard.

How to deploy to Streamlit Community Cloud (short):
1. Put this file at the repo root (e.g. streamlit_inrix_dashboard.py).
2. Add a requirements.txt containing the Python packages listed below.
3. Commit and push to GitHub. In share.streamlit.io choose the repo and this file as the main app file.

Minimal requirements.txt (example):
streamlit
pandas
numpy
altair
folium
streamlit-folium
matplotlib
python-dateutil
geopy

Notes:
- This app is written to work with CSV/DB file paths you choose via the UI. For large datasets, host them on cloud storage (S3 / GCS) or load a small demo sample into the repo.
- If you previously used local-only resources (SQLite files, local TomTom tokens), do NOT commit sensitive tokens to GitHub.

"""
import st

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil import parser
import io

st.set_page_config(page_title="INRIX & AZ511 Dashboard", layout="wide")

# ------------------------------- Helpers ----------------------------------
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        # pandas will parse datetimes if possible
        df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception:
        # try excel
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
    return df

@st.cache_data
def parse_datetime_column(df, cols):
    for c in cols:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
            except Exception:
                df[c] = pd.to_datetime(df[c].astype(str), errors='coerce')
    return df

@st.cache_data
def make_sample_distribution(df, group_col, duration_col, label):
    if df is None or duration_col not in df.columns:
        return None
    series = df[duration_col].dropna()
    if series.dtype == 'timedelta64[ns]':
        vals = series.dt.total_seconds()/60.0
    else:
        # assume numeric (minutes) or string
        vals = pd.to_numeric(series, errors='coerce').dropna()
    return vals

# ------------------------------- UI --------------------------------------
st.title("INRIX / AZ511 Event & Speed Dashboard — Streamlit"")

st.sidebar.header("Data inputs")
uploaded_inrix = st.sidebar.file_uploader("Upload INRIX speed CSV/data (optional)", type=['csv','parquet','feather','xlsx'])
uploaded_incidents = st.sidebar.file_uploader("Upload INRIX incidents CSV (optional)", type=['csv','xlsx'])
uploaded_az511 = st.sidebar.file_uploader("Upload AZ511 events CSV (optional)", type=['csv','xlsx'])

st.sidebar.markdown("---")

# Time-range selector
st.sidebar.header("Temporal filters")
use_one_week = st.sidebar.checkbox("Focus on a single week (recommended for temporal matching)", value=True)

# Date inputs - default to last week
today = datetime.now().date()
default_end = today
default_start = today - timedelta(days=7)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)
if start_date > end_date:
    st.sidebar.error("Start must be <= End")

st.sidebar.markdown("---")

# Map and visualization options
st.sidebar.header("Visualization options")
map_center_lat = st.sidebar.number_input("Map center lat", value=33.4484)
map_center_lon = st.sidebar.number_input("Map center lon", value=-112.0740)
map_zoom = st.sidebar.slider("Map zoom", min_value=8, max_value=15, value=11)

# Load data
inrix_df = load_csv(uploaded_inrix) if uploaded_inrix is not None else None
incidents_df = load_csv(uploaded_incidents) if uploaded_incidents is not None else None
az511_df = load_csv(uploaded_az511) if uploaded_az511 is not None else None

# Basic parsing for common column names
if incidents_df is not None:
    incidents_df = parse_datetime_column(incidents_df, ['timestamp','start_time','end_time','event_time'])
if az511_df is not None:
    az511_df = parse_datetime_column(az511_df, ['start_time','end_time','timestamp'])
if inrix_df is not None:
    inrix_df = parse_datetime_column(inrix_df, ['time','timestamp','datetime'])

# show data summary
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.metric("INRIX rows", len(inrix_df) if inrix_df is not None else "No file")
with col2:
    st.metric("INRIX incidents rows", len(incidents_df) if incidents_df is not None else "No file")
with col3:
    st.metric("AZ511 rows", len(az511_df) if az511_df is not None else "No file")

st.markdown("---")

# ------------------------------- Map -------------------------------------
st.header("Map: Incident Locations")
map_col1, map_col2 = st.columns([2,1])

with map_col1:
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=map_zoom)
    # add incidents
    def add_points_from_df(df, lat_col='lat', lon_col='lon', popup_cols=None, color='red'):
        if df is None:
            return
        # try common names
        lat_candidates = [c for c in df.columns if c.lower() in ('lat','latitude','y')]
        lon_candidates = [c for c in df.columns if c.lower() in ('lon','longitude','x')]
        if not lat_candidates or not lon_candidates:
            return
        latc = lat_candidates[0]
        lonc = lon_candidates[0]
        for _, r in df.iterrows():
            try:
                folium.CircleMarker(location=[float(r[latc]), float(r[lonc])], radius=4, color=color, fill=True,
                                    popup=('<br>'.join([f"{c}: {r.get(c,'') }" for c in (popup_cols or [])])), opacity=0.7).add_to(m)
            except Exception:
                continue

    add_points_from_df(incidents_df, popup_cols=['event_type','start_time','end_time'], color='blue')
    add_points_from_df(az511_df, popup_cols=['event_type','start_time','end_time'], color='green')

    st_data = st_folium(m, width=700, height=500)

with map_col2:
    st.write("Map legend")
    st.markdown("- **Blue**: INRIX incidents (if provided)\n- **Green**: AZ511 events (if provided)")

st.markdown("---")

# ------------------------------- Distributions ---------------------------
st.header("Event duration distributions — 4 panels")

# We will create 4 distributions: ERS incidents, ERS non-incidents, INRIX incidents, INRIX non-incidents
# To do this we need to guess which df columns indicate ERS and incident labels. We'll allow user to choose columns.

st.subheader("Select columns for labeling and duration")
label_col = st.selectbox("Select column that indicates 'ERS' label (True/False) in AZ511 (if present)", options=(['(none)'] + (list(az511_df.columns) if az511_df is not None else [])))
duration_col_az = st.selectbox("Select duration column (AZ511)", options=(['(none)'] + (list(az511_df.columns) if az511_df is not None else [])))

label_col_inrix = st.selectbox("Select column that indicates INRIX incident label (in INRIX incidents file)", options=(['(none)'] + (list(incidents_df.columns) if incidents_df is not None else [])))
duration_col_inrix = st.selectbox("Select duration column (INRIX incidents)", options=(['(none)'] + (list(incidents_df.columns) if incidents_df is not None else [])))

plot_cols = st.columns(2)
with plot_cols[0]:
    st.write("AZ511 duration distributions")
with plot_cols[1]:
    st.write("INRIX duration distributions")

# helper plot function
def plot_distribution_panel(vals, title):
    if vals is None or len(vals)==0:
        st.write(f"No data for {title}")
        return
    # use Altair to show a smooth histogram / density
    dfv = pd.DataFrame({ 'minutes': vals })
    chart = alt.Chart(dfv).transform_density(
        'minutes', as_=['minutes','density']).mark_area(opacity=0.5).encode(
            x=alt.X('minutes:Q', title='Duration (minutes)'),
            y='density:Q'
    ).properties(width=350, height=250, title=title)
    st.altair_chart(chart)

# AZ511 panels
az_ers_vals = None
az_noners_vals = None
if az511_df is not None and duration_col_az != '(none)':
    df = az511_df.copy()
    # assume label column is boolean-like
    if label_col != '(none)' and label_col in df.columns:
        df[label_col] = df[label_col].astype(str).str.lower().isin(['1','true','yes','ers','y'])
        az_ers_vals = make_sample_distribution(df[df[label_col]==True], None, duration_col_az, 'AZ511 ERS')
        az_noners_vals = make_sample_distribution(df[df[label_col]!=True], None, duration_col_az, 'AZ511 Non-ERS')
    else:
        az_noners_vals = make_sample_distribution(df, None, duration_col_az, 'AZ511 All')

# INRIX panels
inrix_inc_vals = None
inrix_noninc_vals = None
if incidents_df is not None and duration_col_inrix != '(none)':
    df2 = incidents_df.copy()
    if label_col_inrix != '(none)' and label_col_inrix in df2.columns:
        df2[label_col_inrix] = df2[label_col_inrix].astype(str).str.lower().isin(['1','true','yes','incident','y'])
        inrix_inc_vals = make_sample_distribution(df2[df2[label_col_inrix]==True], None, duration_col_inrix, 'INRIX incidents')
        inrix_noninc_vals = make_sample_distribution(df2[df2[label_col_inrix]!=True], None, duration_col_inrix, 'INRIX non-incidents')
    else:
        inrix_inc_vals = make_sample_distribution(df2, None, duration_col_inrix, 'INRIX All')

# Draw 4 panels
cols = st.columns(2)
with cols[0]:
    plot_distribution_panel(az_ers_vals, 'AZ511 ERS — duration (minutes)')
    plot_distribution_panel(az_noners_vals, 'AZ511 Non-ERS — duration (minutes)')
with cols[1]:
    plot_distribution_panel(inrix_inc_vals, 'INRIX incidents — duration (minutes)')
    plot_distribution_panel(inrix_noninc_vals, 'INRIX non-incidents — duration (minutes)')

st.markdown("---")

# ------------------------------- Duration vs Severity --------------------
st.header("Duration vs Severity analysis")
severity_col_az = st.selectbox("AZ511 severity column (if any)", options=(['(none)'] + (list(az511_df.columns) if az511_df is not None else [])))
severity_col_inrix = st.selectbox("INRIX severity column (if any)", options=(['(none)'] + (list(incidents_df.columns) if incidents_df is not None else [])))

def scatter_duration_vs_severity(df, duration_col, severity_col, title):
    if df is None or duration_col not in df.columns or severity_col not in df.columns:
        st.write(f"Not enough data for {title}")
        return
    df2 = df[[duration_col, severity_col]].dropna()
    # convert duration
    if df2[duration_col].dtype == 'timedelta64[ns]':
        df2['dur_min'] = df2[duration_col].dt.total_seconds()/60.0
    else:
        df2['dur_min'] = pd.to_numeric(df2[duration_col], errors='coerce')
    df2['sev'] = pd.to_numeric(df2[severity_col], errors='coerce')
    df2 = df2.dropna()
    if df2.empty:
        st.write(f"No numeric duration/severity for {title}")
        return
    chart = alt.Chart(df2).mark_circle(size=30, opacity=0.5).encode(
        x=alt.X('dur_min:Q', title='Duration (min)'),
        y=alt.Y('sev:Q', title='Severity'),
        tooltip=['dur_min','sev']
    ).properties(width=600, height=300, title=title)
    st.altair_chart(chart)

scatter_duration_vs_severity(az511_df, duration_col_az, severity_col_az, 'AZ511: duration vs severity')
scatter_duration_vs_severity(incidents_df, duration_col_inrix, severity_col_inrix, 'INRIX: duration vs severity')

st.markdown("---")

# ------------------------------- Temporal matching ----------------------
st.header("Temporal matching: incidents vs speed drops (shorter segment / week)")

st.markdown("Use this section to inspect one specific road segment and one week of data. Select an ID or coordinate and a time window.")
seg_id_col = st.selectbox("Segment ID column in INRIX speed data (if any)", options=(['(none)'] + (list(inrix_df.columns) if inrix_df is not None else [])))
seg_select = None
if seg_id_col != '(none)' and inrix_df is not None:
    seg_unique = inrix_df[seg_id_col].dropna().unique().tolist()[:200]
    seg_select = st.selectbox('Choose segment id', options=['(none)'] + seg_unique)

# Choose a week window
if use_one_week:
    # default use start_date/end_date selected earlier
    ds = pd.to_datetime(start_date)
    de = pd.to_datetime(end_date) + pd.Timedelta(days=1)
else:
    ds = pd.to_datetime(start_date)
    de = pd.to_datetime(end_date) + pd.Timedelta(days=1)

st.write(f"Inspecting data from {ds.date()} to { (de - pd.Timedelta(days=1)).date() }")

# Filter INRIX speed series for selected segment and time window
if inrix_df is not None:
    df_speed = inrix_df.copy()
    # try common time column
    time_candidates = [c for c in df_speed.columns if c.lower() in ('time','timestamp','datetime')]
    if time_candidates:
        timecol = time_candidates[0]
        df_speed = df_speed[(df_speed[timecol] >= ds) & (df_speed[timecol] < de)]
        if seg_select and seg_select!='(none)' and seg_id_col in df_speed.columns:
            df_speed = df_speed[df_speed[seg_id_col]==seg_select]

    st.subheader('Speed time series (selected segment)')
    if df_speed.empty:
        st.write('No speed rows after applying filters')
    else:
        # try to pick a speed column
        speed_col_candidates = [c for c in df_speed.columns if c.lower() in ('speed','currentSpeed','avg_speed','travel_time')]
        if speed_col_candidates:
            spc = speed_col_candidates[0]
            # plot time series using altair
            df_plot = df_speed[[timecol, spc]].dropna()
            df_plot = df_plot.sort_values(by=timecol)
            line = alt.Chart(df_plot).mark_line().encode(x=timecol+':T', y=spc+':Q').properties(width=900, height=300)
            st.altair_chart(line)
        else:
            st.write('No obvious speed column found in INRIX data')

# Overlay incidents in the chosen week
st.subheader('Incidents in the selected week (list)')
if incidents_df is not None:
    # choose incident time column
    tcols = [c for c in incidents_df.columns if c.lower() in ('timestamp','time','start_time','event_time')]
    tcol = tcols[0] if tcols else None
    filtered_inc = incidents_df.copy()
    if tcol:
        filtered_inc = filtered_inc[(filtered_inc[tcol] >= ds) & (filtered_inc[tcol] < de)]
    if seg_select and seg_select!='(none)' and seg_id_col in filtered_inc.columns:
        filtered_inc = filtered_inc[filtered_inc[seg_id_col]==seg_select]
    st.dataframe(filtered_inc.head(200))
else:
    st.write('No INRIX incidents file uploaded')

if az511_df is not None:
    st.subheader('AZ511 events in the selected week (list)')
    tcols2 = [c for c in az511_df.columns if c.lower() in ('timestamp','time','start_time','event_time')]
    tcol2 = tcols2[0] if tcols2 else None
    filtered_az = az511_df.copy()
    if tcol2:
        filtered_az = filtered_az[(filtered_az[tcol2] >= ds) & (filtered_az[tcol2] < de)]
    st.dataframe(filtered_az.head(200))

st.markdown("---")

st.info("Deployment: commit this file and a requirements.txt to GitHub and connect the repo to share.streamlit.io. For large data, point the app to a cloud storage location rather than uploading files to the app.")

# Footer
st.caption("Converted to Streamlit by your assistant. If you want I can also produce a small README and a requirements.txt file for the repo.")
