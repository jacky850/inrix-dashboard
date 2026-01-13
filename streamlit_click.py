# streamlit_inrix_click.py
# Purpose: Streamlit app that displays INRIX incidents on a map.
# Clicking (or clicking near) an incident shows the speed time series for that segment.
# Data sources: default repo-relative paths data/incidents.csv and data/speed.csv
# OR set env vars INCIDENT_CSV_URL and SPEED_CSV_URL to point to raw GitHub/S3 URLs.

import os
import math
import io
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="INRIX incidents — click to show speed")

# ---------------- Config / data locations ----------------
# Prefer environment URLs if provided (raw GitHub / S3). Otherwise relative repo files.
INCIDENT_CSV_URL = os.environ.get("INCIDENT_CSV_URL", None)
SPEED_CSV_URL = os.environ.get("SPEED_CSV_URL", None)

DEFAULT_INCIDENT_PATH = Path("data/incidents.csv")
DEFAULT_SPEED_PATH = Path("data/speed.csv")

# helper: load csv robustly (local path or URL)
@st.cache_data
def load_csv_path_or_url(path_or_url):
    if path_or_url is None:
        return None
    try:
        if str(path_or_url).lower().startswith("http"):
            df = pd.read_csv(path_or_url, low_memory=False)
        else:
            df = pd.read_csv(path_or_url, low_memory=False)
    except Exception:
        # fallback to try excel
        df = pd.read_excel(path_or_url)
    df.columns = [c.strip() for c in df.columns]
    return df

# attempt loads (order: env url -> repo path)
if INCIDENT_CSV_URL:
    inc_df = load_csv_path_or_url(INCIDENT_CSV_URL)
else:
    inc_df = load_csv_path_or_url(DEFAULT_INCIDENT_PATH) if DEFAULT_INCIDENT_PATH.exists() else None

if SPEED_CSV_URL:
    speed_df = load_csv_path_or_url(SPEED_CSV_URL)
else:
    speed_df = load_csv_path_or_url(DEFAULT_SPEED_PATH) if DEFAULT_SPEED_PATH.exists() else None

# If missing, show clear instructions and stop (we don't want upload UI)
if inc_df is None:
    st.error(
        "Incident CSV not found. Put `data/incidents.csv` in the repo or set env var INCIDENT_CSV_URL to a raw URL."
    )
    st.stop()
if speed_df is None:
    st.error(
        "Speed CSV not found. Put `data/speed.csv` in the repo or set env var SPEED_CSV_URL to a raw URL."
    )
    st.stop()

# ---------------- Parse minimal necessary columns (robust) ----------------
def parse_incidents_for_latlon_and_time(df):
    df = df.copy()
    # find lat/lon
    lat_col = next((c for c in df.columns if 'lat' in c.lower() and 'lon' not in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower() or 'long' in c.lower()), None)
    if lat_col and lon_col:
        df["Latitude"] = pd.to_numeric(df[lat_col], errors='coerce')
        df["Longitude"] = pd.to_numeric(df[lon_col], errors='coerce')
    else:
        # try combined 'Location' like "lat, lon" or WKT
        combined_col = next((c for c in df.columns if 'lat' in c.lower() and 'lon' in c.lower()), None)
        if combined_col is None:
            combined_col = next((c for c in df.columns if c.lower().strip() == 'location'), None)
        def parse_latlon(val):
            if pd.isna(val): return (np.nan, np.nan)
            s = str(val).strip()
            # comma separated
            if "," in s:
                a,b = [p.strip() for p in s.split(",")[:2]]
                try:
                    return float(a), float(b)
                except:
                    pass
            # fallback: extract floats
            import re
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])
            return (np.nan, np.nan)
        if combined_col:
            latlon = df[combined_col].apply(parse_latlon)
            df["Latitude"] = latlon.apply(lambda t: t[0])
            df["Longitude"] = latlon.apply(lambda t: t[1])
        else:
            df["Latitude"] = np.nan
            df["Longitude"] = np.nan

    # find start time column
    time_col = next((c for c in df.columns if any(k in c.lower() for k in ['start','time','date','timestamp','reported','occurrence'])), None)
    if time_col:
        df["start_ts"] = pd.to_datetime(df[time_col], errors='coerce')
    else:
        df["start_ts"] = pd.NaT

    # segment id
    seg_col = next((c for c in df.columns if any(k in c.lower() for k in ['segment','tmc','segment id'])), None)
    if seg_col:
        df["segment_id"] = df[seg_col].astype(str)
    else:
        # try common names
        seg_col = next((c for c in df.columns if 'id' in c.lower() and ('segment' in c.lower() or 'tmc' in c.lower())), None)
        df["segment_id"] = df[seg_col].astype(str) if seg_col else df.index.astype(str)

    return df

def parse_speed_for_timestamp_segment_speed(df):
    df = df.copy()
    # segment id
    seg_col = next((c for c in df.columns if 'segment' in c.lower() or 'tmc' in c.lower()), None)
    if not seg_col:
        raise ValueError("Speed file: cannot find segment id column.")
    df["segment_id"] = df[seg_col].astype(str)

    # time col
    time_col = next((c for c in df.columns if any(k in c.lower() for k in ['time','date','timestamp','datetime'])), None)
    if time_col:
        df["timestamp_parsed"] = pd.to_datetime(df[time_col], errors='coerce')
    else:
        # try numeric epoch
        possible = pd.to_numeric(df.iloc[:,0], errors='coerce')
        df["timestamp_parsed"] = pd.to_datetime(possible, unit='s', errors='coerce')

    # speed col
    speed_col = next((c for c in df.columns if 'speed' in c.lower() or 'current' in c.lower() or 'travel' in c.lower()), None)
    if not speed_col:
        raise ValueError("Speed file: cannot find speed column.")
    df["currentSpeed"] = pd.to_numeric(df[speed_col], errors='coerce')

    # compute freeFlowSpeed fallback (85th percentile)
    ff_col = next((c for c in df.columns if 'ref' in c.lower() and 'speed' in c.lower()), None)
    if ff_col:
        df["freeFlowSpeed"] = pd.to_numeric(df[ff_col], errors='coerce')
    else:
        df["freeFlowSpeed"] = df.groupby("segment_id")["currentSpeed"].transform(lambda x: np.nanpercentile(x.dropna(), 85) if x.notna().any() else np.nan)

    df = df.dropna(subset=["timestamp_parsed"])
    df = df.sort_values(["segment_id", "timestamp_parsed"]).reset_index(drop=True)
    return df

# parse
inc_df = parse_incidents_for_latlon_and_time(inc_df)
try:
    speed_df = parse_speed_for_timestamp_segment_speed(speed_df)
except Exception as e:
    st.error(f"Error parsing speed CSV: {e}")
    st.stop()

# filter valid coords
inc_df = inc_df.dropna(subset=["Latitude","Longitude"]).copy()
if inc_df.empty:
    st.error("No incidents with valid Latitude/Longitude found.")
    st.stop()

# ---------------- UI layout ----------------
st.title("INRIX incidents — click on map to show speed time series")
left, right = st.columns([2,1])

# build the folium map
center = [inc_df["Latitude"].mean(), inc_df["Longitude"].mean()]
m = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=11, control_scale=True)
for _, r in inc_df.iterrows():
    lat = float(r["Latitude"]); lon = float(r["Longitude"])
    seg = str(r.get("segment_id",""))
    st_iso = ""
    if pd.notna(r.get("start_ts")):
        st_iso = pd.to_datetime(r["start_ts"]).isoformat()
    title = str(r.get("Event Text", r.get("Description", r.get("RoadwayName", "Incident"))))
    popup_html = f"<b>{title}</b><br/>seg: {seg}<br/>start: {st_iso}"
    folium.CircleMarker(location=[lat, lon], radius=5, color="blue",
                        popup=folium.Popup(popup_html, max_width=300)).add_to(m)

with left:
    st.markdown("### Map (click a point)")
    # st_folium returns last_clicked when user clicks on map
    mf = st_folium(m, width=700, height=600)

# helper: haversine
def haversine(lat1, lon1, lat2, lon2):
    R=6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

selected_info = None

# If user clicked on map, st_folium returns last_clicked lat/lng
if mf and isinstance(mf, dict) and mf.get("last_clicked"):
    lc = mf["last_clicked"]
    latc = lc.get("lat"); lonc = lc.get("lng")
    # find nearest incident within 0.3 km (300 m). If none, increase to 1 km.
    inc_df["__dist_km"] = inc_df.apply(lambda r: haversine(latc, lonc, float(r["Latitude"]), float(r["Longitude"])), axis=1)
    near = inc_df.sort_values("__dist_km").iloc[0]
    if near["__dist_km"] > 1.0:
        selected_info = None
        st.sidebar.info("Clicked location is farther than 1 km from any incident marker. Try clicking closer to a marker.")
    else:
        selected_info = dict(segment_id=str(near["segment_id"]), start_ts=near.get("start_ts"), title=str(near.get("Event Text", near.get("Description", "Incident"))))
else:
    st.sidebar.info("Click an incident on the map (left) to show its speed time series here.")

# right panel: show timeseries if selected
with right:
    st.markdown("### Selected event")
    if selected_info is None:
        st.write("No event selected.")
    else:
        st.write(f"**{selected_info['title']}**")
        seg = selected_info["segment_id"]
        start_dt = pd.to_datetime(selected_info["start_ts"], errors='coerce') if selected_info.get("start_ts") is not None else pd.NaT
        if pd.isna(start_dt):
            # fallback: earliest timestamp for segment
            subx = speed_df[speed_df["segment_id"] == seg]
            if not subx.empty:
                start_dt = subx["timestamp_parsed"].min()
        if pd.isna(start_dt):
            st.error("Could not determine event time for this segment.")
        else:
            PRE = 20; POST = 60
            window_start = start_dt - pd.Timedelta(minutes=PRE)
            window_end   = start_dt + pd.Timedelta(minutes=POST)
            sub = speed_df[(speed_df["segment_id"] == seg) & (speed_df["timestamp_parsed"] >= window_start) & (speed_df["timestamp_parsed"] <= window_end)].copy()
            if sub.empty:
                st.warning("No speed records for this segment in the window.")
            else:
                fig, ax = plt.subplots(figsize=(6,3))
                ax.plot(sub["timestamp_parsed"], sub["currentSpeed"], label="currentSpeed", linewidth=1.6)
                ax.plot(sub["timestamp_parsed"], sub["freeFlowSpeed"], label="freeFlowSpeed", linewidth=1.0, linestyle='--')
                ratio = (sub["currentSpeed"] / sub["freeFlowSpeed"]).replace([np.inf, -np.inf], np.nan)
                ax2 = ax.twinx()
                ax2.plot(sub["timestamp_parsed"], ratio, label="speed_ratio", linewidth=1, alpha=0.8)
                ax.axvline(start_dt, color='red', linestyle=':', linewidth=1)
                ax.set_xlabel("Time")
                ax.set_ylabel("Speed")
                ax2.set_ylabel("speed_ratio")
                ax.legend(loc='upper left', fontsize=8)
                ax2.legend(loc='upper right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                st.caption(f"Window: {PRE} min before / {POST} min after")
