# streamlit_inrix_click_on_demand.py
# Purpose: Streamlit app that shows INRIX incidents on a map.
# Clicking a point triggers on-demand, chunked filtering of the large speed CSV
# so the app doesn't try to read the whole huge file at startup.
#
# Usage:
# - Put this file in your repo (root).
# - Optionally add small demo files in data/demo_incidents.csv and data/demo_speed.csv for fallback.
# - Set environment variables in Streamlit Cloud:
#     INCIDENT_CSV_URL = "https://....incidents.csv"
#     SPEED_CSV_URL    = "https://....speed.csv"
#
# Behavior:
# - incidents are loaded at startup (from URL or repo demo)
# - speed data is filtered on click by streaming the CSV in chunks

import os
import math
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium
import matplotlib.pyplot as plt
import requests

st.set_page_config(layout="wide", page_title="INRIX incidents — click to show speed (on-demand)")

# ---------------- Config ----------------
INCIDENT_CSV_URL = os.environ.get("INCIDENT_CSV_URL", None)
SPEED_CSV_URL = os.environ.get("SPEED_CSV_URL", None)

# demo fallback files (put small samples into repo/data/ if you want immediate demo)
DEMO_INCIDENT_PATH = Path("data/demo_incidents.csv")
DEMO_SPEED_PATH = Path("data/demo_speed.csv")

# size threshold to prefer demo instead of attempting direct streaming (in bytes)
# if remote HEAD says the file is larger than this, we will use demo fallback for fast preview
SIZE_LIMIT_BYTES = 200 * 1024 * 1024  # 200 MB

# ---------------- Helpers ----------------
def try_head_content_length(url, timeout=20):
    """Return content-length in bytes or None on failure."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code >= 400:
            return None
        cl = r.headers.get("Content-Length")
        if cl is None:
            return None
        return int(cl)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_incidents(path_or_url, demo_path=DEMO_INCIDENT_PATH):
    """
    Load incidents: try path_or_url (URL or local), if large or fails, fallback to demo_path.
    Returns pandas DataFrame with Latitude, Longitude, start_ts, and segment_id columns (not guaranteed names).
    """
    if path_or_url is None:
        # No URL provided -> try demo
        if demo_path.exists():
            df = pd.read_csv(demo_path, low_memory=False)
        else:
            return None
    else:
        s = str(path_or_url)
        # if URL, check accessibility & size
        if s.lower().startswith("http"):
            # attempt HEAD to check status
            try:
                length = try_head_content_length(s)
                # head ok: if 'too big', still safe to load incidents because it's usually small file.
                # Try to read and fallback to demo on any error.
                df = pd.read_csv(s, low_memory=False)
            except Exception:
                if demo_path.exists():
                    df = pd.read_csv(demo_path, low_memory=False)
                else:
                    # last resort: try direct read (let exception bubble)
                    df = pd.read_csv(s, low_memory=False)
        else:
            # local file
            df = pd.read_csv(s, low_memory=False)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def parse_incidents_for_latlon_and_time(df):
    """Add Latitude, Longitude, start_ts, segment_id if possible."""
    df = df.copy()
    # find lat/lon
    lat_col = next((c for c in df.columns if c.lower() in ('lat','latitude','y')), None)
    lon_col = next((c for c in df.columns if c.lower() in ('lon','longitude','x')), None)
    if lat_col and lon_col:
        df["Latitude"] = pd.to_numeric(df[lat_col], errors='coerce')
        df["Longitude"] = pd.to_numeric(df[lon_col], errors='coerce')
    else:
        # try combined location-like column
        combined_col = next((c for c in df.columns if 'location' in c.lower() or 'coord' in c.lower()), None)
        def parse_latlon(val):
            if pd.isna(val): return (np.nan, np.nan)
            s = str(val)
            if "," in s:
                a,b = [p.strip() for p in s.split(",")[:2]]
                try:
                    return float(a), float(b)
                except:
                    pass
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
    # time
    time_col = next((c for c in df.columns if any(k in c.lower() for k in ['start','time','date','timestamp','reported','occurrence','event'])), None)
    if time_col:
        df["start_ts"] = pd.to_datetime(df[time_col], errors='coerce')
    else:
        df["start_ts"] = pd.NaT
    # segment id
    seg_col = next((c for c in df.columns if any(k in c.lower() for k in ['segment','tmc','segment id','locationid','linkid'])), None)
    if seg_col:
        df["segment_id"] = df[seg_col].astype(str)
    else:
        # fallback to index as id
        df["segment_id"] = df.index.astype(str)
    return df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

# --- streaming filter for speed file (on-demand) ---
def filter_speed_by_segment(url_or_path, segment_id, chunksize=200_000):
    """
    Stream-filter speed CSV (remote URL or local) in chunks and return DataFrame for the requested segment_id.
    This avoids loading the whole file into memory.
    """
    if url_or_path is None:
        return pd.DataFrame()
    s = str(url_or_path)
    # pandas can read from URL with chunksize (it will stream)
    try:
        it = pd.read_csv(s, chunksize=chunksize, low_memory=False)
    except Exception as e:
        # If direct pd.read_csv(url, chunksize=...) fails (some servers), try streaming via requests to a temp file
        try:
            with requests.get(s, stream=True, timeout=300) as r:
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpf:
                    for chunk in r.iter_content(chunk_size=4*1024*1024):
                        if chunk:
                            tmpf.write(chunk)
                    tmp_name = tmpf.name
            it = pd.read_csv(tmp_name, chunksize=chunksize, low_memory=False)
        except Exception as e2:
            raise RuntimeError(f"Failed to open speed CSV for streaming: {e2}")
    found = []
    seg_col_found = None
    for chunk in it:
        chunk.columns = [c.strip() for c in chunk.columns]
        if seg_col_found is None:
            seg_candidates = [c for c in chunk.columns if 'segment' in c.lower() or 'tmc' in c.lower() or c.lower()=='segment_id' or 'linkid' in c.lower()]
            if seg_candidates:
                seg_col_found = seg_candidates[0]
            else:
                # try id-like columns
                seg_candidates = [c for c in chunk.columns if 'id' in c.lower()]
                seg_col_found = seg_candidates[0] if seg_candidates else None
        if seg_col_found is None:
            raise RuntimeError("Cannot find segment id column in speed file (streaming).")
        # do equality compare as string
        sub = chunk[chunk[seg_col_found].astype(str) == str(segment_id)]
        if not sub.empty:
            found.append(sub)
    if not found:
        return pd.DataFrame()
    df = pd.concat(found, ignore_index=True)
    # parse time column
    time_cols = [c for c in df.columns if any(k in c.lower() for k in ['time','timestamp','date','datetime'])]
    if time_cols:
        df['timestamp_parsed'] = pd.to_datetime(df[time_cols[0]], errors='coerce')
    else:
        df['timestamp_parsed'] = pd.NaT
    return df.sort_values('timestamp_parsed')

# ---------------- Load incidents at startup (fast) ----------------
st.title("INRIX incidents — click on map to show speed (on-demand)")

# load incidents (use URL or demo)
inc_df_raw = load_incidents(INCIDENT_CSV_URL, DEMO_INCIDENT_PATH)
if inc_df_raw is None or inc_df_raw.empty:
    st.error("No incidents data found. Provide INCIDENT_CSV_URL or put data/demo_incidents.csv in repo.")
    st.stop()

inc_df = parse_incidents_for_latlon_and_time(inc_df_raw)
# drop invalid coords
inc_df = inc_df.dropna(subset=["Latitude","Longitude"]).copy()
if inc_df.empty:
    st.error("No incidents with valid Latitude/Longitude found.")
    st.stop()

left, right = st.columns([2,1])

# build map
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
    mf = st_folium(m, width=700, height=600)

# selection logic
selected_info = None
if mf and isinstance(mf, dict) and mf.get("last_clicked"):
    lc = mf["last_clicked"]
    latc = lc.get("lat"); lonc = lc.get("lng")
    inc_df["__dist_km"] = inc_df.apply(lambda r: haversine(latc, lonc, float(r["Latitude"]), float(r["Longitude"])), axis=1)
    near = inc_df.sort_values("__dist_km").iloc[0]
    if near["__dist_km"] > 1.0:
        selected_info = None
        st.sidebar.info("Clicked location is farther than 1 km from any incident marker. Try clicking closer to a marker.")
    else:
        selected_info = dict(segment_id=str(near["segment_id"]), start_ts=near.get("start_ts"), title=str(near.get("Event Text", near.get("Description", "Incident"))))

else:
    st.sidebar.info("Click an incident on the map (left) to show its speed time series here.")

with right:
    st.markdown("### Selected event")
    if selected_info is None:
        st.write("No event selected.")
    else:
        st.write(f"**{selected_info['title']}**")
        seg = selected_info["segment_id"]
        start_dt = pd.to_datetime(selected_info["start_ts"], errors='coerce') if selected_info.get("start_ts") is not None else pd.NaT
        if pd.isna(start_dt):
            # fallback: earliest timestamp for segment in demo/speed if present
            st.info("Event time not recorded; using available speed timestamps for that segment.")
        # Now stream-filter the speed CSV for this segment
        st.info("Filtering speed data for the selected segment (this may take a while for the first request)...")
        try:
            # if remote file seems very large, prefer demo fallback to keep UI responsive
            prefer_demo = False
            if SPEED_CSV_URL and SPEED_CSV_URL.lower().startswith("http"):
                cl = try_head_content_length(SPEED_CSV_URL)
                if cl is not None and cl > SIZE_LIMIT_BYTES and DEMO_SPEED_PATH.exists():
                    prefer_demo = True
            if prefer_demo:
                speed_src = str(DEMO_SPEED_PATH)
            else:
                speed_src = SPEED_CSV_URL or str(DEMO_SPEED_PATH)

            speed_seg_df = filter_speed_by_segment(speed_src, seg, chunksize=150_000)
        except Exception as e:
            st.error(f"Failed to filter speed data: {e}")
            speed_seg_df = pd.DataFrame()

        if speed_seg_df is None or speed_seg_df.empty:
            st.warning("No speed records found for this segment in the available data (or the filtering returned nothing).")
        else:
            # choose speed column
            spc = next((c for c in speed_seg_df.columns if 'speed' in c.lower() or 'current' in c.lower()), None)
            ff = next((c for c in speed_seg_df.columns if 'free' in c.lower() and 'speed' in c.lower()), None)
            if spc and 'timestamp_parsed' in speed_seg_df.columns:
                fig, ax = plt.subplots(figsize=(6,3))
                ax.plot(speed_seg_df["timestamp_parsed"], speed_seg_df[spc], label="currentSpeed", linewidth=1.6)
                if ff:
                    ax.plot(speed_seg_df["timestamp_parsed"], speed_seg_df[ff], label="freeFlowSpeed", linewidth=1.0, linestyle='--')
                # speed ratio
                if ff:
                    ratio = (speed_seg_df[spc] / speed_seg_df[ff]).replace([np.inf, -np.inf], np.nan)
                    ax2 = ax.twinx()
                    ax2.plot(speed_seg_df["timestamp_parsed"], ratio, label="speed_ratio", linewidth=1, alpha=0.8)
                    ax2.set_ylabel("speed_ratio")
                if pd.notna(start_dt):
                    ax.axvline(start_dt, color='red', linestyle=':', linewidth=1)
                ax.set_xlabel("Time")
                ax.set_ylabel("Speed")
                ax.legend(loc='upper left', fontsize=8)
                if ff:
                    ax2.legend(loc='upper right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Window: showing all available speed rows for this segment (first request may be slow).")
            else:
                st.warning("Speed column or timestamp not found in filtered data for this segment.")
