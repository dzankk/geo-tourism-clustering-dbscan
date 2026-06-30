# app.py
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN

# Force premium wide widescreen view
st.set_page_config(page_title="BiH Spatial Clustering Engine", layout="wide")

# Academic Header
st.title("DBSCAN Spatial Clustering: BiH Heritage Corridors")
st.caption("Seminar Project: Analyzing the influence of parameters ε (Epsilon) and MinPts on spatial clustering results.")
st.markdown("---")

# 1. Sidebar Parameter Controls
st.sidebar.header(" DBSCAN Configuration")
eps_km = st.sidebar.slider("Epsilon Radius (ε in Kilometers)", min_value=5.0, max_value=60.0, value=25.0, step=2.5)
min_samples = st.sidebar.slider("Minimum Points per Cluster (MinPts)", min_value=2, max_value=10, value=4, step=1)

# 2. Stable Data Ingestion
try:
    df = pd.read_csv("data/heritage_sites_bih.csv")
    df.columns = df.columns.str.strip()
    # Alphabetical sorting for seamless menu browsing
    df = df.sort_values(by="name").reset_index(drop=True)
except Exception as e:
    st.error(f"Error loading CSV data file. Details: {e}")
    st.stop()

# 3. DBSCAN Core Engine (Haversine Metric Space)
X_deg = df[['latitude', 'longitude']].values
X_rad = np.radians(X_deg)
EARTH_RADIUS_KM = 6371.009

# Convert kilometer metric directly to radians for the Haversine calculation
eps_rad = eps_km / EARTH_RADIUS_KM
db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
df['cluster_id'] = db.fit_predict(X_rad)

# Summary calculation vectors
total_sites = len(df)
labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
num_noise = list(labels).count(-1)

# 4. Summary Metric Row
col1, col2, col3 = st.columns(3)
col1.metric("Total Sites Loaded", total_sites)
col2.metric("Identified Spatial Corridors", num_clusters)
col3.metric("Noise Points (Isolated Sites)", num_noise)

st.markdown("---")

# 5. Initialize Map Component and Color Configuration
st.subheader("Interactive Spatial Density Map")

# Map stays locked in a stable country view
m = folium.Map(location=[44.15, 17.80], zoom_start=8, tiles="CartoDB dark_matter")

colors = ['#00d2ff', '#70ff00', '#ff0076', '#ffb300', '#a000ff', '#00ffaa', '#ff5722', '#eccc68']
fun_fact_col = 'fun_fact' if 'fun_fact' in df.columns else ('fun_facts' if 'fun_facts' in df.columns else None)

# 6. Session State Tracker for the Dropdown Choice
if "last_selected" not in st.session_state:
    st.session_state["last_selected"] = "-- Select a Site to Trigger Map Popup --"
if "map_render_key" not in st.session_state:
    st.session_state["map_render_key"] = 0

# Render individual markers
for idx, row in df.iterrows():
    cid = row['cluster_id']
    lat, lon = row['latitude'], row['longitude']
    
    if cid == -1:
        color = '#ffffff'
        radius = 5
        fill_op = 0.2
        tag = "Noise / Spatial Outlier"
    else:
        color = colors[cid % len(colors)]
        radius = 7.5
        fill_op = 0.8
        tag = f"Corridor Group {cid + 1}"

    popup_html = f"""
    <div style="font-family: Arial, sans-serif; width: 250px; background: #222; color: #fff; padding: 12px; border-radius: 6px; border-left: 5px solid {color}; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
        <span style="font-size: 9px; font-weight: bold; color: {color}; text-transform: uppercase; letter-spacing: 0.5px;">{tag}</span>
        <h4 style="margin: 4px 0 6px 0; font-size: 14px; color: #fff; border-bottom: 1px solid #444; padding-bottom: 4px;">{row['name']}</h4>
        <p style="margin: 0 0 6px 0; font-size: 11px; color: #aaa;"><b>Municipality:</b> {row['location']}</p>
        <p style="margin: 0 0 8px 0; font-size: 11px; line-height: 1.4; color: #eee; font-style: italic;">"{row['description']}"</p>
    """
    
    if fun_fact_col and pd.notna(row[fun_fact_col]):
        popup_html += f"""
        <div style="background: rgba(0, 210, 255, 0.15); border: 1px solid rgba(0, 210, 255, 0.3); padding: 6px; border-radius: 4px; font-size: 11px; color: #00d2ff; margin-top: 4px;">
            💡 <b>Fun Fact:</b> {row[fun_fact_col]}
        </div>
        """
    popup_html += "</div>"
    
    # Check if this marker is the one chosen in our bottom menu dropdown
    # If yes, add 'show=True' so the popup box renders completely wide open on load
    should_show_popup = (st.session_state["last_selected"] == row['name'])
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=290, show=should_show_popup),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=fill_op,
        weight=1.5,
        tooltip=row['name']
    ).add_to(m)

# Render the map widget
st_folium(m, width="100%", height=650, key=f"bih_map_tracker_{st.session_state['map_render_key']}", returned_objects=[])

st.markdown("---")

# 7. Menu Navigation Component (Positioned Underneath Map)
st.subheader("Heritage Site Navigator")

selected_site = st.selectbox(
    "Choose any site from the dataset to instantly trigger its informational box on the map canvas above:",
    options=["-- Select a Site to Trigger Map Popup --"] + list(df['name'].unique())
)

# If the dropdown selection alters, bump the rendering key to force map to refresh with open popup
if selected_site != st.session_state["last_selected"]:
    st.session_state["last_selected"] = selected_site
    st.session_state["map_render_key"] += 1
    st.rerun()
