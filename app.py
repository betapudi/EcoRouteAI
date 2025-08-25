import os
import math
import json
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from datetime import datetime
from itertools import product
from scipy.spatial import cKDTree
import lightgbm as lgb
from tqdm import tqdm
import streamlit as st
import pydeck as pdk
import folium
from streamlit_folium import st_folium
import networkx as nx
import osmnx as ox
from utils.preprocessing import preprocess_live_data
from utils.prediction import load_model, predict_eco_scores
from branca.colormap import LinearColormap
import requests
import math

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="EcoRouteAI - Eco-Friendly Routes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# INPUTS
# -----------------------------
city_name = "Vizianagaram"
air_csv_path = "data/airquality_sample/AirQuality_03-07-2025_19-00.csv"
ndvi_tif = "data/ndvi/VZ_ClrSky_NDVI_Mar2025.tif"
elev_tif = "data/elev/VZ_ClrSky_ELEV_Mar2025.tif"

# Load initial data if available
if os.path.exists(air_csv_path):
    live_air_df = pd.read_csv(air_csv_path)
else:
    live_air_df = None

# -----------------------------
# ECOSITES IN VIZIANAGARAM
# -----------------------------
eco_sites_df = pd.DataFrame([
    {"site_name": "Thatipudi Reservoir", "latitude": 18.17210954564298, "longitude": 83.19173474704027},
    {"site_name": "Raamateertham", "latitude": 18.167303, "longitude": 83.491762},
    {"site_name": "Gosthani Sarovar Vihar", "latitude": 18.123, "longitude": 83.4},
    {"site_name": "Bobbili Fort", "latitude": 18.490, "longitude": 83.41},
    {"site_name": "Punyagiri", "latitude": 18.62, "longitude": 83.39},
    {"site_name": "Venkateswara Alayam Govindapuram", "latitude": 18.09, "longitude": 83.35},
    {"site_name": "Kumili", "latitude": 18.133, "longitude": 83.367},
    {"site_name": "Vizianagaram Fort", "latitude": 18.110982908881866, "longitude": 83.41201395826927},
    {"site_name": "Ganta Stambham (Clock Tower)", "latitude": 18.115876, "longitude": 83.409422},
    {"site_name": "Sri. Gurajada Apparao House", "latitude": 18.112176207472082, "longitude": 83.41210045943946},
])
eco_sites_df["geometry"] = eco_sites_df.apply(lambda r: Point(r["longitude"], r["latitude"]), axis=1)
eco_sites_df = gpd.GeoDataFrame(eco_sites_df, geometry="geometry", crs="EPSG:4326")

# -----------------------------
# ROUTE OPTIMIZATION FUNCTIONS
# -----------------------------
import requests
import math

def get_osrm_route(start_lat, start_lon, end_lat, end_lon, profile="driving"):
    """
    Get route from OSRM API (Open Source Routing Machine).
    Uses the public demo server. Consider self-hosting for production.
    """
    try:
        # Use HTTPS to avoid mixed-content blocks in some environments
        url = (
            f"https://router.project-osrm.org/route/v1/{profile}/"
            f"{start_lon},{start_lat};{end_lon},{end_lat}"
            f"?overview=full&geometries=geojson&steps=true&annotations=false"
        )
        resp = requests.get(url, timeout=20)
        if not resp.ok:
            return None

        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None

        route = data["routes"][0]
        coords = route["geometry"]["coordinates"]
        # geojson is [lon, lat] -> folium needs [lat, lon]
        route_geometry = [[c[1], c[0]] for c in coords]

        distance_km = route["distance"] / 1000.0
        duration_min = route["duration"] / 60.0

        # instructions
        instructions = []
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                name = step.get("name") or ""
                mtype = step.get("maneuver", {}).get("type", "")
                distm = step.get("distance", 0.0)
                if name and mtype != "depart":
                    instructions.append(f"{mtype} onto {name} for {distm:.0f} m")

        return {
            "geometry": route_geometry,
            "distance_km": distance_km,
            "duration_min": duration_min,
            "instructions": instructions
        }

    except Exception as e:
        st.warning(f"OSRM routing failed: {e}. Falling back to straight line.")
        distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
        return {
            "geometry": [[start_lat, start_lon], [end_lat, end_lon]],
            "distance_km": distance,
            "duration_min": distance * 2,
            "instructions": [f"Head directly to destination ({distance:.1f} km)"]
        }

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def find_optimal_routes_with_osm(start_point, scored_df, transport_mode, alpha=1.0, beta=0.5, top_n=3):
    """
    Find optimal routes based on eco-score and real OSM routing
    """
    routes = []
    start_lat, start_lon = start_point
    
    for _, site in scored_df.iterrows():
        # Get actual route from OSRM
        route_data = get_osrm_route(start_lat, start_lon, site["latitude"], site["longitude"], transport_mode)
        
        if route_data:
            # Calculate combined score
            eco_normalized = site["pred_eco_score"] / 100
            distance_normalized = 1 / (route_data["distance_km"] + 1)  # Avoid division by zero
            
            # Combined score: higher is better
            score = distance_normalized * alpha + eco_normalized * beta
            
            routes.append({
                "destination": site["site_name"],
                "dest_lat": site["latitude"],
                "dest_lon": site["longitude"],
                "route_geometry": route_data["geometry"],
                "distance_km": route_data["distance_km"],
                "duration_min": route_data["duration_min"],
                "instructions": route_data["instructions"],
                "score": score,
                "eco_score": site["pred_eco_score"]
            })
    
    # Sort by combined score (highest first)
    routes.sort(key=lambda x: x["score"], reverse=True)
    
    return routes[:top_n]

def calculate_route_score(route, scored_df, alpha=1.0, beta=0.5):
    """
    Calculate a combined score for a route considering distance and eco-friendliness.
    Higher scores are better.
    """
    if not route or len(route) < 2:
        return 0
    
    # Get destination site
    dest_site = route[-1]
    site_data = scored_df[scored_df["site_name"] == dest_site].iloc[0]
    
    # Calculate route metrics (simplified for PoC)
    # In a real implementation, you would use actual routing APIs
    distance_km = len(route) * 5  # Placeholder: assume 5km per segment
    
    # Normalize eco score (0-100 to 0-1)
    eco_normalized = site_data["pred_eco_score"] / 100
    
    # Combined score: higher is better
    # alpha weights distance, beta weights eco-friendliness
    score = (1 / (distance_km + 1)) * alpha + eco_normalized * beta
    
    return score

def find_optimal_routes(start_point, scored_df, alpha=1.0, beta=0.5, top_n=3):
    """
    Find optimal routes based on eco-score and distance.
    Returns top N routes with their scores.
    """
    routes = []
    
    for _, site in scored_df.iterrows():
        # Simplified route calculation (in real app, use routing API)
        route = [start_point, site["site_name"]]
        score = calculate_route_score(route, scored_df, alpha, beta)
        
        routes.append({
            "destination": site["site_name"],
            "route": route,
            "score": score,
            "eco_score": site["pred_eco_score"],
            "distance_km": len(route) * 5  # Placeholder
        })
    
    # Sort by combined score (highest first)
    routes.sort(key=lambda x: x["score"], reverse=True)
    
    return routes[:top_n]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🌱 EcoRouteAI")
st.sidebar.subheader("Navigation")

page = st.sidebar.radio("Go to", ["📥 Upload Data", "📊 Dashboard", "🗺️ Map View", "🛣️ Route Planner", "⚙️ Settings"])

eco_threshold = st.sidebar.slider("EcoScore Threshold", 0, 100, 75)
aggregate_hours = st.sidebar.checkbox("Aggregate Hours", value=True)

# -----------------------------
# COLOR SCALING FOR ECOSCORE
# -----------------------------
def score_to_color(score):
    """
    Maps eco_score (0–100) to a color scale from dark red to dark green.
    """
    # clamp value
    score = max(0, min(100, score))
    # interpolate R (red goes down), G (green goes up)
    r = int(255 - (score * 2.55))   # from 255 → 0
    g = int(score * 2.55)           # from 0 → 255
    b = 60                          # constant to keep dark tone
    return f"rgb({r},{g},{b})"

# -----------------------------
# MAIN CONTENT
# -----------------------------
if page == "📥 Upload Data":
    st.title("📥 Upload Air Quality Data")

    uploaded_file = st.file_uploader("Upload Air Quality CSV", type=["csv"])

    # Case 1: User uploads a CSV
    if uploaded_file is not None:
        st.session_state["air_data"] = pd.read_csv(uploaded_file)
        st.success("✅ Data uploaded successfully!")
        st.write(st.session_state["air_data"].head())

    # Case 2: User clicks "Use Example Data"
    if st.button("Use Example Data (Vizianagaram)"):
        if live_air_df is not None:
            st.session_state["air_data"] = live_air_df.copy()
            st.success("✅ Example data loaded successfully!")
            st.write(st.session_state["air_data"].head())
        else:
            st.warning("⚠️ Example data file not found. Please upload your own data.")

    # If neither happened yet
    if "air_data" not in st.session_state:
        st.info("ℹ️ Please upload a CSV file or click 'Use Example Data (Vizianagaram)'.")

elif page == "📊 Dashboard":
    st.title("📊 EcoScore Dashboard")

    if "air_data" not in st.session_state:
        st.warning("⚠️ Please upload data first in 'Upload Data' tab.")
    else:
        # Preprocess + Predict
        try:
            model, features = load_model("models/eco_score_lightgbm_final.pkl")
            processed_df = preprocess_live_data(city_name, st.session_state["air_data"], 
                                                ndvi_tif, elev_tif, eco_sites_df, aggregate_hours=False)
            scored_df = predict_eco_scores(model, features, processed_df)

            # Filter
            filtered = scored_df[scored_df["pred_eco_score"] >= eco_threshold]

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average EcoScore", round(scored_df["pred_eco_score"].mean(), 2))
            with col2:
                st.metric("Best Site Score", round(filtered["pred_eco_score"].max(), 2))
            with col3:
                best_site = filtered.loc[filtered["pred_eco_score"].idxmax(), "site_name"]
                st.metric("Best Site", best_site)

            # EcoScore distribution
            st.subheader("EcoScore Distribution")
            st.bar_chart(scored_df.set_index("site_name")["pred_eco_score"])

            # Site ranking table
            st.subheader("EcoScore by Site (Ranked)")
            ranked_sites = scored_df[["site_name", "pred_eco_score"]].sort_values("pred_eco_score", ascending=False)
            ranked_sites["Rank"] = range(1, len(ranked_sites) + 1)
            ranked_sites = ranked_sites[["Rank", "site_name", "pred_eco_score"]]
            st.dataframe(ranked_sites.style.background_gradient(subset=["pred_eco_score"], cmap="RdYlGn"))
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Make sure the model file exists at 'models/eco_score_lightgbm_final.pkl'")

elif page == "🗺️ Map View":
    st.title("🗺️ Eco-Friendly Map")
    
    if "air_data" not in st.session_state:
        st.warning("⚠️ Please upload data first in 'Upload Data' tab.")
    else:
        try:
            model, features = load_model("models/eco_score_lightgbm_final.pkl")
            processed_df = preprocess_live_data(city_name, st.session_state["air_data"], 
                                                ndvi_tif, elev_tif, eco_sites_df, aggregate_hours=False)
            scored_df = predict_eco_scores(model, features, processed_df)

            # Create Folium Map        
            m = folium.Map(location=[scored_df["latitude"].mean(),
                                    scored_df["longitude"].mean()],
                        zoom_start=12,
                        tiles="OpenStreetMap")

            # Add eco-sites with color by score
            bounds = []
            for _, row in scored_df.iterrows():
                color = score_to_color(row["pred_eco_score"])
                folium.CircleMarker(
                    [row["latitude"], row["longitude"]],
                    radius=10,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=folium.Popup(
                        f"<b>{row['site_name']}</b><br>"
                        f"EcoScore: {row['pred_eco_score']:.2f}<br>"
                        f"PM2.5: {row.get('PM2_5_idw', 'N/A'):.2f}<br>"
                        f"NDVI: {row.get('ndvi_mean', 'N/A'):.3f}",
                        max_width=250
                    )
                ).add_to(m)
                bounds.append([row["latitude"], row["longitude"]])

            # Fit map to all markers
            if bounds:
                m.fit_bounds(bounds)

            # Add eco-score legend (red→green)
            colormap = LinearColormap(
                colors=['darkred', 'orange', 'yellow', 'green', 'darkgreen'],
                vmin=0, vmax=100,
                caption="Eco Score"
            )
            colormap.add_to(m)

            # Display in Streamlit
            st_folium(m, width=1000, height=600)
            
            # Add site details below map
            st.subheader("Site Details")
            display_df = scored_df[["site_name", "pred_eco_score", "PM2_5_idw", "PM10_idw", "CO2_idw", "ndvi_mean"]].copy()
            display_df.columns = ["Site Name", "Eco Score", "PM2.5", "PM10", "CO2", "NDVI"]
            st.dataframe(display_df.sort_values("Eco Score", ascending=False))
            
        except Exception as e:
            st.error(f"Error generating map: {str(e)}")

elif page == "🛣️ Route Planner":
    st.title("🛣️ Eco-Friendly Route Planner with OSM Routing")
    
    if "air_data" not in st.session_state:
        st.warning("⚠️ Please upload data first in 'Upload Data' tab.")
        st.stop()
    
    try:
        # Load model and data only once using session state
        if "model" not in st.session_state:
            with st.spinner("Loading model and processing data..."):
                model, features = load_model("models/eco_score_lightgbm_final.pkl")
                processed_df = preprocess_live_data(city_name, st.session_state["air_data"], 
                                                    ndvi_tif, elev_tif, eco_sites_df, aggregate_hours=False)
                scored_df = predict_eco_scores(model, features, processed_df)
                st.session_state.model = model
                st.session_state.features = features
                st.session_state.scored_df = scored_df
        else:
            model = st.session_state.model
            features = st.session_state.features
            scored_df = st.session_state.scored_df
        
        # Route planning options
        st.sidebar.subheader("Route Planning Options")
        transport_mode = st.sidebar.selectbox(
            "Transport Mode", 
            ["driving", "cycling", "walking"],
            help="Select your preferred mode of transportation",
            key="transport_mode"
        )
        
        alpha = st.sidebar.slider("Distance Weight (α)", 0.0, 3.0, 1.0, 0.1, key="alpha")
        beta = st.sidebar.slider("Eco Score Weight (β)", 0.0, 3.0, 1.5, 0.1, key="beta")
        avoid_pollution = st.sidebar.checkbox("Avoid Polluted Areas", value=True, key="avoid_pollution")
        
        # Location input methods
        st.subheader("Select Your Starting Point")
        location_method = st.radio("Location input method:", 
                                  ["Enter Coordinates", "Click on Map", "Use My Location"],
                                  key="location_method")
        
        # Initialize session state for routes if not exists
        if "routes" not in st.session_state:
            st.session_state.routes = None
        if "start_point" not in st.session_state:
            st.session_state.start_point = None
        
        start_lat, start_lon = None, None
        
        if location_method == "Enter Coordinates":
            col1, col2 = st.columns(2)
            with col1:
                start_lat = st.number_input("Start Latitude", value=float(scored_df["latitude"].mean()), format="%.6f", key="start_lat")
            with col2:
                start_lon = st.number_input("Start Longitude", value=float(scored_df["longitude"].mean()), format="%.6f", key="start_lon")
        
        elif location_method == "Click on Map":
            st.info("Click on the map below to select your starting location")
            # Create a map for clicking
            m = folium.Map(location=[scored_df["latitude"].mean(), scored_df["longitude"].mean()],
                          zoom_start=12, tiles="OpenStreetMap")
            
            # Add eco-sites with color coding
            for _, row in scored_df.iterrows():
                color = score_to_color(row["pred_eco_score"])
                folium.CircleMarker(
                    [row["latitude"], row["longitude"]],
                    radius=10,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(
                        f"<b>{row['site_name']}</b><br>"
                        f"EcoScore: {row['pred_eco_score']:.2f}<br>"
                        f"PM2.5: {row.get('PM2_5_idw', 'N/A'):.2f}",
                        max_width=250
                    )
                ).add_to(m)
            
            # Add click handler
            m.add_child(folium.LatLngPopup())
            
            # Display map - use a unique key to preserve state
            map_data = st_folium(m, width=700, height=400, key="click_map")
            
            if map_data and map_data.get("last_clicked"):
                start_lat = map_data["last_clicked"]["lat"]
                start_lon = map_data["last_clicked"]["lng"]
                st.session_state.last_clicked = (start_lat, start_lon)
                st.success(f"Selected location: {start_lat:.6f}, {start_lon:.6f}")
            elif "last_clicked" in st.session_state:
                start_lat, start_lon = st.session_state.last_clicked
                st.info(f"Using previously selected location: {start_lat:.6f}, {start_lon:.6f}")
        
        elif location_method == "Use My Location":
            # For demo purposes - in a real app, you'd use JavaScript geolocation
            if st.button("Simulate Current Location", key="simulate_loc"):
                start_lat = scored_df["latitude"].mean() + 0.01
                start_lon = scored_df["longitude"].mean() + 0.01
                st.session_state.simulated_loc = (start_lat, start_lon)
                st.success(f"Simulated location: {start_lat:.6f}, {start_lon:.6f}")
            elif "simulated_loc" in st.session_state:
                start_lat, start_lon = st.session_state.simulated_loc
                st.info(f"Using simulated location: {start_lat:.6f}, {start_lon:.6f}")
        
        # Route calculation
        calculate_routes = False
        if start_lat and start_lon:
            st.session_state.start_point = (start_lat, start_lon)
            st.success(f"Starting from: {start_lat:.6f}, {start_lon:.6f}")
            
            # Use a form to batch the route calculation
            with st.form("route_calculation_form"):
                calculate_routes = st.form_submit_button("Find Optimal Eco Routes")
            
            if calculate_routes:
                with st.spinner("Calculating optimal eco-routes..."):
                    routes = find_optimal_routes_with_osm(
                        (start_lat, start_lon), 
                        scored_df, 
                        transport_mode,
                        alpha, 
                        beta, 
                        top_n=3
                    )
                
                # Store routes in session state
                st.session_state.routes = routes
                st.session_state.routes_calculated = True
                st.rerun()  # Trigger a single rerun to display results
        
        # Display routes if they exist in session state
        if st.session_state.get("routes_calculated", False) and st.session_state.routes:
            st.subheader("Recommended Eco-Friendly Routes")
            
            for i, route in enumerate(st.session_state.routes, 1):
                with st.expander(f"Route #{i}: {route['destination']} (Score: {route['score']:.2f})"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Eco Score", f"{route['eco_score']:.2f}")
                    with col2:
                        st.metric("Distance", f"{route['distance_km']:.1f} km")
                    with col3:
                        st.metric("Duration", f"{route['duration_min']:.1f} min")
                    with col4:
                        st.metric("Route Score", f"{route['score']:.2f}")
                    
                    # Create map with actual OSM route
                    route_map = folium.Map(
                        location=[st.session_state.start_point[0], st.session_state.start_point[1]], 
                        zoom_start=12,
                        tiles="OpenStreetMap"
                    )
                    
                    # Add start marker
                    folium.Marker(
                        [st.session_state.start_point[0], st.session_state.start_point[1]],
                        popup="Start Location",
                        icon=folium.Icon(color="green", icon="user", prefix="fa")
                    ).add_to(route_map)
                    
                    # Add destination marker
                    folium.Marker(
                        [route['dest_lat'], route['dest_lon']],
                        popup=f"{route['destination']} (Eco: {route['eco_score']:.2f})",
                        icon=folium.Icon(color="blue", icon="flag", prefix="fa")
                    ).add_to(route_map)
                    
                    # Add the actual route from OSM
                    if route['route_geometry']:
                        folium.PolyLine(
                            route['route_geometry'],
                            color="blue",
                            weight=5,
                            opacity=0.7,
                            popup=f"Route to {route['destination']} - {route['distance_km']:.1f} km"
                        ).add_to(route_map)
                    
                    # Display the map with a unique key for each route
                    st_folium(route_map, width=700, height=400, key=f"route_map_{i}")
                    
                    # Add route instructions
                    if route['instructions']:
                        st.subheader("Route Instructions")
                        for j, instruction in enumerate(route['instructions'][:10], 1):  # Show first 10 instructions
                            st.write(f"{j}. {instruction}")
                        if len(route['instructions']) > 10:
                            st.info(f"... and {len(route['instructions']) - 10} more instructions")
        
        # Show all sites for reference
        st.subheader("All Eco Sites")
        display_df = scored_df[["site_name", "pred_eco_score", "latitude", "longitude"]].copy()
        display_df["Eco Score"] = display_df["pred_eco_score"].round(2)
        display_df = display_df.rename(columns={
            "site_name": "Site Name",
            "latitude": "Latitude",
            "longitude": "Longitude"
        }).drop(columns=["pred_eco_score"])
        
        st.dataframe(
            display_df.sort_values("Eco Score", ascending=False),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error in route planning: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.info("For full routing functionality, ensure you have a stable internet connection")

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.write("Model: eco_score_lightgbm_final.pkl")
    st.write("Interpolation: IDW (k=5, power=2)")
    st.write("Mode: Aggregate" if aggregate_hours else "Mode: Per-hour")
    
    # Model information
    st.subheader("Model Information")
    if st.button("Load Model Details"):
        try:
            model, features = load_model("models/eco_score_lightgbm_final.pkl")
            st.success("Model loaded successfully!")
            st.write(f"Number of features: {len(features)}")
            st.write("Features:", features)
        except:
            st.error("Could not load model. Please check the file path.")
    
    # Data information
    st.subheader("Data Information")
    if "air_data" in st.session_state:
        st.write(f"Data shape: {st.session_state['air_data'].shape}")
        st.write("Columns:", list(st.session_state['air_data'].columns))
    else:
        st.info("No data uploaded yet.")
    
    st.info("Future: toggle interpolation method, update model dynamically.")