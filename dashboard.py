import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect

# 1. Page Configuration (Midnight & Teal Engineering Theme)
st.set_page_config(
    page_title="MSL Mission Control", 
    layout="wide", 
    page_icon="📡",
    initial_sidebar_state="collapsed"
)

# Custom CSS for that "Engineering" Look
st.markdown("""
    <style>
    .main { background-color: #0F172A; }
    .stMetric { background-color: #1E293B; padding: 15px; border-radius: 10px; border-left: 5px solid #2DD4BF; }
    </style>
""", unsafe_allow_escaping=True)

st.title("🛰️ MSL Identity-Aware Localization")
st.markdown("### Power-Optimized Mobile Localization System")
st.markdown("---")

# 2. Database Connection Handling
db_url_input = st.sidebar.text_input("Render DB URL", type="password")

if db_url_input:
    # Prefix fix for SQLAlchemy
    if db_url_input.startswith("postgres://"):
        db_url_input = db_url_input.replace("postgres://", "postgresql://", 1)

    try:
        engine = create_engine(db_url_input)
        insp = inspect(engine)
        existing_tables = insp.get_table_names()

        # --- NEW NAVIGATION HUB (Clickable Icons) ---
        st.markdown("#### 🛠️ Select Data View")
        
        # Mapping our tables to professional icons and names
        # Adjust names ('raw_scans', etc.) to match your models.py
        view_map = {
            "📡 Live Scans": "raw_scans",
            "📱 Registered Devices": "devices",
            "📍 ML Fingerprints": "fingerprints"
        }
        
        # Filter only tables that actually exist in your DB
        available_views = [k for k, v in view_map.items() if v in existing_tables]
        
        if not available_views:
            st.error(f"No project tables found! Detected: {existing_tables}")
            selected_view = None
        else:
            # Horizontal Selection (Segmented Control / Radio)
            selected_label = st.radio(
                "Navigate Database:", 
                available_views, 
                horizontal=True,
                label_visibility="collapsed"
            )
            selected_view = view_map[selected_label]

        # 3. Data Retrieval & Display
        if selected_view:
            query = f"SELECT * FROM {selected_view} ORDER BY id DESC LIMIT 100;"
            df = pd.read_sql(query, engine)

            # Metric Bar
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Entries in View", len(df))
            with m2:
                # Count unique people if user_name column exists
                unique_users = df['user_name'].nunique() if 'user_name' in df.columns else "N/A"
                st.metric("Active Personnel", unique_users)
            with m3:
                st.button("🔄 Force Refresh", on_click=lambda: st.rerun(), use_container_width=True)

            # Interactive Table
            st.markdown(f"### {selected_label} Data Output")
            
            # Formatting the dataframe for better aesthetics
            if not df.empty:
                # If we are looking at scans, let's highlight the Identity column
                st.dataframe(
                    df.style.highlight_max(axis=0, subset=['user_name'] if 'user_name' in df.columns else []),
                    use_container_width=True, 
                    height=500
                )
            else:
                st.info("Table is empty. Send a scan from the Android app to begin.")

    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")

else:
    st.info("🗝️ Enter your Render Database URL in the sidebar to begin monitoring.")
    st.image("https://img.icons8.com/clouds/500/satellite.png", width=250)