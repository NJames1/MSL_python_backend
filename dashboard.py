import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime

st.set_page_config(page_title="MSL Lecturer Dashboard", layout="wide")

# Database Connection (Ensure you use the EXTERNAL URL)
DB_URL = "postgresql://msl_db_bx39_user:NfV5fQ7kLojvg920DrpmcqLg8RwLFMEQ@dpg-d71ak3vgi27c73fav4ug-a.oregon-postgres.render.com/msl_db_bx39"
engine = create_engine(DB_URL)

# 1. FORCE CACHE CLEAR (This is the most important part)
@st.cache_data(ttl=5) # Reduced to 5 seconds for live debugging
def get_live_data():
    try:
        return pd.read_sql("SELECT * FROM raw_scans WHERE timestamp >= CURRENT_DATE", engine)
    except Exception as e:
        st.error(f"Database Query Error: {e}")
        return pd.DataFrame()

df = get_live_data()

st.title("👨‍🏫 Lecturer Command Center")
st.markdown("---")

if not df.empty:
    # 2. SAFETY CHECK: If column is missing, create a dummy one so the app doesn't crash
    if 'proximity_verified' not in df.columns:
        st.warning("⚠️ Column 'proximity_verified' not found in DB. Showing raw data instead.")
        df['proximity_verified'] = False # Temporary dummy column
    
    if 'user_name' not in df.columns:
         df['user_name'] = "Unknown"

    # Metrics Calculations
    total_students = df['user_name'].nunique()
    verified_df = df[df['proximity_verified'] == True].drop_duplicates(subset=['user_name'])
    flagged_df = df[df['proximity_verified'] == False].drop_duplicates(subset=['user_name'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", total_students)
    with col2:
        if st.button(f"✅ In Proximity: {len(verified_df)}"):
            st.session_state.view = "verified"
    with col3:
        if st.button(f"🚩 Outside Pool: {len(flagged_df)}"):
            st.session_state.view = "flagged"
    with col4:
        rate = (len(verified_df) / total_students * 100) if total_students > 0 else 0
        st.metric("Integrity Rate", f"{int(rate)}%")

    # Activity Chart
    st.subheader("Real-Time Engagement")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    fig = px.line(df.resample('15min', on='timestamp').count().reset_index(), 
                 x='timestamp', y='id', title="Scan Volume", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Drill-Down Logic
    if 'view' in st.session_state:
        st.divider()
        view_type = st.session_state.view
        st.write(f"### {'Verified Students' if view_type == 'verified' else 'Flagged Devices'}")
        display_df = verified_df if view_type == 'verified' else flagged_df
        st.dataframe(display_df, use_container_width=True)
else:
    st.info("📡 Dashboard connected. Waiting for scans from the American Wing...")