import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="MSL Lecturer Dashboard", layout="wide")

# Custom CSS to mimic the rounded card look from your sample image
st.markdown("""
    <style>
    .metric-card {
        background-color: #fdf2e9;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE CONNECTION ---
# Use your External Connection String from Render
DB_URL = "postgresql://james:YmeXArVGRY19ermS7lXy1Op4fVv00Yro@dpg-d7n6k868bjmc738msds0-a.oregon-postgres.render.com/msl_live_demo"

@st.cache_data(ttl=60) # Refresh data every 60 seconds
def load_data():
    engine = create_engine(DB_URL)
    query = "SELECT * FROM raw_scans WHERE timestamp >= CURRENT_DATE"
    df = pd.read_sql(query, engine)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not connect to Render DB: {e}")
    st.stop()

# --- 3. LOGIC PROCESSING ---
total_scans = len(df)
# Verified = Student's CID matched the Lecturer's CID
verified_df = df[df['proximity_verified'] == True].drop_duplicates(subset=['user_name'])
flagged_df = df[df['proximity_verified'] == False].drop_duplicates(subset=['user_name'])

# --- 4. TOP SECTION: ENGAGEMENT METRICS ---
st.title("👨‍🏫 Lecturer Command Center")
st.write(f"Showing activity for **{datetime.now().strftime('%A, %d %B %Y')}**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Scans", total_scans, delta="Live", delta_color="normal")

with col2:
    if st.button(f"✅ Verified: {len(verified_df)}"):
        st.session_state.view = "verified"

with col3:
    if st.button(f"🚩 Flagged: {len(flagged_df)}"):
        st.session_state.view = "flagged"

with col4:
    attendance_rate = (len(verified_df) / 50 * 100) # Assuming class size of 50
    st.metric("Attendance Rate", f"{attendance_rate}%", delta=f"{len(verified_df)}/50")

# --- 5. MIDDLE SECTION: ACTIVITY CHART ---
st.subheader("Arrival Activity")
if not df.empty:
    # Resample to show scans per hour
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    chart_data = df.set_index('timestamp').resample('1H').count().reset_index()
    
    fig = px.area(chart_data, x='timestamp', y='id', 
                  title="Students entering American Wing over time",
                  color_discrete_sequence=['#ff4b4b'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Waiting for the first scan of the day...")

# --- 6. DRILL-DOWN: STUDENT NAMES ---
if 'view' in st.session_state:
    st.divider()
    if st.session_state.view == "verified":
        st.success("### Students Confirmed in Proximity")
        if not verified_df.empty:
            st.table(verified_df[['user_name', 'timestamp', 'rf_prediction']])
        else:
            st.write("No students verified yet.")
            
    elif st.session_state.view == "flagged":
        st.error("### Devices Outside Proximity Pool")
        st.warning("These devices scanned but their Cell ID does not match the room anchors.")
        if not flagged_df.empty:
            st.table(flagged_df[['user_name', 'timestamp', 'serving_cell']])
        else:
            st.write("No suspicious activity detected.")