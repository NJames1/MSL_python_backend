import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect

# 1. Page Configuration (Dark Mode & Wide Layout)
st.set_page_config(page_title="Live Telemetry | MSL", layout="wide", page_icon="📡")

st.title("📡 Power-Optimized Localization")
st.markdown("### Real-Time Data Monitoring Dashboard")
st.markdown("---")

# 2. Database Connection Inputs
col1, col2 = st.columns([3, 1])
with col1:
    db_url_input = st.text_input("Enter Render External Database URL:", type="password")
with col2:
    # Failsafe: You can change this live in the web browser if the table name is different
    table_name = st.text_input("Database Table Name:", value="scans") 

if db_url_input:
    # Fix SQLAlchemy's prefix requirement (postgres:// to postgresql://)
    if db_url_input.startswith("postgres://"):
        db_url_input = db_url_input.replace("postgres://", "postgresql://", 1)

    try:
        # Connect to Render PostgreSQL
        engine = create_engine(db_url_input)
        
        # Built-in diagnostic: Check what tables actually exist in your database
        insp = inspect(engine)
        existing_tables = insp.get_table_names()
        
        if table_name not in existing_tables:
            st.error(f"Table '{table_name}' not found! Found these tables instead: {existing_tables}")
            st.info("Update the 'Database Table Name' box above to match one of the tables found.")
        else:
            # Fetch the latest data
            query = f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 50;"
            df = pd.read_sql(query, engine)

            # Top Control Panel
            col_btn1, col_btn2 = st.columns([8, 1])
            with col_btn2:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    st.rerun()

            # Dashboard Metrics & Tables
            if not df.empty:
                st.success("Secure Connection Established. Streaming live telemetry...")
                
                st.metric("Total Scans Retrieved", len(df))
                
                st.markdown("### 📡 Raw Hardware Payloads")
                # Display the dataframe as an interactive table
                st.dataframe(df, use_container_width=True, height=400)
                
            else:
                st.info("Database connected successfully, but no scans found yet. Press 'Scan' on the Android device!")

    except Exception as e:
        st.error(f"Connection Failed. Check your URL or network. Error: {e}")
else:
    st.warning("Awaiting Database Credentials...")