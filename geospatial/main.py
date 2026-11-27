import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import numpy as np
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="US Accidents Analysis - New Jersey",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("US Traffic Accidents Analysis - New Jersey")
st.markdown("""
This application visualizes traffic accident data in New Jersey, helping identify patterns and high-risk areas.
""")

@st.cache_data
def load_data():
    """Load and cache the accident data"""
    try:
        df = pd.read_csv('../data/raw/US_Accidents_March23_NJ.csv')
        df['Start_Time'] = pd.to_datetime(df['Start_Time'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = df['Start_Time'].min()
    max_date = df['Start_Time'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    # Severity filter
    severity_options = sorted(df['Severity'].unique())
    selected_severity = st.sidebar.multiselect(
        "Select Severity Levels",
        severity_options,
        default=severity_options
    )
    
    # Filter data based on selections
    mask = (
        (df['Start_Time'].dt.date >= date_range[0]) &
        (df['Start_Time'].dt.date <= date_range[1]) &
        (df['Severity'].isin(selected_severity))
    )
    filtered_df = df[mask]
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📍 Map View", "📊 Time Analysis", "📈 Statistics"])
    
    with tab1:
        st.subheader("Accident Locations")
        
        # Create map centered on NJ
        m = folium.Map(
            location=[filtered_df['Start_Lat'].mean(), filtered_df['Start_Lng'].mean()],
            zoom_start=8
        )
        
        # Add markers for accidents
        for idx, row in filtered_df.sample(min(len(filtered_df), 1000)).iterrows():
            folium.CircleMarker(
                [row['Start_Lat'], row['Start_Lng']],
                radius=3,
                color='red',
                fill=True,
                popup=f"Severity: {row['Severity']}<br>Date: {row['Start_Time']}<br>Street: {row['Street']}",
            ).add_to(m)
        
        # Display map
        folium_static(m)
    
    with tab2:
        st.subheader("Temporal Analysis")
        
        # Accidents by hour
        filtered_df['Hour'] = filtered_df['Start_Time'].dt.hour
        hourly_accidents = filtered_df.groupby('Hour').size().reset_index(name='count')
        
        fig_hourly = px.line(
            hourly_accidents,
            x='Hour',
            y='count',
            title='Accidents by Hour of Day'
        )
        st.plotly_chart(fig_hourly)
        
        # Accidents by day of week
        filtered_df['DayOfWeek'] = filtered_df['Start_Time'].dt.day_name()
        daily_accidents = filtered_df.groupby('DayOfWeek').size().reset_index(name='count')
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_accidents['DayOfWeek'] = pd.Categorical(daily_accidents['DayOfWeek'], categories=day_order, ordered=True)
        daily_accidents = daily_accidents.sort_values('DayOfWeek')
        
        fig_daily = px.bar(
            daily_accidents,
            x='DayOfWeek',
            y='count',
            title='Accidents by Day of Week'
        )
        st.plotly_chart(fig_daily)
    
    with tab3:
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Accidents", len(filtered_df))
        
        with col2:
            avg_severity = filtered_df['Severity'].mean()
            st.metric("Average Severity", f"{avg_severity:.2f}")
        
        with col3:
            most_common_weather = filtered_df['Weather_Condition'].mode()[0]
            st.metric("Most Common Weather", most_common_weather)
        
        # Top accident locations
        st.subheader("Top 10 Streets by Number of Accidents")
        top_streets = filtered_df['Street'].value_counts().head(10)
        fig_streets = px.bar(
            x=top_streets.values,
            y=top_streets.index,
            orientation='h',
            title='Top 10 Streets with Most Accidents'
        )
        st.plotly_chart(fig_streets)

else:
    st.error("Unable to load the dataset. Please check if the data file exists and is accessible.")