import streamlit as st


def render_projects():
    st.title("Projects")
    st.markdown(
        """
        - **Accident Severity App** — Location-aware CatBoost model that predicts Major/Moderate/Minor crash severity using weather, time, location, and infrastructure features. Includes scenario generation, intervention what-ifs, and interactive risk maps to support planners and safety officials.
        """
    )
