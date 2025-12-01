import streamlit as st

from page_home import render_home
from page_resume import render_resume
from page_projects import render_projects
from page_accident import render_accident_app

st.set_page_config(page_title="Accident Severity & Portfolio", layout="wide")

# Style the menu to be bolder and slightly larger
st.markdown(
    """
    <style>
    div[data-testid="stSegmentedControl"] > label {
       padding: 0.65rem 1.25rem;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        font-size: 2.15rem;
        font-weight: 900;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Single-page rendering with a tab-like selector; sidebar appears only on Accident Severity App
page = st.segmented_control(
    "Menu",
    ["Home", "Resume", "Projects", "Accident Severity App"],
    default="Accident Severity App",
    label_visibility="collapsed",
)

if page == "Home":
    render_home()
elif page == "Resume":
    render_resume()
elif page == "Projects":
    render_projects()
else:
    render_accident_app()
