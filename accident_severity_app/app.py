import streamlit as st

from page_home import render_home
from page_resume import render_resume
from page_projects import render_projects
from page_accident import render_accident_app

st.set_page_config(page_title="Accident Severity & Portfolio", layout="wide")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Resume", "Projects", "Accident Severity App"],
    index=3,
)

if page == "Home":
    render_home()
elif page == "Resume":
    render_resume()
elif page == "Projects":
    render_projects()
else:
    render_accident_app()
