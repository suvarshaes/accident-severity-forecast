import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier


@st.cache_resource
def load_model():
    """Load CatBoost model from the models directory."""
    model = CatBoostClassifier()
    model.load_model("../models/catboost.cbm")
    return model


@st.cache_data
def load_street_options():
    """Load a list of street names for the UI (fallback to a small list if not available)."""
    try:
        df_sample = pd.read_csv("../data/clean/US_Accidents_March23_NY_Clean.csv", usecols=["Street"], nrows=200000)
        streets = sorted(df_sample["Street"].dropna().unique().tolist())
        return streets[:500] if streets else ["Unknown"]
    except Exception:
        return ["Main St", "Broadway", "1st Ave", "State St", "Market St"]
