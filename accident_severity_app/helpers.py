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


@st.cache_data
def load_street_encoder():
    """Return mapping from street name to encoded value used during training."""
    try:
        df_map = pd.read_csv("../data/clean/US_Accidents_March23_NY_Clean.csv", usecols=["Street", "Street_encoded"])
        df_map = df_map.dropna(subset=["Street"])
        mapping = dict(zip(df_map["Street"], df_map["Street_encoded"]))
        return mapping
    except Exception:
        # Minimal fallback: unknown streets map to 0
        return {}


@st.cache_data
def load_county_options():
    """Provide counties seen in training data to avoid unseen categories."""
    try:
        df_counties = pd.read_csv("../data/clean/US_Accidents_March23_NY_Clean.csv", usecols=["County"], nrows=200000)
        counties = sorted(df_counties["County"].dropna().unique().tolist())
        return counties[:200] if counties else ["Unknown"]
    except Exception:
        return ["Middlesex", "Kings", "Queens", "Bronx", "New York"]


@st.cache_data
def load_city_options():
    """Provide cities seen in training data to avoid unseen categories."""
    try:
        df_cities = pd.read_csv("../data/clean/US_Accidents_March23_NY_Clean.csv", usecols=["City"], nrows=200000)
        cities = sorted(df_cities["City"].dropna().unique().tolist())
        return cities[:200] if cities else ["Unknown"]
    except Exception:
        return ["New York", "Brooklyn", "Queens", "Bronx", "Rochester"]


@st.cache_data
def load_city_encoder():
    """Return mapping from city name to encoded value used during training."""
    try:
        df_map = pd.read_csv("../data/clean/US_Accidents_March23_NY_Clean.csv", usecols=["City", "City_encoded"])
        df_map = df_map.dropna(subset=["City"])
        mapping = dict(zip(df_map["City"], df_map["City_encoded"]))
        return mapping
    except Exception:
        return {}


@st.cache_data
def load_category_options():
    """Load categorical vocabularies used during training to avoid unseen categories at inference."""
    defaults = {
        "Weather_Condition": ["Clear", "Rain", "Snow", "Fog", "Overcast", "Thunderstorm", "Cloudy", "Drizzle"],
        "Wind_Direction_clean": ["Calm", "NW", "NNW", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "Variable"],
        "Weather_Short": ["Clear", "Rain", "Snow", "Fog", "Cloudy", "Other"],
        "Season": ["Winter", "Spring", "Summer", "Fall"],
    }
    try:
        df = pd.read_csv(
            "../data/clean/US_Accidents_March23_NY_Clean.csv",
            usecols=["Weather_Condition", "Wind_Direction_clean", "Weather_Short", "Season"],
            nrows=300000,
        )
        return {
            "Weather_Condition": sorted(df["Weather_Condition"].dropna().unique().tolist()) or defaults["Weather_Condition"],
            "Wind_Direction_clean": sorted(df["Wind_Direction_clean"].dropna().unique().tolist()) or defaults["Wind_Direction_clean"],
            "Weather_Short": sorted(df["Weather_Short"].dropna().unique().tolist()) or defaults["Weather_Short"],
            "Season": sorted(df["Season"].dropna().unique().tolist()) or defaults["Season"],
        }
    except Exception:
        return defaults
