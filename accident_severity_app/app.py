import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool

st.title("Accident Severity Prediction 🚦")
st.write(
    """
    Predict the severity of traffic accidents (Fatal, Serious, Minor) using road, weather,
    and environmental factors. This location-aware tool is designed for urban planners
    and safety officials to get instant, site-specific risk insights and plan proactive
    interventions.
    """
)

FEATURES = [
    "Distance(mi)",
    "Temperature(F)",
    "Visibility(mi)",
    "Humidity(%)",
    "Pressure(in)",
    "Wind_Speed(mph)",
    "Hour",
    "Duration_min",
    "Street_encoded",
    "County",
    "Weather_Condition",
    "Wind_Direction",
    "Weather_Short",
    "Season",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Turning_Loop",
]

CATEGORICAL_COLS = [
    "Street_encoded",
    "County",
    "Weather_Condition",
    "Wind_Direction",
    "Weather_Short",
    "Season",
]

# Sidebar inputs (defaults are reasonable mid-range values)
st.sidebar.header("Input Features")
distance = st.sidebar.number_input("Distance (mi)", value=1.0, min_value=0.0, step=0.1)
temperature = st.sidebar.number_input("Temperature (F)", value=70.0, step=0.5)
visibility = st.sidebar.number_input("Visibility (mi)", value=10.0, min_value=0.0, step=0.5)
humidity = st.sidebar.number_input("Humidity (%)", value=50.0, min_value=0.0, max_value=100.0, step=1.0)
pressure = st.sidebar.number_input("Pressure (in)", value=30.0, min_value=0.0, step=0.1)
wind_speed = st.sidebar.number_input("Wind Speed (mph)", value=5.0, min_value=0.0, step=0.5)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
duration = st.sidebar.number_input("Duration (minutes)", value=10.0, min_value=0.0, step=1.0)
latitude = st.sidebar.number_input("Latitude (for map)", value=40.7128, format="%.6f")
longitude = st.sidebar.number_input("Longitude (for map)", value=-74.0060, format="%.6f")

# Street options loaded from clean data (falls back to a small list if unavailable)
@st.cache_data
def load_street_options():
    try:
        df_sample = pd.read_csv("../data/clean/US_Accidents_March23_NY_Clean.csv", usecols=["Street"], nrows=200000)
        streets = sorted(df_sample["Street"].dropna().unique().tolist())
        return streets[:500] if streets else ["Unknown"]
    except Exception:
        return ["Main St", "Broadway", "1st Ave", "State St", "Market St"]

street_options = load_street_options()
street = st.sidebar.selectbox("Street", street_options, index=0)
county = st.sidebar.text_input("County", "Middlesex")
weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    ["Clear", "Rain", "Snow", "Fog", "Overcast", "Thunderstorm", "Cloudy", "Drizzle"],
)
wind_direction = st.sidebar.selectbox(
    "Wind Direction",
    ["Calm", "NW", "NNW", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "Variable"],
)
weather_short = st.sidebar.selectbox("Weather Short", ["Clear", "Rain", "Snow", "Fog", "Cloudy", "Other"])
season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])

# Infrastructure flags (0/1)
amenity = st.sidebar.checkbox("Amenity present", value=False)
bump = st.sidebar.checkbox("Bump", value=False)
crossing = st.sidebar.checkbox("Crossing", value=False)
give_way = st.sidebar.checkbox("Give Way", value=False)
junction = st.sidebar.checkbox("Junction", value=False)
no_exit = st.sidebar.checkbox("No Exit", value=False)
railway = st.sidebar.checkbox("Railway", value=False)
roundabout = st.sidebar.checkbox("Roundabout", value=False)
station = st.sidebar.checkbox("Station", value=False)
stop = st.sidebar.checkbox("Stop", value=False)
traffic_calming = st.sidebar.checkbox("Traffic Calming", value=False)
traffic_signal = st.sidebar.checkbox("Traffic Signal", value=False)
turning_loop = st.sidebar.checkbox("Turning Loop", value=False)


@st.cache_resource
def load_model():
    """Load CatBoost model from the models directory."""
    model = CatBoostClassifier()
    model.load_model("../models/catboost.cbm")
    return model


model = load_model()

if st.sidebar.button("Predict Severity"):
    input_row = {
        "Distance(mi)": distance,
        "Temperature(F)": temperature,
        "Visibility(mi)": visibility,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure,
        "Wind_Speed(mph)": wind_speed,
        "Hour": hour,
        "Duration_min": duration,
        "Street_encoded": street,
        "County": county,
        "Weather_Condition": weather_condition,
        "Wind_Direction": wind_direction,
        "Weather_Short": weather_short,
        "Season": season,
        "Amenity": int(amenity),
        "Bump": int(bump),
        "Crossing": int(crossing),
        "Give_Way": int(give_way),
        "Junction": int(junction),
        "No_Exit": int(no_exit),
        "Railway": int(railway),
        "Roundabout": int(roundabout),
        "Station": int(station),
        "Stop": int(stop),
        "Traffic_Calming": int(traffic_calming),
        "Traffic_Signal": int(traffic_signal),
        "Turning_Loop": int(turning_loop),
    }
    input_data = pd.DataFrame([input_row], columns=FEATURES)

    pool = Pool(input_data, cat_features=[col for col in CATEGORICAL_COLS if col in input_data.columns])
    pred_class = model.predict(pool)[0]
    proba = model.predict_proba(pool)[0]

    # Predicted class
    st.success(f"Predicted Severity Class: {pred_class}")

    # Probability distribution visualization
    class_labels = list(model.classes_) if hasattr(model, "classes_") else [f"class_{i}" for i in range(len(proba))]
    proba_df = pd.DataFrame({"Severity": class_labels, "Probability": proba})
    st.subheader("Probability Distribution")
    st.bar_chart(proba_df.set_index("Severity"))

    # Map visualization (if coordinates provided)
    if latitude is not None and longitude is not None:
        st.subheader("Location Map")
        st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}))

    # Top predictive factors for this prediction (per-instance contributions)
    try:
        contribs = model.get_feature_importance(type="PredictionValuesChange", data=pool)
        contrib_row = contribs[0] if hasattr(contribs, "__len__") else contribs
        fi_df = pd.DataFrame({"feature": FEATURES, "contribution": contrib_row})
        fi_df["abs_contribution"] = fi_df["contribution"].abs()
        top_fi = fi_df.sort_values("abs_contribution", ascending=False).head(5)

        st.subheader("Top Factors Driving This Prediction")
        st.bar_chart(top_fi.set_index("feature")["contribution"])

        drivers = [f for f in top_fi[top_fi["contribution"] > 0]["feature"].tolist()]
        if drivers:
            st.info("Risk elevated by: " + ", ".join(drivers))
        else:
            st.info("No strong positive risk drivers detected for this case.")
    except Exception as e:
        st.warning(f"Could not compute feature-level contributions: {e}")
