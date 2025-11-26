import streamlit as st
import pandas as pd
from catboost import Pool

from helpers import load_model, load_street_options


def render_accident_app():
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

    # Sidebar inputs
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

        class_labels = list(model.classes_) if hasattr(model, "classes_") else [f"class_{i}" for i in range(len(proba))]
        proba_df = pd.DataFrame({"Severity": class_labels, "Probability": proba})
        try:
            pred_idx = class_labels.index(pred_class)
            pred_conf = proba[pred_idx]
        except Exception:
            pred_conf = max(proba)

        st.success(f"Predicted Severity: {pred_class} (confidence: {pred_conf:.1%})")

        st.subheader("Probability Distribution")
        st.bar_chart(proba_df.set_index("Severity"))

        if latitude is not None and longitude is not None:
            st.subheader("Location Map")
            st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}))

        explain = st.checkbox("Show why this result? (top factors)", value=True)
        if explain:
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

        # Explainability dashboard (SHAP-style contribution summary)
        try:
            st.subheader("Explainability Dashboard")
            st.caption("Per-instance contribution summary (SHAP-style). Interpret as relative push toward higher severity.")
            # Normalize contributions to a simple 0–100 scale for readability
            fi_df_norm = fi_df.copy()
            fi_df_norm["impact(%)"] = fi_df_norm["contribution"] / fi_df_norm["abs_contribution"].sum() * 100
            shap_like = fi_df_norm.sort_values("impact(%)", key=abs, ascending=False).head(5)
            st.bar_chart(shap_like.set_index("feature")["impact(%)"])

            # Plain-language summaries
            summary_lines = []
            for _, row in shap_like.iterrows():
                direction = "increase" if row["contribution"] > 0 else "lower"
                magnitude = abs(row["impact(%)"])
                summary_lines.append(f"{row['feature']} tends to {direction} severity by ~{magnitude:.0f}%.")
            if summary_lines:
                st.info(" | ".join(summary_lines))
        except Exception:
            pass

    # -----------------------
    # Interactive risk maps (synthetic hotspot view; replace with real geospatial outputs as available)
    # -----------------------
    st.subheader("Interactive Risk Maps")
    st.caption("Filter by time, weather, or road type to visualize severity hotspots (placeholder data).")

    hotspot_data = pd.DataFrame(
        [
            {"lat": 40.75, "lon": -73.99, "severity": "Major", "time": "evening", "weather": "Rain", "road": "Arterial"},
            {"lat": 40.73, "lon": -73.98, "severity": "Moderate", "time": "morning", "weather": "Clear", "road": "Arterial"},
            {"lat": 40.71, "lon": -74.01, "severity": "Major", "time": "night", "weather": "Snow", "road": "Highway"},
            {"lat": 40.78, "lon": -73.95, "severity": "Minor", "time": "day", "weather": "Clear", "road": "Local"},
            {"lat": 40.69, "lon": -73.99, "severity": "Moderate", "time": "evening", "weather": "Rain", "road": "Highway"},
            {"lat": 40.77, "lon": -73.97, "severity": "Major", "time": "night", "weather": "Fog", "road": "Arterial"},
            {"lat": 40.72, "lon": -74.02, "severity": "Moderate", "time": "day", "weather": "Cloudy", "road": "Local"},
            {"lat": 40.74, "lon": -73.93, "severity": "Minor", "time": "morning", "weather": "Clear", "road": "Local"},
            {"lat": 40.70, "lon": -73.90, "severity": "Major", "time": "evening", "weather": "Rain", "road": "Highway"},
            {"lat": 40.68, "lon": -73.92, "severity": "Moderate", "time": "night", "weather": "Fog", "road": "Arterial"},
        ]
    )

    time_filter = st.selectbox("Time of day", ["All", "morning", "day", "evening", "night"], index=0)
    weather_filter = st.selectbox("Weather", ["All", "Clear", "Rain", "Snow", "Fog", "Cloudy"], index=0)
    road_filter = st.selectbox("Road type", ["All", "Highway", "Arterial", "Local"], index=0)

    filtered = hotspot_data.copy()
    if time_filter != "All":
        filtered = filtered[filtered["time"] == time_filter]
    if weather_filter != "All":
        filtered = filtered[filtered["weather"] == weather_filter]
    if road_filter != "All":
        filtered = filtered[filtered["road"] == road_filter]

    st.map(filtered[["lat", "lon"]])

    st.dataframe(
        filtered[["severity", "time", "weather", "road", "lat", "lon"]]
        .reset_index(drop=True)
        .rename(columns={"severity": "Severity", "time": "Time", "weather": "Weather", "road": "Road Type"})
    )
