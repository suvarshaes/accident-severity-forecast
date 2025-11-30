import random
from itertools import product

import random
from itertools import product

import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
from catboost import Pool

from helpers import (
    load_model,
    load_street_options,
    load_street_encoder,
    load_county_options,
    load_city_options,
    load_city_encoder,
    load_category_options,
)


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
        "City_encoded",
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
        "City_encoded",
        "County",
        "Weather_Condition",
        "Wind_Direction",
        "Weather_Short",
        "Season",
    ]

    # Sidebar inputs (Planner-focused)
    st.sidebar.header("Planner Input Features")
    latitude = st.sidebar.number_input("Latitude (for map)", value=40.7128, format="%.6f")
    longitude = st.sidebar.number_input("Longitude (for map)", value=-74.0060, format="%.6f")
    street_options = load_street_options()
    street_mapping = load_street_encoder()
    street_max = min(len(street_options), 30)
    street_choices = ["All"] + street_options
    street_selected = st.sidebar.multiselect(
        "Streets",
        street_choices,
        default=["All"],
        help="Select specific streets or choose All (uses up to 30 most common).",
    )
    streets_to_use = street_options[:street_max] if "All" in street_selected or not street_selected else street_selected

    county_options = load_county_options()
    default_county_index = county_options.index("Middlesex") if "Middlesex" in county_options else 0
    county = st.sidebar.selectbox("County", county_options, index=default_county_index)

    city_options = load_city_options()
    city_mapping = load_city_encoder()
    city_max = min(len(city_options), 20)
    city_choices = ["All"] + city_options
    city_selected = st.sidebar.multiselect(
        "Cities",
        city_choices,
        default=["All"],
        help="Select specific cities or choose All (uses up to 20 most common).",
    )
    cities_to_use = city_options[:city_max] if "All" in city_selected or not city_selected else city_selected
    duration = st.sidebar.number_input("Duration (minutes)", value=10.0, min_value=0.0, step=1.0)
    high_risk_preset = st.sidebar.checkbox("Use high-risk preset (low vis, higher wind/humidity)", value=False)
    base_distance = st.sidebar.number_input("Distance (mi)", value=1.0, min_value=0.0, max_value=50.0, step=0.1)
    base_temperature = st.sidebar.number_input("Temperature (F)", value=70.0, min_value=-50.0, max_value=130.0, step=1.0)
    base_visibility = st.sidebar.number_input("Visibility (mi)", value=10.0, min_value=0.1, max_value=30.0, step=0.1)
    base_humidity = st.sidebar.slider("Humidity (%)", value=50, min_value=0, max_value=100, step=1)
    base_pressure = st.sidebar.number_input("Pressure (in)", value=30.0, min_value=25.0, max_value=32.5, step=0.1)
    base_wind_speed = st.sidebar.number_input("Wind Speed (mph)", value=5.0, min_value=0.0, max_value=60.0, step=0.5)
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

    category_options = load_category_options()
    WEATHER_CONDITIONS = category_options["Weather_Condition"]
    WEATHER_CONDITIONS_FAST = WEATHER_CONDITIONS[:5] if len(WEATHER_CONDITIONS) > 5 else WEATHER_CONDITIONS
    WIND_DIRECTIONS = category_options["Wind_Direction"]
    WIND_DIRECTIONS_FAST = WIND_DIRECTIONS[:6] if len(WIND_DIRECTIONS) > 6 else WIND_DIRECTIONS
    WEATHER_SHORTS = category_options["Weather_Short"]
    WEATHER_SHORTS_FAST = WEATHER_SHORTS[:5] if len(WEATHER_SHORTS) > 5 else WEATHER_SHORTS
    SEASONS = category_options["Season"]
    HOURS = list(range(24))

    use_all_hours = st.sidebar.checkbox("Evaluate all 24 hours", value=True)
    hours_to_use = HOURS if use_all_hours else [0, 6, 12, 18]
    use_detailed_weather = st.sidebar.checkbox("Use detailed weather/wind grid", value=True)
    scenario_cap = st.sidebar.slider("Max scenarios to evaluate", 50, 2000, 800, step=50)

    current_flags = {
        "Bump": int(bump),
        "Crossing": int(crossing),
        "Give_Way": int(give_way),
        "Junction": int(junction),
        "No_Exit": int(no_exit),
        "Roundabout": int(roundabout),
        "Stop": int(stop),
        "Traffic_Calming": int(traffic_calming),
        "Traffic_Signal": int(traffic_signal),
        "Turning_Loop": int(turning_loop),
    }

    intervention_map = {
        "Add bump": {"Bump": 1, "Traffic_Calming": 1},
        "Add crossing": {"Crossing": 1, "Give_Way": 1},
        "Add give-way": {"Give_Way": 1},
        "Add junction": {"Junction": 1},
        "Mark no-exit": {"No_Exit": 1},
        "Add roundabout": {"Roundabout": 1},
        "Add stop sign": {"Stop": 1},
        "Add traffic calming": {"Traffic_Calming": 1},
        "Add traffic signal": {"Traffic_Signal": 1},
        "Add turning loop": {"Turning_Loop": 1},
    }

    # Filter out interventions that are already present in the user's inputs
    available_interventions = {
        name: overrides
        for name, overrides in intervention_map.items()
        if any(current_flags.get(flag, 0) == 0 for flag in overrides)
    }

    intervention_choices = st.sidebar.multiselect(
        "Try infrastructure interventions",
        list(available_interventions.keys()),
        default=list(available_interventions.keys())[:2],
        help="Simulate adding calming or control measures and see how severity changes. Options already present are hidden.",
    )

    weather_conditions_list = WEATHER_CONDITIONS if use_detailed_weather else WEATHER_CONDITIONS_FAST
    weather_shorts_list = WEATHER_SHORTS if use_detailed_weather else WEATHER_SHORTS_FAST
    wind_directions_list = WIND_DIRECTIONS if use_detailed_weather else WIND_DIRECTIONS_FAST

    available_weather_conditions = sorted(set(WEATHER_CONDITIONS + weather_conditions_list))
    available_weather_shorts = sorted(set(WEATHER_SHORTS + weather_shorts_list))
    available_wind_directions = sorted(set(WIND_DIRECTIONS + wind_directions_list))

    selected_weathers = st.sidebar.multiselect(
        "Weather conditions to test (from CSV values)",
        available_weather_conditions,
        default=available_weather_conditions,
    ) or available_weather_conditions
    selected_weather_shorts = st.sidebar.multiselect(
        "Weather short categories (from CSV values)",
        available_weather_shorts,
        default=available_weather_shorts,
    ) or available_weather_shorts
    selected_winds = st.sidebar.multiselect(
        "Wind directions (from CSV values)",
        available_wind_directions,
        default=available_wind_directions,
    ) or available_wind_directions
    selected_seasons = st.sidebar.multiselect(
        "Seasons",
        SEASONS,
        default=SEASONS,
    ) or SEASONS
    selected_hours = st.sidebar.multiselect(
        "Hours of day",
        HOURS,
        default=HOURS if use_all_hours else hours_to_use,
    ) or (HOURS if use_all_hours else hours_to_use)

    def build_scenarios(hours, weather_conditions, weather_shorts, wind_directions, seasons, streets, cities, cap):
        combos = list(product(hours, weather_conditions, weather_shorts, wind_directions, seasons, streets, cities))
        if cap and len(combos) > cap:
            random.seed(42)
            combos = random.sample(combos, cap)
        return combos

    def choose_worst_label(counts_series):
        """Pick worst severity among Major > Moderate > Minor; break ties by frequency."""
        severity_priority = {"major": 3, "moderate": 2, "minor": 1}

        def score(label, count):
            priority = severity_priority.get(str(label).lower(), 0)
            return (priority, count)

        best_label, _ = max(counts_series.items(), key=lambda item: score(item[0], item[1]))
        return best_label

    scenarios = build_scenarios(
        selected_hours,
        selected_weathers,
        selected_weather_shorts,
        selected_winds,
        selected_seasons,
        streets_to_use,
        cities_to_use,
        scenario_cap,
    )

    def run_scenarios(scenario_list, overrides):
        results = []
        for hour, weather_condition, weather_short, wind_direction, season, street_name, city_name in scenario_list:
            street_code = street_mapping.get(street_name, 0)
            city_code = city_mapping.get(city_name, 0)
            base_conditions = {
                "Distance(mi)": 0.2 if high_risk_preset else base_distance,
                "Temperature(F)": 32.0 if high_risk_preset else base_temperature,
                "Visibility(mi)": 1.0 if high_risk_preset else base_visibility,
                "Humidity(%)": 90 if high_risk_preset else base_humidity,
                "Pressure(in)": 28.5 if high_risk_preset else base_pressure,
                "Wind_Speed(mph)": 25.0 if high_risk_preset else base_wind_speed,
            }
            input_row = {
                "Distance(mi)": base_conditions["Distance(mi)"],
                "Temperature(F)": base_conditions["Temperature(F)"],
                "Visibility(mi)": base_conditions["Visibility(mi)"],
                "Humidity(%)": base_conditions["Humidity(%)"],
                "Pressure(in)": base_conditions["Pressure(in)"],
                "Wind_Speed(mph)": base_conditions["Wind_Speed(mph)"],
                "Hour": hour,
                "Duration_min": duration,
                "Street_encoded": street_code,
                "City_encoded": city_code,
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
            input_row.update(overrides)
            input_data = pd.DataFrame([input_row], columns=FEATURES)
            pool = Pool(input_data, cat_features=[col for col in CATEGORICAL_COLS if col in input_data.columns])
            pred_class = model.predict(pool, prediction_type="Class")[0]
            if isinstance(pred_class, (list, tuple, np.ndarray)):
                pred_class = pred_class[0]
            proba = model.predict_proba(pool)[0]
            class_labels = list(model.classes_) if hasattr(model, "classes_") else [f"class_{i}" for i in range(len(proba))]
            results.append({
                "Weather_Condition": weather_condition,
                "Weather_Short": weather_short,
                "Wind_Direction": wind_direction,
                "Season": season,
                "Hour": hour,
                "Street": street_name,
                "City": city_name,
                "Predicted_Severity": str(pred_class),
                "Probabilities": dict(zip(class_labels, proba))
            })
        return pd.DataFrame(results)

    if st.sidebar.button("Predict Severity Across Weather Scenarios"):
        baseline_df = run_scenarios(scenarios, {})
        baseline_labels = baseline_df["Predicted_Severity"].astype(str)
        severity_counts = baseline_labels.value_counts()
        worst_label = choose_worst_label(severity_counts)
        prob_keys = sorted({k for probs in baseline_df["Probabilities"] for k in probs.keys()})
        baseline_prob_mean = (
            pd.DataFrame(list(baseline_df["Probabilities"]))
            .reindex(columns=prob_keys)
            .mean()
        )

        st.subheader("Severity Predictions Summary")
        col1, col2 = st.columns(2)
        col1.metric("Most common outcome", worst_label, delta=None)
        col2.metric("Scenario count", len(baseline_df))
        st.bar_chart(severity_counts)
        st.caption("Average predicted probabilities (baseline)")
        st.bar_chart(baseline_prob_mean)
        if hasattr(model, "classes_"):
            st.caption(f"Model classes: {', '.join(map(str, model.classes_))}")

        top_risky = baseline_df[baseline_labels == worst_label].copy()
        top_risky["Confidence"] = top_risky["Probabilities"].apply(lambda d: max(d.values()) if isinstance(d, dict) else 0)
        st.write("Top scenarios leading to highest severity (sorted by confidence):")
        st.dataframe(
            top_risky.sort_values("Confidence", ascending=False)
            .head(10)[["Weather_Condition", "Weather_Short", "Wind_Direction", "Season", "Hour", "Street", "City", "Predicted_Severity", "Confidence"]]
        )

        focus_top = top_risky.sort_values("Confidence", ascending=False).head(20)
        high_risk_scenarios = list(
            focus_top[["Hour", "Weather_Condition", "Weather_Short", "Wind_Direction", "Season", "Street", "City"]].itertuples(index=False, name=None)
        )

        with st.expander("Full scenario table (filtered)"):
            st.dataframe(
                baseline_df.assign(Predicted_Severity=baseline_labels)[
                    ["Weather_Condition", "Weather_Short", "Wind_Direction", "Season", "Hour", "Predicted_Severity"]
                ]
            )

        st.subheader("What-if: intervention suggestions")
        suggestions = []
        all_labels = set(baseline_labels.unique())
        baseline_focus_df = focus_top.copy()
        baseline_focus_labels = baseline_focus_df["Predicted_Severity"].astype(str)
        baseline_focus_counts = baseline_focus_labels.value_counts().reindex(all_labels, fill_value=0)
        baseline_focus_prob = (
            pd.DataFrame(list(baseline_focus_df["Probabilities"]))
            .reindex(columns=prob_keys)
            .mean()
        )
        baseline_worst_prob = baseline_focus_prob.get(worst_label, 0)

        st.caption("Baseline (top high-severity scenarios)")
        st.dataframe(
            baseline_focus_df[
                ["Weather_Condition", "Weather_Short", "Wind_Direction", "Season", "Hour", "Street", "City", "Predicted_Severity", "Confidence"]
            ]
        )

        suggestions_data = []
        for name in intervention_choices:
            overrides = available_interventions.get(name, {})
            df = run_scenarios(high_risk_scenarios, overrides)
            labels = df["Predicted_Severity"].astype(str)
            counts = labels.value_counts().reindex(all_labels, fill_value=0)
            delta_worst = counts.get(worst_label, 0) - baseline_focus_counts.get(worst_label, 0)
            prob_mean = (
                pd.DataFrame(list(df["Probabilities"]))
                .reindex(columns=prob_keys)
                .mean()
            )
            delta_prob = prob_mean.get(worst_label, 0) - baseline_worst_prob
            change_text = "fewer high-severity cases" if delta_worst < 0 or delta_prob < 0 else "no improvement"
            suggestions_data.append(
                {
                    "name": name,
                    "delta_worst": delta_worst,
                    "delta_prob": delta_prob,
                    "change_text": change_text,
                }
            )
            suggestions.append(f"{name}: {change_text} (delta {delta_worst} count, delta {delta_prob:.3f} prob for '{worst_label}')")
            st.write(f"Results for {name} (top 20 high-risk scenarios):")
            df["Confidence"] = df["Probabilities"].apply(lambda d: max(d.values()) if isinstance(d, dict) else 0)
            st.dataframe(
                df[
                    ["Weather_Condition", "Weather_Short", "Wind_Direction", "Season", "Hour", "Street", "City", "Predicted_Severity", "Confidence"]
                ]
            )
        if suggestions:
            for s in suggestions:
                st.write("- ", s)
        else:
            st.write("No interventions selected or no change observed.")

        summary_lines = [
            f"We reviewed the top {len(focus_top)} highest-risk situations. The most severe outcome was '{worst_label}'."
        ]
        improvements = [s for s in suggestions_data if s["delta_worst"] < 0 or s["delta_prob"] < 0]
        if improvements:
            best = sorted(improvements, key=lambda x: (x["delta_worst"], x["delta_prob"]))[0]
            summary_lines.append(
                f"Adding {best['name'].lower()} reduced the number of '{worst_label}' predictions and lowered risk confidence."
            )
        else:
            summary_lines.append("None of the tested changes reduced the highest severity in this set.")
        st.info(" ".join(summary_lines))

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

    if not filtered.empty:
        view_state = pdk.ViewState(latitude=filtered["lat"].mean(), longitude=filtered["lon"].mean(), zoom=11, pitch=0)
    else:
        view_state = pdk.ViewState(latitude=40.75, longitude=-73.99, zoom=11, pitch=0)

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=filtered,
        get_position="[lon, lat]",
        get_fill_color=[200, 30, 0, 160],
        get_radius=120,
        pickable=True,
    )

    tooltip = {
        "text": "Severity: {severity}\nTime: {time}\nWeather: {weather}\nRoad: {road}\nLat: {lat}\nLon: {lon}"
    }

    st.pydeck_chart(pdk.Deck(layers=[scatter], initial_view_state=view_state, tooltip=tooltip))

    severity_counts = filtered["severity"].value_counts()
    if not filtered.empty:
        st.caption(
            f"{len(filtered)} hotspots shown. Most frequent severity: {severity_counts.idxmax()}."
        )
        st.bar_chart(severity_counts)
    else:
        st.caption("No hotspots match the current filters.")

    st.dataframe(
        filtered[["severity", "time", "weather", "road", "lat", "lon"]]
        .reset_index(drop=True)
        .rename(columns={"severity": "Severity", "time": "Time", "weather": "Weather", "road": "Road Type"})
    )

