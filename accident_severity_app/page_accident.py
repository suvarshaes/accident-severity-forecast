import random
from datetime import date, datetime
from itertools import product

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
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
        Predict the severity of traffic accidents (Major, Moderate, Minor) using road, weather,
        and environmental factors. This location-aware tool is designed for urban planners
        and safety officials to get instant, site-specific risk insights and plan proactive
        interventions.
        """
    )

    st.markdown(
        """
        ### What this page does
        - **Problem:** Forecast how severe a traffic incident could be (Major / Moderate / Minor) given weather, time, location, and infrastructure context.
        - **Approach:** A trained CatBoost model consumes categorical (city, county, weather, wind, season) and numeric factors (distance, visibility, wind speed, etc.), plus infrastructure flags (signals, crossings, bumps).
        - **How to use:** Adjust the sidebar filters and run predictions. The summary highlights common outcomes, top risky situations, and how interventions (signals, bumps, crossings) change risk.
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
        "Wind_Direction_clean",
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
        "Wind_Direction_clean",
        "Weather_Short",
        "Season",
    ]

    def season_from_month(month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        return "Fall"

    model = load_model()

    category_options = load_category_options()
    WIND_DIRECTIONS = category_options["Wind_Direction_clean"]
    WEATHER_SHORTS = category_options["Weather_Short"]
    SEASONS = category_options["Season"]
    HOURS = list(range(24))

    @st.cache_data
    def load_city_options_for_county(county_name: str):
        """Return cities filtered by county; fallback to all cities if none found or on error."""
        try:
            df = pd.read_csv(
                "../data/clean/US_Accidents_March23_NY_Clean.csv",
                usecols=["County", "City"],
                nrows=200000,
            )
            cities = sorted(df.loc[df["County"] == county_name, "City"].dropna().unique().tolist())
            return cities if cities else load_city_options()
        except Exception:
            return load_city_options()

    def get_region_defaults(county_name: str, city_name: str):
        fallback = {
            "Distance(mi)": 1.0,
            "Temperature(F)": 70.0,
            "Visibility(mi)": 10.0,
            "Humidity(%)": 50.0,
            "Pressure(in)": 30.0,
            "Wind_Speed(mph)": 5.0,
            "Hour": 12,
            "Wind_Direction_clean": "Calm",
        }
        try:
            df = pd.read_csv(
                "../data/clean/US_Accidents_March23_NY_Clean.csv",
                usecols=[
                    "County",
                    "City",
                    "Distance(mi)",
                    "Temperature(F)",
                    "Visibility(mi)",
                    "Humidity(%)",
                    "Pressure(in)",
                    "Wind_Speed(mph)",
                    "Hour",
                    "Wind_Direction_clean",
                ],
                nrows=200000,
            )
            region = df[(df["County"] == county_name) & (df["City"] == city_name)]
            if region.empty:
                region = df[df["County"] == county_name]
            if region.empty:
                return fallback

            def med(col, default):
                series = region[col].dropna()
                return float(series.median()) if not series.empty else default

            def mode(col, default):
                series = region[col].dropna()
                return series.mode().iloc[0] if not series.empty else default

            return {
                "Distance(mi)": med("Distance(mi)", fallback["Distance(mi)"]),
                "Temperature(F)": med("Temperature(F)", fallback["Temperature(F)"]),
                "Visibility(mi)": med("Visibility(mi)", fallback["Visibility(mi)"]),
                "Humidity(%)": med("Humidity(%)", fallback["Humidity(%)"]),
                "Pressure(in)": med("Pressure(in)", fallback["Pressure(in)"]),
                "Wind_Speed(mph)": med("Wind_Speed(mph)", fallback["Wind_Speed(mph)"]),
                "Hour": int(mode("Hour", fallback["Hour"])),
                "Wind_Direction_clean": mode("Wind_Direction_clean", fallback["Wind_Direction_clean"]),
            }
        except Exception:
            return fallback

    # Sidebar inputs (Planner-focused)
    st.sidebar.header("User inputs")
    # Style the primary action button in the sidebar
    st.sidebar.markdown(
        """
        <style>
        div[data-testid="stSidebar"] button {
            background: linear-gradient(90deg, #0b74da, #0a5fb3);
            color: blue;
            border-radius: 10px;
            border: 1px solid #084f93;
            font-weight: 700;
            letter-spacing: 0.2px;
            padding: 0.65rem 1.4rem;
            box-shadow: 0 4px 10px rgba(11, 116, 218, 0.25);
        }
        div[data-testid="stSidebar"] button:hover {
            background: linear-gradient(90deg, #0a5fb3, #094b8d);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Provide location, time-on-road, distance, and existing infrastructure. Weather and risk factors are auto-generated.")
    street_options = load_street_options()
    street_mapping = load_street_encoder()
    default_street = street_options[0] if street_options else "Unknown"
    streets_to_use = [default_street]

    county_options = load_county_options()
    default_county_index = county_options.index("Albany") if "Albany" in county_options else 0
    county = st.sidebar.selectbox("County", county_options, index=default_county_index)

    city_options_all = load_city_options()
    city_mapping = load_city_encoder()
    city_options = load_city_options_for_county(county) or city_options_all
    default_city_index = city_options.index("Albany") if "Albany" in city_options else 0
    city = st.sidebar.selectbox("City", city_options, index=default_city_index)
    region_defaults = get_region_defaults(county, city)
    cities_to_use = [city]

    duration = st.sidebar.number_input("Duration (minutes)", value=15.0, min_value=0.0, step=1.0)
    base_distance = st.sidebar.number_input("Distance (mi)", value=5.0, min_value=0.0, max_value=50.0, step=0.1)

    st.sidebar.markdown("**Existing infrastructure (set if already present)**")
    amenity = st.sidebar.checkbox("Amenity", value=False)
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
    WIND_DIRECTIONS = category_options["Wind_Direction_clean"]
    WEATHER_SHORTS = category_options["Weather_Short"]
    SEASONS = category_options["Season"]
    HOURS = list(range(24))

    # Default extreme-leaning samples for numeric factors (stress-test scenarios)
    def numeric_samples():
        temp_samples = sorted(
            {
                -20.0,
                max(-30.0, region_defaults["Temperature(F)"] - 25.0),
                region_defaults["Temperature(F)"],
                min(120.0, region_defaults["Temperature(F)"] + 25.0),
                110.0,
            }
        )
        visibility_samples = sorted({0.1, 0.25, 0.5, max(0.1, region_defaults["Visibility(mi)"] * 0.5)})
        humidity_samples = sorted({85.0, 95.0, 100.0, min(100.0, max(0.0, region_defaults["Humidity(%)"] + 20.0))})
        pressure_samples = sorted({28.0, 28.5, 29.5, max(25.0, region_defaults["Pressure(in)"] - 1.0)})
        wind_speed_samples = sorted({20.0, 35.0, 50.0, max(0.0, region_defaults["Wind_Speed(mph)"] + 25.0)})
        return (
            temp_samples,
            visibility_samples,
            humidity_samples,
            pressure_samples,
            wind_speed_samples,
        )

    temp_samples_default, visibility_samples_default, humidity_samples_default, pressure_samples_default, wind_speed_samples_default = numeric_samples()

    st.sidebar.markdown("### Prediction grid inputs")
    st.sidebar.caption("We always evaluate high-risk conditions using all seasons, winds, and weather-short values unless you override them below.")
    scenario_cap = st.sidebar.slider("Max scenarios to evaluate", 50, 5000, 1000, step=50)
    use_advanced = st.sidebar.checkbox("Use advanced scenario inputs", value=False)
    high_risk_hours = [18, 19, 20, 21, 22, 23]
    if use_advanced:
        # Default advanced selections now use relevant low/median/high samples around the region defaults
        def low_med_high(values, default):
            vals = sorted(values)
            median = default
            low = vals[0] if vals else default
            high = vals[-1] if vals else default
            return sorted({low, median, high})

        def clamp_defaults(options, picks):
            filtered = [v for v in picks if v in options]
            return filtered if filtered else options

        extreme_temps = clamp_defaults(temp_samples_default, low_med_high(temp_samples_default, region_defaults["Temperature(F)"]))
        extreme_visibility = clamp_defaults(visibility_samples_default, low_med_high(visibility_samples_default, region_defaults["Visibility(mi)"]))
        extreme_humidity = clamp_defaults(humidity_samples_default, low_med_high(humidity_samples_default, region_defaults["Humidity(%)"]))
        extreme_pressure = clamp_defaults(pressure_samples_default, low_med_high(pressure_samples_default, region_defaults["Pressure(in)"]))
        extreme_wind_speeds = clamp_defaults(wind_speed_samples_default, low_med_high(wind_speed_samples_default, region_defaults["Wind_Speed(mph)"]))

        selected_hours = st.sidebar.multiselect("Hours (24h) for scenarios", HOURS, default=high_risk_hours) or high_risk_hours
        selected_seasons = st.sidebar.multiselect("Seasons for scenarios", SEASONS, default=SEASONS) or SEASONS
        selected_winds = st.sidebar.multiselect("Wind directions", WIND_DIRECTIONS, default=WIND_DIRECTIONS) or WIND_DIRECTIONS
        selected_weather_shorts = st.sidebar.multiselect("Weather (short)", WEATHER_SHORTS, default=WEATHER_SHORTS) or WEATHER_SHORTS
        selected_temps = st.sidebar.multiselect("Temperature samples (F)", temp_samples_default, default=extreme_temps) or extreme_temps
        selected_visibility = st.sidebar.multiselect("Visibility samples (mi)", visibility_samples_default, default=extreme_visibility) or extreme_visibility
        selected_humidity = st.sidebar.multiselect("Humidity samples (%)", humidity_samples_default, default=extreme_humidity) or extreme_humidity
        selected_pressure = st.sidebar.multiselect("Pressure samples (in)", pressure_samples_default, default=extreme_pressure) or extreme_pressure
        selected_wind_speeds = st.sidebar.multiselect("Wind speed samples (mph)", wind_speed_samples_default, default=extreme_wind_speeds) or extreme_wind_speeds
    else:
        selected_hours = high_risk_hours
        selected_seasons = SEASONS
        # Default to a limited set of wind directions to keep the grid lighter by default
        selected_winds = WIND_DIRECTIONS[:10] if len(WIND_DIRECTIONS) > 10 else WIND_DIRECTIONS
        selected_weather_shorts = WEATHER_SHORTS
        selected_temps = temp_samples_default
        selected_visibility = visibility_samples_default
        selected_humidity = humidity_samples_default
        selected_pressure = pressure_samples_default
        selected_wind_speeds = wind_speed_samples_default

    current_flags = {
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

    infra_features = list(current_flags.keys())

    st.sidebar.markdown("### Try infrastructure interventions")
    st.sidebar.caption("Select features to add (set to 1) during scenario evaluation.")
    available_to_add = [f for f in infra_features if current_flags.get(f, 0) == 0]
    effective_priority = ["Traffic_Signal", "Traffic_Calming", "Roundabout", "Crossing", "Bump", "Stop"]
    prioritized_options = [f for f in effective_priority if f in (available_to_add if available_to_add else infra_features)]
    remaining_options = [
        f for f in (available_to_add if available_to_add else infra_features) if f not in prioritized_options
    ]
    ordered_options = prioritized_options + remaining_options
    default_effective = prioritized_options[:4] if prioritized_options else ["Traffic_Signal", "Traffic_Calming", "Crossing"]
    intervention_choices = st.sidebar.multiselect(
        "Interventions to add",
        ordered_options,
        default=default_effective,
        help="Effective controls are preselected (signals, calming, roundabout/crossing). Add or remove as needed.",
    )

    def build_scenarios(
        hours,
        weather_shorts,
        wind_directions,
        seasons,
        temps,
        visibility_vals,
        humidity_vals,
        pressure_vals,
        wind_speed_vals,
        cities,
        cap,
    ):
        combos = list(
            product(
                hours,
                weather_shorts,
                wind_directions,
                seasons,
                temps,
                visibility_vals,
                humidity_vals,
                pressure_vals,
                wind_speed_vals,
                cities,
            )
        )
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
        selected_weather_shorts,
        selected_winds,
        selected_seasons,
        selected_temps,
        selected_visibility,
        selected_humidity,
        selected_pressure,
        selected_wind_speeds,
        cities_to_use,
        scenario_cap,
    )

    st.subheader("Scenario inputs being used")
    st.caption("These values form the cartesian grid (capped by the slider) before running predictions.")
    scenario_inputs_df = pd.DataFrame(
        [
            {"Factor": "Hours", "Count": len(selected_hours), "Values": ", ".join(map(str, selected_hours))},
            {"Factor": "Weather", "Count": len(selected_weather_shorts), "Values": ", ".join(map(str, selected_weather_shorts))},
            {"Factor": "Wind Direction", "Count": len(selected_winds), "Values": ", ".join(map(str, selected_winds))},
            {"Factor": "Season", "Count": len(selected_seasons), "Values": ", ".join(map(str, selected_seasons))},
            {"Factor": "Temperature(F)", "Count": len(selected_temps), "Values": ", ".join(map(str, selected_temps))},
            {"Factor": "Visibility(mi)", "Count": len(selected_visibility), "Values": ", ".join(map(str, selected_visibility))},
            {"Factor": "Humidity(%)", "Count": len(selected_humidity), "Values": ", ".join(map(str, selected_humidity))},
            {"Factor": "Pressure(in)", "Count": len(selected_pressure), "Values": ", ".join(map(str, selected_pressure))},
            {"Factor": "Wind Speed", "Count": len(selected_wind_speeds), "Values": ", ".join(map(str, selected_wind_speeds))},
            {"Factor": "Cities", "Count": len(cities_to_use), "Values": ", ".join(map(str, cities_to_use))},
            {"Factor": "Scenario cap", "Count": scenario_cap, "Values": f"Cap at {scenario_cap}"},
        ]
    )
    st.dataframe(scenario_inputs_df)
    
    def run_scenarios(scenario_list, overrides):
        results = []
        for (
            hour,
            weather_short,
            wind_direction,
            season,
            temp_val,
            visibility_val,
            humidity_val,
            pressure_val,
            wind_speed_val,
            city_name,
        ) in scenario_list:
            street_code = street_mapping.get(default_street, 0)
            city_code = city_mapping.get(city_name, 0)
            base_conditions = {
                "Distance(mi)": base_distance,
                "Temperature(F)": float(temp_val),
                "Visibility(mi)": float(visibility_val),
                "Humidity(%)": float(humidity_val),
                "Pressure(in)": float(pressure_val),
                "Wind_Speed(mph)": float(wind_speed_val),
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
                "Weather_Condition": weather_short,  # reuse short form to populate required column
                "Wind_Direction_clean": wind_direction,
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
                "Weather_Short": weather_short,
                "Wind_Direction_clean": wind_direction,
                "Season": season,
                "Hour": hour,
                "Temperature(F)": temp_val,
                "Visibility(mi)": visibility_val,
                "Humidity(%)": humidity_val,
                "Pressure(in)": pressure_val,
                "Wind_Speed(mph)": wind_speed_val,
                "City": city_name,
                "Predicted_Severity": str(pred_class),
                "Probabilities": dict(zip(class_labels, proba))
            })
        return pd.DataFrame(results)
    
    if st.sidebar.button("Predict Severity Across Weather Scenarios"):
        st.session_state["run_predict"] = True

    if st.session_state.get("run_predict"):
        baseline_df = run_scenarios(scenarios, {})
        baseline_labels = baseline_df["Predicted_Severity"].astype(str)
        baseline_df["Confidence"] = baseline_df["Probabilities"].apply(lambda d: max(d.values()) if isinstance(d, dict) else 0)
        severity_counts = baseline_labels.value_counts()
        most_common_label = severity_counts.idxmax() if not severity_counts.empty else "N/A"
        worst_label = choose_worst_label(severity_counts) if not severity_counts.empty else "N/A"
        prob_keys = sorted({k for probs in baseline_df["Probabilities"] for k in probs.keys()})
        baseline_prob_mean = (
            pd.DataFrame(list(baseline_df["Probabilities"]))
            .reindex(columns=prob_keys)
            .mean()
        )

        st.subheader("Severity Predictions Summary")
        col1, col2 = st.columns(2)
        col1.metric("Most common outcome", most_common_label, delta=None)
        col2.metric("Scenario count", len(baseline_df))
        st.markdown(
            f"*Location-aware insight:* For the city of **{city} in {county} county**, we evaluated high-risk evening/night scenarios across extreme weather and road conditions right in this county/city context. "
            f"The most frequent predicted outcome was **{most_common_label}** (worst observed: **{worst_label}**). "
            "Use this as a quick guide for planners and safety teams to target proactive interventions at this location."
        )
        st.bar_chart(severity_counts)
        st.caption("Average predicted probabilities (baseline)")
        st.bar_chart(baseline_prob_mean)
        top_prob_label = baseline_prob_mean.idxmax()
        top_prob_value = baseline_prob_mean.max()
        st.info(
            f"Overall, the model leans toward **{top_prob_label}** with an average confidence of {top_prob_value:.2f}. "
            "Use this as a quick headline: higher average confidence means the model is more certain about that outcome across the tested situations."
        )
        classes_text = ", ".join(map(str, model.classes_)) if hasattr(model, "classes_") else "Major, Moderate, Minor"
        # Intentionally removed detailed model summary text per request
        top_risky = baseline_df[baseline_labels == worst_label].copy()
        top_risky["Confidence"] = top_risky["Probabilities"].apply(lambda d: max(d.values()) if isinstance(d, dict) else 0)
        st.write("Top scenarios leading to highest severity (sorted by confidence):")
        st.dataframe(
            top_risky.sort_values("Confidence", ascending=False)
            .head(10)[
                [
                    "Hour",
                    "Weather_Short",
                    "Wind_Direction_clean",
                    "Season",
                    "Temperature(F)",
                    "Visibility(mi)",
                    "Humidity(%)",
                    "Pressure(in)",
                    "Wind_Speed(mph)",
                    "City",
                    "Predicted_Severity",
                    "Confidence",
                ]
            ]
            .rename(
                columns={
                    "Weather_Short": "Weather",
                    "Wind_Direction_clean": "Wind Direction",
                    "Wind_Speed(mph)": "Wind Speed",
                    "Predicted_Severity": "Severity",
                }
            )
        )

        focus_top = top_risky.sort_values("Confidence", ascending=False).head(20)
        high_risk_scenarios = list(
            focus_top[
                [
                    "Hour",
                    "Weather_Short",
                    "Wind_Direction_clean",
                    "Season",
                    "Temperature(F)",
                    "Visibility(mi)",
                    "Humidity(%)",
                    "Pressure(in)",
                    "Wind_Speed(mph)",
                    "City",
                ]
            ].itertuples(index=False, name=None)
        )

        with st.expander("Full scenario table (filtered)"):
            st.dataframe(
                baseline_df.assign(Predicted_Severity=baseline_labels)[
                [
                        "Hour",
                        "Weather_Short",
                        "Wind_Direction_clean",
                        "Season",
                        "Temperature(F)",
                        "Visibility(mi)",
                        "Humidity(%)",
                        "Pressure(in)",
                        "Wind_Speed(mph)",
                        "City",
                        "Predicted_Severity",
                        "Confidence",
                    ]
                ]
                .rename(
                    columns={
                        "Weather_Short": "Weather",
                        "Wind_Direction_clean": "Wind Direction",
                        "Wind_Speed(mph)": "Wind Speed",
                        "Predicted_Severity": "Severity",
                    }
                )
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
            [
                    "Hour",
                    "Weather_Short",
                    "Wind_Direction_clean",
                    "Season",
                    "Temperature(F)",
                    "Visibility(mi)",
                    "Humidity(%)",
                    "Pressure(in)",
                    "Wind_Speed(mph)",
                    "City",
                    "Predicted_Severity",
                    "Confidence",
                ]
            ]
            .rename(
                columns={
                    "Weather_Short": "Weather",
                    "Wind_Direction_clean": "Wind Direction",
                    "Wind_Speed(mph)": "Wind Speed",
                    "Predicted_Severity": "Severity",
                }
            )
        )

        suggestions_data = []
        for name in intervention_choices:
            overrides = {name: 1}
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
            change_text = "lower risk" if delta_worst < 0 or delta_prob < 0 else "no improvement or higher risk"
            suggestions_data.append(
                {
                    "name": name,
                    "delta_worst": delta_worst,
                    "delta_prob": delta_prob,
                    "change_text": change_text,
                }
            )
            direction = "down" if delta_worst < 0 or delta_prob < 0 else "up or unchanged"
            suggestions.append(
                f"{name}: {change_text} — count {delta_worst}, prob {delta_prob:.3f} ({direction} for '{worst_label}')"
            )
            st.write(f"Results for {name} (top 20 high-risk scenarios):")
            df["Confidence"] = df["Probabilities"].apply(lambda d: max(d.values()) if isinstance(d, dict) else 0)
            st.dataframe(
                df[
                [
                        "Hour",
                        "Weather_Short",
                        "Wind_Direction_clean",
                        "Season",
                        "Temperature(F)",
                        "Visibility(mi)",
                        "Humidity(%)",
                        "Pressure(in)",
                        "Wind_Speed(mph)",
                        "City",
                        "Predicted_Severity",
                        "Confidence",
                    ]
                ]
                .rename(
                    columns={
                        "Weather_Short": "Weather",
                        "Wind_Direction_clean": "Wind Direction",
                        "Wind_Speed(mph)": "Wind Speed",
                        "Predicted_Severity": "Severity",
                    }
                )
            )
        if suggestions:
            st.caption("Note: count is change in worst-severity cases; prob is change in worst-class confidence (positive = higher risk, negative = lower risk).")
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
        st.caption("Model-scored hotspots: ranked by highest predicted severity/confidence.")
    
        hotspot_data = pd.DataFrame(
            [
                {"lat": 40.75, "lon": -73.99, "severity": "Major", "time": "evening", "weather": "Rain", "road": "Arterial", "city": "New York", "street": "8th Ave"},
                {"lat": 40.73, "lon": -73.98, "severity": "Moderate", "time": "morning", "weather": "Clear", "road": "Arterial", "city": "New York", "street": "5th Ave"},
                {"lat": 40.71, "lon": -74.01, "severity": "Major", "time": "night", "weather": "Snow", "road": "Highway", "city": "New York", "street": "Broadway"},
                {"lat": 40.78, "lon": -73.95, "severity": "Minor", "time": "day", "weather": "Clear", "road": "Local", "city": "New York", "street": "Madison Ave"},
                {"lat": 40.69, "lon": -73.99, "severity": "Moderate", "time": "evening", "weather": "Rain", "road": "Highway", "city": "Brooklyn", "street": "Flatbush Ave"},
                {"lat": 40.77, "lon": -73.97, "severity": "Major", "time": "night", "weather": "Fog", "road": "Arterial", "city": "New York", "street": "Lexington Ave"},
                {"lat": 40.72, "lon": -74.02, "severity": "Moderate", "time": "day", "weather": "Cloudy", "road": "Local", "city": "New York", "street": "Hudson St"},
                {"lat": 40.74, "lon": -73.93, "severity": "Minor", "time": "morning", "weather": "Clear", "road": "Local", "city": "Queens", "street": "Roosevelt Ave"},
                {"lat": 40.70, "lon": -73.90, "severity": "Major", "time": "evening", "weather": "Rain", "road": "Highway", "city": "Queens", "street": "Queens Blvd"},
                {"lat": 40.68, "lon": -73.92, "severity": "Moderate", "time": "night", "weather": "Fog", "road": "Arterial", "city": "Brooklyn", "street": "Atlantic Ave"},
            ]
        )
    
        # Simple filters with persistent state; apply immediately without a separate submit
        st.session_state.setdefault("hotspot_time", "All")
        st.session_state.setdefault("hotspot_weather", "All")
        st.session_state.setdefault("hotspot_road", "All")
        st.selectbox(
            "Time of day",
            ["All", "morning", "day", "evening", "night"],
            key="hotspot_time",
        )
        st.selectbox(
            "Weather",
            ["All", "Clear", "Rain", "Snow", "Fog", "Cloudy"],
            key="hotspot_weather",
        )
        st.selectbox(
            "Road type",
            ["All", "Highway", "Arterial", "Local"],
            key="hotspot_road",
        )
        time_filter = st.session_state["hotspot_time"]
        weather_filter = st.session_state["hotspot_weather"]
        road_filter = st.session_state["hotspot_road"]
    
        filtered = hotspot_data.copy()
        if time_filter != "All":
            filtered = filtered[filtered["time"] == time_filter]
        if weather_filter != "All":
            filtered = filtered[filtered["weather"] == weather_filter]
        if road_filter != "All":
            filtered = filtered[filtered["road"] == road_filter]

        # Predict severity for filtered hotspots using today's date/time context
        time_to_hour = {"morning": 8, "day": 12, "evening": 18, "night": 23}
        season_today = season_from_month(date.today().month)
        pred_records = []
        map_infra_off = {
            "Amenity": 0,
            "Bump": 0,
            "Crossing": 0,
            "Give_Way": 0,
            "Junction": 0,
            "No_Exit": 0,
            "Railway": 0,
            "Roundabout": 0,
            "Station": 0,
            "Stop": 0,
            "Traffic_Calming": 0,
            "Traffic_Signal": 0,
            "Turning_Loop": 0,
        }
        for _, row in filtered.iterrows():
            hour = time_to_hour.get(row["time"], 12)
            street_code = street_mapping.get(default_street, 0)
            city_code = city_mapping.get(row["city"], 0)
            base_conditions = {
                "Distance(mi)": base_distance,
                "Temperature(F)": float(min(selected_temps)) if selected_temps else region_defaults["Temperature(F)"],
                "Visibility(mi)": float(min(selected_visibility)) if selected_visibility else region_defaults["Visibility(mi)"],
                "Humidity(%)": float(max(selected_humidity)) if selected_humidity else region_defaults["Humidity(%)"],
                "Pressure(in)": float(min(selected_pressure)) if selected_pressure else region_defaults["Pressure(in)"],
                "Wind_Speed(mph)": float(max(selected_wind_speeds)) if selected_wind_speeds else region_defaults["Wind_Speed(mph)"],
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
                "Weather_Condition": row["weather"],
                "Wind_Direction_clean": selected_winds[0] if isinstance(selected_winds, list) and selected_winds else selected_winds,
                "Weather_Short": row["weather"],
                "Season": season_today,
            }
            input_row.update(map_infra_off)
            input_df = pd.DataFrame([input_row], columns=FEATURES)
            pool = Pool(input_df, cat_features=[col for col in CATEGORICAL_COLS if col in input_df.columns])
            pred_class = model.predict(pool, prediction_type="Class")[0]
            if isinstance(pred_class, (list, tuple, np.ndarray)):
                pred_class = pred_class[0]
            proba = model.predict_proba(pool)[0]
            street_display = row.get("street", default_street)
            pred_records.append(
                {
                    **row.to_dict(),
                    "Predicted_Severity": str(pred_class),
                    "Confidence": float(max(proba)) if len(proba) else 0.0,
                    "street": street_display,
                }
            )

        if pred_records:
            pred_df = pd.DataFrame(pred_records)
        else:
            pred_df = pd.DataFrame(
                columns=["Predicted_Severity", "Confidence", "city", "street", "time", "weather", "road", "lat", "lon"]
            )

        # Select the highest-risk points (sorted by predicted severity priority then confidence)
        severity_priority = {"Major": 3, "Moderate": 2, "Minor": 1}
        pred_df["severity_score"] = pred_df["Predicted_Severity"].map(lambda x: severity_priority.get(str(x), 0))
        pred_df_sorted = pred_df.sort_values(["severity_score", "Confidence"], ascending=[False, False])
        top_hotspots = pred_df_sorted.head(50) if not pred_df_sorted.empty else pred_df_sorted

        if not top_hotspots.empty:
            view_state = pdk.ViewState(latitude=top_hotspots["lat"].mean(), longitude=top_hotspots["lon"].mean(), zoom=11, pitch=0)
        else:
            view_state = pdk.ViewState(latitude=40.75, longitude=-73.99, zoom=11, pitch=0)

        scatter = pdk.Layer(
            "ScatterplotLayer",
            data=top_hotspots,
            get_position="[lon, lat]",
            get_fill_color=[200, 30, 0, 160],
            get_radius=120,
            pickable=True,
        )

        tooltip = {
            "text": "Predicted: {Predicted_Severity}\nConfidence: {Confidence}\nCity: {city}\nStreet: {street}\nTime: {time}\nWeather: {weather}\nRoad: {road}\nLat: {lat}\nLon: {lon}"
        }

        st.pydeck_chart(pdk.Deck(layers=[scatter], initial_view_state=view_state, tooltip=tooltip))

        severity_counts = top_hotspots.get("Predicted_Severity", pd.Series(dtype=str)).value_counts()
        if not top_hotspots.empty:
            st.caption(f"{len(top_hotspots)} highest-risk hotspots shown (ordered by severity and confidence). Most frequent severity: {severity_counts.idxmax()}.")
        else:
            st.caption("No hotspots match the current filters.")

        st.dataframe(
            top_hotspots.reindex(
                columns=["Predicted_Severity", "Confidence", "city", "street", "time", "weather", "road", "lat", "lon"]
            )
            .reset_index(drop=True)
            .rename(
                columns={
                    "Predicted_Severity": "Predicted Severity",
                    "city": "City",
                    "street": "Street",
                    "time": "Time",
                    "weather": "Weather",
                    "road": "Road Type",
                }
            )
        )
    
