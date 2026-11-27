# Accident Severity Forecast

This repository brings together two related accident-severity projects:

- The original analysis notebooks, models, and Streamlit application are in the repository root, `models/`, and `accident_severity_app/`.
- The geospatial analysis notebooks and mapping application are in [`geospatial/`](geospatial/), imported from the separate geospatial repository.

Each Streamlit application has its own `requirements.txt`; install the dependencies for the application you want to run.

## Geospatial application

The geospatial app expects the New Jersey accident dataset at `data/raw/US_Accidents_March23_NJ.csv`. The dataset is not included in this repository.

From the repository root, install and run it with:

```powershell
pip install -r geospatial\requirements.txt
streamlit run geospatial\main.py
```
