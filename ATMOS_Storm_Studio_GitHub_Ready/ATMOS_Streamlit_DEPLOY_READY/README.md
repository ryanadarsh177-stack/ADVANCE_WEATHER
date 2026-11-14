# ATMOS Storm Studio

A futuristic, animated Streamlit dashboard for live weather and educational climate-data modelling.

## What is included

- Live city search using Open-Meteo geocoding and forecast APIs.
- Current temperature, apparent temperature, humidity, wind and precipitation.
- Five-day high/low temperature outlook and precipitation probability.
- Interactive climate prediction lab using two Multiple Linear Regression models.
- RMSE and R² evaluation on a reproducible 80:20 train/test split.
- Dataset explorer, interactive charts, CSV upload and cleaned-data download.
- Animated cloud banks, atmospheric haze, falling rain, floating weather icon, bouncy cards and buttons, and scroll-reveal effects where supported.
- Responsive layout and reduced-motion accessibility support.

The cloud and rain animations are visual ambience. Live weather values are retrieved from Open-Meteo. Climate model outputs are educational estimates, not official weather forecasts or validated climate projections.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Extract this ZIP and upload the **contents** of `ATMOS_Streamlit_DEPLOY_READY` to the repository root.
3. Open Streamlit Community Cloud and select **Create app**.
4. Choose your repository, branch (`main`), and main file path `app.py`.
5. Deploy. The climate dataset must remain at `data/climate_change_dataset.csv`.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Data requirements

The bundled dataset is `data/climate_change_dataset.csv`. For the climate modelling section, a replacement CSV must include these columns exactly:

- `Avg Temperature (°C)`
- `CO2 Emissions (Tons/Capita)`
- `Rainfall (mm)`
- `Sea Level Rise (mm)`

Optional `Country` and `Year` columns are used by some dataset explorer summaries.
