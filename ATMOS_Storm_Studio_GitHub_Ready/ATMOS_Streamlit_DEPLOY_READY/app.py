from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="ATMOS — Storm Studio | Weather & Climate Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "climate_change_dataset.csv"
TEMP_COL = "Avg Temperature (°C)"
CO2_COL = "CO2 Emissions (Tons/Capita)"
RAINFALL_COL = "Rainfall (mm)"
SEA_COL = "Sea Level Rise (mm)"
FEATURES = [TEMP_COL, CO2_COL]
TARGETS = [RAINFALL_COL, SEA_COL]

# ------------------------------------------------------------
# Premium visual styling (CSS-only animations; no fragile JS)
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#eaf2ff; --muted:#9eb0c8; --panel:rgba(15,29,53,.78); --line:rgba(166,195,235,.16); --cyan:#67e8f9; --blue:#8ba8ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 12% 0%, rgba(35,95,156,.27), transparent 34%), radial-gradient(ellipse at 88% 8%, rgba(83,64,160,.22), transparent 30%), linear-gradient(150deg,#07111f 0%,#0a1528 45%,#0a1020 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(7,17,31,.65); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(10,24,43,.98),rgba(7,15,29,.98)); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#dce9ff; }
.block-container { max-width:1440px; padding-top:1.6rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; color:#f4f8ff; }
p, label, .stMarkdown { color:#d3deef; }
[data-testid="stMetric"] { background:linear-gradient(135deg,rgba(22,42,71,.84),rgba(14,28,49,.72)); border:1px solid var(--line); padding:17px 18px; border-radius:18px; box-shadow:0 10px 28px rgba(0,0,0,.12); transition:transform .22s ease,border-color .22s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); border-color:rgba(103,232,249,.45); }
[data-testid="stMetricLabel"] { color:#9fb5d3 !important; }
[data-testid="stMetricValue"] { color:#f4f8ff !important; font-family:'Space Grotesk',sans-serif; }
.stButton > button, .stDownloadButton > button { border:1px solid rgba(103,232,249,.38); border-radius:12px; background:linear-gradient(115deg,#2dd4bf,#60a5fa); color:#061222; font-weight:700; transition:all .2s ease; box-shadow:0 6px 20px rgba(45,212,191,.12); }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 26px rgba(96,165,250,.24); border-color:#a5f3fc; color:#061222; }
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"], .stFileUploader section { background:rgba(12,27,47,.85) !important; color:#eaf2ff !important; border-color:var(--line) !important; border-radius:12px !important; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:12px 18px; color:#9eb0c8; }
.stTabs [aria-selected="true"] { color:#67e8f9 !important; border-bottom-color:#67e8f9 !important; }
hr { border-color:var(--line); }
.hero { position:relative; overflow:hidden; padding:34px 36px 32px; border-radius:28px; border:1px solid rgba(157,190,255,.2); background:linear-gradient(115deg,rgba(20,49,82,.95),rgba(24,30,66,.85) 55%,rgba(19,38,67,.78)); box-shadow:0 24px 65px rgba(0,0,0,.2); margin-bottom:24px; }
.hero:before { content:''; position:absolute; width:360px;height:360px;right:-70px;top:-175px;border-radius:50%;background:radial-gradient(circle,rgba(103,232,249,.28),rgba(96,165,250,.05) 62%,transparent 70%);animation:floatGlow 7s ease-in-out infinite; }
.hero:after { content:''; position:absolute;width:210px;height:210px;right:150px;bottom:-160px;border-radius:50%;border:1px solid rgba(103,232,249,.18);box-shadow:0 0 0 24px rgba(103,232,249,.025),0 0 0 48px rgba(103,232,249,.02); }
.hero-kicker { text-transform:uppercase; letter-spacing:.22em; font-size:.74rem; font-weight:700; color:#8ee9f4; margin-bottom:12px; }
.hero h1 { font-size:clamp(2.2rem,4.5vw,4.25rem); line-height:1.02; margin:0 0 14px; position:relative; z-index:1; }
.hero p { color:#b9cbe3; max-width:710px; font-size:1.04rem; line-height:1.7; position:relative; z-index:1; margin-bottom:0; }
.hero-badge { display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border:1px solid rgba(103,232,249,.25); border-radius:999px; color:#d8faff; background:rgba(103,232,249,.07); font-size:.8rem; margin-top:20px; }
.section-eyebrow { color:#70dce9; text-transform:uppercase; letter-spacing:.17em; font-size:.72rem; font-weight:700; margin:22px 0 7px; }
.panel { padding:20px 22px; border:1px solid var(--line); background:linear-gradient(135deg,rgba(18,36,61,.8),rgba(11,25,44,.72)); border-radius:18px; margin-bottom:14px; }
.panel h3 { margin-top:0; }
.mini-note { color:#94a9c4; font-size:.84rem; line-height:1.55; }
.status-dot { display:inline-block;width:8px;height:8px;border-radius:50%;background:#5eead4;box-shadow:0 0 12px #5eead4;margin-right:8px; }
@keyframes floatGlow { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-24px,22px) scale(1.08)} }
@keyframes riseIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
.hero, [data-testid="stMetric"], .panel { animation:riseIn .65s ease both; }
footer {visibility:hidden;}
#MainMenu {visibility:hidden;}
@media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation:none !important; transition:none !important; } }
</style>
<style>
/* ATMOS STORM STUDIO: ambient weather, motion and tactile interactions */
:root{--storm-cyan:#72efff;--storm-blue:#7ca8ff;--storm-violet:#a58bff;--storm-glass:rgba(14,31,57,.70)}
.stApp{background:
 radial-gradient(ellipse at 14% 0%,rgba(30,132,190,.24),transparent 34%),
 radial-gradient(ellipse at 88% 13%,rgba(105,75,190,.25),transparent 33%),
 radial-gradient(ellipse at 55% 100%,rgba(19,113,145,.16),transparent 38%),
 linear-gradient(155deg,#050b18 0%,#09172b 48%,#090e20 100%);background-attachment:fixed;overflow-x:hidden}
/* A fixed, low-contrast cloudscape follows the viewport while the page scrolls. */
.stApp:before{content:"";position:fixed;inset:-10vh -15vw;z-index:0;pointer-events:none;opacity:.20;background:
 radial-gradient(ellipse 25vw 7vw at 10% 17%,rgba(125,188,222,.28),transparent 72%),
 radial-gradient(ellipse 29vw 8vw at 37% 9%,rgba(115,158,204,.24),transparent 72%),
 radial-gradient(ellipse 31vw 9vw at 83% 23%,rgba(144,155,220,.23),transparent 72%),
 radial-gradient(ellipse 22vw 7vw at 67% 76%,rgba(91,171,201,.17),transparent 72%);
 filter:blur(16px);animation:cloudDrift 46s ease-in-out infinite alternate}
.stApp:after{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.23;background-image:
 repeating-linear-gradient(108deg,transparent 0 34px,rgba(139,220,255,.24) 35px,transparent 36px 74px);
 mask-image:linear-gradient(180deg,transparent 0%,#000 22%,#000 78%,transparent 100%);animation:rainFall .72s linear infinite}
.block-container{position:relative;z-index:1}
[data-testid="stSidebar"]{position:relative;z-index:2}
/* Slow-moving cloud banks */
.cloud-bank{position:fixed;z-index:0;pointer-events:none;top:9vh;left:-35vw;width:48vw;height:130px;opacity:.16;filter:blur(2px);animation:bankTravel 65s linear infinite}
.cloud-bank:before,.cloud-bank:after{content:"";position:absolute;border-radius:999px;background:linear-gradient(180deg,rgba(195,228,255,.58),rgba(74,130,184,.08));box-shadow:30px 4px 32px rgba(114,198,236,.12)}
.cloud-bank:before{width:70%;height:54px;left:12%;top:40px}.cloud-bank:after{width:34%;height:88px;left:28%;top:0}
.cloud-bank.two{top:63vh;left:auto;right:-45vw;transform:scale(.72);animation:bankTravelReverse 82s linear infinite;opacity:.11}
.rain-layer{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;opacity:.18}
.rain-layer i{position:absolute;top:-15vh;width:1px;height:55px;background:linear-gradient(transparent,rgba(139,220,255,.85),transparent);animation:rainDrop linear infinite}
.rain-layer i:nth-child(3n){height:34px;opacity:.55}.rain-layer i:nth-child(4n){height:72px;opacity:.8}
@keyframes cloudDrift{0%{transform:translate3d(-2%,0,0)}100%{transform:translate3d(3%,2%,0)}}
@keyframes bankTravel{from{transform:translateX(0)}to{transform:translateX(170vw)}}
@keyframes bankTravelReverse{from{transform:translateX(0) scale(.72)}to{transform:translateX(-170vw) scale(.72)}}
@keyframes rainFall{from{background-position:0 -120px}to{background-position:0 120px}}
@keyframes rainDrop{0%{transform:translate3d(0,-10vh,0);opacity:0}10%{opacity:.8}100%{transform:translate3d(-32px,120vh,0);opacity:0}}
@keyframes floatCloud{0%,100%{transform:translateY(0) translateX(0)}50%{transform:translateY(-12px) translateX(9px)}}
@keyframes bounceIn{0%{opacity:0;transform:translateY(22px) scale(.96)}65%{opacity:1;transform:translateY(-5px) scale(1.012)}100%{transform:translateY(0) scale(1)}}
@keyframes iconBob{0%,100%{transform:translateY(0) rotate(-4deg)}50%{transform:translateY(-9px) rotate(4deg)}}
@keyframes electricPulse{0%,100%{box-shadow:0 0 0 rgba(90,220,255,0)}50%{box-shadow:0 0 34px rgba(90,220,255,.15)}}
.hero{min-height:300px;background:linear-gradient(112deg,rgba(11,39,68,.92),rgba(29,38,83,.80) 55%,rgba(16,53,78,.78));border:1px solid rgba(133,218,255,.24);box-shadow:0 25px 85px rgba(0,0,0,.28),inset 0 1px rgba(255,255,255,.08);animation:bounceIn .9s cubic-bezier(.2,.8,.2,1) both}
.hero:before{width:490px;height:490px;right:-70px;top:-245px;background:radial-gradient(circle,rgba(92,229,255,.28),rgba(104,133,255,.09) 48%,transparent 70%);animation:floatGlow 8s ease-in-out infinite, electricPulse 5s ease-in-out infinite}
.hero:after{right:190px;bottom:-155px;width:250px;height:250px;border-color:rgba(117,227,255,.22);box-shadow:0 0 0 30px rgba(103,232,249,.035),0 0 0 62px rgba(103,232,249,.022),0 0 70px rgba(103,232,249,.08)}
.hero-weather-art{position:absolute;right:5%;top:16px;width:270px;height:220px;z-index:1;pointer-events:none;animation:floatCloud 7s ease-in-out infinite;opacity:.92}
.hero-cloud{position:absolute;right:38px;top:58px;width:170px;height:58px;border-radius:60px;background:linear-gradient(180deg,#d9f5ff,#6b9fcb 88%);filter:drop-shadow(0 15px 22px rgba(46,177,224,.25))}
.hero-cloud:before,.hero-cloud:after{content:"";position:absolute;border-radius:50%;background:linear-gradient(180deg,#e4f8ff,#7faed2)}
.hero-cloud:before{width:83px;height:83px;left:25px;top:-38px}.hero-cloud:after{width:104px;height:104px;right:20px;top:-52px}
.hero-rain{position:absolute;right:80px;top:133px;display:flex;gap:23px;transform:skew(-12deg)}
.hero-rain b{display:block;width:3px;height:40px;border-radius:9px;background:linear-gradient(transparent,#7ceaff);animation:rainDrop 1.2s linear infinite}
.hero-rain b:nth-child(2){animation-delay:-.4s;height:30px}.hero-rain b:nth-child(3){animation-delay:-.8s;height:46px}.hero-rain b:nth-child(4){animation-delay:-.2s;height:34px}
.panel,[data-testid="stMetric"],.stTabs [data-baseweb="tab-list"]{backdrop-filter:blur(18px);box-shadow:0 13px 38px rgba(0,0,0,.16),inset 0 1px rgba(255,255,255,.035);transition:transform .28s cubic-bezier(.2,.8,.2,1),border-color .28s,box-shadow .28s}
.panel:hover,[data-testid="stMetric"]:hover{transform:translateY(-6px) scale(1.012);border-color:rgba(103,232,249,.5);box-shadow:0 20px 48px rgba(0,0,0,.23),0 0 24px rgba(103,232,249,.06)}
.stButton>button,.stDownloadButton>button{border-radius:15px;position:relative;overflow:hidden;transition:transform .18s cubic-bezier(.2,.8,.2,1),box-shadow .2s}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-4px) scale(1.025);box-shadow:0 14px 32px rgba(45,212,191,.24),0 0 0 4px rgba(103,232,249,.07)}
.stButton>button:active,.stDownloadButton>button:active{transform:translateY(1px) scale(.97)}
.stTabs [data-baseweb="tab"]{transition:transform .2s,color .2s}.stTabs [data-baseweb="tab"]:hover{transform:translateY(-3px);color:#8df3ff}
[data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stPlotlyChart"]{animation:bounceIn .65s ease both}
[data-testid="stPlotlyChart"]:hover{filter:drop-shadow(0 10px 24px rgba(76,195,237,.09))}
.weather-icon{display:inline-block;animation:iconBob 4s ease-in-out infinite;filter:drop-shadow(0 8px 18px rgba(89,209,255,.25))}
/* Scroll-triggered reveal where supported; normal entrance animation remains fallback. */
@supports (animation-timeline:view()){.panel,[data-testid="stMetric"],[data-testid="stPlotlyChart"]{animation:riseIn .7s ease both;animation-timeline:view();animation-range:entry 0% entry 35%}}
@media(max-width:850px){.hero-weather-art{opacity:.25;right:-30px;transform:scale(.8)}.hero{padding:27px 23px}}
@media(prefers-reduced-motion:reduce){.stApp:before,.stApp:after,.cloud-bank,.rain-layer i,.hero-weather-art,.hero-rain b{animation:none!important}.panel,[data-testid="stMetric"],.stButton>button{transition:none!important}}
</style>\n""", unsafe_allow_html=True)

# Persistent animated weather layers. These are decorative and do not alter API data.
st.markdown("""<div class="cloud-bank"></div><div class="cloud-bank two"></div><div class="rain-layer">""" + "".join(f"<i style=\"left:{(i*37)%100}%;animation-delay:-{(i%13)*0.31:.2f}s;animation-duration:{1.1+(i%7)*0.19:.2f}s\"></i>" for i in range(42)) + "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Data + model helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_data(show_spinner=False, ttl=1800)
def geocode_city(city: str):
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return results[0] if results else None

@st.cache_data(show_spinner=False, ttl=600)
def fetch_weather(latitude: float, longitude: float):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 5,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()

def weather_label(code):
    mapping = {
        0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
        45: ("Fog", "🌫️"), 48: ("Depositing rime fog", "🌫️"), 51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"),
        55: ("Dense drizzle", "🌧️"), 56: ("Freezing drizzle", "🌧️"), 57: ("Freezing drizzle", "🌧️"),
        61: ("Light rain", "🌦️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"), 66: ("Freezing rain", "🌧️"),
        67: ("Heavy freezing rain", "🌧️"), 71: ("Light snow", "🌨️"), 73: ("Snow", "🌨️"), 75: ("Heavy snow", "❄️"),
        77: ("Snow grains", "🌨️"), 80: ("Rain showers", "🌦️"), 81: ("Rain showers", "🌧️"), 82: ("Violent rain showers", "⛈️"),
        85: ("Snow showers", "🌨️"), 86: ("Heavy snow showers", "❄️"), 95: ("Thunderstorm", "⛈️"),
        96: ("Thunderstorm with hail", "⛈️"), 99: ("Thunderstorm with hail", "⛈️"),
    }
    return mapping.get(int(code), ("Conditions unavailable", "🌍"))

def clean_dataset(frame):
    missing = [col for col in FEATURES + TARGETS if col not in frame.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    cleaned = frame.copy()
    for col in FEATURES + TARGETS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + TARGETS)
    if len(cleaned) < 10:
        raise ValueError("At least 10 complete rows are needed to train and evaluate the models.")
    return cleaned

@st.cache_data(show_spinner=False)
def evaluate_models(frame):
    usable = clean_dataset(frame)
    X = usable[FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(X, usable[TARGETS], test_size=0.2, random_state=42)
    metrics = {}
    for target in TARGETS:
        model = LinearRegression()
        model.fit(X_train, y_train[target])
        pred = model.predict(X_test)
        metrics[target] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test[target], pred))),
            "r2": float(r2_score(y_test[target], pred)),
            "model": model,
        }
    # Train final models on all available rows for interactive predictions.
    final_models = {}
    for target in TARGETS:
        model = LinearRegression().fit(X, usable[target])
        final_models[target] = model
    return usable, metrics, final_models

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-family:Space Grotesk;font-size:1.35rem;font-weight:700;letter-spacing:-.04em'>🌦️ ATMOS<span style='color:#67e8f9'>.</span></div>", unsafe_allow_html=True)
    st.caption("WEATHER · CLIMATE · DATA")
    st.divider()
    st.markdown("**Workspace**")
    st.markdown("<span class='status-dot'></span> Live weather API", unsafe_allow_html=True)
    st.markdown("<span class='status-dot'></span> Linear regression lab", unsafe_allow_html=True)
    st.markdown("<span class='status-dot'></span> Dataset explorer", unsafe_allow_html=True)
    st.divider()
    st.markdown("**About this project**")
    st.caption("A student-built weather dashboard with a small climate dataset and two baseline regression models.")
    st.caption("Weather data: Open-Meteo. Climate predictions are educational estimates, not scientific forecasts.")

# ------------------------------------------------------------
# Hero
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-weather-art"><div class="hero-cloud"></div><div class="hero-rain"><b></b><b></b><b></b><b></b></div></div>
  <div class="hero-kicker">Atmospheric intelligence · Interactive weather system</div>
  <h1>Feel the forecast.<br><span style="background:linear-gradient(90deg,#67e8f9,#8ba8ff,#c4a6ff);-webkit-background-clip:text;background-clip:text;color:transparent">Read the planet.</span></h1>
  <p>Track live conditions, follow the clouds, and experiment with climate data. A weather experience designed to move, respond, and make the atmosphere feel alive.</p>
  <div class="hero-badge">✦ Live weather &nbsp; · &nbsp; 5-day outlook &nbsp; · &nbsp; Climate prediction lab</div>
</div>
""", unsafe_allow_html=True)

# Dataset can be replaced with a CSV upload.
base_df = load_default_data()
with st.sidebar:
    uploaded = st.file_uploader("Use another climate CSV", type=["csv"], help="Optional: upload a dataset with the same required column names.")
if uploaded is not None:
    try:
        source_df = pd.read_csv(uploaded)
        data_source_label = "Uploaded dataset"
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    source_df = base_df
    data_source_label = "Bundled Kaggle dataset"

if source_df is None:
    st.error("The bundled dataset could not be found. Upload your climate_change_dataset.csv file in the sidebar.")
    st.stop()

try:
    clean_df, model_metrics, models = evaluate_models(source_df)
except Exception as exc:
    st.error(f"Dataset or model setup issue: {exc}")
    st.info("Required columns: Avg Temperature (°C), CO2 Emissions (Tons/Capita), Rainfall (mm), Sea Level Rise (mm).")
    st.stop()

# Top-level snapshot metrics
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Records analysed", f"{len(clean_df):,}", help="Rows with complete numeric values in the four model columns.")
with m2: st.metric("Input signals", "02", help="Average temperature and CO₂ emissions per capita.")
with m3: st.metric("Prediction targets", "02", help="Rainfall and sea-level rise.")
with m4: st.metric("Model family", "Linear regression", help="A baseline model chosen for this educational project.")

weather_tab, climate_tab, explore_tab, about_tab = st.tabs(["☁️  Live weather", "🧪  Prediction lab", "📊  Data explorer", "✦  Methodology"])

# ------------------------------------------------------------
# Live weather tab
# ------------------------------------------------------------
with weather_tab:
    st.markdown("<div class='section-eyebrow'>01 / Live conditions</div>", unsafe_allow_html=True)
    st.header("The sky, right now")
    st.markdown("Search a city to view current conditions and a five-day outlook. The cloudscape and rainfall motion are ambient visual effects; the readings below come from Open-Meteo.")
    city_suggestions = st.columns(5)
    for suggestion_col, suggestion in zip(city_suggestions, ["Delhi", "London", "Tokyo", "New York", "Singapore"]):
        with suggestion_col:
            if st.button(suggestion, key=f"city_{suggestion}", use_container_width=True):
                st.session_state["weather_city_input"] = suggestion
                st.rerun()
    if "weather_city_input" not in st.session_state:
        st.session_state["weather_city_input"] = "Delhi"
    city_col, btn_col = st.columns([4, 1])
    with city_col:
        city = st.text_input("City or place", key="weather_city_input", placeholder="Try London, Tokyo, New York…", label_visibility="collapsed")
    with btn_col:
        get_weather = st.button("Get weather ↗", use_container_width=True)
    if "weather_result" not in st.session_state:
        st.session_state.weather_result = None
        st.session_state.weather_city = ""
    if get_weather:
        if not city.strip():
            st.warning("Enter a city name first.")
        else:
            try:
                with st.spinner("Finding your city and fetching live conditions…"):
                    place = geocode_city(city.strip())
                    if not place:
                        st.session_state.weather_result = None
                        st.warning("We couldn't find that place. Try a larger city or check the spelling.")
                    else:
                        wx = fetch_weather(place["latitude"], place["longitude"])
                        st.session_state.weather_result = (place, wx)
                        st.session_state.weather_city = city.strip()
            except requests.RequestException as exc:
                st.error(f"Weather service could not be reached. Check your internet connection and try again. Details: {exc}")
    if st.session_state.weather_result:
        place, wx = st.session_state.weather_result
        current = wx.get("current", {})
        label, icon = weather_label(current.get("weather_code", -1))
        location = ", ".join([str(x) for x in [place.get("name"), place.get("admin1"), place.get("country")] if x])
        st.markdown(f"<div class='panel'><div class='mini-note'>CURRENT LOCATION</div><h2 style='margin:.3rem 0'>{location}</h2><span class='mini-note'>{current.get('time','Local time unavailable').replace('T','  ·  ')}</span></div>", unsafe_allow_html=True)
        a, b = st.columns([1.1, 2])
        with a:
            st.markdown(f"<div class='panel' style='text-align:center;padding:30px 16px'><div class='weather-icon' style='font-size:4rem'>{icon}</div><div style='font-family:Space Grotesk;font-size:3rem;font-weight:700;color:#f4f8ff'>{current.get('temperature_2m','—')}°</div><div style='font-size:1.05rem;color:#67e8f9'>{label}</div><div class='mini-note' style='margin-top:12px'>Feels like {current.get('apparent_temperature','—')}°C</div></div>", unsafe_allow_html=True)
        with b:
            w1, w2, w3 = st.columns(3)
            with w1: st.metric("Humidity", f"{current.get('relative_humidity_2m','—')}%")
            with w2: st.metric("Wind", f"{current.get('wind_speed_10m','—')} km/h")
            with w3: st.metric("Precipitation", f"{current.get('precipitation','—')} mm")
            daily = wx.get("daily", {})
            if daily and daily.get("time"):
                forecast = pd.DataFrame({
                    "Date": pd.to_datetime(daily["time"]),
                    "High (°C)": daily.get("temperature_2m_max", []),
                    "Low (°C)": daily.get("temperature_2m_min", []),
                    "Rain chance (%)": daily.get("precipitation_probability_max", []),
                })
                st.markdown("#### Five-day outlook")
                chart = go.Figure()
                chart.add_trace(go.Scatter(x=forecast["Date"], y=forecast["High (°C)"], mode="lines+markers", name="High", line=dict(color="#67e8f9", width=3), marker=dict(size=7)))
                chart.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Low (°C)"], mode="lines+markers", name="Low", line=dict(color="#8ba8ff", width=3), marker=dict(size=7), fill="tonexty", fillcolor="rgba(139,168,255,.08)"))
                chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=15,b=5), height=300, legend=dict(orientation="h", y=1.15), xaxis_title=None, yaxis_title="Temperature (°C)")
                st.plotly_chart(chart, use_container_width=True)
                st.dataframe(forecast.assign(Date=forecast["Date"].dt.strftime("%a, %d %b")), hide_index=True, use_container_width=True)
    else:
        st.markdown("<div class='panel'><h3>Start with a city</h3><p>Try <b>Delhi</b>, <b>London</b>, <b>Singapore</b> or <b>New York</b>. The app uses Open-Meteo's geocoding and weather APIs; an internet connection is required.</p><div class='mini-note'>Tip: if you just opened the app, click <b>Get weather</b> to load the first result.</div></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Prediction lab
# ------------------------------------------------------------
with climate_tab:
    st.markdown("<div class='section-eyebrow'>02 / Machine learning</div>", unsafe_allow_html=True)
    st.header("Climate prediction lab")
    st.markdown("Move the inputs and explore what the fitted baseline models estimate from the supplied dataset.")
    left, right = st.columns([.9, 1.1], gap="large")
    with left:
        st.markdown("<div class='panel'><h3>Input parameters</h3><p class='mini-note'>Both inputs should be within a plausible range represented in your dataset. CO₂ here means tons per capita, not atmospheric concentration.</p></div>", unsafe_allow_html=True)
        tmin = float(clean_df[TEMP_COL].min()); tmax = float(clean_df[TEMP_COL].max())
        cmin = float(clean_df[CO2_COL].min()); cmax = float(clean_df[CO2_COL].max())
        default_temp = float(clean_df[TEMP_COL].median())
        default_co2 = float(clean_df[CO2_COL].median())
        temperature = st.slider("Average temperature (°C)", min_value=float(np.floor(tmin)), max_value=float(np.ceil(tmax)), value=float(np.clip(default_temp, np.floor(tmin), np.ceil(tmax))), step=0.1)
        co2 = st.slider("CO₂ emissions (tons per capita)", min_value=float(np.floor(cmin)), max_value=float(np.ceil(cmax)), value=float(np.clip(default_co2, np.floor(cmin), np.ceil(cmax))), step=0.1)
        predict_clicked = st.button("Generate estimate ↗", use_container_width=True)
        st.caption(f"Training data range: {tmin:.1f}–{tmax:.1f}°C and {cmin:.1f}–{cmax:.1f} tons per capita.")
    with right:
        row = pd.DataFrame([{TEMP_COL: temperature, CO2_COL: co2}])
        rainfall_prediction = float(models[RAINFALL_COL].predict(row)[0])
        sea_prediction = float(models[SEA_COL].predict(row)[0])
        st.markdown("<div class='panel'><div class='mini-note'>MODEL ESTIMATES</div><h3 style='margin-bottom:4px'>Predicted indicators</h3><p class='mini-note'>Values update when you adjust the sliders. Select Generate estimate to highlight the result.</p></div>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Rainfall estimate", f"{rainfall_prediction:,.1f} mm")
        with p2:
            st.metric("Sea-level rise estimate", f"{sea_prediction:,.2f} mm")
        if predict_clicked:
            st.success("Estimate generated from the fitted Linear Regression models.")
        st.markdown("<div class='mini-note'>These are model outputs, not validated forecasts. The dataset is small and observational, and the inputs alone cannot capture the physical processes that determine rainfall or sea-level change.</div>", unsafe_allow_html=True)
        st.markdown("#### Test-set model performance")
        for target, label in [(RAINFALL_COL, "Rainfall"), (SEA_COL, "Sea-level rise")]:
            met = model_metrics[target]
            st.write(f"**{label}** · RMSE: `{met['rmse']:.2f}` · R²: `{met['r2']:.3f}`")
        st.caption("RMSE is measured in the target's units; lower is better. R² can be negative when the model performs worse than predicting the test-set mean.")

# ------------------------------------------------------------
# Data explorer
# ------------------------------------------------------------
with explore_tab:
    st.markdown("<div class='section-eyebrow'>03 / Dataset analytics</div>", unsafe_allow_html=True)
    st.header("Explore the data")
    st.markdown(f"Source: **{data_source_label}** · Showing the cleaned rows used for model training.")
    d1, d2, d3 = st.columns(3)
    with d1: st.metric("Rows", f"{len(clean_df):,}")
    with d2: st.metric("Countries represented", str(clean_df["Country"].nunique()) if "Country" in clean_df.columns else "—")
    with d3: st.metric("Years covered", f"{int(clean_df['Year'].min())}–{int(clean_df['Year'].max())}" if "Year" in clean_df.columns else "—")
    plot_left, plot_right = st.columns(2, gap="large")
    with plot_left:
        fig = px.scatter(clean_df, x=TEMP_COL, y=RAINFALL_COL, color=CO2_COL, hover_data=["Country"] if "Country" in clean_df.columns else None, color_continuous_scale="Tealgrn", title="Temperature vs rainfall")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=50,b=5), height=370)
        st.plotly_chart(fig, use_container_width=True)
    with plot_right:
        fig2 = px.scatter(clean_df, x=CO2_COL, y=SEA_COL, color=TEMP_COL, hover_data=["Country"] if "Country" in clean_df.columns else None, color_continuous_scale="IceFire", title="CO₂ per capita vs sea-level rise")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=50,b=5), height=370)
        st.plotly_chart(fig2, use_container_width=True)
    with st.expander("View dataset rows"):
        st.dataframe(clean_df, use_container_width=True, hide_index=True)
    csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned dataset ↓", data=csv_bytes, file_name="climate_dataset_cleaned.csv", mime="text/csv")
    corr = clean_df[FEATURES + TARGETS].corr(numeric_only=True)
    fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="Tealgrn", zmin=-1, zmax=1, title="Feature and target correlation")
    fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=50,b=5), height=430)
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------
# Methodology
# ------------------------------------------------------------
with about_tab:
    st.markdown("<div class='section-eyebrow'>04 / How it works</div>", unsafe_allow_html=True)
    st.header("Behind the dashboard")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='panel'><h3>01 · Live weather</h3><p>City search uses Open-Meteo geocoding to find coordinates. The forecast API returns current temperature, apparent temperature, humidity, wind, precipitation, and a five-day temperature outlook.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'><h3>02 · Regression models</h3><p>Two Linear Regression models use average temperature and CO₂ emissions per capita to estimate rainfall and sea-level rise. A reproducible 80/20 train-test split evaluates RMSE and R².</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><h3>03 · Data quality</h3><p>Required model columns are converted to numeric values, invalid values are removed, and rows with missing model inputs or targets are excluded. The app shows how many complete records remain.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel'><h3>04 · Important limitation</h3><p>Correlation does not establish causation. A small tabular dataset and two inputs cannot model the complex atmospheric, oceanic, geographic, and time-dependent processes that control real climate outcomes.</p></div>", unsafe_allow_html=True)
    st.markdown("#### Project stack")
    st.write("**Python** · **Streamlit** · **Pandas** · **NumPy** · **scikit-learn** · **Plotly** · **Open-Meteo API**")

st.markdown("<hr><div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#7f95b3;font-size:.8rem'><span>ATMOS. · Weather & Climate Intelligence</span><span>Built for learning · Not a scientific forecasting service</span></div>", unsafe_allow_html=True)
