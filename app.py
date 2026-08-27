import streamlit as st
import pandas as pd
import joblib
from catboost import Pool


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# DARK GREEN + BROWN THEME
# Sections stacked vertically, 4 fields per row, one
# consistent color treatment across every card.
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GLOBAL
   ===================================================== */

html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
    background: #07130D !important;
}

/* Fully remove Streamlit's default toolbar — it includes a
   red "Deploy" button and status dot that were the last
   leftover red on the page. */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Kill any red focus rings Streamlit's default theme injects
   on inputs, links, spinners, etc. */
*:focus,
*:focus-visible {
    outline-color: #16A34A !important;
}

a, a:visited {
    color: #4ADE80 !important;
}

::selection {
    background: #14532D !important;
    color: #ECFDF5 !important;
}

.block-container {
    max-width: 90% !important;
    width: 90% !important;
    padding-top: 0.8rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"] {
    background: transparent !important;
}

[data-testid="stVerticalBlock"] { gap: 0.25rem !important; }
[data-testid="stHorizontalBlock"] { gap: 0.7rem !important; }

/* =====================================================
   TEXT
   ===================================================== */

p, label, span, div { color: #D1FAE5; }

h1, h2, h3 { color: #86EFAC !important; }

/* =====================================================
   AUTHOR CARD
   ===================================================== */

.author-card {
    background: transparent !important;
    border: none;
    padding: 0;
    margin: 4px auto 2px auto;
    max-width: 520px;
    text-align: center;
}

.author-label {
    color: #B08D57 !important;
    font-size: 9px;
    font-weight: bold;
    letter-spacing: 1.5px;
    margin: 0 !important;
}

.author-name {
    color: #8FBFA0 !important;
    font-size: 12px;
    font-weight: 600;
    margin: 1px 0 0 0 !important;
}

/* =====================================================
   TITLE — the clear hero of the page
   ===================================================== */

.hero-wrap {
    text-align: center;
    padding: 10px 0 16px 0;
    margin-bottom: 6px;
    border-bottom: 1px solid #14532D;
}

.main-title {
    color: #ECFDF5 !important;
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.25;
    margin: 0 !important;
    padding: 0 !important;
}

.main-title .accent {
    color: #4ADE80 !important;
}

.subtitle {
    color: #B08D57 !important;
    text-align: center;
    font-size: 13px;
    font-weight: 500;
    margin: 6px 0 0 0 !important;
}

/* =====================================================
   SECTION CARDS — one consistent color treatment,
   stacked full-width, one under the other
   ===================================================== */

.section-card {
    background: #0B2E1A ;
    border: 1px solid #0B2E1A;
    border-left: 4px solid #86EFAC;
    border-radius: 12px;
    padding: 10px 16px 6px 16px;
    margin-bottom: 10px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.20);
}

.section-title {
    color: #86EFAC !important;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 6px !important;
    padding-bottom: 5px;
    
}

/* =====================================================
   INPUT LABELS
   ===================================================== */

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #BBF7D0 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    margin-bottom: 1px !important;
}

/* =====================================================
   NUMBER INPUT — left in its native/default Streamlit look,
   only spacing is adjusted to fit the layout
   ===================================================== */

div[data-testid="stNumberInput"] { margin-bottom: 2px !important; }

div[data-testid="stNumberInput"] > div {
    min-height: 32px !important;
}

div[data-testid="stNumberInput"] input {
    height: 32px !important;
    min-height: 32px !important;
    font-size: 12px !important;
}

div[data-testid="stNumberInput"] button {
    height: 30px !important;
}

/* =====================================================
   SELECT BOX — left in its native/default Streamlit look,
   only spacing is adjusted to fit the layout
   ===================================================== */

div[data-testid="stSelectbox"] { margin-bottom: 2px !important; }

div[data-baseweb="select"] > div {
    min-height: 32px !important;
    height: 32px !important;
}

div[data-baseweb="select"] span {
    font-size: 12px !important;
}

/* =====================================================
   BUTTON
   ===================================================== */

div.stButton { margin-top: 4px !important; margin-bottom: 4px !important; }

div.stButton > button {
    width: 100% !important;
    height: 42px !important;
    background: #16A34A !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}

div.stButton > button:hover {
    background: #78350F !important;
    color: white !important;
}

/* =====================================================
   SUCCESS RESULT
   ===================================================== */

div[data-testid="stAlert"] {
    background: #0B2E1A !important;
    border: 1px solid #22C55E !important;
    border-left: 4px solid #78350F !important;
    border-radius: 9px !important;
    padding: 8px 12px !important;
    margin: 4px 0 !important;
}


/* =====================================================
   DIVIDER
   ===================================================== */

hr { border-color: #14532D !important; margin: 5px 0 !important; }

/* =====================================================
   FOOTER
   ===================================================== */

div[data-testid="stCaptionContainer"] {
    color: #B08D57 !important;
    text-align: center !important;
    font-size: 9px !important;
    margin-top: 2px !important;
}

/* =====================================================
   HIDE STREAMLIT MENU / FOOTER
   ===================================================== */

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* =====================================================
   MOBILE / SMALL SCREEN
   ===================================================== */

@media (max-width: 900px) {
    .block-container { padding-left: 0.7rem !important; padding-right: 0.7rem !important; }
    .main-title { font-size: 22px; }
    .section-card { padding: 8px; }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# AUTHOR
# =========================================================

# st.markdown("""
# <div class="author-card">
#     <div class="author-label">AUTHOR</div>
#     <div class="author-name">Kareem Hussien Abdelmonm Tawfik</div>
# </div>
# """, unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown("""
    <div class="author-card">
        <div class="author-label">AUTHOR</div>
        <div class="author-name">Kareem Hussien Abdelmonm Tawfik</div>
    </div>  
    
    <div class="hero-wrap">
        <div class="main-title">🌱 <span class="accent">Fertilizer</span> Recommendation System</div>
        <div class="subtitle">Smart fertilizer recommendation using Machine Learning</div>
    </div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/catboost_fertilizer_model.pkl"
    )


model = load_model()


# =========================================================
# SOIL SECTION (stacked, 4 fields per row)
# =========================================================

st.markdown("""
<div class="section-card">
    <div class="section-title">🌱 Soil Information</div>
""", unsafe_allow_html=True)

row1 = st.columns(4)
row2 = st.columns(4)

with row1[0]:
    soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Black", "Red"])
with row1[1]:
    soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.07, step=0.01)
with row1[2]:
    soil_moisture = st.number_input("Soil Moisture", min_value=0.0, value=34.98, step=0.01)
with row1[3]:
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=0.32, step=0.01)

with row2[0]:
    electrical_conductivity = st.number_input("Electrical Conductivity", min_value=0.0, value=1.87, step=0.01)
with row2[1]:
    nitrogen_level = st.number_input("Nitrogen Level", min_value=0.0, value=61.0, step=1.0)
with row2[2]:
    phosphorus_level = st.number_input("Phosphorus Level", min_value=0.0, value=44.0, step=1.0)
with row2[3]:
    potassium_level = st.number_input("Potassium Level", min_value=0.0, value=84.0, step=1.0)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# CROP SECTION (stacked, 4 fields per row)
# =========================================================

st.markdown("""
<div class="section-card">
    <div class="section-title">🌾 Crop Information</div>
""", unsafe_allow_html=True)

row3 = st.columns(4)
row4 = st.columns(4)

with row3[0]:
    crop_type = st.selectbox("Crop Type", ["Cotton", "Rice", "Wheat", "Maize", "Sugarcane"])
with row3[1]:
    crop_growth_stage = st.selectbox(
        "Crop Growth Stage",
        ["Seedling", "Vegetative", "Flowering", "Fruiting", "Harvest", "Maturity"]
    )
with row3[2]:
    previous_crop = st.selectbox("Previous Crop", ["Cotton", "Rice", "Wheat", "Maize", "Sugarcane"])
with row3[3]:
    irrigation_type = st.selectbox("Irrigation Type", ["Canal", "Drip", "Sprinkler", "Rainfed"])

with row4[0]:
    fertilizer_used_last_season = st.number_input(
        "Fertilizer Used Last Season", min_value=0.0, value=297.15, step=0.01
    )
with row4[1]:
    yield_last_season = st.number_input("Yield Last Season", min_value=0.0, value=1.19, step=0.01)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# ENVIRONMENT SECTION (stacked, 4 fields per row)
# =========================================================

st.markdown("""
<div class="section-card">
    <div class="section-title">🌦️ Environment</div>
""", unsafe_allow_html=True)

row5 = st.columns(4)

with row5[0]:
    temperature = st.number_input("Temperature", min_value=-50.0, max_value=100.0, value=19.84, step=0.01)
with row5[1]:
    humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=83.31, step=0.01)
with row5[2]:
    rainfall = st.number_input("Rainfall", min_value=0.0, value=1693.22, step=0.01)
with row5[3]:
    season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])

row6 = st.columns(4)

with row6[0]:
    region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

if st.button("🌱  Recommend Fertilizer"):

    input_data = pd.DataFrame({
        "Soil_Type": [soil_type],
        "Soil_pH": [soil_ph],
        "Soil_Moisture": [soil_moisture],
        "Organic_Carbon": [organic_carbon],
        "Electrical_Conductivity": [electrical_conductivity],
        "Nitrogen_Level": [nitrogen_level],
        "Phosphorus_Level": [phosphorus_level],
        "Potassium_Level": [potassium_level],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Rainfall": [rainfall],
        "Crop_Type": [crop_type],
        "Crop_Growth_Stage": [crop_growth_stage],
        "Season": [season],
        "Irrigation_Type": [irrigation_type],
        "Previous_Crop": [previous_crop],
        "Region": [region],
        "Fertilizer_Used_Last_Season": [fertilizer_used_last_season],
        "Yield_Last_Season": [yield_last_season]
    })

    cat_features = [
        "Soil_Type",
        "Crop_Type",
        "Crop_Growth_Stage",
        "Season",
        "Irrigation_Type",
        "Previous_Crop",
        "Region"
    ]

    input_pool = Pool(data=input_data, cat_features=cat_features)

    prediction = model.predict(input_pool)

    fertilizer = prediction[0]

    if isinstance(fertilizer, (list, tuple)):
        fertilizer = fertilizer[0]

    st.success(f"🌿 Recommended Fertilizer: {fertilizer[0]}")


# =========================================================
# FOOTER
# =========================================================

st.caption("Fertilizer Recommendation System • Machine Learning Project")
