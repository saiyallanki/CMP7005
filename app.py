import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ==================================================
# GLOBAL SETTINGS
# ==================================================
st.set_page_config(
    page_title="India Air Quality Analysis & Prediction",
    layout="wide"
)

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)

st.markdown("""
<style>
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    text-align: center;
    min-height: 120px;
}

/* Fix text wrapping issues */
.card h2 {
    color: #2563eb;
    margin-bottom: 5px;
    font-size: 26px;
    word-break: keep-all;        
    overflow-wrap: normal;
    white-space: nowrap;         
    text-overflow: ellipsis;    

/* Smaller subtitle */
.card p {
    color: #374151;
    font-size: 15px;
}
</style>

""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA & MODEL
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("combined_df.csv", parse_dates=["Date"])

@st.cache_resource
def load_model():
    with open("aqi_random_forest_model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# -------------------------------
# Home page styling 
# -------------------------------
st.markdown("""
<style>

/* Overall app background */
.main {
    background-color: #f4f7fb;
}

/* Home page container */
.home-card {
    background: linear-gradient(135deg, #ffffff, #eef3f9);
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Section headers */
.home-title {
    color: #1f4e79;
    font-size: 42px;
    font-weight: 700;
}

.home-subtitle {
    color: #3b6ea5;
    font-size: 22px;
    margin-bottom: 20px;
}

/* Highlight boxes */
.highlight-box {
    background-color: #eaf2fb;
    padding: 18px;
    border-radius: 12px;
    margin-top: 15px;
    border-left: 6px solid #1f4e79;
}

/* Bullet text */
.home-text {
    font-size: 16px;
    color: #2c3e50;
    line-height: 1.6;
}

/* Footer note */
.footer-note {
    background-color: #e8f5e9;
    padding: 15px;
    border-radius: 10px;
    color: #1b5e20;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Data Overview",
        "Exploratory Data Analysis",
        "Spatio-Temporal AQI Heatmap",
        "Temporal Trends",
        "Model Evaluation & Features",
        "AQI Prediction"
    ]
)

# =========================================================
# 1️⃣ HOME — 
# =========================================================
if section == "Home":

    # ---------- Custom CSS ----------
    st.markdown("""
    <style>
    .main {
        background-color: #f7f9fb;
    }
    .home-header {
        background: linear-gradient(90deg, #74ebd5, #acb6e5);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: #1f2937;
    }
    .home-header h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }
    .home-header h3 {
        font-weight: 400;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
        text-align: center;
    }
    .card h2 {
        color: #2563eb;
        margin-bottom: 5px;
        font-size: 26px;
    	word-break: keep-all;        
    	overflow-wrap: normal;
    	white-space: nowrap;        
    	text-overflow: ellipsis;
    }
    .card p {
        color: #374151;
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Header ----------
    st.markdown("""
    <div class="home-header">
        <h1>India Air Quality Intelligence Platform</h1>
        <h3>Data Analytics & Machine Learning–Based AQI Prediction</h3>
    </div>
    """, unsafe_allow_html=True)

    # ---------- AQI EXPLANATION ----------
    st.markdown("### 🧪 What is AQI (Air Quality Index)?")
    st.markdown("""
    The **Air Quality Index (AQI)** is a standardized indicator used to communicate
    how polluted the air currently is or how polluted it is forecast to become.

    Instead of reporting individual pollutant concentrations, AQI converts multiple
    air pollutants into a **single numerical value** that is easier for the public to understand.

    AQI is calculated using major pollutants such as:
    - **PM2.5 & PM10** – fine particulate matter affecting lungs and heart  
    - **NO₂ (Nitrogen Dioxide)** – traffic and industrial emissions  
    - **SO₂ (Sulphur Dioxide)** – power plants and fossil fuels  
    - **O₃ (Ozone)** – secondary pollutant formed by sunlight reactions  
    - **CO (Carbon Monoxide)** – incomplete combustion  

    👉 **Higher AQI = Poorer air quality = Greater health risk**
    """)

    # ---------- AQI LEGEND ----------
    st.markdown("### 🎨 AQI Categories & Health Impact")
    st.markdown("""
    | AQI Range | Category | Health Implications |
    |----------|----------|--------------------|
    | 0–50 | 🟢 Good | Clean air, minimal risk |
    | 51–100 | 🟡 Satisfactory | Minor breathing discomfort |
    | 101–200 | 🟠 Moderate | Discomfort for sensitive groups |
    | 201–300 | 🔴 Poor | Respiratory illness likely |
    | 301–400 | 🟣 Very Poor | Serious health effects |
    | 401+ | ⚫ Severe | Emergency conditions |
    """)

    # ---------- PROJECT CONTEXT ----------
    st.markdown("### 🌍 Why This Project Matters")
    st.markdown("""
    India experiences some of the highest air-pollution levels globally.
    Continuous monitoring and intelligent analysis of AQI data
    are essential for **public health awareness, policy formulation,
    and sustainable urban planning**.

    This application transforms **raw environmental data**
    into **actionable insights and predictions**.
    """)

    # ---------- OBJECTIVES ----------
    st.markdown("### 🎯 Project Objectives")
    st.markdown("""
    ✔ Analyse spatial and temporal air-quality trends  
    ✔ Identify key pollutants influencing AQI  
    ✔ Apply robust preprocessing and feature engineering  
    ✔ Train interpretable machine learning models  
    ✔ Deploy an interactive decision-support system  
    """)

    # ---------- PIPELINE ----------
    st.markdown("### 🧠 Machine Learning Workflow")
    st.markdown("""
    **Data Collection → Cleaning → Feature Engineering → Model Training → Evaluation → Deployment**

    - Temporal features (year, month, weekday, weekend)  
    - Random Forest model for non-linear pollutant interactions  
    - Feature importance for explainability  
    """)

    # ---------- HOW TO USE ----------
    st.markdown("### 🚀 How to Use This Application")
    st.markdown("""
    1️⃣ Review dataset structure and quality  
    2️⃣ Explore pollutant distributions and AQI trends  
    3️⃣ Analyse relationships between pollutants  
    4️⃣ Examine model performance and key drivers  
    5️⃣ Predict AQI using real-time user inputs  
    """)

    st.success("✅ This platform demonstrates a complete end-to-end data science workflow — from environmental data to real-world decision support.")


# ==================================================
# 2️⃣ DATA OVERVIEW
# ==================================================
elif section == "Data Overview":
    st.title("📊 Data Overview")

    # -------------------------------
    # High-level dataset metrics
    # -------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Cities Covered", df["City"].nunique())
    start_date = df["Date"].min().date()
    end_date = df["Date"].max().date()
    col3.metric("Date Range", f"{start_date.year}–{end_date.year}")
    col3.caption(f"{start_date} → {end_date}")
    col4.metric("Average AQI", f"{df['AQI'].mean():.1f}")

    st.divider()

    # -------------------------------
    # Sample data preview
    # -------------------------------
    st.subheader("📄 Sample Records")
    st.markdown(
        "A preview of the dataset helps verify structure, formatting, "
        "and the nature of pollutant measurements."
    )
    st.dataframe(df.head(10), use_container_width=True)

    # -------------------------------
    # Column data types
    # -------------------------------
    st.subheader("🧬 Column Data Types")
    st.markdown(
        "Understanding data types is essential for selecting appropriate "
        "analysis and preprocessing techniques."
    )

    dtype_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str)
    })
    st.dataframe(dtype_df, use_container_width=True)

    # -------------------------------
    # Statistical summary (NUMERIC ONLY)
    # -------------------------------
    st.subheader("📐 Statistical Summary (Numeric Variables)")
    st.markdown(
        "The table below summarises central tendency and dispersion "
        "for all numeric air-quality variables."
    )

    numeric_cols = [
        'PM2.5','PM10','NO','NO2','NOx','NH3',
        'CO','SO2','O3','Benzene','Toluene','Xylene','AQI'
    ]

    # Safety: ensure numeric
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    stats_df = df[numeric_cols].describe().T.round(2)
    st.dataframe(stats_df, use_container_width=True)

    st.markdown("""
    📌 **Interpretation:**  
    Particulate matter (PM2.5 and PM10) exhibits high variability across observations,
    indicating substantial spatial and temporal differences in pollution levels.
    AQI values show a wide range, highlighting periods of both moderate and hazardous air quality.
    """)

    # -------------------------------
    # Missing value analysis
    # -------------------------------
    st.subheader("⚠️ Missing Values Analysis")
    st.markdown(
        "Missing values can bias statistical analysis and model performance. "
        "Identifying them is a critical preprocessing step."
    )

    missing_df = (
        df.isnull()
        .sum()
        .reset_index()
        .rename(columns={"index": "Column", 0: "Missing Count"})
    )

    st.dataframe(missing_df, use_container_width=True)

    st.markdown("""
    📌 **Interpretation:**  
    Certain gaseous pollutants and VOCs contain missing observations,
    which were handled during preprocessing using city-level median imputation
    to preserve distributional characteristics.
    """)


# ==================================================
# 3️⃣ EXPLORATORY DATA ANALYSIS
# ==================================================
elif section == "Exploratory Data Analysis":
    st.title("📈 Exploratory Data Analysis")

    pollutant = st.selectbox(
        "Select Pollutant",
        ['PM2.5','PM10','NO2','CO','SO2','O3']
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df[pollutant].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"{pollutant} Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(y=df[pollutant], ax=ax, color="orange")
        ax.set_title(f"{pollutant} Boxplot")
        st.pyplot(fig)

    st.subheader("Pollutant vs AQI Relationship")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df[pollutant],
        y=df["AQI"],
        alpha=0.4,
        color="purple",
        ax=ax
    )
    ax.set_xlabel(pollutant)
    ax.set_ylabel("AQI")
    ax.set_title(f"{pollutant} vs AQI")
    st.pyplot(fig)

# =========================================================
# 🌍 SPATIO-TEMPORAL AQI HEATMAP
# =========================================================
elif section == "Spatio-Temporal AQI Heatmap":

    st.title("🌍 Spatio-Temporal AQI Heatmap")

    st.markdown("""
    This section presents **city-wise AQI patterns over time**, helping identify
    **pollution hotspots**, **regional disparities**, and **temporal variations**
    in air quality across India.
    """)

    # ---------- KPI SUMMARY ----------
    avg_aqi = round(df["AQI"].mean(), 1)
    worst_city = df.groupby("City")["AQI"].mean().idxmax()
    best_city = df.groupby("City")["AQI"].mean().idxmin()
    city_count = df["City"].nunique()

    st.markdown("### 📊 National AQI Summary")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="card">
            <h2>{avg_aqi}</h2>
            <p>Average AQI</p>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="card">
            <h2>{worst_city}</h2>
            <p>Most Polluted City</p>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="card">
            <h2>{best_city}</h2>
            <p>Cleanest City</p>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="card">
            <h2>{city_count}</h2>
            <p>Cities Analysed</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---------- YEAR SELECTION ----------
    df["Year"] = df["Date"].dt.year

    selected_year = st.slider(
        "Select Year",
        int(df["Year"].min()),
        int(df["Year"].max()),
        int(df["Year"].max())
    )

    st.info(f"Displaying **average AQI by city** for the year **{selected_year}**")

    year_df = df[df["Year"] == selected_year]

    # ---------- CITY-WISE AQI AGGREGATION ----------
    city_year_avg = (
        year_df.groupby("City")["AQI"]
        .mean()
        .sort_values(ascending=False)
        .to_frame()
    )

    # ---------- HEATMAP ----------
    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(
        city_year_avg,
        cmap="RdYlGn_r",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Average AQI"},
        ax=ax
    )

    ax.set_title(f"Average AQI by City ({selected_year})")
    ax.set_xlabel("")
    ax.set_ylabel("City")

    st.pyplot(fig)

    # ---------- INTERPRETATION ----------
    st.markdown("### 🔍 Interpretation")

    st.markdown("""
    - 🔴 **Red shades** indicate cities with consistently poor air quality  
    - 🟢 **Green shades** represent relatively cleaner environments  
    - Changes across years highlight **seasonal effects, industrial activity,
      and urban growth patterns**

    This spatio-temporal analysis supports **policy planning, environmental monitoring,
    and public health risk assessment**.
    """)

# ==================================================
# 4️⃣ TEMPORAL TRENDS
# ==================================================
elif section == "Temporal Trends":
    st.title("📆 AQI Temporal Trends")

    city = st.selectbox("Select City", sorted(df["City"].unique()))
    temp_df = df[df["City"] == city].sort_values("Date")

    temp_df["AQI_7day_avg"] = temp_df["AQI"].rolling(7).mean()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(temp_df["Date"], temp_df["AQI"], color="crimson", alpha=0.4, label="Daily AQI")
    ax.plot(temp_df["Date"], temp_df["AQI_7day_avg"],
            color="black", linestyle="--", label="7-day Average")
    ax.set_title(f"AQI Trend – {city}")
    ax.set_ylabel("AQI")
    ax.legend()
    st.pyplot(fig)

# ==================================================
# 5️⃣ MODEL EVALUATION & FEATURES
# ==================================================
elif section == "Model Evaluation & Features":
    st.title("🧠 Model Evaluation")

    st.markdown("""
    **Random Forest Performance (Test Set)**  
    - **R² Score:** 0.903  
    - **MAE:** 15.02  
    - **RMSE:** 31.79  
    """)

    st.subheader("Feature Importance")

    feature_names = [
        'PM2.5','PM10','NO','NO2','NOx','NH3','CO',
        'SO2','O3','Benzene','Toluene','Xylene',
        'City_le','year','month','day','dayofweek','is_weekend'
    ]

    importances = model.named_steps['rf'].feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=fi.head(12).values, y=fi.head(12).index, palette="magma", ax=ax)
    ax.set_title("Top 12 Important Features")
    st.pyplot(fig)

# ==================================================
# 6️⃣ AQI PREDICTION (FIXED & DEPLOYMENT SAFE)
# ==================================================
elif section == "AQI Prediction":
    st.title("🤖 AQI Prediction")

    col1, col2 = st.columns(2)
    input_data = {}

    # ----------- RAW POLLUTANT INPUTS -----------
    with col1:
        for col in [
            'PM2.5','PM10','NO','NO2','NOx','NH3',
            'CO','SO2','O3','Benzene','Toluene','Xylene'
        ]:
            input_data[col] = st.number_input(
                col, min_value=0.0, value=10.0
            )

    # ----------- METADATA INPUTS -----------
    with col2:
        city_selected = st.selectbox(
            "City", sorted(df["City"].unique())
        )
        date_selected = st.date_input("Date")

    # ----------- PREDICT -----------
    if st.button("Predict AQI"):

        date_selected = pd.to_datetime(date_selected)

        # ----------- CREATE RAW FEATURE ROW -----------
        input_row = {
            **input_data,
            "City": city_selected,
            "Date": date_selected
        }

        input_df = pd.DataFrame([input_row])

        # ----------- DERIVE DATE FEATURES (SAME AS TRAINING) -----------
        input_df["year"] = input_df["Date"].dt.year
        input_df["month"] = input_df["Date"].dt.month
        input_df["day"] = input_df["Date"].dt.day
        input_df["dayofweek"] = input_df["Date"].dt.dayofweek
        input_df["is_weekend"] = (input_df["dayofweek"] >= 5).astype(int)

        # ----------- SAFE DEFAULTS FOR LAG / ROLLING FEATURES -----------
        input_df["PM2.5_roll7"] = input_df["PM2.5"]
        input_df["PM2.5_roll30"] = input_df["PM2.5"]
        input_df["PM10_roll7"] = input_df["PM10"]
        input_df["PM10_roll30"] = input_df["PM10"]
        input_df["PM2.5_lag1"] = input_df["PM2.5"]
        input_df["AQI_lag1"] = df["AQI"].mean()

        input_df["PM_ratio"] = input_df["PM2.5"] / (input_df["PM10"] + 1e-6)
        input_df["NO2_to_NOx"] = input_df["NO2"] / (input_df["NOx"] + 1e-6)
        input_df["VOC_sum"] = (
            input_df["Benzene"] +
            input_df["Toluene"] +
            input_df["Xylene"]
        )

        # ----------- DROP UNUSED COLUMNS -----------
        input_df = input_df.drop(columns=["Date"])

        # ----------- PREDICTION (PIPELINE HANDLES EVERYTHING) -----------
        prediction = model.predict(input_df)[0]

        st.success(f"✅ Predicted AQI: **{prediction:.2f}**")


