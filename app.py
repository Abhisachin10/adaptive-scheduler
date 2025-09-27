import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- Page Config ---
st.set_page_config(page_title="🤖 AI-Driven Adaptive Scheduling", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        background-color: beige;
        color: brown;
    }
    h1, h2, h3, h4, h5, h6 {
        text-align: center;
        color: brown;
    }
    .stButton>button {
        background-color: brown;
        color: beige;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #8B4513;
        color: white;
        transform: scale(1.05);
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: #fff8dc;
        color: brown;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    .metric-card:hover {
        box-shadow: 0px 4px 16px rgba(0,0,0,0.3);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🤖 AI-Driven Adaptive Scheduling")

# --- Feature Engineering Function ---
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Production_Load" in df and "Deadline_Hours" in df:
        df["urgency"] = df["Production_Load"] / (df["Deadline_Hours"] + 1e-3)
    if "Available_Operators" in df and "Available_Machines" in df:
        df["operator_machine_ratio"] = df["Available_Operators"] / (df["Available_Machines"] + 1)
    if "Expected_Runtime_Min" in df and "Machine_Efficiency" in df:
        df["adjusted_runtime"] = df["Expected_Runtime_Min"] / (df["Machine_Efficiency"] + 1e-3)
    if "Production_Load" in df and "Available_Operators" in df:
        df["load_per_operator"] = df["Production_Load"] / (df["Available_Operators"] + 1)
    if "Shift" in df:
        df["shift_binary"] = df["Shift"].apply(lambda x: 1 if str(x).lower() == "night" else 0)
    return df

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df = add_engineered_features(df)

    engineered_features = ["urgency", "operator_machine_ratio", "adjusted_runtime",
                           "load_per_operator", "shift_binary"]

    st.write("✅ Dataset loaded successfully with engineered features!")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    st.subheader("⚙ Select Features and Target Columns")

    input_cols = st.multiselect(
        "Select Input Columns (X)", 
        [c for c in all_columns if c not in engineered_features],
        default=[c for c in all_columns if c not in engineered_features]
    )
    output_cols = st.multiselect(
        "Select Output Columns (y)", 
        [c for c in all_columns if c not in engineered_features],
        default=[c for c in all_columns if c not in engineered_features and c not in input_cols]
    )

    if input_cols and output_cols and st.button("🚀 Train Model"):
        X = df[input_cols]
        y = df[output_cols]

        X_encoded = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred, multioutput="uniform_average")

        st.subheader("📊 Model Accuracy")
        st.write(f"✅ R² Score: {r2*100:.2f}%")

        st.session_state["model"] = model
        st.session_state["features"] = X_encoded.columns
        st.session_state["output_cols"] = output_cols
        st.session_state["input_cols"] = input_cols
        st.session_state["df"] = df

if "model" in st.session_state:
    st.subheader("🔧 Predict for New Input")

    df = st.session_state["df"]
    input_cols = st.session_state["input_cols"]
    output_cols = st.session_state["output_cols"]

    input_data = {}
    for col in input_cols:
        if df[col].dtype in ["int64", "float64"]:
            val = st.number_input(
                f"{col}", 
                min_value=0.0, 
                max_value=10000.0, 
                value=float(df[col].mean())
            )
            input_data[col] = val
        else:
            options = df[col].unique().tolist()
            val = st.selectbox(f"{col}", options)
            input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        input_df = add_engineered_features(input_df)

        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=st.session_state["features"], fill_value=0)

        prediction = st.session_state["model"].predict(input_encoded).flatten()

        st.success("🎯 Predictions:")
        for i, col in enumerate(output_cols):
            if col.lower() in ["machine", "manpower"]:
                val = int(round(prediction[i]))
            else:
                val = round(prediction[i], 2)
            st.markdown(f'<div class="metric-card">{col}: {val}</div>', unsafe_allow_html=True)
else:
    st.info("📥 Please upload a CSV, select columns, and click 🚀 Train Model")
