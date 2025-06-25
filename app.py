import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import os
from datetime import datetime

# ======================================================================================
# Page Configuration
# ======================================================================================
st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================
# Load Model and Scaler
# ======================================================================================
# --- CHANGE #1: Correct the base directory calculation ---
base_dir = Path(__file__).resolve().parent
MODEL_PATH = base_dir / 'models' / 'final_model.pkl'
SCALER_PATH = base_dir / 'models' / 'scaler.pkl'
# --- CHANGE #2: Correct the history path ---
HISTORY_PATH = base_dir / 'appdata' / 'user_prediction_history.csv'


# ======================================================================================
# Ensure history directory exists
# ======================================================================================
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

# ======================================================================================
# Debug Info
# ======================================================================================
show_debug = not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH))
if show_debug:
    st.sidebar.write("--- Debug Info ---")
    st.sidebar.write(f"Base Directory: `{base_dir}`")
    st.sidebar.write(f"Attempting to load model from: `{MODEL_PATH}`")
    st.sidebar.write(f"Model Exists: `{os.path.exists(MODEL_PATH)}`")
    st.sidebar.write(f"Attempting to load scaler from: `{SCALER_PATH}`")
    st.sidebar.write(f"Scaler Exists: `{os.path.exists(SCALER_PATH)}`")
    st.sidebar.write("--------------------")

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    try:
        with open(MODEL_PATH, 'rb') as f_model:
            model = pickle.load(f_model)
        with open(SCALER_PATH, 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
        return model, scaler
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        return None, None

model, scaler = load_model_and_scaler()

# ======================================================================================
# Application Constants
# ======================================================================================
FEATURES = ["EC", "Cl", "TDS", "Na"]
LABEL_MAPPING = {
    0: "Excellent",
    1: "Good",
    2: "Poor",
    3: "Unsuitable for Drinking",
    4.0: "Very Poor yet Drinkable"
}
RESULT_STYLES = {
    "Excellent": {"color": "green", "advice": "This water is of the highest quality."},
    "Good": {"color": "blue", "advice": "This water is safe and of good quality."},
    "Poor": {"color": "orange", "advice": "Caution: This water is of poor quality and may not be safe for drinking."},
    "Very Poor yet Drinkable": {"color": "red", "advice": "Warning: This water is of very poor quality. Treatment is highly recommended before consumption."},
    "Unsuitable for Drinking": {"color": "darkred", "advice": "DANGER: This water is unsuitable for drinking and may pose significant health risks."}
}

# ======================================================================================
# Sidebar Navigation
# ======================================================================================
st.sidebar.title("Navigation")
main_option = st.sidebar.radio("Go to", [
    "🏠 Home",
    "🧪 Try the Model",
    "📊 Model Development"
])

# ======================================================================================
# History Saving Function
# ======================================================================================
def save_prediction_to_history(input_data, predicted_label, confidence):
    record = input_data.copy()
    record["Timestamp"] = datetime.now().isoformat(timespec='seconds')
    record["Prediction"] = predicted_label
    record["Confidence (%)"] = round(confidence, 2)
    df_record = pd.DataFrame([record])

    if HISTORY_PATH.exists():
        try:
            df_existing = pd.read_csv(HISTORY_PATH)
            df_all = pd.concat([df_existing, df_record], ignore_index=True)
        except pd.errors.EmptyDataError:
            df_all = df_record
    else:
        df_all = df_record

    df_all.to_csv(HISTORY_PATH, index=False)

# ======================================================================================
# Main Content
# ======================================================================================
if model is None or scaler is None:
    st.error("Model or scaler not found. Please check the paths in the debug info in the sidebar and ensure the files exist.")
    st.stop()

if main_option == "🏠 Home":
    st.title("💧 Water Quality Classification App")
    st.markdown("---")
    st.markdown("Welcome to the **Water Quality Predictor**! This application leverages a machine learning model to instantly classify water quality based on four key chemical parameters.")
    st.header("🎯 Project Goal")
    st.info("To provide a fast, accessible, and reliable tool for assessing water quality, overcoming the time and cost barriers of traditional laboratory testing.")
    st.header("🛠️ How It Works")
    st.markdown("""
    The application uses a **Random Forest** model that has been trained on thousands of water samples.
    1.  Go to the **"🧪 Try the Model"** page.
    2.  Input values for the four required features.
    3.  Click the **"Predict"** button to get an instant classification.
    """)

elif main_option == "🧪 Try the Model":
    st.title("🧪 Try the Water Quality Model")
    st.markdown("Enter the chemical parameters of a water sample below to classify its quality.")

    with st.form("prediction_form"):
        st.subheader("Input Water Quality Features")
        col1, col2 = st.columns(2)
        input_data = {}
        with col1:
            input_data["EC"] = st.number_input("Enter Electrical Conductivity (EC)", min_value=0.0, format="%.2f")
            input_data["Cl"] = st.number_input("Enter Chloride (Cl)", min_value=0.0, format="%.2f")
        with col2:
            input_data["TDS"] = st.number_input("Enter Total Dissolved Solids (TDS)", min_value=0.0, format="%.2f")
            input_data["Na"] = st.number_input("Enter Sodium (Na)", min_value=0.0, format="%.2f")
        submitted = st.form_submit_button("Predict Water Quality")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_df_log = input_df.copy()
        for col in FEATURES:
            input_df_log[col] = np.log1p(input_df_log[col])

        scaled_input = scaler.transform(input_df_log)
        prediction = model.predict(scaled_input)[0]
        probs = model.predict_proba(scaled_input)[0]
        confidence = np.max(probs) * 100
        predicted_label = LABEL_MAPPING.get(prediction, "Unknown")

        st.subheader("Prediction Result")
        style = RESULT_STYLES.get(predicted_label, {"color": "gray", "advice": "No specific advice available."})
        st.markdown(f"The model predicts the water quality is: <strong style='color:{style['color']}; font-size: 24px;'>{predicted_label}</strong>", unsafe_allow_html=True)
        st.info(f"**Advice:** {style['advice']}")
        st.success(f"**Model Confidence:** {confidence:.2f}%")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": [LABEL_MAPPING.get(i, str(i)) for i in model.classes_],
            "Probability": probs
        })
        st.bar_chart(prob_df.set_index("Class"))

        # Save history
        save_prediction_to_history(input_data, predicted_label, confidence)

    # Show history if available
    if HISTORY_PATH.exists():
        try:
            history_df = pd.read_csv(HISTORY_PATH)
            if not history_df.empty:
                st.markdown("---")
                st.subheader("📜 Prediction History")
                st.dataframe(history_df)
                if st.button("🗑️ Clear Prediction History"):
                    HISTORY_PATH.unlink()
                    st.success("Prediction history cleared.")
            else:
                st.info("No predictions have been made yet.")
        except pd.errors.EmptyDataError:
            st.info("No predictions have been made yet.")
    else:
        st.info("No predictions have been made yet.")

elif main_option == "📊 Model Development":
    st.title("📊 Model Development Workflow")
    st.markdown("This section showcases the key visualizations generated during the development and evaluation of the model.")
    figures_dir = base_dir / 'reports' / 'figures'

    if not figures_dir.is_dir():
        st.error(f"Figures directory not found at '{figures_dir}'. Please ensure the project structure is correct.")
    else:
        st.subheader("Target Variable Distribution")
        st.image(str(figures_dir / "target_variable_distribution.png"), caption="Distribution of classes in the dataset, highlighting the class imbalance.", use_container_width=True)
        st.subheader("Feature Distributions (Top 8)")
        st.image(str(figures_dir / "top_features_distribution.png"), caption="Distributions of the top 8 features, showing significant right-skewness before transformation.", use_container_width=True)
        st.subheader("Model Performance Comparison")
        st.image(str(figures_dir / "confusion_matrix_Tuned_Random_Forest.png"), caption="Confusion matrix for the final, tuned Random Forest model.", use_container_width=True)
        st.image(str(figures_dir / "feature_importance_Random_Forest.png"), caption="Feature importance plot for the final model.", use_container_width=True)

st.markdown("---")
st.markdown("This application uses a fine-tuned **Random Forest Classifier** to predict water quality.")