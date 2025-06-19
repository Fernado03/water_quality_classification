import streamlit as st
import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from nbconvert import HTMLExporter
import nbformat
import streamlit.components.v1 as components
from PIL import Image
import os

st.set_page_config(page_title="Notebook Viewer", layout="wide")

# Paths to your notebooks
NOTEBOOK_PATHS = {
    "Phase 1: EDA & Feature Selection": "notebooks/01_eda_and_feature_selection.ipynb",
    "Phase 2: Model Training & Evaluation": "notebooks/02_model_training_and_evaluation.ipynb",
    "Phase 3: Hyperparameter Tuning": "notebooks/03_hyperparameter_tuning.ipynb"
}

# Load trained model and scaler
MODEL_PATH = 'models/final_gradient_boosting_model.pkl'
SCALER_PATH = 'models/min_max_scaler.pkl'

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

FEATURES = ["EC", "Cl", "TDS", "Na"]

label_mapping = {
    0: "Excellent",
    1: "Good",
    2: "Poor",
    3: "Unsuitable for Drinking",
    4: "Very Poor yet Drinkable"
}

# 🟢 Safe method using HTMLExporter instead of subprocess
def render_notebook_html(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_node = nbformat.read(f, as_version=4)

        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True  # Hide code cells
        html_exporter.exclude_output_prompt = True
        html_exporter.template_name = "lab"  # Can be 'classic' or 'lab'

        body, _ = html_exporter.from_notebook_node(notebook_node)
        components.html(body, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"Failed to render notebook: {e}")

# Sidebar Navigation
st.sidebar.title("Navigation")
main_option = st.sidebar.radio("Go to", [
    "Home",
    "Model Development",
    "Try the Model",
    "Model Visualizations"
])

# ========== Home ==========
if main_option == "Home":
    st.title("Home")
    st.title("💧 Water Quality Classification App")

    st.markdown("""
    Welcome to the **Water Quality Predictor**!  
    This app uses a machine learning model to classify water samples based on key features like:

    - Electrical Conductivity (EC)
    - Chloride (Cl)
    - Sodium (Na)
    - Total Dissolved Solids (TDS)

    ---

    ### 🔧 How to Use

    1. Go to the **"Try the Model"** section from the sidebar.
    2. Input the values for the 4 water quality features.
    3. Click the **"Predict"** button.
    4. The app will display the predicted class and label (e.g., **Safe** or **Unsafe** water).

    Ready to begin? Head over to the model tab and try it out! 🚀
    """)

# ========== Model Development ==========
elif main_option == "Model Development":
    st.title("Model Development")
    st.subheader("Dataset")
    st.write("`water_quality.csv`")
    data = pd.read_csv("data/raw/water_quality.csv")
    st.write(data)

    st.title("Jupyter Notebooks")
    tab_titles = list(NOTEBOOK_PATHS.keys())
    tabs = st.tabs([f"📘 {title}" for title in tab_titles])

    for i, title in enumerate(tab_titles):
        with tabs[i]:
            st.markdown(f"### {title}")
            notebook_file = NOTEBOOK_PATHS[title]
            render_notebook_html(notebook_file)

# ========== Try the Model ==========
elif main_option == "Try the Model":
    st.title("Try the Model")

    sub_option = st.sidebar.selectbox("Choose input method:", ["Manual Input", "Upload CSV"])

    if sub_option == "Manual Input":
        selected_columns = ["EC", "Cl", "TDS", "Na", "WQI", "Water Quality Classification"]
        data_test_self = pd.read_csv("data/raw/water_quality.csv", usecols=selected_columns)
        data_test_self = data_test_self[selected_columns]
        st.header("Sample Data for Reference")
        st.write(data_test_self)

        input_data = {feature: st.number_input(f"{feature}:", format="%.4f") for feature in FEATURES}

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            probs = model.predict_proba(scaled_input)[0]
            confidence = np.max(probs) * 100
            predicted_label = label_mapping.get(prediction, "Unknown")

            st.success(f"Predicted Class Number: **{prediction}**")
            st.success(f"Predicted Label: **{predicted_label}**")
            st.info(f"📈 Model Confidence: **{confidence:.2f}%**")

            # --- Bar Chart of Probabilities ---
            st.subheader("🔍 Class Probabilities")
            class_names = model.classes_
            prob_df = pd.DataFrame({
                "Class": [label_mapping.get(i, str(i)) for i in class_names],
                "Probability": probs
            })

            st.bar_chart(prob_df.set_index("Class"))

    elif sub_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            df_uploaded = pd.read_csv(uploaded_file)
            if all(feature in df_uploaded.columns for feature in FEATURES):
                scaled = scaler.transform(df_uploaded[FEATURES])
                predictions = model.predict(scaled)
                df_uploaded['Prediction'] = predictions
                st.dataframe(df_uploaded)

                csv = df_uploaded.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
            else:
                st.error("Uploaded CSV must contain all required features.")

# ========== Model Visualizations ==========
elif main_option == "Model Visualizations":
    st.title("Model Visualizations")

    image_dir = Path(os.path.abspath(".")) / "notebooks" / "static_figures"
    image_files = sorted([f for f in image_dir.glob("*.png")])

    if not image_files:
        st.error("No images found in the directory.")
    else:
        image_index = st.slider("Use arrow to browse images:", min_value=0, max_value=len(image_files) - 1, format="Image %d")
        image = Image.open(image_files[image_index])
        st.image(image, caption=image_files[image_index].name, use_container_width=True)

# ========== Footer ==========
st.markdown("---")
st.markdown("This model uses Gradient Boosting for classification based on chemical and physical water parameters.")
