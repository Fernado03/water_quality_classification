import streamlit as st

import pandas as pd
import numpy as np
import pickle

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

# Define feature input names (adjust based on your actual data columns)
FEATURES = [
    "EC", "Cl", "TDS", "Na"
]

label_mapping = {
    0: "Excellent",
    1: "Good",
    2: "Poor",
    3: "Unsuitable for Drinking",
    4: "Very Poor yet Drinkable"
}

# Sidebar for navigation
st.sidebar.title("Navigation")
main_option = st.sidebar.radio("Go to",
    ["Home",
    "Model Development",
    "Try the Model",
    "Model Visualizations"
    ]
)

if main_option == "Home":
    st.title("Home")
    # App title
    st.title("Water Quality Classification")
    st.markdown("Enter water sample features below to predict quality class.")


elif main_option == "Model Development":
    st.title("Model Development")
    st.subheader("Dataset")
    st.write("water_quality.csv")
    data = pd.read_csv("data/raw/water_quality.csv")
    st.write(data)







    

elif main_option == "Try the Model":
    st.title("Try the Model")

    # Sub-option selector only visible here
    sub_option = st.sidebar.selectbox(
        "Choose input method:",
        ["Manual Input", "Upload CSV"]
    )

    # ========== Manual Input ==========
    if sub_option == "Manual Input":
        # Define feature input names (adjust based on your actual data columns)
        FEATURES = [
            "EC", "Cl", "TDS", "Na"
        ]

        selected_columns = ["EC", "Cl", "TDS", "Na", "WQI", "Water Quality Classification"]
        
        data_test_self = pd.read_csv("data/raw/water_quality.csv", usecols=selected_columns)
        st.header("data from dataset for manual testing")
        st.write(data_test_self)

        input_data = {}
        for feature in FEATURES:
            input_data[feature] = st.number_input(f"{feature}:", format="%.4f")

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            st.success(f"Predicted Water Quality Class: **{prediction}**")

            predicted_label = label_mapping.get(prediction, "Unknown")
            st.success(f"Predicted Water Quality Class: **{predicted_label}**")

    # ========== CSV Upload ==========
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

elif main_option == "Model Visualizations":
    st.title("Model Visualizations")

    # Define relative path to the image
    from pathlib import Path
    from PIL import Image

    import os
    image_dir = Path(os.path.abspath(".")) / "notebooks" / "static_figures"


    # Get list of image files (filter by file extension)
    image_files = sorted([f for f in image_dir.glob("*.png")])

    # Check if images exist
    if not image_files:
        st.error("No images found in the directory.")
    else:
        # Slider to select image index
        image_index = st.slider(
            "Use arrow to browse images:",
            min_value=0,
            max_value=len(image_files) - 1,
            format="Image %d"
        )

        # Load and display the selected image
        image = Image.open(image_files[image_index])
        st.image(image, caption=image_files[image_index].name, use_container_width=True)

# ========== Footer ==========
st.markdown("---")
st.markdown("This model uses Gradient Boosting for classification based on chemical and physical water parameters.")
