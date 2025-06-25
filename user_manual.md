# User Manual: AI for Water Quality Classification

-----

## 1\. Introduction

Welcome to the user manual for the AI Water Quality Classification application. This guide will walk you through the entire process of setting up the project on your local machine, launching the application, and using it to predict water quality.

This application is designed to be a user-friendly prototype that demonstrates how machine learning can be used for rapid environmental analysis.

-----

## 2\. Installation and Setup

To run the application, you first need to set up the project and its dependencies. This process ensures that the application has all the necessary software components to run correctly.

### Step 2.1: Get the Project Files

First, you need to download the project files from the repository onto your computer.

1.  Open your terminal (Command Prompt, PowerShell, or Terminal).
2.  Navigate to the directory where you want to store the project.
3.  Run the following command to clone the repository:
    ```bash
    git clone <your-repository-url>
    ```
4.  Navigate into the newly created project folder:
    ```bash
    cd <your-repository-folder>
    ```

### Step 2.2: Set Up the Virtual Environment

⚠️ Important: This project is compatible only with Python 3.11.
Make sure you have Python 3.11 installed before continuing.
Check your version by running:

```bash
python --version
```
If it's not Python 3.11, download it from the official site:
👉 https://www.python.org/downloads/release/python-3110/

A virtual environment is a private, isolated space for the project's libraries. This prevents conflicts with other Python projects on your system.

  * **For macOS/Linux users:**
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```
  * **For Windows users:**
    ```bash
    py -3.11 -m venv .venv
    .\.venv\Scripts\activate
    ```

After running this, you should see `(.venv)` at the beginning of your terminal prompt, indicating the environment is active.

### Step 2.3: Install Required Libraries

The `requirements.txt` file lists all the Python libraries the project needs. Install them with this single command:

```bash
pip install -r requirements.txt
```

This will download and install pandas, scikit-learn, streamlit, and all other necessary packages into your virtual environment.

-----

## 3\. Launching the Application

With the setup complete, you can now launch the interactive web application.

1.  Ensure you are in the main project directory in your terminal.
2.  Ensure your virtual environment is active (you see `(.venv)` in the prompt).
3.  Run the following command:
    ```bash
    streamlit run app.py
    ```

This command starts a local web server, and your default web browser should automatically open a new tab with the application running.

-----

## 4\. Navigating the Application

The application interface is organized into three main sections, accessible via the sidebar on the left.

### 4.1. 🏠 Home

This is the main landing page. It provides a welcome message, an overview of the project's purpose, and a brief guide on how to use the application.

### 4.2. 🧪 Try the Model

This is the core interactive part of the application. Here, you can input chemical parameter values to get an instant water quality classification. This page is designed for hands-on use of the trained machine learning model.

### 4.3. 📊 Model Development

This section provides a high-level overview of the project's development workflow. It displays key visualizations that were generated during the data analysis and model evaluation phases, such as:

  - The distribution of water quality classes in the dataset.
  - The distribution of the most important features.
  - The final confusion matrix and feature importance plot for the model.

-----

## 5\. How to Get a Prediction

To classify a water sample, follow these steps on the **"🧪 Try the Model"** page.

1.  **Locate the Input Form**: On the "Try the Model" page, you will see a form titled "Input Water Quality Features".
2.  **Enter the Values**: Fill in the four required input fields with the data from your water sample:
      * **Electrical Conductivity (EC)**
      * **Chloride (Cl)**
      * **Total Dissolved Solids (TDS)**
      * **Sodium (Na)**
3.  **Submit for Prediction**: Click the **"Predict Water Quality"** button at the bottom of the form.
4.  **Review the Results**: The application will display the prediction results instantly:
      * **Predicted Label**: A clear text label (e.g., "Excellent", "Poor", "Unsuitable for Drinking") indicating the predicted class.
      * **Advice**: A short, helpful message based on the predicted quality.
      * **Model Confidence**: A percentage indicating how confident the model is in its prediction.
      * **Class Probabilities**: A bar chart showing the model's calculated probability for each of the five possible classes.

-----

## 6\. Troubleshooting

### "Model or scaler not found" Error

If you see this error when launching the app, it means the trained model files are missing.

  * **Solution**: You must run the project's Jupyter Notebooks in order to generate the model files.
    1.  Activate your virtual environment and start Jupyter Lab: `jupyter lab`
    2.  Run the notebooks in numerical order:
          - `01_eda_and_data_cleaning.ipynb`
          - `02_model_training_and_evaluation.ipynb`
          - `03_hyperparameter_tuning_and_finalization.ipynb`
    3.  After the third notebook completes, the `models/final_model.pkl` and `models/scaler.pkl` files will be created. You can then relaunch the Streamlit app.