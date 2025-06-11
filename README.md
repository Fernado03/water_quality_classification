# AI for Water Quality Classification

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Project Workflow](#project-workflow)
3.  [Dataset](#dataset)
4.  [Directory Structure](#directory-structure)
5.  [Installation & Setup](#installation--setup)
6.  [How to Run](#how-to-run)
7.  [Technology Stack](#technology-stack)
8.  [License](#license)

-----

## Project Overview

This project addresses the critical need for rapid and accessible water quality assessment. Traditional methods of testing water quality are often time-consuming, expensive, and require specialized laboratory equipment. This high barrier can delay necessary interventions, posing significant health risks and contributing to environmental damage.

To overcome these challenges, this project implements a machine learning solution to automate water quality classification. The model predicts the quality of water into one of five categories based on key chemical properties:

  * Excellent
  * Good
  * Poor
  * Very Poor
  * Unsuitable for Drinking

The final product is a simple web application where a user can input four chemical parameter values and receive an instant classification of the water's quality, making this a practical tool for environmental monitoring.

## Project Workflow

The project was executed in several phases, from data acquisition to model deployment in a prototype application.

1.  **Foundation & Data Acquisition**: The project environment was set up, and the dataset was acquired. Initial Exploratory Data Analysis (EDA) was performed to understand data distributions and identify key variables.
2.  **Data Preparation & Modeling**:
      * **Feature Selection**: Based on a correlation analysis with the Water Quality Index (WQI), four key features were selected: **Electrical Conductivity (EC)**, **Chloride (Cl)**, **Total Dissolved Solids (TDS)**, and **Sodium (Na)**.
      * **Preprocessing**: The selected features were scaled using Min-Max scaling, and the categorical target variable was encoded.
      * **Model Training**: The data was split using an 80/20 stratified split to handle class imbalance. Several classification models (e.g., Logistic Regression, Random Forest, Gradient Boosting, kNN) were trained and evaluated.
3.  **Optimization & Prototyping**: The best-performing model was optimized using `RandomizedSearchCV` with cross-validation to find the best hyperparameters. A user-friendly prototype was developed using Streamlit.
4.  **Deliverables**: The project is summarized in this repository, including the final model, source code, and reports.

## Dataset

The primary dataset used for this project is publicly available from a [GitHub Repository](https://github.com/shahsanjanav/DL-WaterQuality-Classifier).

  * **Original Dimensions**: 19,029 rows × 24 features.
  * **Final Features Used for Modeling**: `Electrical Conductivity (EC)`, `Chloride (Cl)`, `Total Dissolved Solids (TDS)`, `Sodium (Na)`.
  * **Target Variable**: `Water Quality Classification`.

## Directory Structure

The project is organized into a clear and modular structure:

```
water_quality_classifier/
│
├── .venv/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda_and_feature_selection.ipynb
│   ├── 02_model_training_and_evaluation.ipynb
│   └── 03_hyperparameter_tuning.ipynb
├── src/
│   ├── data_preprocessing.py
│   └── model_training.py
├── app/
│   └── app.py
├── models/
│   ├── final_model.pkl
│   └── scaler.pkl
├── reports/
│   ├── figures/
│   └── final_report.md
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation & Setup

To set up this project on your local machine, please follow the steps below.

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd water_quality_classifier
```

**2. Create and Activate a Virtual Environment**

  * **For macOS/Linux:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
  * **For Windows:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

**3. Install Required Libraries**
Once the virtual environment is activated, install all the necessary packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## How to Run

There are two main ways to interact with this project: running the analysis notebooks or launching the final application.

**1. Running the Jupyter Notebooks**

The notebooks in the `notebooks/` directory detail the project's analytical process. To explore them, start Jupyter Lab:

```bash
jupyter lab
```

Then, open and run the notebooks in numerical order (`01`, `02`, `03`) to follow the workflow from EDA to model tuning.

**2. Launching the Streamlit Application**

The final prototype is a web application built with Streamlit. To run it, execute the following command in your terminal from the project's root directory:

```bash
streamlit run app/app.py
```

This will launch a local web server, and you can access the application in your browser. The UI will prompt you to enter values for the four key features to get a water quality prediction.

## Technology Stack

  * **Data Analysis & Manipulation**: `pandas`, `numpy`
  * **Machine Learning**: `scikit-learn`
  * **Data Visualization**: `matplotlib`, `seaborn`
  * **Web Application Framework**: `streamlit`
  * **Environment Management**: `venv`

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.