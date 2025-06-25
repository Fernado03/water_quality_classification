# Final Report: AI for Water Quality Classification

---

## 1. Executive Summary

This report details the end-to-end development of a machine learning solution for rapid water quality classification. Traditional water testing methods are often slow and costly, creating a barrier to timely environmental monitoring. To address this, we developed a system that classifies water quality into one of five categories (`Excellent`, `Good`, `Poor`, `Very Poor yet Drinkable`, `Unsuitable for Drinking`) using just four key chemical parameters.

The project culminated in the creation of a **Random Forest Classifier** with a final accuracy of **97.08%** on unseen test data. This high-performance model was then integrated into an interactive web application using Streamlit, providing a practical and accessible tool for on-the-spot water quality assessment.

---

## 2. Methodology

The project followed a structured data science workflow, from data acquisition and cleaning to model deployment and evaluation.

### 2.1. Data Sourcing and Initial Analysis

* **Dataset**: The project utilized the `water_quality.csv` dataset, containing 19,029 initial records with 24 features.
* **Exploratory Data Analysis (EDA)**: A thorough EDA revealed several key characteristics:
    * The dataset contained significant data entry errors, including impossible geographic coordinates and negative chemical concentrations, which needed to be cleaned.
    * The target variable, `Water Quality Classification`, was imbalanced, with a majority of samples falling into "Poor" or "Unsuitable" categories.
    * The key predictive features were found to be severely right-skewed.

### 2.2. Data Cleaning and Preprocessing

A rigorous data preparation pipeline was established to ensure model robustness:

1.  **Data Cleaning**: Removed 7 duplicate rows and 162 rows containing invalid data (e.g., invalid coordinates, negative concentrations), resulting in a final, reliable dataset of **18,860 rows**.
2.  **Feature Selection**: Based on a strong correlation with the `Water Quality Index (WQI)`, the following four features were selected for the model:
    * `Electrical Conductivity (EC)`
    * `Chloride (Cl)`
    * `Total Dissolved Solids (TDS)`
    * `Sodium (Na)`
3.  **Logarithmic Transformation**: To correct for the severe right-skewness identified during EDA, a `log1p` transformation was applied to all selected features. This was a critical step for improving model performance.
4.  **Feature Scaling**: A `MinMaxScaler` was used to scale all features to a uniform range of [0, 1].

### 2.3. Model Training and Selection

Several baseline models were trained and evaluated on the preprocessed data, including Logistic Regression, K-Nearest Neighbors, Naïve Bayes, Gradient Boosting, and Random Forest.

The **Random Forest Classifier** emerged as the clear top performer, achieving a baseline accuracy of **97.11%**. Its strong performance, combined with its robustness and ability to handle complex interactions, made it the ideal candidate for further optimization.

### 2.4. Hyperparameter Tuning

The selected Random Forest model was fine-tuned using `RandomizedSearchCV` with 5-fold cross-validation. The search optimized for the `f1_macro` score to ensure strong performance across all classes, particularly the minority ones.

---

## 3. Results and Evaluation

### 3.1. Final Model Performance

The final, tuned Random Forest model demonstrated excellent and well-balanced performance on the held-out test set.

* **Overall Accuracy**: **97.08%**
* **Macro Average F1-Score**: **0.97**
* **Weighted Average F1-Score**: **0.97**

The classification report below shows high precision and recall for all five classes, confirming the model's reliability.

| Class | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
| Excellent | 0.99 | 0.96 | 0.97 | 149 |
| Good | 0.95 | 0.93 | 0.94 | 307 |
| Poor | 0.96 | 0.97 | 0.97 | 1056 |
| Unsuitable for Drinking | 0.99 | 0.99 | 0.99 | 1320 |
| Very Poor yet Drinkable | 0.96 | 0.96 | 0.96 | 940 |
| | | | | |
| **accuracy** | | | **0.97** | **3772** |
| **macro avg** | **0.97** | **0.96** | **0.97** | **3772** |
| **weighted avg** | **0.97** | **0.97** | **0.97** | **3772** |

### 3.2. Feature Importance

The analysis of the final model confirmed that **Electrical Conductivity (EC)** is the most influential feature, followed by **Total Dissolved Solids (TDS)**. This aligns with chemical principles, as both are measures of the total ionic content in the water.

---

## 4. Deployment

The deployment phase involved integrating the trained machine learning model and the associated data preprocessing steps into an interactive and user-friendly web application using the **Streamlit** framework.

The final **Random Forest Classifier** was serialized into `final_model.pkl`, and the corresponding **MinMaxScaler** used for feature normalization was saved as `scaler.pkl`. These files are loaded at runtime within the `app.py` script, which forms the backbone of the web interface.

### Key Application Features

- **Input Interface**:  
  Users can input four essential chemical parameters:
  - **EC** (Electrical Conductivity)
  - **Cl** (Chloride)
  - **TDS** (Total Dissolved Solids)
  - **Na** (Sodium)

- **Preprocessing and Prediction**:  
  Upon form submission:
  - Inputs are log-transformed using `log1p` to correct for skewness.
  - Features are scaled with the loaded `MinMaxScaler`.
  - The preprocessed data is passed to the trained model.
  - The model returns a predicted class (e.g., "Excellent", "Good", "Poor", etc.) along with a **confidence score**, displayed with color-coded styling and health advisories.

- **User Navigation**:
  - 🏠 **Home** – Introduction to the app's purpose and background.
  - 🧪 **Try the Model** – Main interface for submitting input and viewing predictions.
  - 📊 **Model Development** – Visual summaries of training results (e.g., class distributions, feature importance).
  - 📜 **Prediction History** – Logs past user predictions with timestamps, inputs, predicted class, and confidence.

- **Persistent Logging**:  
  Each user prediction is recorded in `user_prediction_history.csv` for future reference. The app also allows users to view or clear this history.

### Local Setup Instructions

To run the app locally, follow these steps:

```bash
# Step 1: Create and activate virtual environment
# For macOS/Linux:
python3.11 -m venv .venv
source .venv/bin/activate

# For Windows:
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch the app
streamlit run app/app.py



## 5. Conclusion

This project successfully demonstrates the power of machine learning to create a fast, accurate, and accessible tool for environmental monitoring. By following a structured workflow of data cleaning, preprocessing, and model optimization, we developed a highly reliable Random Forest Classifier. The final Streamlit application serves as a practical proof-of-concept, translating a complex model into a usable tool for a broader audience.
