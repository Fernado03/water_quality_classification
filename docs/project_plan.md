# Detailed Plan: AI for Water Quality Classification (Final)

**Primary Dataset:** [`dataset/water_quality.csv`](dataset/water_quality.csv)

**Justification for Choosing this Dataset:**

1.  **Real-world Impact**: Ensuring water quality is a critical global concern. An AI prototype in this area can demonstrate significant real-world impact, aligning with the project's objective of "Real-World Impact in Critical Sectors".
2.  **Clear Problem Definition**: The task is to classify water quality into several categories (e.g., "Excellent", "Good", "Poor", "Very Poor yet Drinkable", "Unsuitable for Drinking") based on its chemical and physical properties. This is a well-defined machine learning problem (multi-class classification), fitting the project's aim to "develop predictive AI prototypes".
3.  **Data Availability**: The local dataset [`dataset/water_quality.csv`](dataset/water_quality.csv) provides a rich set of features for this task.
4.  **Suitability for Prototyping**: A model predicting water quality based on input parameters can be effectively prototyped into a simple UI (e.g., Streamlit, Flask), as required by the technical guidelines.
5.  **Alignment with Target Domains**: This project falls under the "Smart Technologies" domain, specifically addressing an environmental/health-related smart solution.

---

**Phase 1: Foundation & Data Acquisition**

1.  **Project Setup & Environment Configuration:**
    *   Create a dedicated project directory.
    *   Initialize a Python virtual environment (e.g., `venv`, `conda`).
    *   Install core libraries: `pandas` (data manipulation), `numpy` (numerical operations), `scikit-learn` (modeling), `matplotlib`/`seaborn` (visualization).
    *   Install UI framework: `streamlit` or `flask`.
    *   Initialize Git for version control.
2.  **Data Acquisition and Initial Understanding:**
    *   Load data from [`dataset/water_quality.csv`](dataset/water_quality.csv) into a `pandas` DataFrame.
    *   Conduct initial Exploratory Data Analysis (EDA):
        *   Dataset dimensions (`df.shape`), data types (`df.info()`), statistical summary (`df.describe()`).
        *   Identify features and the target variable: `Water Quality Classification` (multi-class, including "Excellent", "Good", "Poor", etc.).
        *   Analyze target variable distribution (class balance across multiple classes).
        *   **Crucial EDA Task: Confirm that `Water Quality Classification` is derived from `WQI` to justify `WQI` exclusion (as per current understanding).**
        *   Examine distributions of all features. Identify potential identifiers (e.g., `Well_ID`) that might be excluded from modeling.
        *   Initial assessment of categorical features (`State`, `District`, `Block`, `Village`, `Year`) and numerical features.

---

**Phase 2: Data Preparation & Modeling**

3.  **Data Preprocessing:**
    *   **Feature Selection/Exclusion**:
        *   Based on EDA, exclude clear identifiers (e.g., `Well_ID`).
        *   **Crucial: Exclude the `WQI` column from the feature set. Since `Water Quality Classification` is understood to be derived from `WQI`, including `WQI` as a feature would lead to data leakage and an unrealistic model. The objective is to predict the classification from the other raw parameters.**
        *   Consider the utility of `Latitude` and `Longitude` for initial models (may be dropped or used as simple numerical features initially, or explored for more complex feature engineering later).
    *   Missing Value Imputation: Identify and handle missing values in the remaining features (e.g., mean/median/mode imputation, or more advanced techniques if warranted, with justification).
    *   Outlier Detection & Treatment (if necessary): Use visualization (box plots) and statistical methods for numerical features; decide on capping, transformation, or removal with justification.
    *   **Feature Encoding**:
        *   Apply appropriate encoding (e.g., One-Hot Encoding, Target Encoding, or others like `pd.get_dummies`) to categorical features such as `State`, `District`, `Block`, `Village`.
        *   Decide on handling for `Year` (treat as numerical, categorical, or derive features like 'age of reading').
    *   Feature Scaling: Normalize/standardize numerical features (e.g., `StandardScaler`, `MinMaxScaler`) for models sensitive to feature magnitudes (e.g., SVM, NNs, Logistic Regression).
4.  **Model Selection, Training, and Baseline Evaluation:**
    *   **Data Splitting**: Divide data into training and testing sets (e.g., 70/30 or 80/20 split). **Crucially, use `stratify` based on the multi-class `Water Quality Classification` target to ensure representative class proportions in both sets, especially if class imbalance is significant.**
    *   **Model Candidates**: Select models suitable for **multi-class classification**:
        *   Logistic Regression (with multi-class capability, e.g., `multi_class='ovr'` or `'multinomial'`).
        *   Support Vector Machine (SVM) (with multi-class capability).
        *   Decision Trees / Random Forest.
        *   Gradient Boosting Machines (e.g., XGBoost, LightGBM, CatBoost).
        *   (Consider a simple Neural Network if initial results warrant and time permits).
    *   **Training**: Train models on the preprocessed training set.
    *   **Evaluation**: Assess models on the test set using metrics appropriate for multi-class classification:
        *   Accuracy.
        *   Precision, Recall, F1-score (macro, micro, and weighted averages).
        *   Confusion Matrix (for detailed class-wise performance).
        *   Cohen's Kappa (optional, for inter-rater agreement if applicable conceptually).
    *   **Feature Importance Analysis**: For tree-based models (e.g., Random Forest, Gradient Boosting), extract and analyze feature importances. This will provide insights into which water parameters are most influential.

---

**Phase 3: Optimization & Prototyping**

5.  **Hyperparameter Tuning & Model Optimization:**
    *   Select the best performing model(s) from the baseline evaluation.
    *   Employ techniques like `GridSearchCV` or `RandomizedSearchCV` with cross-validation for hyperparameter tuning. Ensure the scoring metric used in tuning aligns with project goals (e.g., weighted F1-score if class imbalance is an issue).
    *   Retrain the optimized model on the full training set (or using the best cross-validation setup).
    *   Perform a final robust evaluation on the held-out test set.
6.  **Prototype Development (UI):**
    *   **Save Model**: Serialize the final trained model and the preprocessing pipeline (e.g., `joblib`, `pickle`). It's important to save the pipeline to ensure consistent preprocessing of new input.
    *   **UI Implementation**: Develop a simple UI using Streamlit or Flask.
        *   Allow users to input values for the numerical features used by the model (e.g., `pH`, `EC`, `Cl`, `SO4`, `NO3`, `TH`, `Ca`, `Mg`, `Na`, `K`, `F`, `TDS`).
        *   **Categorical Feature Input**: For categorical features (e.g., `State`, `District`, `Block`, `Village`, and `Year` if treated as categorical), the UI should present user-friendly input methods, such as dropdown menus. These menus should be populated with the unique values observed in the training dataset for each respective feature. The user's selection will then be passed to the saved preprocessing pipeline to be encoded consistently with how the model was trained (e.g., one-hot encoding).
        *   **Input Validation**: Implement basic validation for all user inputs:
            *   For numerical features: ensure inputs are numeric and fall within reasonable expected ranges (these ranges could be informed by the min/max values from the training data). Handle non-numeric entries gracefully.
            *   For categorical features (dropdowns): ensure a selection is made.
        *   Load the saved model and preprocessing pipeline.
        *   Preprocess input data consistently with the training pipeline.
        *   Display the predicted `Water Quality Classification` (e.g., "Excellent", "Good", "Poor", "Unsuitable for Drinking") clearly.

---

**Phase 4: Deliverables & Presentation**

7.  **Project Documentation & Reporting:**
    *   Ensure all code (e.g., Jupyter notebooks or Python scripts) is well-commented and follows good practices.
    *   Prepare a 5-page Project Report covering:
        *   Problem statement, relevance, and objectives.
        *   Dataset description (source [`dataset/water_quality.csv`](dataset/water_quality.csv), features, multi-class target), detailed EDA findings (including class distribution, feature characteristics, justification for `WQI` exclusion, handling of categorical features).
        *   Detailed preprocessing steps (missing value imputation, outlier handling, encoding, scaling) with justifications.
        *   Model selection rationale, training process, and comprehensive evaluation (all relevant multi-class metrics, confusion matrices).
        *   Feature importance findings and their interpretation.
        *   Hyperparameter tuning methodology and results.
        *   Prototype design (UI elements, input validation) and demonstration.
        *   Conclusions, system impact (linking back to feature importances and real-world application), limitations, and future potential.
    *   Package deliverables: Project code, trained model file (and preprocessing pipeline), runnable prototype (with clear instructions).
8.  **Video Showcase Preparation:**
    *   Create a 5-8 minute video presentation that:
        *   Clearly explains the problem and its real-world relevance.
        *   Showcases the dataset ([`dataset/water_quality.csv`](dataset/water_quality.csv)), key EDA insights (especially regarding multi-class target and `WQI`).
        *   Details the model, preprocessing pipeline (highlighting encoding, scaling, `WQI` exclusion).
        *   Demonstrates the prototype with example input/output, highlighting input validation and clear classification results.
        *   Discusses feature importances and their implications for water quality assessment.
        *   Concludes with system impact, limitations, and future potential.

---

**Project Workflow Diagram:**

```mermaid
graph TD
    A[Phase 1: Foundation & Data Acquisition] --> B(1. Project Setup & Env Config);
    A --> C(2. Data Acq. from dataset/water_quality.csv & EDA - Focus on Multi-class Target, WQI relationship, Categorical/Numerical Features);
    C --> D[Phase 2: Data Preparation & Modeling];
    D --> E(3. Data Preprocessing - Feature Selection incl. WQI EXCLUSION, Encoding Categorical, Scaling);
    E --> F(4. Model Selection for Multi-class, Training, Baseline Eval & Feature Importance);
    F --> G[Phase 3: Optimization & Prototyping];
    G --> H(5. Hyperparameter Tuning for Multi-class Models);
    H --> I(6. Prototype Development - UI with Input Validation, Multi-class Output);
    I --> J[Phase 4: Deliverables & Presentation];
    J --> K(7. Comprehensive Documentation & Reporting - Reflecting Multi-class Project & WQI decision);
    K --> L(8. Video Showcase Prep - Highlighting Key Decisions & Results);

    subgraph "Iterative Nature"
        direction LR
        E <--> F;
        F <--> H;
    end