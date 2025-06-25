# AI for Water Quality Classification

## Project Overview

This project implements a machine learning solution to automate water quality classification. The final product is a user-friendly web application built with Streamlit that classifies water quality based on four key chemical parameters: `EC`, `Cl`, `TDS`, and `Na`.

The model predicts water quality into one of five categories: `Excellent`, `Good`, `Poor`, `Very Poor yet Drinkable`, and `Unsuitable for Drinking`.

-----

## Final Model & Performance

After a comprehensive analysis involving data cleaning, preprocessing, and the evaluation of multiple baseline models, the **Random Forest Classifier** was selected as the final model.

  - **Hyperparameter Tuning**: The model was fine-tuned using `RandomizedSearchCV` to optimize for the Macro F1-score, ensuring robust performance across imbalanced classes.
  - **Final Accuracy**: The optimized model achieved a final accuracy of **97.08%** on the held-out test set.

-----

## How to Run the Application

The final product is a web application. To run it on your local machine, follow these steps.

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd <your-repository-folder>
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

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit App**

Once the setup is complete, launch the application with the following command:

```bash
streamlit run app.py
```

This will open the application in your web browser.

-----

## Project Workflow & Development

The project was executed in three main phases, detailed in the Jupyter notebooks located in the `notebooks/` directory:

1.  **Phase 1: EDA & Data Cleaning**: Initial data inspection, visualization, and cleaning.
2.  **Phase 2: Model Training & Evaluation**: Training baseline models and selecting the best performer.
3.  **Phase 3: Hyperparameter Tuning & Finalization**: Optimizing the selected model and saving the final artifacts.

You can explore these notebooks using Jupyter Lab:

```bash
jupyter lab
```

## Directory Structure

```
water_quality_classifier/
│
├── app.py
│   
├── data/
│   ├── raw/
│   │   └── water_quality.csv
│   └── processed/
│       └── water_quality_cleaned.csv
├── models/
│   ├── final_model.pkl     # The tuned Random Forest model
│   └── scaler.pkl          # The MinMaxScaler object
├── notebooks/
│   ├── 01_eda_and_data_cleaning.ipynb
│   ├── 02_model_training_and_evaluation.ipynb
│   └── 03_hyperparameter_tuning_and_finalization.ipynb
├── reports/
│   └── figures/            # Saved plots and figures
├── .gitignore
├── requirements.txt
└── README.md
```

## Technology Stack

  * **Data Analysis**: `pandas`, `numpy`
  * **Machine Learning**: `scikit-learn`
  * **Data Visualization**: `matplotlib`, `seaborn`
  * **Web Application**: `streamlit`
  * **Image Processing**: `Pillow`
  * **Environment Management**: `venv`

-----

## License

This project is licensed under the MIT License.