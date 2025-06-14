import os
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


from data_preprocessing import load_dataset, preprocess_data

# Load and preprocess data
df = load_dataset()
X_train, X_test, y_train, y_test, scaler, class_names = preprocess_data(df)

# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "MLP Classifier": MLPClassifier(max_iter=1000, random_state=42)
}

# Track best model by F1-score (macro)
best_model = None
best_model_name = ""
best_f1 = 0

# Make sure the output folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# Train, evaluate and save results for each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {accuracy:.4f} | Macro F1: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Update best model tracker
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    # Save confusion matrix figure
    fig_path = f"reports/figures/confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(fig_path)
    print(f"Confusion matrix saved to '{fig_path}'")
    plt.close()

# Save the best model
if best_model:
    print(f"\nBest Model: {best_model_name} with F1 Score: {best_f1:.4f}")

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("models/min_max_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("models/class_names.pkl", "wb") as f:
        pickle.dump(class_names, f)

    print("Best model, scaler, and class names saved in 'models/'")
