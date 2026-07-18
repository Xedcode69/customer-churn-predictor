import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

df = pd.read_csv("../data/customer_churn_data.csv")

df.drop(columns=["CustomerID", "Age", "Gender"], inplace=True)

x = df.drop("Churn", axis=1)
y = df["Churn"]

df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})

numerical_features = x.select_dtypes("int64", "Float64").columns.to_list()

categorical_features = x.select_dtypes("object").columns.tolist()

numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

rf_model = pipeline.fit(x_train, y_train)

model_2_prediction = rf_model.predict(x_test)

joblib.dump(rf_model, "random_forest_model.pkl")


print("Model 2 (Random Forest Classifier) Evaluation:")

# accuracy_score
accuracy_score = accuracy_score(y_test, model_2_prediction)
print(f"Accuracy Score(model_2): {accuracy_score:.2f}")

# precision_score
precision_score = precision_score(y_test, model_2_prediction)
print(f"Precision Score(model_2): {precision_score:.2f}")

# recall_score
recall_score = recall_score(y_test, model_2_prediction)
print(f"Recall Score(model_2): {recall_score:.2f}")

# f1_score
f1_score = f1_score(y_test, model_2_prediction)
print(f"F1 Score(model_2): {f1_score:.2f}")

# confusion_matrix
confusion_matrix = confusion_matrix(y_test, model_2_prediction)
print("Confusion Matrix(model_2):")
print(confusion_matrix)

# classification_report
classification_report = classification_report(y_test, model_2_prediction)
print("Classification Report(model_2):")
print(classification_report)

# predict probability
model_2_pred_prob = rf_model.predict_proba(x_test)[:, 1]

# roc_auc_score
auc_score = roc_auc_score(y_test, model_2_pred_prob)
print("rou-auc(model_2):", auc_score)

# roc_curve
false_positive_rate, true_positive_rate, threshold = roc_curve(
    y_test, model_2_pred_prob
)

# visual chart for roc_curve using matplotlib
plt.figure(figsize=(8, 6))

plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=f"ROC Curve(model_2), AUC = {auc_score:.2f}",
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Customer Churn ROC Curve(model_2)")
plt.legend()
plt.show()
