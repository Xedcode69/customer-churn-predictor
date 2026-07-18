import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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

df.drop(columns=["CustomerID", "Gender", "Age"], inplace=True)

x = df.drop("Churn", axis=1)
y = df["Churn"]

df["Churn"] = df["Churn"].map({1: "Yes", 0: "No"})

numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_features = x.select_dtypes(include=["object"]).columns.tolist()

print(numeric_features)
print(categorical_features)

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# train the model using logistic regression algorithm
lr_model = pipeline.fit(x_train, y_train)

# predict the test data using the trained model
model_1_prediction = lr_model.predict(x_test)

joblib.dump(lr_model, "logical_regression_model.pkl")


# output metrics

print("Model 1 (Logistic Regression) Evaluation:")

# accuracy_score for model_1
accuracy_score = accuracy_score(y_test, model_1_prediction)
print(f"Accuracy Score(model_1): {accuracy_score:.2f}")

# precision_score for model_1
precision_score = precision_score(y_test, model_1_prediction)
print(f"Precision Score(model_1): {precision_score:.2f}")

# recall_score for model_1
recall_score = recall_score(y_test, model_1_prediction)
print(f"Recall Score(model_1): {recall_score:.2f}")

# f1_score
f1_score = f1_score(y_test, model_1_prediction)
print(f"F1 Score(model_1): {f1_score:.2f}")

# confusion_matrix
confusion_matrix = confusion_matrix(y_test, model_1_prediction)
print("Confusion Matrix(model_1):")
print(confusion_matrix)

# classification_report
classification_report = classification_report(y_test, model_1_prediction)
print("Classification Report(model_1):")
print(classification_report)

# predict probability
model_1_pred_prob = lr_model.predict_proba(x_test)[:, 1]

# roc_auc_score
auc_score = roc_auc_score(y_test, model_1_pred_prob)
print("rou-auc(model_1):", auc_score)

# roc_curve
false_positive_rate, true_positive_rate, threshold = roc_curve(
    y_test, model_1_pred_prob
)

# visual chart for roc_curve using matplotlib
plt.figure(figsize=(8, 6))

plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=f"ROC Curve(model_1), AUC = {auc_score:.2f}",
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Customer Churn ROC Curve(model_1)")
plt.legend()
plt.show()


# comparing to radom forest algorithm the precision, recall, f1, accuracy all are lower in logical regression algorithm
# Hence the rf_model will be used as the main prediction model
