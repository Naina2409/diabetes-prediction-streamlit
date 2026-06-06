import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
df = pd.read_csv("diabetes.csv")

print("\nFirst 5 Rows\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(6,4))

sns.countplot(
    x="Outcome",
    data=df
)

plt.title("Diabetes Distribution")

plt.savefig(
    "diabetes_distribution.png",
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "correlation_heatmap.png",
    bbox_inches="tight"
)

plt.close()

X = df.drop(columns="Outcome")

Y = df["Outcome"]
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled,
    Y,
    test_size=0.2,
    random_state=2,
    stratify=Y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

model = SVC(
    kernel="linear",
    probability=True
)

model.fit(
    X_train,
    Y_train
)

print("\nModel Trained Successfully")
train_prediction = model.predict(
    X_train
)

train_accuracy = accuracy_score(
    Y_train,
    train_prediction
)

print(
    "\nTraining Accuracy:",
    round(train_accuracy*100,2),
    "%"
)
test_prediction = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    Y_test,
    test_prediction
)

print(
    "\nTesting Accuracy:",
    round(test_accuracy*100,2),
    "%"
)
cm = confusion_matrix(
    Y_test,
    test_prediction
)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

print("\nClassification Report\n")

print(
    classification_report(
        Y_test,
        test_prediction
    )
)
pickle.dump(
    model,
    open("diabetes_model.pkl","wb")
)

pickle.dump(
    scaler,
    open("scaler.pkl","wb")
)
print("Model successful.")