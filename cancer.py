import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.datasets import load_breast_cancer

df = pd.read_csv(r"C:\Users\Admin\Downloads\archive (2)\diabetes.csv")
print(X.head())

# Split the dataset into features and target variable
X = df.drop('Outcome', axis=1) # Features
y = df['Outcome'] # Target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)


cm = confusion_matrix(y_test, knn_pred)
print("KNN Confusion Matrix:\n", cm)
Accuracy = accuracy_score(y_test, knn_pred)
print("KNN Accuracy: {:.2%}".format(Accuracy))
Error = 1 - Accuracy
print("KNN Error: {:.2%}".format(Error))
precision = precision_score(y_test, knn_pred)
print("KNN Precision: {:.2%}".format(precision))
recall = recall_score(y_test, knn_pred)
print("KNN Recall: {:.2%}".format(recall))