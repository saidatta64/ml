import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# A. Data Pre-processing
df = pd.read_csv("1 Admission_Predict.csv")
df.columns = df.columns.str.strip()

# Creating binary target variable
df['Admitted'] = (df['Chance of Admit'] > 0.8).astype(int)

# Features & target
X = df[['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research']]
y = df['Admitted']

print("--- Initial Data Head (First 5 Rows) ---")
print(X.head())

print("\n--- Target Labels Count ---")
print(y.value_counts())

print("\n--- Missing Values Count ---")
print(X.isnull().sum())

# B. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot decision tree
plt.figure(figsize=(18, 10))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Admitted", "Admitted"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()