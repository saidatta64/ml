# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Step 1: Load dataset
df = pd.read_csv("diabetes.csv")

# Step 2: Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize (KNN is distance-based)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = knn.predict(X_test)

# Step 7: Compute evaluation metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Step 8: Print results
print("Confusion Matrix:\n", cm)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))
print("Error Rate: {:.2f}%".format(error_rate * 100))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
