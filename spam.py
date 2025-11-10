# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

# Step 1: Load dataset
df = pd.read_csv("emails.csv")

# Check data
print("Dataset shape:", df.shape)
print(df.head())

# Step 2: Check for missing values
df.dropna(inplace=True)

# Step 3: Convert labels (Spam = 1, Not Spam = 0)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Category'])  # assuming 'Category' column

# Step 4: Split data into features (X) and target (y)
X = df['Message']   # assuming 'Message' column has email text
y = df['label']

# Step 5: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = vectorizer.fit_transform(X)

# Step 6: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Step 7: Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Step 8: Train SVM model
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Step 9: Evaluate both models
def evaluate_model(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"\n🔹 {name} Performance:")
    print("Confusion Matrix:\n", cm)
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Precision: {:.2f}".format(prec))
    print("Recall: {:.2f}".format(rec))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Step 10: Analyze performance
evaluate_model("K-Nearest Neighbors", y_test, y_pred_knn)
evaluate_model("Support Vector Machine", y_test, y_pred_svm)
