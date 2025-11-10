# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv("sales_data_sample.csv", encoding='latin1')

# Step 2: Select relevant numerical features
# (You can change columns depending on dataset)
X = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES']]

# Step 3: Handle missing values (if any)
X = X.dropna()

# Step 4: Standardize the data (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Determine the number of clusters using the Elbow Method
inertia = []
K = range(1, 11)  # try k = 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Step 6: Plot the Elbow curve
plt.figure(figsize=(6, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Step 7: Choose optimal k (e.g., k=3) and fit K-Means
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 8: View cluster summary
print(df.groupby('Cluster')[['QUANTITYORDERED', 'PRICEEACH', 'SALES']].mean())

# Step 9: Visualize clusters (optional, 2D plot)
plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 2], c=df['Cluster'], cmap='viridis')
plt.xlabel('Quantity Ordered (scaled)')
plt.ylabel('Sales (scaled)')
plt.title('K-Means Clustering')
plt.show()
