# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Load dataset
df = pd.read_csv("sales_data_sample.csv", encoding='latin1')

# Step 2: Select numeric columns for clustering
X = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES']].dropna()

# Step 3: Standardize the data (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Determine optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)  # test k = 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Step 5: Plot the Elbow Curve
plt.figure(figsize=(6, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Step 6: Choose optimal k (based on the elbow point, e.g., k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 7: View cluster centers and summaries
print("\nCluster Centers (scaled):\n", kmeans.cluster_centers_)
print("\nAverage values per cluster:")
print(df.groupby('Cluster')[['QUANTITYORDERED', 'PRICEEACH', 'SALES']].mean())

# Step 8: Visualize the clusters
plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 2], c=df['Cluster'], cmap='viridis')
plt.xlabel('Quantity Ordered (scaled)')
plt.ylabel('Sales (scaled)')
plt.title('K-Means Clustering Visualization')
plt.show()
