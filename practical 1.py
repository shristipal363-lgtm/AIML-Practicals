import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Creating customer dataset
data = {
    "Customer_ID": [1, 2, 3, 4, 5, 6,
                    7, 8, 9, 10, 11, 12],

    "Annual_Income": [15, 16, 17, 20, 22, 70,
                      72, 75, 80, 85, 90, 95],

    "Spending_Score": [39, 42, 45, 50, 52, 20,
                       25, 30, 75, 80, 85, 90]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Display dataset
print("Customer Dataset:")
print(df)

# Selecting features for clustering
X = df[["Annual_Income", "Spending_Score"]]

# Creating K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Applying K-Means
df["Cluster"] = kmeans.fit_predict(X)

# Display clustered data
print("\nClustered Data:")
print(df)

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Visualizing clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    s=100
)

# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation Using K-Means Clustering")
plt.grid()

plt.show()