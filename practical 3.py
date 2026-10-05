import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Creating the dataset
data = {
    "Mathematics": [85, 78, 92, 65, 72, 88, 55, 95, 70, 82],
    "Physics": [80, 75, 88, 70, 68, 84, 60, 92, 73, 79],
    "Chemistry": [82, 80, 90, 68, 75, 86, 58, 94, 69, 84],
    "Computer": [90, 85, 95, 72, 78, 92, 65, 98, 75, 88]
}

# Creating DataFrame
df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Applying PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Creating reduced DataFrame
pca_df = pd.DataFrame(
    principal_components,
    columns=["Principal_Component_1", "Principal_Component_2"]
)

print("\nReduced Dataset:")
print(pca_df)

# Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Total explained variance
total_variance = pca.explained_variance_ratio_.sum()

print("\nTotal Variance Explained:")
print(total_variance)

# Visualization
plt.figure(figsize=(8, 5))

plt.scatter(
    pca_df["Principal_Component_1"],
    pca_df["Principal_Component_2"],
    s=100
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Dimensionality Reduction")
plt.grid()

plt.show()