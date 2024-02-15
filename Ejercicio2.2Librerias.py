import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt 

# Load the dataset
df = pd.read_csv('bank_transactions.csv', encoding='utf-8')

# Select variables
variables = ['CustAccountBalance', 'TransactionAmount (INR)']

# Preprocessing
scaler = StandardScaler()
df[variables] = scaler.fit_transform(df[variables])

imputer = SimpleImputer(strategy='mean')  
df[variables] = imputer.fit_transform(df[variables])

# PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[variables])

# Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=3)  
gmm.fit(pca_components)
clusters = gmm.predict(pca_components)  # Use predict in place of fit_predict

# Visualization using matplotlib
plt.scatter(
    pca_components[:, 0], pca_components[:, 1], 
    c=clusters, cmap='viridis', s=50, alpha=0.7
)
plt.title('Customer Segmentation')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')  # Add colorbar 
plt.show()

# Cluster interpretation (This part remains as before)
for i in range(3):
    cluster_data = df.loc[clusters == i, variables]  
    print(f"Cluster {i + 1}:")
    print(cluster_data.describe())  
