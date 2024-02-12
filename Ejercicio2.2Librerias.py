import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Carga del conjunto de datos
with open('bank_transactions.csv', 'rb') as f:
    df = pd.read_csv(f, encoding='utf-8', errors='replace')



# Selecci�n de variables
variables = ['CustAccountBalance', 'TransactionAmount (INR)', 'TransactionFrequency']

# Escalado de variables num�ricas
scaler = StandardScaler()
df[variables] = scaler.fit_transform(df[variables])

# An�lisis de Componentes Principales (PCA)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[variables])

# Implementaci�n de modelos de mezcla con diferentes n�meros de clusters
modelos = {}
for n_clusters in range(2, 6):
  modelos[n_clusters] = GaussianMixture(n_components=n_clusters)
  modelos[n_clusters].fit(pca_components)

# Evaluaci�n de la calidad de los clusters
silhouette_scores = {}
for n_clusters, model in modelos.items():
  silhouette_scores[n_clusters] = model.score_samples(pca_components)

# Selecci�n del n�mero de clusters
mejor_n_clusters = 3

# Visualizaci�n de clusters
import matplotlib.pyplot as plt
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=modelos[mejor_n_clusters].predict(pca_components), s=50, alpha=0.7)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# Interpretaci�n de clusters
for i in range(mejor_n_clusters):
  print(f"Cluster {i + 1}:")
  print(f"  - Saldo promedio: {df['CustAccountBalance'].iloc[modelos[mejor_n_clusters].predict(pca_components) == i].mean():.2f}")
  print(f"  - Monto promedio de transacci�n: {df['TransactionAmount (INR)'].iloc[modelos[mejor_n_clusters].predict(pca_components) == i].mean():.2f}")
  print(f"  - Frecuencia de transacciones: {df['TransactionFrequency'].iloc[modelos[mejor_n_clusters].predict(pca_components) == i].mean():.2f}")

