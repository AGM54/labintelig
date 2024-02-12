import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Carga del conjunto de datos
with open('bank_transactions.csv', 'rb') as f:
    df = pd.read_csv(f, encoding='utf-8', errors='replace')



# Selección de variables
variables = ['CustAccountBalance', 'TransactionAmount (INR)', 'TransactionFrequency']

# Escalado de variables numéricas
scaler = StandardScaler()
df[variables] = scaler.fit_transform(df[variables])

# Análisis de Componentes Principales (PCA)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[variables])

# Implementación de modelos de mezcla con diferentes números de clusters
modelos = {}
for n_clusters in range(2, 6):
  modelos[n_clusters] = GaussianMixture(n_components=n_clusters)
  modelos[n_clusters].fit(pca_components)

# Evaluación de la calidad de los clusters
silhouette_scores = {}
for n_clusters, model in modelos.items():
  silhouette_scores[n_clusters] = model.score_samples(pca_components)

# Selección del número de clusters
mejor_n_clusters = 3

# Visualización de clusters
import matplotlib.pyplot as plt
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=modelos[mejor_n_clusters].predict(pca_components), s=50, alpha=0.7)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# Interpretación de clusters
for i in range(mejor_n_clusters):
  print(f"Cluster {i + 1}:")
  print(f"  - Saldo promedio: {df['CustAccountBalance'].iloc[modelos[mejor_n_clusters].predict(pca_components) == i].mean():.2f}")
  print(f"  - Monto promedio de transacción: {df['TransactionAmount (INR)'].iloc[modelos[mejor_n_clusters].predict(pca_components) == i].mean():.2f}")
  print(f"  - Frecuencia de transacciones: {df['TransactionFrequency'].iloc[modelos[mejor_n_clusters].predict(pca_components) == i].mean():.2f}")

