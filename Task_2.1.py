import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import seaborn as sns
#cargar el dataset
data = pd.read_csv("bank_transactions.csv")

#Processado de la data:

data = data[['CustAccountBalance', 'TransactionAmount (INR)']]

#escalar la data numérica para que no haya desbalacnes en el peso de cada uno, hacer que w(Account) ~ w(INR)
for numeric in ['CustAccountBalance', 'TransactionAmount (INR)']:
    mean = data[numeric].mean()
    sd = data[numeric].std()
    data[numeric] = data[numeric].apply(lambda x: (x-mean)/sd)

#eliminar todos los nan
# Imputación de valores faltantes (NaN)
imputer = SimpleImputer(strategy='mean')
data[['CustAccountBalance', 'TransactionAmount (INR)']] = imputer.fit_transform(data[['CustAccountBalance', 'TransactionAmount (INR)']])

#implementar k means

#elbow method
distances_df = []
for clusters in range(1,11):
    km = KMeans(n_clusters=clusters)
    km.fit(data)
    distances_df.append([clusters,km.inertia_])
distances_df = pd.DataFrame(distances_df,columns=["clusters","distance"])

#graficar
distances_df.set_index("clusters").plot()
plt.xlabel("cantidad de centroides")
plt.ylabel("distancia cuadrada entre centroides")
plt.title("Kmeans Elbow")
plt.show()

# se usa 3 después de revisar la gráfica del elbow
km = KMeans(n_clusters=3)
data["cluster"] = km.fit_predict(data)
centroids = km.cluster_centers_


custom_palette = ["red", "green", "blue"]
sns.scatterplot(x="CustAccountBalance", y="TransactionAmount (INR)", hue=data['cluster'], data=data,palette=custom_palette)
plt.scatter(x=centroids[0][0],y=centroids[0][1], color="red" , marker="X")
plt.scatter(x=centroids[1][0],y=centroids[1][1], color="green",  marker="X")
plt.scatter(x=centroids[2][0],y=centroids[2][1], color="blue",  marker="X")
plt.title('K-means Clustering with 2 dimensions')
plt.show()