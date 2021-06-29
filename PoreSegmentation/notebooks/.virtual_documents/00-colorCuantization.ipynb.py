import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter
from pylab import rcParams

rcParams['figure.figsize'] = 10, 10


img = plt.imread("../imgs/porosidadalta.jpg")

rcParams["figure.figsize"] = 10,10
plt.imshow(img)


img.shape


img[:2]


img = img/255
img.shape


img[:2]




m, l, k = img.shape[0], img.shape[1], img.shape[2]

df = pd.DataFrame(img.reshape(m*l,k))
print(df)


df.columns = ["R", "G", "B"]
print(df.head())


n_clusters = 3
kmeans = KMeans(n_clusters = n_clusters)
kmeans.fit(df)
df["clusters"] = kmeans.labels_


df.sample(5)



colors = df.groupby(["clusters"]).mean()
colors


index_larger = colors.sort_values(by=["B"], ascending=False)


por = index_larger.iloc[0].name


s=df.clusters.value_counts(normalize=True,sort=False).mul(100) # mul(100) is == *100

print(s) #series.to_frame() returns a dataframe



print(s.loc[por])


centers = pd.DataFrame(kmeans.cluster_centers_)
centers["clusters"] = range(n_clusters)


df["ind"] = df.index
df = df.merge(centers)
df = df.sort_values("ind")
df = df.drop("ind", 1)


quant_img = df.iloc[:,4:7].values

quant_img = quant_img.reshape(img.shape[0], img.shape[1], img.shape[2])


plt.imshow(quant_img)


again_img = quant_img*255
print(again_img)


df2 = pd.DataFrame(again_img.reshape(m*l,k))
df2.columns = ["R", "G", "B"]


print(df2)



print(df2.R.unique())
print(df2.G.unique())
print(df2.B.unique())


contados = df2.apply(pd.value_counts).sum(axis=1)
print(contados)


total = pd.unique(contados)
print(total)


suma = total.sum()
porcentaje = (total/suma)*100
print(porcentaje)
print("______________________ \n")
print("POROSIDAD = ", porcentaje[0])






