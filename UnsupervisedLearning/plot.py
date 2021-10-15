import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn import metrics

df = pd.read_csv('agaricus-lepiota.csv')
df_dummies = pd.get_dummies(df.drop('edibility', axis='columns'))

plt.close()

pca = PCA(n_components=3)
df_pca = pca.fit_transform(df_dummies)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

k_means = KMeans(n_clusters=9)
k_means.fit(df_dummies)
clusters = k_means.predict(df_dummies)

colors = ['green',
          'blue',
          'red',
          'yellow',
          'cyan',
          'orange',
          'purple',
          'grey',
          'pink']
print(clusters)

for i, e in enumerate(df_pca):
    ax.scatter(*e, color=colors[clusters[i]])

plt.show()