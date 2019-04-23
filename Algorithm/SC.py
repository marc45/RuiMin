from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd

filename='../Datasets/maweiweather.csv'
data = pd.read_csv(filename).drop('日期', axis=1)
x = np.array(data)

clustering = SpectralClustering(n_clusters=4,
        assign_labels="discretize",
        random_state=0).fit(x)
print(clustering.labels_)