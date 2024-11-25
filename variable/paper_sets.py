import numpy as np
from sklearn import cluster, datasets, mixture, metrics
from sklearn.decomposition import PCA
from clustering import Donkey

def iris():
    data, target = datasets.load_iris(return_X_y = True)
    X_reduced = PCA(n_components=3).fit_transform(data)
    clus = Donkey(log=True)
    clus.fit(X_reduced)
    breakpoint()

iris()