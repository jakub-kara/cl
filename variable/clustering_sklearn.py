import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets, mixture, metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from clustering import Donkey

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
np.random.seed(0)
seed = 30
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# ============
# Set up cluster parameters
# ============

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": True,
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 3,
    "random_state": 42,
}

datasets = [
    # (
    #     noisy_moons,
    #     {
    #         "damping": 0.75,
    #         "preference": -220,
    #         "n_clusters": 2,
    #         "min_samples": 7,
    #         "xi": 0.1,
    #     },
    # ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (
        varied,
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        noisy_circles,
        {
            "damping": 0.77,
            "eps": 0.25,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    # (
    #     aniso,
    #     {
    #         "eps": 0.15,
    #         "n_neighbors": 2,
    #         "min_samples": 7,
    #         "xi": 0.1,
    #         "min_cluster_size": 0.2,
    #     },
    # ),
    #(no_structure, {}),
]

fig, axs = plt.subplots(3,6)
plt.subplots_adjust(wspace=0, hspace=0)
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    dbscan = cluster.DBSCAN(eps=params["eps"])
    hdbscan = cluster.HDBSCAN(
        min_samples=params["hdbscan_min_samples"],
        min_cluster_size=params["hdbscan_min_cluster_size"],
        allow_single_cluster=params["allow_single_cluster"],
    )
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )

    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"],
        covariance_type="full",
        random_state=params["random_state"],
    )

    donkey = Donkey()

    clustering_algorithms = (
        ("DONKEY", donkey),
        ("DBSCAN", dbscan),
        ("HDBSCAN", hdbscan),
        ("OPTICS", optics),
        ("Gaussian Mix.", gmm),
        ("Mean Shift", ms),
    )

    plot_num = 0
    for name, algorithm in clustering_algorithms:
        ax = axs[i_dataset, plot_num]
        time1 = time.time()
        algorithm.fit(X)
        time2 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        if plot_num == 0:
            ref = y_pred.copy()
        else:
            Donkey.align_labels(y_pred, ref)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        ax.set_box_aspect(1)

        if i_dataset==0: ax.set_title(name, size=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(X[:, 0], X[:, 1], s=3, color=colors[y_pred])
        plot_num += 1
        print(i_dataset, f"{name:30}", f"{metrics.adjusted_rand_score(y, y_pred):+.5f}", f"{metrics.adjusted_mutual_info_score(y, y_pred):+.5f}", f"{metrics.v_measure_score(y, y_pred):+.5f}", f"{(time2-time1):.5f}")
fig.set_size_inches(10,5.045)
plt.savefig("/mnt/c/Users/karaj/Desktop/UoO/cl/paper/model_all.png", dpi=600, bbox_inches="tight")
breakpoint()