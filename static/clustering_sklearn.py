import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import clustering
from report_figures import activate_tex

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

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
c = 0

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
}

datasets = [
    (no_structure, {}),
    (
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
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
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    
]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    print(i_dataset)
    if i_dataset in [0,1,2,3,4]:
        continue
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset
    Xplot = np.empty_like(X)
    np.copyto(Xplot, X)
    Xplot[:,0] -= np.amin(Xplot[:,0])
    Xplot[:,0] /= np.amax(Xplot[:,0])
    Xplot[:,1] -= np.amin(Xplot[:,1])
    Xplot[:,1] /= np.amax(Xplot[:,1])

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )
    dbscan = cluster.DBSCAN(eps=params["eps"])
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )
    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"], covariance_type="full"
    )

    cl = clustering.Clustering()

    activate_tex()
    clustering_algorithms = (
        ("$Ward$", ward),
        ("$k-Means$", two_means),
        ("$Gaussian \: Mixture$", gmm),
        ("$Mean \: Shift$", ms),
        ("$OPTICS$", optics),
        ("$Our \: Method$", cl) 
    )

    fig = plt.figure(figsize=(2*2.1/0.98, 2*3.2/0.98))
    axs = fig.subplots(3,2)
    plt.subplots_adjust(
        left=0.01,
        bottom=0.01,
        right=0.99, 
        top=0.99,
        wspace=0.1, 
        hspace=0.1)

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        if plot_num == 6:
            algorithm.fit(Xplot)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                    + "connectivity matrix is [0-9]{1,2}"
                    + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding"
                    + " may not work as expected.",
                    category=UserWarning,
                )
                algorithm.fit(X)

        t1 = time.time()
        if plot_num == 6:
            y_pred = algorithm.labels_
        else:
            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(X)

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
                            "#2020a0",
                            "#ffe4c4"
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )

        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        ax = axs[(plot_num-1)//2, (plot_num-1)%2]
        #ax.set_title(name, fontsize=8)

        ax.set_aspect(1)

        ax.scatter(Xplot[:, 0], Xplot[:, 1], s=1, color=colors[y_pred])

        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks([])
        ax.set_yticks([])

        
        ax.text(
            (ax.get_xlim()[1] - ax.get_xlim()[0])*0.03+ax.get_xlim()[0], 
            -(ax.get_ylim()[1] - ax.get_ylim()[0])*0.03+ax.get_ylim()[1],
            "$({})$".format(list(string.ascii_lowercase)[plot_num-1]), ha="left", va="top")

        plot_num += 1
    plt.savefig("report/images/results_fig.png", bbox_inches="tight", dpi=1200)
    plt.show()
    plot_num=1
    breakpoint()
