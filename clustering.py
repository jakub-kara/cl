import numpy as np
from functools import partial
import multiprocessing
import time, os
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from itertools import islice, cycle
from scipy.sparse.csgraph import minimum_spanning_tree

from density import clus

class Donkey:
    # TODO: implement initial cov guess
    # TODO: implement restarts
    # TODO: implement reader from output
    # TODO: rerun steps with different params
    def __init__(self, alpha=1., beta=1, gamma=0., log=False, name="clusters", abramson=True):
        self.alpha = alpha
        self.beta = np.exp(-beta)
        self.gamma = gamma
        self.do_log = log
        self.name = name
        self._abr = abramson

    @property
    def n_features(self):
        return self.data.shape[0]

    @property
    def n_traj(self):
        return self.data.shape[1]

    def get_density(self, x, pts=None):
        res = 0
        if pts is None:
            data = self.data
            icovs = self.icov
            dets = self.det
        else:
            data = self.data[:,pts]
            icovs = self.icov[:,:,pts]
            dets = self.det[pts]
        im = self.image(x, data)
        return -clus.get_density(res, im, icovs, dets, self.alpha)/self.n_traj

    def get_gradient(self, x, data=None):
        if data is None: data = self.data
        im = self.image(x, data)
        return -clus.get_gradient(im, self.icov, self.det, self.alpha)/self.n_traj

    def get_hessian(self, x, data=None):
        if data is None: data = self.data
        im = self.image(x, data)
        return -clus.get_hessian(im, self.icov, self.det, self.alpha)/self.n_traj

    def log(self, *args):
        if self.do_log:
            with open(f"{self.name}.log", "a") as logfile:
                for text in args:
                    logfile.write(text + "\n")

    def setup(self, data, ranges, pbc):
        if not os.path.exists("donkey"):
            os.mkdir("donkey")
        os.chdir("donkey")
        if self.do_log:
            with open(f"{self.name}.log", "w"): pass

        self.set_data(data)
        self.feature_ranges = ranges
        self.minmax_data()

        self.pbc = np.zeros(self.n_features, dtype=int)
        self.pbc[pbc] = 1
        self.labels_ = np.zeros(self.n_traj, dtype=int)

        self.log(
            "DONKEY Clustering",
            f"Number of datapoints: {self.n_traj}",
            f"Number of features:   {self.n_features}",
            f"Periodic features:    {pbc}",
            "")

    def set_data(self, data: np.ndarray):
        self.data = np.asfortranarray(data.T)

    def image(self, ref, data):
        n = ref.shape[0]
        im = np.zeros_like(data)
        for i in range(n):
            im[i] = ref[i,None] - data[i,:]
            if self.pbc[i]:
                im[i] = np.mod(im[i], 1)
                im[i] -= (im[i] > 0.5)
        return im

    def minmax_data(self):
        for j in range(self.n_features):
            if j in self.feature_ranges.keys():
                self.data[j,:] -= self.feature_ranges[j][0]
                self.data[j,:] /= self.feature_ranges[j][1] - self.feature_ranges[j][0]
                if np.amin(self.data[j,:]) < 0:
                    print(f"Illegal minimum range in feature {j}")
                    exit(1)
                if np.amax(self.data[j,:]) > 1:
                    print(f"Illegal maximum range in feature {j}")
                    exit(1)
            else:
                self.data[j,:] -= np.amin(self.data[j,:])
                self.data[j,:] /= np.amax(self.data[j,:])

    def idcov_data(self):
        cov = np.cov(self.data)
        lam, vec = np.linalg.eig(cov)
        self.trans = np.diag(1/np.sqrt(lam)) @ vec.T
        self.data = self.trans @ self.data

    def positions_to_labels(self):
        uniques = np.unique(self.cluster_positions.T, axis=0)
        self.n_clus = np.shape(uniques)[0]
        for i in range(self.n_traj):
            for j, clus in enumerate(uniques):
                if np.all(self.cluster_positions[:,i] == clus):
                    self.labels_[i] = j

    def reduce_cluster_labels(self):
        for i in range(len(np.unique(self.labels_[self.labels_ >= 0]))):
            while(i not in self.labels_):
                self.labels_[self.labels_ > i] -= 1
        self.n_clus = int(i+1)

    @staticmethod
    def align_labels(labels, ref):
        def pair_labels():
            nref = np.amax(ref)+1
            nlab = np.amax(labels)+1
            pairs = np.full(nlab, -1, dtype=int)
            overlap = np.zeros((nref, nlab), dtype=int)
            for i in range(n):
                if ref[i] >= 0 and labels[i] >= 0:
                    overlap[ref[i], labels[i]] += 1

            for i in range(nlab):
                mx = np.argmax(overlap)
                if np.amax(overlap) == -1:
                    break
                else:
                    pairs[mx%nlab] = mx//nlab
                    overlap[mx//nlab] = np.full(nlab, -1)
                    overlap[:,mx%nlab] = np.full(nref, -1)

            while -1 in pairs:
                pairs[list(pairs).index(-1)] = max(np.amax(pairs), nref)+1

            return pairs

        n = labels.shape[0]
        pairs = pair_labels()
        for old, new in enumerate(pairs):
            labels[labels == old] = new + n
        labels[labels >= 0] -= n

    # TODO: clean up!
    def plot_clusters(self, ax=None, axes=[0,1], data=None, pts=True, grid=False, axis_labels=None, axis_limits=None, maxima=False, saddles=False, density=False, title=None, ticks=False, angle=None):
        if data is None:
            data = self.data

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
                    int(max(self.labels_) + 1),
                )
            )
        )
        colors = np.append(colors, ["#000000"])

        if density:
            if ax is None:
                fig = plt.figure(figsize=(5,5))
                ax = fig.add_subplot(projection="3d")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            ax.scatter(
                self.data[axes[0]],
                self.data[axes[1]],
                [-self.get_density(x) for x in self.data.T],
                s=10,
                color = colors[self.labels_]
            )

            ax.scatter(
                self.saddles[axes[0]].flatten(),
                self.saddles[axes[1]].flatten(),
                [-self.get_density(x) for x in self.saddles.reshape((self.n_features, -1)).T],
                s = 15,
                color = "k",
                marker = "+"
            )

            n = 300
            x = np.linspace(min(data[axes[0]]), max(data[axes[0]]), n)
            y = np.linspace(min(data[axes[1]]), max(data[axes[1]]), n)
            X,Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i,j] = -self.get_density(np.array([X[i,j], Y[i,j]]))
            ax.plot_surface(X, Y, Z, alpha=0.3)

        elif len(axes) == 2:
            if ax is None:
                fig = plt.figure(figsize=(5,5))
                ax = fig.add_subplot()

            if grid:
                n = 100
                x = np.linspace(min(data[0]), max(data[0]), n)
                y = np.linspace(min(data[1]), max(data[1]), n)
                X,Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i,j] = np.log(-self.get_density(np.array([X[i,j], Y[i,j]])))
                plt.contourf(X, Y, Z, 15, alpha=.75, cmap=plt.cm.Greys)

            if not ticks:
                plt.xticks([])
                plt.yticks([])

            if title: plt.title(title, size=24)

            if axis_labels:
                plt.xlabel(axis_labels[0])
                plt.ylabel(axis_labels[1])

            if pts:
                plt.scatter(
                    self.data[axes[0]],
                    self.data[axes[1]],
                    s=10,
                    color = colors[self.labels_]
                )

            if axis_limits:
                ax.set_xlim(axis_limits[0])
                ax.set_ylim(axis_limits[1])

            if maxima:
                plt.scatter(
                    self.cluster_positions[axes[0],:],
                    self.cluster_positions[axes[1],:],
                    marker="*",
                    c="r",
                    s=25)

                for i in range(self.n_traj):
                    plt.annotate(
                        self.labels_[i], (self.minima[0,i], self.minima[1,i])
                    )

            if saddles:
                plt.scatter(
                    self.saddles[axes[0]].flatten(),
                    self.saddles[axes[1]].flatten(),
                    s = 15,
                    color = "k",
                    marker = "+"
                )

        elif len(axes) == 3:
            if ax is None:
                fig = plt.figure(figsize=(5,5))
                ax = fig.add_subplot(projection='3d', computed_zorder=False)
                ax.set_proj_type("ortho")

            if pts:
                ax.scatter(
                    data[axes[0]],
                    data[axes[1]],
                    data[axes[2]],
                    s=6,
                    color=colors[self.labels_]
                )

            if maxima:
                ax.scatter(
                    self.cluster_positions[axes[0],:],
                    self.cluster_positions[axes[1],:],
                    self.cluster_positions[axes[2],:],
                    marker="*",
                    c="r",
                    s=25)

            if saddles:
                ax.scatter(
                    self.saddles[axes[0]].flatten(),
                    self.saddles[axes[1]].flatten(),
                    self.saddles[axes[2]].flatten(),
                    s = 15,
                    color = "k",
                    marker = "+"
                )

            if angle:
                ax.view_init(azim=angle[0], elev=angle[1], roll=angle[2])

            ax.grid(False)
            if ticks:
                ax.tick_params(labelsize=16)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

            if axis_limits:
                ax.set_xlim(axis_limits[0])
                ax.set_ylim(axis_limits[1])
                ax.set_zlim(axis_limits[2])

            if title: plt.title(title, size=24)

            if axis_labels:
                font = {"size": 16}
                ax.set_xlabel(axis_labels[0], fontdict=font, labelpad=10)
                ax.set_ylabel(axis_labels[1], fontdict=font, labelpad=10)
                ax.set_zlabel(axis_labels[2], fontdict=font, labelpad=8)

        else:
            print("Please use 2 or 3 axes")
            return
        return ax

    def optimise_cov(self):
        self.cov = np.zeros((self.n_features, self.n_features, self.n_traj), dtype=float, order="F")
        self.icov = np.zeros((self.n_features, self.n_features, self.n_traj), dtype=float, order="F")
        self.det = np.zeros(self.n_traj, dtype=float)

        t1 = time.time()
        self.cov0 = self.mlloof()
        if self._abr:
            self.cov = self.abramson(self.cov0)
        else:
            self.cov = self.cov0
        t2 = time.time()
        self.log(f"Total time taken for covariance optimisation: {(t2-t1):.4} s", "")
        for i in range(self.n_traj):
            self.icov[:,:,i] = np.linalg.inv(self.cov[:,:,i])
            self.det[i] = np.linalg.det(self.cov[:,:,i])

    def mlloof(self):
        thresh = 1e-8
        cov = np.eye(self.n_features, order="F")

        self.log("1. Covariance Optimisation", f"Convergence threshold: {thresh}", "", "Initial guess", f"{cov}")
        nit = clus.mlloof(np.asfortranarray(cov), self.data, self.pbc, thresh)

        self.log(f"Convergence reached in {nit} iterations", f"{cov}", "")
        ret = np.zeros_like(self.cov)
        for i in range(self.n_traj): ret[:,:,i] = cov
        return ret

    def abramson(self, cov0: np.ndarray):
        self.log("Abramson correction active")

        f = np.zeros(self.n_traj)
        for i in range(self.n_traj):
            ref = self.data[:,i]
            im = self.image(ref, self.data)
            f += 1/np.sqrt(np.linalg.det(cov0[:,:,i])) * np.exp(-1/2*np.einsum("in,ij,jn->n", im, np.linalg.inv(cov0[:,:,i]), im))
        f /= self.n_traj
        g = np.exp(np.sum(np.log(f))/self.n_traj)
        lam = (f/g)**(-1/2)
        self.log(f"Local covariance matrices saved in {self.name}.cov.npy",
                f"Local coefficients",
                f"{lam}",
                "")
        out = np.einsum("ijn,n->ijn", cov0, lam)
        np.save(f"{self.name}.cov.npy", out)
        return out

    def find_maxima(self):
        self.minima = np.zeros((self.n_features, self.n_traj), order="F")
        trust = 1e-3
        thresh = 1e-6
        self.log(
            "2. Maxima detection",
            f"Broadening coefficient: {self.alpha}",
             "Selected method:        Trust-region Newton-Raphson",
            f"Trust radius:           {trust}",
            f"Convergence threshold:  {thresh}",
            "")

        t1 = time.time()
        with multiprocessing.Pool(8) as pool:
            res = pool.map(partial(self.newton_raphson_pbc, r0=trust, tol=thresh), [i for i in self.data.T])
            self.minima = np.array([i[0] for i in res]).T
        self.cluster_positions = np.round(self.minima, 3)
        self.positions_to_labels()
        self.log("", "Initial labels", f"{self.labels_}", "")
        t2 = time.time()
        self.log(f"Total time taken for maxima detection: {(t2-t1):.4} s", "")

    def newton_raphson_pbc(self, x0, r0, tol):
        return clus.newton_raphson(x0, self.data, self.icov, self.det, self.pbc, self.alpha, r0, tol)

    def merge_clusters(self):
        trust = 1e-2
        thresh = 1e-6

        self.log(
            "3. Cluster merging",
            f"Merging threshold:     {self.beta}",
             "Selected method:       Smith + Trust-region Newton-Raphson",
            f"Trust radius:          {trust}",
            f"Convergence threshold: {thresh}",
            "")

        t1 = time.time()
        count = 0
        self.saddles = np.zeros((self.n_features, self.n_clus, self.n_clus), order="F")
        self.merge = np.zeros((self.n_clus, self.n_clus), order="F")
        for c in range(self.n_clus):
            self.merge[c,c] = self.get_density(self.minima[:, self.labels_==c][:,0])

        contr = np.zeros((self.n_clus, self.n_traj), order="F")
        for i in range(self.n_traj):
            ref = self.data[:,i]
            im = self.image(ref, self.data)
            contr[:,i] = clus.get_density_by_clusters(im, self.icov, self.det, self.labels_, self.alpha, self.n_clus)

        for c1 in range(self.n_clus):
            for c2 in range(c1):
                count += 1
                pts = np.append(np.argwhere(self.labels_==c1), np.argwhere(self.labels_==c2))
                prod = contr[c1] * contr[c2]
                desc = np.argsort(prod)[::-1]

                if prod[desc[0]] < 1e-10:
                    continue
                temp = np.sum(self.labels_==c1) + np.sum(self.labels_==c2)
                closest_pts = desc[:temp]

                # centroid with PBC
                # https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
                centroid = np.zeros(self.n_features)
                for i in range(self.n_features):
                    if self.pbc[i]:
                        theta = 2*np.pi*self.data[i, closest_pts]
                        avg1 = np.mean(np.cos(theta))
                        avg2 = np.mean(np.sin(theta))
                        centroid[i] = (np.arctan2(-avg2, -avg1) + np.pi)/(2*np.pi)
                    else:
                        centroid[i] = np.average(self.data[i, closest_pts], weights=prod[closest_pts])

                pos, en, nit = self.get_saddle_smith(centroid, trust, thresh, pts)

                if np.abs(en) > 0:
                    self.log(f"Saddle search between clusters {c1:2} and {c2:2} succeeded in {nit} steps")
                else:
                    self.log(f"Saddle search between clusters {c1:2} and {c2:2} terminated after {nit} steps")

                if en == 0:
                    pos, en, nit = self.get_saddle_smith(self.data[:,desc[0]], trust, thresh, pts)

                im = self.image(pos, self.data)
                temp = clus.get_density_by_clusters(im, self.icov, self.det, self.labels_, self.alpha, self.n_clus)
                csum = temp[c1] + temp[c2]
                if csum == 0 or temp[c1] / csum < 0.01 or temp[c2] / csum < 0.01:
                    en = 0

                if en == 0:
                    continue
                if pos is not None:
                    en = self.get_density(pos)
                    self.saddles[:,c1,c2] = pos
                    self.saddles[:,c2,c1] = pos
                else:
                    en = None
                    self.saddles[:,c1,c2] = None
                self.merge[c1, c2] = en
                self.merge[c2, c1] = en
        self.log("")
        np.save(f"{self.name}.merge.npy", self.merge)

        labels = np.arange(self.n_clus)
        while True:
            to_merge = None
            metric = 0
            for c1 in range(self.n_clus):
                for c2 in range(c1):
                    temp = self.merge[c1, c2]/max(self.merge[c1, c1], self.merge[c2, c2])
                    if temp <= self.beta or temp > 1: continue
                    if temp > metric:
                        metric = temp
                        to_merge = [c1, c2]

            if to_merge is None: break

            newmin = min(self.merge[to_merge[0], to_merge[0]], self.merge[to_merge[1], to_merge[1]])
            l1 = np.argwhere(labels == labels[to_merge[0]]).flatten()
            l2 = np.argwhere(labels == labels[to_merge[1]]).flatten()
            for i in l1:
                self.merge[i, i] = newmin
                for j in l2:
                    self.merge[i, j] = 0
                    self.merge[j, i] = 0
                    self.merge[j, j] = newmin

            self.labels_[self.labels_==labels[to_merge[1]]] = labels[to_merge[0]]
            labels[labels==labels[to_merge[1]]] = labels[to_merge[0]]

            self.log(f"Clusters {to_merge[0]:2} and {to_merge[1]:2} merged with value {metric:.4}")

        self.reduce_cluster_labels()
        self.log("", "Merged labels", f"{self.labels_}", "")
        t2 = time.time()
        self.log(f"Total time taken for cluster merging: {(t2-t1):.4} s", "")

    # TODO: translate to FORTRAN
    def get_saddle_smith(self, x0, trust, thresh, pts):
        def gprime(x):
            grad = self.get_gradient(x)
            hess = self.get_hessian(x)
            lam, vec = np.linalg.eig(hess)
            gp = vec.T @ grad
            i = np.argmin(lam)
            gp[i] *= -1
            lam[i] *= -1
            return vec @ gp, vec @ np.diag(lam) @ vec.T

        nd = x0.shape[0]
        xk = 1*x0
        k = 1
        while True:
            temp = self.get_density(xk, pts)
            fk = self.get_density(xk)
            if np.abs(temp) < 1e-8 or np.abs(temp/fk) < 0.6 or k > 10000:
                return xk, 0, k

            gk, hk = gprime(xk)
            lam, vec = np.linalg.eig(hk)
            dxk = np.zeros(nd)
            gbar = vec.T @ gk
            for n in range(nd):
                dxk -= gbar[n]/np.abs(lam[n])*vec[:,n]
            if np.linalg.norm(dxk) < trust: xk += dxk
            else: xk += dxk/np.linalg.norm(dxk)*trust

            for n in range(nd):
                if self.pbc[n]:
                    xk = np.mod(xk, 1)

            if np.linalg.norm(dxk) < thresh:
                return xk, self.get_density(xk), k
            k += 1

    def find_outliers(self):
        self.log(
            "4. Outlier detection",
            f"Outlier threshold:   {self.gamma}")

        for i in range(self.n_traj):
            if self.get_density(self.data[:,i])/self.get_density(self.minima[:, self.labels_==self.labels_[i]][:,0]) < self.gamma:
                self.labels_[i] = -1
        self.log("", "Final labels", f"{self.labels_}")
        self.log(f"Final labels saved in {self.name}.lab.npy", "")
        np.save(f"{self.name}.lab.npy", self.labels_)

    def fit(self, data, ranges={}, pbc=[]):
        """
        Master fitting method calling all the successive steps
        Clustering assignments can be accessed via the labels_ attribute

        Args:
            data (np.ndarray): A 2D array containing data to be fitted. Shape required to be (n_traj, n_features)
        """

        t1 = time.time()
        self.setup(data, ranges, pbc)

        print("Optimising covariances")
        self.optimise_cov()

        print("Finding maxima")
        self.find_maxima()
        # breakpoint()

        print("Merging clusters")
        self.merge_clusters()


        print("Detecting outliers")
        self.find_outliers()
        t2 = time.time()
        self.log(f"Total time taken for fitting: {(t2-t1):.4} s")
        print(f"Finished in {(t2-t1):.4} s")
        os.chdir("..")