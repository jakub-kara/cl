import numpy as np
import time, os
import matplotlib.pyplot as plt
import matplotlib.colors as clr

import fmodules.density_routines as dr

def progress_bar(frac):
    if frac < 0: frac = 0
    if frac > 1: frac = 1
    size = 20
    prog = int(size*frac)
    print(" [" + prog*"=" + (size-prog)*" " + "]  " + f"{frac*100:.2f}%", end="\r", flush=True)
    if frac == 1: print()

class Donkey:
    # TODO: implement initial cov guess
    # TODO: implement restarts
    # TODO: implement reader from output
    # TODO: rerun steps with different params
    def __init__(self, alpha=1., beta=1/np.e, gamma=0., log=False, name="clusters"):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.do_log = log
        self.name = name

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
        return -dr.get_density(res, im, icovs, dets, self.alpha)/self.n_traj

    def get_gradient(self, x, data=None):
        if data is None: data = self.data
        im = self.image(x, data)
        return -dr.get_gradient(im, self.icov, self.det, self.alpha)/self.n_traj

    def get_hessian(self, x, data=None):
        if data is None: data = self.data
        im = self.image(x, data)
        return -dr.get_hessian(im, self.icov, self.det, self.alpha)/self.n_traj

    def log(self, *args):
        if self.do_log:
            with open(f"{self.name}.log", "a") as logfile:
                for text in args:
                    logfile.write(text + "\n")

    def setup(self, data, ranges, pbc):
        if not os.path.exists("donkey"): os.mkdir("donkey")
        os.chdir("donkey")
        if self.log:
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
        self.data = data.T.copy()

    def image(self, ref, data):
        n = ref.shape[0]
        im = np.zeros_like(data)
        for i in range(n):
            im[i] = ref[i,None] - data[i,:]
            if self.pbc[i]:
                im[i] -= (im[i] > 0.5)
                im[i] += (im[i] <= -0.5)
        return im

    def minmax_data(self):
        """
        Normalises the data to be fitted
        """

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
        """
        Converts local maxima positions to cluster labels
        """

        uniques = np.unique(self.cluster_positions.T, axis=0)
        self.n_clus = np.shape(uniques)[0]
        for i in range(self.n_traj):
            for j, clus in enumerate(uniques):
                if np.all(self.cluster_positions[:,i] == clus):
                    self.labels_[i] = j

    def reduce_cluster_labels(self):
        """
        Removes any "holes" in cluster labels, so that all labels in [0, n_clus-1] are used
        """
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

        # for i in range(len(np.unique(labels[labels >= 0]))):
        #     while(i not in labels):
        #         labels[labels > i] -= 1

    def plot_marginal(self, axis, limits=None, n_pts=201):
        if limits is None:
            mn = np.min(self.data[axis])
            mx = np.max(self.data[axis])
            limits = [mn - (mx-mn)*0.1, mx + (mx-mn)*0.1]

        x = np.linspace(*limits, n_pts)
        prob = np.zeros((self.n_clus, n_pts))

        for k in range(self.n_traj):
            a = self.icov[axis, axis, k]
            temp = [i for i in range(self.n_features) if i != axis]
            if len(temp) == 1:
                b = self.icov[[temp], axis, k]
                B = self.icov[[temp], [temp], k]
            else:
                b = self.icov[temp, axis, k]
                B = self.icov[temp][:,temp,k]
            lam, vec = np.linalg.eig(B)
            prod = 0
            for i in range(self.n_features-1):
                for j in range(self.n_features-1):
                    prod += (b[j]*vec[j,i])**2/lam[i]

            # prob[self.labels_[k]] += 1/np.sqrt(self.det[k]) * np.exp(-1/2*a*(x-self.data[axis,k])**2) * np.exp(1/2*(x-self.data[axis,k])**2*prod) / np.sqrt(np.prod(lam))
            log = -1/2*np.log(self.det[k]) - 1/2*(x-self.data[axis,k])**2*(a-prod) - 1/2*np.sum(np.log(lam))
            prob[self.labels_[k]] += np.exp(log)
        for i in range(self.n_clus):
            plt.plot(x, prob[i])
        plt.show()
        breakpoint()

    # TODO: clean up!
    def plot_clusters(self, axes=[0,1], data=None, pts=True, grid=False, axis_labels=None, axis_limits=None, maxima=False, vec=False, save=None, title=None, ticks=False, angle=None):
        """
        Plots clusters if the number of features is 2 or 3

        Args:
            plot_grid (bool, optional): Requests background density grid generation (2D only). Defaults to False.
            grid_res (int, optional): Resolution of background density grid. Defaults to 300.
            axis_labels (np.ndarray, optional): Labels of axes. Defaults to None.
        """
        if data is None:
            data = self.data

        if len(axes) == 2:
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot()
            colours = [(0.5,0.5,0.5) if self.labels_[i] < 0 \
                else clr.hsv_to_rgb([self.labels_[i]/self.n_clus,1,0.75]) for i in range(self.n_traj)]

            if grid:
                n = 100
                x = np.linspace(min(data[0]), max(data[0]), n)
                y = np.linspace(min(data[1]), max(data[1]), n)
                X,Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i,j] = np.log(-self.get_density(np.array([X[i,j], Y[i,j]])))
                        # Z[i,j] = -self.get_density(np.array([X[i,j], Y[i,j]]))
                plt.contourf(X, Y, Z, 15, alpha=.75, cmap=plt.cm.Greys)
                #plt.contour(X, Y, Z, 15, colors='black')

            # plt.gca().set_box_aspect(1)
            if not ticks:
                plt.xticks([])
                plt.yticks([])

            if title: plt.title(title, size=24)

            if axis_labels:
                plt.xlabel(axis_labels[0])
                plt.ylabel(axis_labels[1])

            if pts:
                for i in range(np.max(self.labels_) + 1):
                    plt.scatter(
                        self.data[axes[0],self.labels_==i],
                        self.data[axes[1],self.labels_==i],
                        s=10
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

                # for i in range(self.n_traj):
                    # plt.annotate(
                        # self.labels_[i], (self.minima[0,i], self.minima[1,i])
                    # )

            if vec:
                for i in range(self.n_traj):
                    evals, evecs = np.linalg.eig(self.cov[:,:,i])
                    evals = np.sqrt(evals)
                    plt.arrow(self.data[0,i], self.data[1,i], evals[0]*evecs[0,0], evals[0]*evecs[1,0])
                    plt.arrow(self.data[0,i], self.data[1,i], evals[1]*evecs[0,1], evals[1]*evecs[1,1])
                    plt.arrow(self.data[0,i], self.data[1,i],-evals[0]*evecs[0,0],-evals[0]*evecs[1,0])
                    plt.arrow(self.data[0,i], self.data[1,i],-evals[1]*evecs[0,1],-evals[1]*evecs[1,1])

        elif len(axes) == 3:
            colours = [(0.5,0.5,0.5) if self.labels_[i] < 0 \
                else clr.hsv_to_rgb([self.labels_[i]/self.n_clus,1,0.75]) for i in range(self.n_traj)]

            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(projection='3d', computed_zorder=False)

            if pts:
                for i in range(np.max(self.labels_) + 1):
                    ax.scatter(
                        data[axes[0],self.labels_==i],
                        data[axes[1],self.labels_==i],
                        data[axes[2],self.labels_==i],
                        s=6)

                # geoms = np.array([
                #     [2.47576463, 1.54885609, 1.94317856, 1.94317856, 2.11731496],
                #     [0.        , 0.        , 0.49874606,-0.49874606, 0.],
                #     [1.32974303, 1.53039729, 1.44273745, 1.44273745, 1.41118084]])
                # markers = ["s", "D", "^", "^", "o"]
                # for i in range(5): ax.scatter(geoms[0,i], geoms[1,i], geoms[2,i], marker=markers[i], edgecolor="k", facecolor="gray", s=30, zorder=10)


            if angle:
                ax.view_init(azim=angle[0], elev=angle[1], roll=angle[2])

            if not ticks:
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
                ax.set_xlabel(axis_labels[0], fontdict=font)
                ax.set_ylabel(axis_labels[1], fontdict=font)
                ax.set_zlabel(axis_labels[2], fontdict=font)

        else:
            print("Please use 2 or 3 axes")
            return


        if save:
            if len(axes) == 2 or axis_labels is None:
                plt.savefig(save, dpi=600, bbox_inches="tight", transparent=True)
            else:
                plt.subplots_adjust(left=0)
                plt.savefig(save, dpi=600)
            plt.gca().clear()
        else:
            plt.show()

    def compare_pdfs(self, fun):
        n = 200
        x = np.linspace(2*min(self.data[0]), 2*max(self.data[0]), n)
        y = np.linspace(2*min(self.data[1]), 2*max(self.data[1]), n)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        X,Y = np.meshgrid(x, y)
        est = np.zeros_like(X)
        ground = np.zeros_like(X)
        for i in range(n):
            print(i)
            for j in range(n):
                est[i,j] = self.get_density(np.array([X[i,j], Y[i,j]]))
                ground[i,j] = fun(np.array([X[i,j], Y[i,j]]))
        tot = np.sum(est)*dx*dy
        est /= tot
        temp = np.abs(np.log(est) - np.log(ground))
        mn = min(np.min(est), np.min(ground), np.min(temp))
        mx = max(np.max(est), np.max(ground), np.max(temp))


        fig, axs = plt.subplots(1,3, squeeze=True)
        est[0,0] = mn
        est[-1,-1] = mx
        axs[0].pcolormesh(np.log(est))
        ground[0,0] = mn
        ground[-1,-1] = mx
        axs[1].pcolormesh(np.log(ground))
        temp[0,0] = mn
        temp[-1,-1] = mx
        axs[2].pcolormesh(temp)

        # return (est - ground)/ground

    def optimise_cov(self):
        self.cov = np.zeros((self.n_features, self.n_features, self.n_traj), dtype=float, order="F")
        self.icov = np.zeros((self.n_features, self.n_features, self.n_traj), dtype=float, order="F")
        self.det = np.zeros(self.n_traj, dtype=float)

        t1 = time.time()
        self.cov0 = self.mlloof()
        self.cov = self.abramson(self.cov0)
        t2 = time.time()
        self.log(f"Total time taken for covariance optimisation: {(t2-t1):.4} s", "")
        for i in range(self.n_traj):
            self.icov[:,:,i] = np.linalg.inv(self.cov[:,:,i])
            self.det[i] = np.linalg.det(self.cov[:,:,i])

    def mlloof(self):
        thresh = 1e-8
        it = 0
        cov = np.eye(self.n_features)

        self.log("1. Covariance Optimisation", f"Convergence threshold: {thresh}", "", "Initial guess", f"{cov}")
        while True:
            it += 1
            temp = np.zeros_like(cov)
            for i in range(self.n_traj):
                ref = self.data[:,i]
                num = np.zeros_like(cov)
                den = 0
                im = self.image(ref, self.data)
                kcov = np.exp(-1/2*np.einsum("in,ij,jn->n", im, np.linalg.inv(cov), im))
                mat = np.einsum("in,jn->ijn", im, im)
                num = np.einsum("ijn,n->ij", mat, kcov)
                den = np.sum(kcov) - 1
                if den > 1e-10: temp += num/den
            err = np.sum(np.abs(cov - temp/self.n_traj))
            cov = temp/self.n_traj
            progress_bar(np.log(err)/np.log(thresh))
            if err < thresh: break
            self.log(f"Iteration {it}", f"Error: {err}", f"{cov}")

        self.log(f"Convergence reached in {it} iterations", f"{cov}", "")
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
        for i in range(self.n_traj):
            self.minima[:,i], nit = self.newton_raphson_pbc(self.data[:,i], trust, thresh)
            self.log(f"Point {i} converged in {nit} steps")
            progress_bar((i+1)/self.n_traj)
        self.cluster_positions = np.round(self.minima, 3)
        self.positions_to_labels()
        self.log("", "Initial labels", f"{self.labels_}", "")
        t2 = time.time()
        self.log(f"Total time taken for maxima detection: {(t2-t1):.4} s", "")

    def newton_raphson_pbc(self, x0, r0, thresh):
        nd = x0.shape[0]
        xk = 1*x0
        k = 1
        while True:
            # plt.scatter(xk[0], xk[1], marker=".", c="k")
            gk = self.get_gradient(xk)
            hk = self.get_hessian(xk)
            lam, vec = np.linalg.eig(hk)
            dxk = np.zeros(nd)
            gbar = vec.T @ gk
            for n in range(nd):
                dxk -= gbar[n]/np.abs(lam[n])*vec[:,n]
            if np.linalg.norm(dxk) < r0: xk += dxk
            else: xk += dxk/np.linalg.norm(dxk)*r0

            for n in range(nd):
                if self.pbc[n]:
                    xk = np.mod(xk, 1)
            if np.linalg.norm(dxk) < thresh: return xk, k
            k += 1

    def merge_clusters(self):
        trust = 1e-3
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
        self.links = np.eye(self.n_clus, dtype=bool)
        for c1 in range(self.n_clus):
            for c2 in range(c1):
                count += 1
                pts = np.append(np.argwhere(self.labels_==c1), np.argwhere(self.labels_==c2))
                closest_pts, dist, vec = dr.find_closest_points(self.data, self.labels_, np.linalg.inv(self.cov0[:,:,0]), self.pbc, c1, c2, np.sqrt(np.sum(self.labels_==c1)*np.sum(self.labels_==c2)))
                closest_pts = closest_pts.flatten()

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
                        centroid[i] = np.mean(self.data[i, closest_pts])

                progress_bar(count/(self.n_clus*(self.n_clus-1)/2))
                pos, en, nit = self.get_saddle_smith(centroid, trust, thresh, pts)
                if np.abs(en) > 0: self.log(f"Saddle search between clusters {c1:2} and {c2:2} succeeded in {nit} steps")
                else: self.log(f"Saddle search between clusters {c1:2} and {c2:2} terminated after {nit} steps")

                if en == 0: continue
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

        while True:
            to_merge = [-1,-1]
            metric = 0
            for c1 in range(self.n_clus):
                for c2 in range(c1):
                    if self.links[c1, c2]: continue
                    temp = self.merge[c1, c2]/max(np.min(np.diag(self.merge)[self.links[c1]]), np.min(np.diag(self.merge)[self.links[c2]]))
                    # try xi in c1 and not in c2 + xj in c2 and not c1 vs xk in c1 and in c2
                    if temp <= self.beta: continue
                    if temp > metric:
                        metric = temp
                        to_merge = [c1, c2]

            if -1 in to_merge: break
            self.links[to_merge[0], to_merge[1]] = True
            self.links[to_merge[1], to_merge[0]] = True

        labels = np.arange(self.n_clus)
        while True:
            to_merge = [-1,-1]
            metric = 0
            for c1 in range(self.n_clus):
                for c2 in range(c1):
                    temp = self.merge[c1, c2]/max(self.merge[c1, c1], self.merge[c2, c2])
                    # try xi in c1 and not in c2 + xj in c2 and not c1 vs xk in c1 and in c2
                    if temp <= self.beta: continue
                    if temp > metric:
                        metric = temp
                        to_merge = [c1, c2]

            if -1 in to_merge: break

            newmin = min(self.merge[to_merge[0], to_merge[0]], self.merge[to_merge[1], to_merge[1]])
            self.merge[to_merge[0], to_merge[1]] = 0
            self.merge[to_merge[1], to_merge[0]] = 0
            self.merge[to_merge[0], to_merge[0]] = newmin
            self.merge[to_merge[1], to_merge[1]] = newmin

            self.labels_[self.labels_==labels[to_merge[1]]] = labels[to_merge[0]]
            labels[labels==labels[to_merge[1]]] = labels[to_merge[0]]
            self.log(f"Clusters {to_merge[0]:2} and {to_merge[1]:2} merged with value {metric:.4}")

        self.reduce_cluster_labels()
        self.log("", "Merged labels", f"{self.labels_}", "")
        t2 = time.time()
        self.log(f"Total time taken for cluster merging: {(t2-t1):.4} s", "")


    def get_saddle_smith(self, x0, trust, thresh, pts):
        def gprime(x):
            grad = self.get_gradient(x)
            hess = self.get_hessian(x)
            lam, vec = np.linalg.eig(hess)
            gp = vec.T @ grad
            i = np.argmin(lam)
            gp[i] *= -1
            return vec @ gp

        nd = x0.shape[0]
        xk = 1*x0
        k = 1
        while True:
            # plt.scatter(xk[0], xk[1], marker=".", c="k")
            temp = self.get_density(xk, pts)
            fk = self.get_density(xk)
            if np.abs(temp) < 1e-8 or np.abs(temp/fk) < 0.5 or k > 10000: return xk, 0, k
            gk = gprime(xk)
            hk = self.get_hessian(xk)
            lam, vec = np.linalg.eig(hk)
            dxk = np.zeros(nd)
            gbar = vec.T @ gk
            for n in range(nd):
                dxk -= gbar[n]/np.abs(lam[n])*vec[:,n]
                #dxk -= gbar[n]/lam[n]*vec[:,n]
            if np.linalg.norm(dxk) < trust: xk += dxk
            else: xk += dxk/np.linalg.norm(dxk)*trust

            for n in range(nd):
                if self.pbc[n]:
                    xk = np.mod(xk, 1)

            if np.linalg.norm(dxk) < thresh: return xk, self.get_density(xk), k
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

        print("Merging clusters")
        self.merge_clusters()

        print("Detecting outliers")
        self.find_outliers()
        t2 = time.time()
        self.log(f"Total time taken for fitting: {(t2-t1):.4} s")
        print(f"Finished in {(t2-t1):.4} s")
        os.chdir("..")