import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import *
import os, timeit
import fmodules.density_routines as dr

class Clustering:
    plot_colours = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00"]

    def __init__(self, coeff=1., merge_threshold=1.):
        """
        Instantiates a Clustering object with experimental parameters

        Args:
            coeff (float, optional): Specifies the width of the Gaussian distributins. Defaults to 1
            merge_threshold (float, optional): Specifies the threshold above which cluster merging will not occur. 
                                               Defaults to 1
        """
        
        self.coeff = coeff
        self.merge_threshold = merge_threshold
    
    def set_data(self, data):
        self.n_traj, self.n_features = np.shape(data)
        self.data = np.zeros((self.n_features, self.n_traj), order="F")
        np.copyto(self.data, data.T)

    def normalise_data(self):
        """
        Normalises the data to be fitted        
        """

        for j in range(self.n_features):
            self.data[j,:] -= np.amin(self.data[j,:])
            self.data[j,:] /= np.amax(self.data[j,:])
        
        #self.data[0] += 0.2
        #self.data[0] %= 1
        #self.data[1] -= 0.2
        #self.data[1] %= 1


    def get_feature_vars(self):
        """
        Calculates variances along each feature, then adjusts them,
        so that the total hypervolume under all Gaussians is 1
        """

        self.feature_vars = np.zeros(self.n_features)

        for f in range(self.n_features):
            self.feature_vars[f] = np.var(self.data[f,:])

    def rescale_data(self):
        temp = np.sqrt((2*np.pi/self.coeff)**(self.n_features)*np.prod(self.feature_vars[:]))
        self.feature_vars *= (self.n_traj*temp)**(-1/self.n_features)
        self.n_pts = int(50/np.sqrt(np.amin(self.feature_vars)))
        self.data *= self.n_pts - 1
        self.feature_vars *= (self.n_pts-1)**2

    def get_clusters(self):
        """
        Performs gradient ascend to assign each data point to a local maximum
        """

        t1 = timeit.default_timer()
        
        self.cluster_positions = np.zeros((self.n_features, self.n_traj), dtype=int)
        self.labels_ = np.zeros(self.n_traj, dtype=int)

        self.cluster_positions = dr.get_clusters(
            self.data, self.feature_vars, self.coeff, self.pbc, self.n_pts)
        t2 = timeit.default_timer()
        print("Maxima assignment: ", f"{t2-t1:10.5f}")

    def get_clusters_cnt(self):
        t1 = timeit.default_timer()
        
        self.cluster_positions = np.zeros((self.n_features, self.n_traj))
        self.labels_ = np.zeros(self.n_traj, dtype=int)

        tol = 1e-10
        for i in range(self.n_traj):
            print(i)
            coords = self.data[:,i]
            coords_old = np.zeros(self.n_features)

            grad = np.zeros(self.n_features)
            grad_old = np.zeros(self.n_features)
            while True:
                grad = 0
                grad = dr.get_gradient(coords, self.data, self.feature_vars, self.coeff)
                #dens = 0
                #dens = dr.get_density(dens, self.data, self.feature_vars, coords, self.coeff)
                #print(coords, dens, grad)
                
                if np.linalg.norm(grad) < tol: break

                #alpha = min(1, np.abs(np.inner(coords - coords_old, grad - grad_old))/np.linalg.norm(grad - grad_old)**2)
                alpha = min(1, np.linalg.norm(coords - coords_old)**2/np.abs(np.inner(coords - coords_old, grad - grad_old)))
                
                coords += alpha * grad
                grad_old = grad
            self.cluster_positions[:,i] = coords

        t2 = timeit.default_timer()
        print("Maxima assignment: ", f"{t2-t1:10.5f}")
    
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

    def relabel_by_contributions(self):
        """
        Calculates contributions to each point's density from all other points,
        then reassigns its cluster membership to maximise own cluster contribution
        """

        t1 = timeit.default_timer()

        self.abs_contributions = dr.get_contributions_at_datapoints(
            self.data, self.feature_vars, self.labels_, 
            self.coeff, self.pbc, self.n_clus, self.n_pts)

        for i in range(self.n_traj):
            if self.labels_[i] != np.argmax(self.abs_contributions[:,i]):
                self.labels_[i] = np.argmax(self.abs_contributions[:,i])
        self.reduce_cluster_labels()

        self.abs_contributions = dr.get_contributions_at_datapoints(
            self.data, self.feature_vars, self.labels_, 
            self.coeff, self.pbc, self.n_clus, self.n_pts)
        
        self.cluster_sizes = np.zeros(self.n_clus)
        for clus1 in range(self.n_clus):
            self.cluster_sizes[clus1] = np.sum(self.labels_==clus1)
        
        t2 = timeit.default_timer()
        print("Relabelling        ", f"{t2-t1:10.5f}")

    def get_avg_nn_distance(self, clus_id):
        """
        Calculates the average nearest neighbour distance in a given cluster

        Args:''
            clus_id (int): label of the cluster of interest

        Returns:
            float: average nearest neighbour distance
        """

        nn_dist = 0
        for i in np.where(self.labels_==clus_id)[0]:
            smallest_dist = self.n_pts
            for j in np.where(self.labels_==clus_id)[0]:
                if (i==j):
                    continue 
                temp = np.linalg.norm(self.data[:,i] - self.data[:,j])
                if (temp < smallest_dist):
                    smallest_dist = temp
            nn_dist += smallest_dist
        nn_dist /= self.cluster_sizes[clus_id]

        return nn_dist
    
    def merge_two_clusters(self, clus1, clus2):
        """
        Calculates the merge metric of two clusters

        Args:
            clus1 (int): First cluster label
            clus2 (int): Second cluster label
        """

        if self.merge_metric[clus1,clus2] != 0:
            return

        nn_dist1 = self.get_avg_nn_distance(clus1)
        nn_dist2 = self.get_avg_nn_distance(clus2)
        nn_dist = min(nn_dist1, nn_dist2)

        cluster_avg1 = np.mean(
            self.abs_contributions[clus1,:][self.labels_==clus1]+self.abs_contributions[clus2,:][self.labels_==clus1])
        cluster_avg2 = np.mean(
            self.abs_contributions[clus1,:][self.labels_==clus2]+self.abs_contributions[clus2,:][self.labels_==clus2])
        cluster_avg = min(cluster_avg1, cluster_avg2)
        for i in range(self.n_traj):
            if (self.labels_[i] == clus1 or self.labels_[i] == clus2):
                if self.abs_contributions[clus1,i]+self.abs_contributions[clus2,i] < cluster_avg:
                    self.labels_[i] -= self.n_traj

        n_pairs = int(np.ceil(self.cluster_sizes[clus1]**(1/self.n_features))) \
                * int(np.ceil(self.cluster_sizes[clus2]**(1/self.n_features)))
        closest_pts = np.zeros((2, n_pairs), dtype=int, order="F")
        pair_dist = np.zeros((n_pairs))

        closest_pts, pair_dist, path_vecs = dr.find_closest_points(
            self.data, self.labels_, self.pbc, clus1, clus2, n_pairs, self.n_pts)
            
        self.merge_metric[clus1,clus2] = dr.compute_merge_metric(
            self.data, self.feature_vars, self.labels_, closest_pts, path_vecs,
            self.coeff, self.pbc, clus1, clus2, self.n_clus, self.n_pts)

        self.merge_metric[clus1,clus2] *= np.sqrt(pair_dist[0])/nn_dist
        
        self.merge_metric[clus2,clus1] = self.merge_metric[clus1,clus2]
        self.cluster_dist[clus1,clus2] = np.sqrt(pair_dist[0])/nn_dist

        for i in range(self.n_traj):
            if self.labels_[i] < -1:
                self.labels_[i] += self.n_traj

    def merge_clusters(self):
        """
        Merges clusters whose merge metric is below merge threshold
        """

        t1 = timeit.default_timer()
        
        self.cluster_maxima = np.unique(self.cluster_positions, axis=0)
        self.merge_metric = np.zeros((self.n_clus, self.n_clus), dtype=float) + np.identity(self.n_clus)
        self.cluster_dist = np.zeros((self.n_clus, self.n_clus))

        self.border_points = []

        for clus1 in range(self.n_clus):
            for clus2 in range(clus1):
                self.merge_two_clusters(clus1, clus2)

            self.cluster_sizes = np.zeros(self.n_clus)
            for clus1 in range(self.n_clus):
                self.cluster_sizes[clus1] = np.sum(self.labels_==clus1)
        
        cluster_series = np.arange(self.n_clus)
        for clus1 in range(self.n_clus):
            for clus2 in range(clus1):
                if self.merge_metric[clus1,clus2] < 1:
                    self.labels_[self.labels_==cluster_series[clus2]] = cluster_series[clus1]
                    cluster_series[cluster_series==cluster_series[clus2]] = cluster_series[clus1]
        
        self.reduce_cluster_labels()
        t2 = timeit.default_timer()
        print("\rMerging            ", f"{t2-t1:10.5f}", " "*20)

    def plot_clusters(self, plot_grid=False, grid_res=300, axis_labels=None):
        """
        Plots clusters if the number of features is 2 or 3

        Args:
            plot_grid (bool, optional): Requests background density grid generation (2D only). Defaults to False.
            grid_res (int, optional): Resolution of background density grid. Defaults to 300.
            axis_labels (np.ndarray, optional): Labels of axes. Defaults to None.
        """

        if self.n_features == 2:
            colours = [(0.5,0.5,0.5) if self.labels_[i] < 0 \
                else clr.hsv_to_rgb([self.labels_[i]/self.n_clus,1,0.75]) for i in range(self.n_traj)]

            if plot_grid:
                self.data *= (grid_res-1)/(self.n_pts-1)
                self.feature_vars *= ((grid_res-1)/(self.n_pts-1))**2
                self.grid = dr.generate_grid_2d(
                    self.data, self.feature_vars, 
                    self.coeff, self.n_pts)

                plt.imshow(self.grid.T, cmap="binary_r", origin="lower")
                plt.xlim([0, grid_res])
                plt.ylim([0, grid_res])

            plt.xticks([])
            plt.yticks([])
            if axis_labels is not None:
                plt.xlabel(axis_labels[0,0])
                plt.ylabel(axis_labels[1,0])
            
            plt.scatter(
                self.data[0,:],
                self.data[1,:],
                c=colours,
                s=10)
            plt.xlim(min(self.data[0]), max(self.data[0]))
            plt.ylim(min(self.data[1]), max(self.data[1]))

            '''
            plt.scatter(
                self.cluster_positions[0,:],
                self.cluster_positions[1,:],
                c="k",
                s=10)
            '''
            if plot_grid:
                self.data /= (grid_res-1)/(self.n_pts-1)
                self.feature_vars /= ((grid_res-1)/(self.n_pts-1))**2
            
            plt.show()
            
        elif self.n_features == 3:
            colours = ["#7f7f7f" if self.labels_[i] < 0 \
                else Clustering.plot_colours[self.labels_[i]] for i in range(self.n_traj)]
            
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(
                self.data[:,0],
                self.data[:,1],
                self.data[:,2],
                c=colours,
                s=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
    
            if axis_labels is not None:
                ax.set_xlabel(axis_labels[0,0])
                ax.set_ylabel(axis_labels[1,0])
                ax.set_zlabel(axis_labels[2,0])
            ax.set_frame_on(False)
            ax.set_xlim([0, self.n_pts])
            ax.set_ylim([0, self.n_pts])
            ax.set_zlim([0, self.n_pts])

            plt.show()

    def fit(self, data, feature_vars=None, pbc=[]):
        """
        Master fitting method calling all the successive steps
        Clustering assignments can be accessed via the labels_ attribute

        Args:
            data (np.ndarray): A 2D array containing data to be fitted. Shape required to be (n_traj, n_features)
        """

        self.set_data(data)

        if feature_vars is None:
            print("Normalising data")
            self.normalise_data()
            print("Calculating feature variances")
            self.get_feature_vars()
        else:
            self.feature_vars = feature_vars
        self.rescale_data()
        
        self.pbc = np.zeros(self.n_features, dtype=int)
        self.pbc[pbc] = 1
        print("Finding maxima")
        self.get_clusters()
        print("Converting labels")
        self.positions_to_labels()
        print("Relabelling by contribution")
        self.relabel_by_contributions()
        
        print("Merging clusters")
        self.merge_clusters()


class ClusteringTD:
    def __init__(self, coeff=1., merge_threshold=1, td_variances=False):
        self.coeff = coeff
        self.merge_threshold = merge_threshold
        self.td_variances = td_variances
    
    def normalise_data(self):
        for j in range(self.n_features):
            self.data[:,:,j] -= np.amin(self.data[:,:,j])
            self.data[:,:,j] /= np.amax(self.data[:,:,j])

    def get_feature_vars(self):
        self.feature_vars = np.zeros(self.n_features)
        for f in range(self.n_features):
            self.feature_vars[f] = np.var(self.data[:,:,f])
    
    def plot_clusters(self, t, axis_labels=None):
        clustering_ti = Clustering()
        clustering_ti.set_data(self.data[t])
        clustering_ti.labels_ = self.labels_[t]
        clustering_ti.n_clus = np.max(clustering_ti.labels_) + 1
        clustering_ti.plot_clusters(axis_labels=axis_labels)

    def pair_labels(self, t):
        clusters_old = self.labels_[t-1]
        clusters_new = self.labels_[t]
        Nold = np.amax(clusters_old)+1
        Nnew = np.amax(clusters_new)+1
        pairs = np.full(Nnew, -1, dtype=int)
        overlap = np.zeros((Nold, Nnew), dtype=int)
        for i in range(self.n_traj):
            if clusters_old[i] >= 0 and clusters_new[i] >= 0:
                overlap[clusters_old[i], clusters_new[i]] += 1

        for i in range(Nnew):
            mx = np.argmax(overlap)
            if np.amax(overlap) == -1:
                break
            else:
                pairs[mx%Nnew] = mx//Nnew
                overlap[mx//Nnew] = np.full(Nnew, -1)
                overlap[:,mx%Nnew] = np.full(Nold, -1)

        while -1 in pairs:
            pairs[list(pairs).index(-1)] = max(np.amax(pairs), Nold)+1

        return pairs

    def convert_labels(self):
        for t in range(1, self.n_ts):
            pairs = self.pair_labels(t)
            for old, new in enumerate(pairs):
                self.labels_[t][self.labels_[t] == old] = new + self.n_traj
            self.labels_[t] -= self.n_traj
        
    def fit(self, data):
        self.n_ts, self.n_traj, self.n_features = np.shape(data)
        self.data = np.empty((self.n_ts, self.n_traj, self.n_features), dtype=float)
        np.copyto(self.data, data)

        self.labels_ = np.zeros((self.n_ts, self.n_traj), dtype=int)

        if not self.td_variances:
            self.normalise_data()
            self.get_feature_vars()

        clustering_ti = Clustering(self.coeff, self.merge_threshold)
        for t in range(self.n_ts):
            print("Step {}/{}".format(t+1, self.n_ts))
            if self.td_variances:
                clustering_ti.fit(data[t])
            else:
                clustering_ti.fit(data[t], self.feature_vars)
            print(clustering_ti.feature_vars)
            self.labels_[t] = clustering_ti.labels_
