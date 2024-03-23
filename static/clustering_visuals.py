import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn import *
import pyqofta as p
import make_data
import ctypes as ct
import numpy.ctypeslib as npc
import timeit

libc = ct.WinDLL("clustering_functions", winmode=0)

def populate_data(data, ensemble, Nts):
    Nts, Ntraj, Nfeatures = np.shape(data)
    for t in range(Nts):
        for i, traj in enumerate(ensemble):
            data[t,i,:] = [traj.geometries[t].bond_length([0,1]), \
                traj.geometries[t].bond_length([1,2]), \
                traj.geometries[t].bond_length([2,3]), \
                traj.geometries[t].bond_length([3,4]), \
                traj.geometries[t].bond_length([4,5]), \
                traj.geometries[t].bond_length([0,5]), \
                np.abs(traj.geometries[t].angle([0,1,2])), \
                np.abs(traj.geometries[t].angle([1,2,3])), \
                np.abs(traj.geometries[t].angle([2,3,4])), \
                np.abs(traj.geometries[t].angle([3,4,5])), \
                np.abs(traj.geometries[t].angle([4,5,0])), \
                np.abs(traj.geometries[t].angle([5,0,1])), \
                np.abs(traj.geometries[t].dihedral([0,1,2,3])), \
                np.abs(traj.geometries[t].dihedral([1,2,3,4])), \
                np.abs(traj.geometries[t].dihedral([2,3,4,5])), \
                np.abs(traj.geometries[t].dihedral([3,4,5,0])), \
                np.abs(traj.geometries[t].dihedral([4,5,0,1])), \
                np.abs(traj.geometries[t].dihedral([5,0,1,2]))]

            """data[t,i,:] = [traj.geometries[t].bond_length([0,1]), \
                traj.geometries[t].bond_length([1,2]), \
                traj.geometries[t].bond_length([2,3]), \
                traj.geometries[t].bond_length([3,4]), \
                traj.geometries[t].bond_length([4,5]), \
                traj.geometries[t].bond_length([0,5]), \
                traj.geometries[t].angle([0,1,2]), \
                traj.geometries[t].angle([1,2,3]), \
                traj.geometries[t].angle([2,3,4]), \
                traj.geometries[t].angle([3,4,5]), \
                traj.geometries[t].angle([4,5,0]), \
                traj.geometries[t].angle([5,0,1]), \
                traj.geometries[t].dihedral([0,1,2,3]), \
                traj.geometries[t].dihedral([1,2,3,4]), \
                traj.geometries[t].dihedral([2,3,4,5]), \
                traj.geometries[t].dihedral([3,4,5,0]), \
                traj.geometries[t].dihedral([4,5,0,1]), \
                traj.geometries[t].dihedral([5,0,1,2])]
            for f in range(6, Nfeatures):
                if data[t,i,f] - data[t-1,i,f] < -180:
                    data[t,i,f] += 360"""
    return data

def normalise_data(data):
    if data.ndim == 3:
        Nts, _, Nfeatures = np.shape(data)
        param_min, param_max = np.zeros(Nfeatures), np.zeros(Nfeatures)
        for j in range(Nfeatures):
            param_min[j] = np.min(data[:,:,j])
            param_max[j] = np.max(data[:,:,j])
            data[:,:,j] -= param_min[j]
            if param_max[j]-param_min[j] != 0:
                data[:,:,j] /= (param_max[j]-param_min[j])
        return data, param_min, param_max-param_min
    elif data.ndim == 2:
        _, Nfeatures = np.shape(data)
        param_min, param_max = np.zeros(Nfeatures), np.zeros(Nfeatures)
        for j in range(Nfeatures):
            param_min[j] = np.min(data[:,j])
            param_max[j] = np.max(data[:,j])
            data[:,j] -= param_min[j]
            data[:,j] /= (param_max[j]-param_min[j])
        return data, param_min, param_max-param_min
            
def standardise_data(data, std=1):
    Nts, _, Nfeatures = np.shape(data)
    feature_means, feature_stds = np.zeros(Nfeatures), np.zeros(Nfeatures)
    for t in range(Nts):
        for j in range(Nfeatures):
            feature_means[j] = np.mean(data[t,:,j])
            feature_stds[j] = np.std(data[t,:,j])
            data[t,:,j] -= feature_means[j]
            data[t,:,j] *= std/feature_stds[j]
    return data, feature_means, feature_stds

def calculate_distance_matrix(data):
    Ntraj, _ = np.shape(data)
    dmatrix = np.zeros((Ntraj, Ntraj))
    for i in range(Ntraj):
        for j in range(i):
            #dmatrix[t,i,j] = np.linalg.norm(data[-1,i,use_features]-data[-1,j,use_features])
            dmatrix[i,j] = np.linalg.norm(data[i,:]-data[j,:])
            dmatrix[j,i] = dmatrix[i,j]
    return dmatrix

def calculate_rmsds(ensemble):
    Ntraj, Nts = ensemble.ntrajs, ensemble.nts_max
    rmsds = np.zeros((Nts, Ntraj, Ntraj))
    for t in range(Nts):
        for i in range(Ntraj):
            for j in range(i):
                rmsds[t,i,j] = p.Molecule.Kabsch_rmsd(ensemble.trajs[i].geometries[t], ensemble.trajs[j].geometries[t], \
                    Hydrogens=False, Mirror=True)
                rmsds[t,j,i] = rmsds[t,i,j]

def distance_matrix_over_time(data):
    Nts, Ntraj, Nfeatures = np.shape(data)
    dmatrix = np.zeros((Ntraj,Ntraj))
    for t in range(Nts):
        for i in range(Ntraj):
            for j in range(i):
                dmatrix[i,j] += np.inner(data[t,i,:]-data[t,j,:],data[t,i,:]-data[t,j,:])
    dmatrix = np.sqrt(dmatrix)
    dmatrix /= Nts
    for i in range(Ntraj):
        for j in range(i):
            dmatrix[j,i] = dmatrix[i,j]
    return dmatrix

def get_distances_dtw(data):
    Nts, Ntraj, Nfeatures = np.shape(data)
    distance_matrix = np.zeros((Ntraj, Ntraj))
    for i in range(Ntraj):
        for j in range(i):
            Zi, Zj = data[:,i,:], data[:,j,:] # each of the two nfeature dimensional trajectories in question
            d = metrics.pairwise.euclidean_distances(Zi,Zj)
            alignment = dtw(d, keep_internals=True, step_pattern=rabinerJuangStepPattern(2, "b")) # do DTW
            distance_matrix[i, j] = distance_matrix[j, i] = alignment.distance # store the cost associated with the minimum warping path
    return distance_matrix

def molecule_to_xyz(molecule, time):
    outstr = str(molecule.natoms) + "\n" + str(time) + "\n"
    for i, at in enumerate(molecule.coordinates):
         outstr += molecule.atom_labels[i] + " "
         outstr += str(at[0]) + " " + str(at[1]) + " " + str(at[2]) + "\n"
    return outstr

def give_label(dih, cis=60, tra=120):
    bins = np.ones(3, dtype=int)
    outstr = ""
    for i in range(3):
        if dih[i] < cis:
            bins[i] = 0
        if dih[i] > tra:
            bins[i] = 2
    if bins[0] > bins[2]:
        bins[[0,2]] = bins[[2,0]]
    outstr += "c"*(int(bins[0]==0)) + "n"*(int(bins[0]==1)) + "t"*(int(bins[0]==2))
    outstr += "Z"*(int(bins[1]==0)) + "N"*(int(bins[1]==1)) + "E"*(int(bins[1]==2))
    outstr += "c"*(int(bins[2]==0)) + "n"*(int(bins[2]==1)) + "t"*(int(bins[2]==2))
    return outstr

def reduce_data(data, freq):
    Nts, Ntraj, Nfeatures = np.shape(data)
    new_data = np.zeros((Nts//freq, Ntraj, Nfeatures))
    for t in range(Nts//freq):
        new_data[t,:,:] = data[freq*t,:,:]
    return new_data, Nts//freq

def features_to_bool(use_features, Nfeatures):
    use_features_bool = np.full(Nfeatures, False, dtype=bool)
    for f in range(Nfeatures):
        if f in use_features:
            use_features_bool[f] = True
    return use_features_bool


def density_score_rel(rel_contributions, clusters):
    Ntraj = len(clusters)
    score = 0
    for i in range(Ntraj):
        if clusters[i] != 0:
            score += rel_contributions[i,clusters[i]]
    score /= Ntraj
    return score
            
def find_outliers(abs_contributions, clusters):
    Ntraj = len(clusters)
    for i in range(Ntraj):
        if (abs_contributions[i,clusters[i]] <= \
            np.sum(abs_contributions[i,clusters[i]+1:]) + np.sum(abs_contributions[i,:clusters[i]])) \
            or (np.sum(abs_contributions[i,:]) < 2):
            clusters[i] = -1
    return clusters

def pair_labels(clusters_old, clusters_new, t=None):
    Ntraj = len(clusters_old)
    Nold = np.amax(clusters_old)+1
    Nnew = np.amax(clusters_new)+1
    pairs = np.full(Nnew, -1, dtype=int)
    overlap = np.zeros((Nold, Nnew), dtype=int)
    for i in range(Ntraj):
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

def convert_labels(clusters):
    Nts, Ntraj = np.shape(clusters)
    for t in range(1,Nts):
        pairs = pair_labels(clusters[t-1], clusters[t], t)
        for old, new in enumerate(pairs):
            clusters[t][clusters[t] == old] = new + Ntraj
        clusters[t] -= Ntraj
    return clusters

def find_convoys(clusters, tolerance=0.9):
    Nts, Ntraj = np.shape(clusters)
    similarity = np.zeros((Ntraj, Ntraj))
    tstart = 0
    while np.all(clusters[tstart] == 0):
        tstart += 1
    for i in range(Ntraj):
        for j in range(i+1):
            similarity[i,j] = np.sum(clusters[tstart:,i] == clusters[tstart:,j])
            similarity[j,i] = similarity[i,j]
    
    similarity /= Nts-tstart
    convoys = np.full(Ntraj, -1, dtype=int)
    c = 0
    for i in range(Ntraj):
        if convoys[i] == -1:
            convoys[similarity[i,:] >= tolerance] = c
            c += 1
        else:
            for j in range(Ntraj):
                if similarity[i,j] >= tolerance:
                    if convoys[j] == -1:
                        convoys[j] = convoys[i]
                    else:
                        convoys[convoys == convoys[j]] = convoys[i]
                        convoys[j] = convoys[i]
    
    occurrence = np.bincount(convoys)
    for i in range(len(occurrence)):
        if occurrence[i] < 2:
            convoys[convoys == i] = -1
    return convoys

def cluster_evolution(clusters):
    Nts, Ntraj = np.shape(clusters)
    Nclus = np.amax(clusters)+1
    colour_map = {}
    for cl in range(Nclus):
        colour_map[cl] = clr.hsv_to_rgb([cl/Nclus, 1, 0.8])
    colours = np.zeros((Ntraj, Nts, 3), dtype=int)
    for t in range(Nts):
        for i in range(Ntraj):
            colours[i,t,:] = np.floor(256*colour_map[clusters[t,i]])
    plt.imshow(colours)

def polar_plot(clusters, data, labels=None, short_labels=False):
    fig, ax = plt.subplots()
    Ntraj, Nfeatures = np.shape(data)
    Nclus = len(np.unique(clusters))
    data = normalise_data(data.reshape((1,Ntraj,Nfeatures)))
    ax.set_axis_off()
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    background = plt.Circle((0,0), 1, edgecolor="black", fill=None, clip_on=False)
    ax.add_patch(background)
    for traj, col in enumerate(clusters):
        Xvals = data[traj,:] * np.sin(2*np.pi*np.arange(Nfeatures)/Nfeatures)
        Yvals = data[traj,:] * np.cos(2*np.pi*np.arange(Nfeatures)/Nfeatures)
        plt.plot(np.append(Xvals, Xvals[0]), np.append(Yvals, Yvals[0]), \
            color=clr.hsv_to_rgb([col/Nclus, 1, 0.75]), linewidth="0.5")
    if labels != None:
        for f in range(Nfeatures):
            plt.text(1.1*np.sin(2*np.pi*f/Nfeatures), 1.1*np.cos(2*np.pi*f/Nfeatures), \
                labels[f][int(short_labels)], rotation=-360*f/Nfeatures-180*(f/Nfeatures>1/4 and f/Nfeatures<3/4), \
                va="center", ha="center")

def polar_plot_subplots(clusters, data, labels=None, short_labels=True):
    Ntraj, Nfeatures = np.shape(data)
    Nclus = len(np.unique(clusters))
    data = normalise_data(data.reshape((1,Ntraj,Nfeatures)))
    data = np.squeeze(data)
    grid_size = int(np.ceil(np.sqrt(Nclus)))
    fig, ax = plt.subplots(int(np.ceil(Nclus/grid_size))+(int(np.ceil(Nclus/grid_size))<2), grid_size, \
        sharex=True, sharey=True, constrained_layout=True)
    ax[0,0].set_xlim([-1,1])
    ax[0,0].set_ylim([-1,1])
    for i in range(int(np.ceil(Nclus/grid_size))):
        for j in range(grid_size):
            ax[i,j].set_axis_off()
            if i*grid_size+j < Nclus:
                background = plt.Circle((0,0), 1, edgecolor="black", fill=None, clip_on=False)
                ax[i,j].add_patch(background)
                if labels is not None:
                    for f in range(Nfeatures):
                        ax[i,j].text(1.1*np.sin(2*np.pi*f/Nfeatures), 1.1*np.cos(2*np.pi*f/Nfeatures), \
                            labels[f][int(short_labels)], rotation=-360*f/Nfeatures-180*(f/Nfeatures>1/4 and f/Nfeatures<3/4), \
                            va="center", ha="center")
    for traj, col in enumerate(clusters):
        Xvals = data[traj,:] * np.sin(2*np.pi*np.arange(Nfeatures)/Nfeatures)
        Yvals = data[traj,:] * np.cos(2*np.pi*np.arange(Nfeatures)/Nfeatures)
        ax[col//grid_size, col%grid_size].plot(np.append(Xvals, Xvals[0]), np.append(Yvals, Yvals[0]), \
            color=clr.hsv_to_rgb([col/Nclus,1,0.75]), linewidth="0.5")

def features_plot(clusters, data, means=None, stds=None, labels=None, short_labels=True):
    Nts, Ntraj, Nfeatures = np.shape(data)
    if means is None:
        means = np.zeros(Nfeatures)
    if stds is None:
        stds = np.ones(Nfeatures)

    Nclus = np.amax(clusters)+1
    outliers = -1 in clusters
    if outliers:
        Noutliers = len(np.nonzero(clusters == -1)[0])
        Nclus += 1
    fig, ax = plt.subplots(Nclus, Nfeatures, sharex=True, sharey="col", constrained_layout=True)
    #fig.tight_layout()
    ax[0,0].set_xlim([0,Nts-1])
    for j in range(Nfeatures):
        if labels is not None:
            ax[0,j].set_title(labels[j][int(short_labels)])
        ax[-1,j].set_xlabel("Timestep")
        for i in range(Nclus-int(outliers)):
            for traj in np.nonzero(clusters == i):
                ax[i,j].plot(np.arange(Nts), stds[j]*data[:,traj,j]+means[j], color=clr.hsv_to_rgb([i/Nclus,1,0.75]), \
                    linewidth="0.5")
        if outliers:
            for k, traj in enumerate(np.nonzero(clusters == -1)[0]):
                ax[-1,j].plot(np.arange(Nts), stds[j]*data[:,traj,j]+means[j], color=clr.hsv_to_rgb([k/Noutliers,1,0.75]), \
                    linewidth="0.5")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

def scores_plot(clus_type, data, limits, param_name="Number of clusters", Nsteps=None, clusters_true=None):
    Nts, Ntraj, Nfeatures = np.shape(data)
    dmatrix = get_distances_dtw(data)

    match clus_type:
        case "affinity_propagation":
            clustering = cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
        case "single_linkage":
            clustering = cluster.AgglomerativeClustering(affinity="precomputed", linkage="single")
        case "complete_linkage":
            clustering = cluster.AgglomerativeClustering(affinity="precomputed", linkage="complete")
        case "average_linkage":
            clustering = cluster.AgglomerativeClustering(affinity="precomputed", linkage="average")
        case _:
            print("No valid clustering type chosen")
        
    if Nsteps == None:
        params = np.arange(limits[0], limits[1]+1)
    else:
        params = np.linspace(limits[0], limits[1], num=Nsteps, endpoint=True)
    
    Nscores = 3+int(clusters_true is not None)
    fig, ax = plt.subplots(Nscores, 1, sharex="all", constrained_layout=True)
    ax[Nscores-1].set_xlabel(param_name)
    ax[0].set_ylabel("Calinski-Harabasz")
    ax[1].set_ylabel("Davies-Bouldin")
    ax[2].set_ylabel("Silhouette")
    if clusters_true is not None:
        ax[3].set_ylabel("Adjusted Rand")
    scores = np.zeros((Nscores,len(params)))
    for i, param in enumerate(params):
        if clus_type=="affinity_propagation":
            clustering.damping = param
        else:
            clustering.n_clusters = param
        clustering.fit(dmatrix)
        scores[0,i] = np.mean([metrics.calinski_harabasz_score(data[j,:,:], clustering.labels_) \
            for j in range(Nts)])
        scores[1,i] = np.mean([metrics.davies_bouldin_score(data[j,:,:], clustering.labels_) \
            for j in range(Nts)])
        scores[2,i] = np.mean([metrics.silhouette_score(data[j,:,:], clustering.labels_) \
            for j in range(Nts)])
        if clusters_true is not None:
            scores[3,i] = metrics.adjusted_rand_score(clusters_true, clustering.labels_)

    for i in range(Nscores):
        ax[i].plot(params, scores[i], "-+", linewidth="0.5")

def scores_plot_3d(clus_type, data, use_features, limits1, limits2, Nsteps1=None, Nsteps2=None, \
    param_name1="xi", param_name2="minPts", clusters_true=None):
    Nts, Ntraj, Nfeatures = np.shape(data)
    use_features_bool = features_to_bool(use_features, Nfeatures)
    dmatrix = np.zeros((Ntraj, Ntraj))
    libc.distance_matrix_over_time(ct.byref(npc.as_ctypes(data)), ct.byref(npc.as_ctypes(dmatrix)), \
        ct.byref(npc.as_ctypes(use_features_bool)), Nts, Ntraj, Nfeatures)
    match clus_type:
        case "affinity_propagation":
            clustering = cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
        case "single_linkage":
            clustering = cluster.AgglomerativeClustering(affinity="precomputed", linkage="single")
        case "complete_linkage":
            clustering = cluster.AgglomerativeClustering(affinity="precomputed", linkage="complete")
        case "average_linkage":
            clustering = cluster.AgglomerativeClustering(affinity="precomputed", linkage="average")
        case "dbscan":
            clustering = cluster.DBSCAN(metric="precomputed")
        case "optics":
            clustering = cluster.OPTICS(metric="precomputed", min_cluster_size=2)
        case _:
            print("No valid clustering type chosen")
        
    if Nsteps1 is None:
        params1 = np.arange(limits1[0], limits1[1]+1)
    else:
        params1 = np.linspace(limits1[0], limits1[1], num=Nsteps1, endpoint=True)
    
    if Nsteps2 is None:
        params2 = np.arange(limits2[0], limits2[1]+1)
    else:
        params2 = np.linspace(limits2[0], limits2[1], num=Nsteps2, endpoint=True)
    
    mesh = np.meshgrid(params1, params2)
    scores = np.zeros((len(params1), len(params2)))

    Nscores = 3+int(clusters_true is not None)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel(param_name1)
    ax.set_ylabel(param_name2)
    ax.set_zlabel("Silhouette Score")
    for i, param1 in enumerate(params1):
        if clus_type == "dbscan":
            clustering.eps = param1
        else:
            clustering.xi = param1
        for j, param2 in enumerate(params2):
            print(param1, param2)
            clustering.min_samples = param2
            clustering.fit(dmatrix)
            if len(np.unique(clustering.labels_)) == 1:
                if np.max(clustering.labels_) == 0:
                    scores[i,j] = 0
                else:
                    scores[i,j] = -1
            else:
                scores[i,j] = np.average([metrics.silhouette_score(data[j,:,:][:,use_features], clustering.labels_) \
                    for j in range(Nts)], weights=np.arange(Nts)+1)
            """scores[i,j] = np.mean([metrics.adjusted_rand_score(clusters_true, clustering.labels_) \
                for j in range(Nts)])"""
    X,Y = mesh
    ax.plot_surface(X, Y, scores.T, cmap="viridis")
    breakpoint()

#============================================================================

if __name__ == "__main__":
    #load ensemble and reference structures
    fpaths = ["R_traj0" +"0"*(i<10) + str(i) + ".xyz" for i in range(100)]
    ensemble = p.Ensemble.load_ensemble(fpaths, "sh")
    ref_struct = [p.Molecule.init_from_xyz("cZc-HT-Transformed.xyz"), p.Molecule.init_from_xyz("s0_s1_CI_Closed-Transformed.xyz"), \
        p.Molecule.init_from_xyz("s0_s1_CI_Open-Transformed.xyz")]

    #define some useful variables and create empty data arrays
    Ntraj, Nts, Nat, Nfeatures = ensemble.ntrajs, ensemble.nts_max, ensemble.trajs[0].geometries[0].natoms, 18
    data = np.zeros((Nts, Ntraj, Nfeatures))
    feature_names = {0: ["C1-C2 length", "BL12"], 1: ["C2-C3 length", "BL23"], 2: ["C3-C4 length", "BL34"], \
        3: ["C4-C5 length", "BL45"], 4: ["C5-C6 length", "BL56"], 5: ["C1-C6 length", "BL16"], \
        6: ["1-2-3 Angle", "A123"], 7: ["2-3-4 Angle", "A234"], 8: ["3-4-5 Angle", "A345"], \
        9: ["4-5-6 Angle", "A456"], 10: ["5-6-1 Angle", "A561"], 11: ["6-1-2 Angle", "A612"], \
        12: ["1-2-3-4 Dihedral", "D1234"], 13: ["2-3-4-5 Dihedral", "D2345"], 14: ["3-4-5-6 Dihedral", "D3456"], \
        15: ["4-5-6-1 Dihedral", "D4561"], 16: ["5-6-1-2 Dihedral", "D5612"], 17: ["6-1-2-3 Dihedral", "D6123"]}
    feature_names_ic = {0: ["C1-C2 length", "BL12"], 1: ["C2-C3 length", "BL23"], 2: ["C3-C4 length", "BL34"], \
        3: ["C4-C5 length", "BL45"], 4: ["C5-C6 length", "BL56"], \
        5: ["1-2-3 Angle", "A123"], 6: ["2-3-4 Angle", "A234"], 7: ["3-4-5 Angle", "A345"], \
        8: ["4-5-6 Angle", "A456"], \
        9: ["1-2-3-4 Dihedral", "D1234"], 10: ["2-3-4-5 Dihedral", "D2345"], 11: ["3-4-5-6 Dihedral", "D3456"]}

    #populate data
    #populate_data(data, ensemble, Nts)
    xyz_data, data, clusters_true = make_data.load_data()
    Nts, Ntraj, Nfeatures = np.shape(data)
    data, Nts = reduce_data(data, 20)
    #convert data to interval [0,1] or standardise
    _, feature_means, feature_stds = normalise_data(data)
    """#perform pca
    pca = skd.PCA(n_components=0.9, svd_solver="full")
    pca_data = pca.fit_transform(data[-1,:,:])
    """
    #data_emb = manifold.TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(data[-1,:,:])

    #calculate distance matrix
    use_features = [9,10,11]
    plot_features = [9,10,11]
    dmatrix = np.zeros((Ntraj, Ntraj))
    use_features_bool = features_to_bool(use_features, Nfeatures)
    libc.distance_matrix_over_time(ct.byref(npc.as_ctypes(data)), ct.byref(npc.as_ctypes(dmatrix)), \
        ct.byref(npc.as_ctypes(use_features_bool)), Nts, Ntraj, Nfeatures)
    #dmatrix = calculate_distance_matrix(data[-1,:,:][:,use_features])
    #scores_plot("average_linkage", data[:,:,use_features], [2,12], clusters_true=clusters_true)
    #breakpoint()
    #dmatrix = get_distances_dtw(data[:,:,use_features])
    #rmsds = calculate_rmsds(data)


    cl = cluster.OPTICS(metric="precomputed", xi=0.03, min_samples=10, min_cluster_size=5).fit(dmatrix)
    #cl = cluster.AffinityPropagation(damping=0.5, affinity="precomputed").fit(dmatrix)
    #cl = cluster.AgglomerativeClustering(n_clusters=6, affinity="euclidean", linkage="average").fit(dmatrix)
    clusters = cl.labels_
    outliers = -1 in clusters
    Nclus = len(np.unique(clusters)) - outliers
    if 0:
        colors = ["g.", "r.", "b.", "y.", "c."]
        space = np.arange(Ntraj)
        labels = cl.labels_[cl.ordering_]
        reachability = cl.reachability_[cl.ordering_]
        for klass, color in zip(range(0, Nclus), colors):
            Xk = space[labels == klass]
            Rk = reachability[labels == klass]
            plt.plot(Xk, Rk, color, alpha=0.3)
        plt.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
        #plt.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
        #plt.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
    for clus in range(-int(outliers), Nclus-int(outliers)):
        out = np.where(clusters == clus)
        print(clus)
        print([str(i) + ": " + give_label(data[-1,i,9:12]) for i in out[0]])
    features_plot(clusters, data[:,:,plot_features], means=feature_means, stds=feature_stds, \
        labels=[feature_names_ic[i] for i in plot_features], outliers=outliers)
    #polar_plot_subplots(clusters, data[-1,:,:][:,plot_features], labels=[feature_names[i] for i in use_features])

    print("Calinski-Harabasz:\t", metrics.calinski_harabasz_score(data[-1,:,:][:,use_features], clusters))
    print("Davies-Bouldin:\t\t", metrics.davies_bouldin_score(data[-1,:,:][:,use_features], clusters))
    print("Silhouette:\t\t", metrics.silhouette_score(data[-1,:,:][:,use_features], clusters))
    #print("Adjusted Rand\t\t", metrics.adjusted_rand_score(clusters_true, clusters))
    breakpoint()