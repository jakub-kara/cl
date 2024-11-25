from clustering import *
from matplotlib import animation

from sklearn import datasets, cluster, decomposition, preprocessing

def animate_clusters(data):
    time_clusters = np.load("clusters_time.npy")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    points = ax.scatter(
        data[0,:,:][:,0],
        data[0,:,:][:,1],
        data[0,:,:][:,2])
    
    title = ax.set_title("")
    ax.set_xlabel("0")
    ax.set_ylabel("1")
    ax.set_zlabel("2")

    def update(t):
        title.set_text("time={}".format(2*t))
        points._offsets3d = (
            data[t,:,:][:,0],
            data[t,:,:][:,1],
            data[t,:,:][:,2])
        points.set_color(["#7f7f7f" if time_clusters[t,i] < 0 \
            else Donkey.plot_colours[time_clusters[t,i]] for i in range(n_traj)])
    
    ani = animation.FuncAnimation(fig, update, frames=np.arange(Nts), blit=False, interval=300)
    plt.show()

    writergif = animation.PillowWriter(fps=4)
    ani.save(f"figures\chd_nd\chd_evolution.gif", writer=writergif)

def populate_data(data, ensemble, Nts):
    Nts, n_traj, Nfeatures = np.shape(data)
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
    return data

if __name__ == "__main__":
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

    #choose either CHD data or fake data
    if 0:
        import pyqofta as p

        #load ensemble and reference structures
        fpaths = ["../R_traj0" +"0"*(i<10) + str(i) + ".xyz" for i in range(100)]
        ensemble = p.Ensemble.load_ensemble(fpaths, "sh")
        #ref_struct = [p.Molecule.init_from_xyz("cZc-HT-Transformed.xyz"), p.Molecule.init_from_xyz("s0_s1_CI_Closed-Transformed.xyz"), \
        #p.Molecule.init_from_xyz("s0_s1_CI_Open-Transformed.xyz")]

        #define some useful variables and create empty data arrays
        n_traj, Nts, Nat, n_features = ensemble.ntrajs, ensemble.nts_max, ensemble.trajs[0].geometries[0].natoms, 18
        use_features = np.array([5,13,14])
        #use_features = np.arange(18)
        plot_features = np.array([13,14])
        data = np.zeros((Nts, n_traj, n_features))
        populate_data(data, ensemble, Nts)
    else:
        """xyz_data, data, clusters_true = make_data.load_data()
        data, Nts = reduce_data(data, 20)
        Nts, n_traj, Nfeatures = np.shape(data)
        use_features = np.array([9,10,11])
        plot_features = np.array([9,10,11])
        """
        
        n_features = 2
        n_samples = 500
        # mixed moons and blobs
        moons, _ = datasets.make_moons(n_samples=60, noise=0.05)
        blobs, _ = datasets.make_blobs(n_samples=60, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
        test_data = np.vstack([moons, blobs])

        #from sklearn
        np.random.seed(0)

        noisy_circles, ycircles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
        noisy_moons, ymoons = datasets.make_moons(n_samples=n_samples, noise=0.05)
        blobs, yblobs = datasets.make_blobs(n_samples=n_samples, random_state=30)
        temp = np.amax(blobs[:,0]) - np.amin(blobs[:,0])
        #blobs[:,0] -= 0.4*temp
        #blobs[:,0] = np.mod(blobs[:,0], temp)

        blob, _ = datasets.make_blobs(n_samples=n_samples, centers=np.array([[0.5,0.5]]), cluster_std=0.05)
        no_structure, _ = np.random.rand(n_samples, 2), None
        blob_noise = np.append(blob, no_structure, axis=0)


        # Anisotropicly distributed data
        random_state = 170
        X, yaniso = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)

        # blobs with varied variances
        stds = [1.0, 2.5, 0.5]
        varied, yvaried, centres = datasets.make_blobs(n_samples=n_samples, cluster_std=stds, random_state=random_state, return_centers=True)
        blobs1, _ = datasets.make_blobs(n_samples=n_samples, random_state=30, centers=np.array([[0,0],[1,0],[1/2,np.sqrt(3)/2]]), cluster_std=0.3*np.ones(3))

        colors = np.array([
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
            "#984ea3",
            "#999999",
            "#e41a1c",
            "#dede00",
            "#000000"
        ])

    clusters = Donkey()
    clusters.fit(X_aniso)
    breakpoint()

    ref = np.zeros(639, dtype=int)
    for t in np.arange(17, dtype=int)*10:
        print(t)
        if 1:
            bonds = np.load("bonds.npy")[:,:,1:]
            veloc = bonds[:,1:,:] - bonds[:,:-1,:]
            bonds = bonds[:,:-1,:]
            nbd = np.hstack((bonds[:,t], veloc[:,t]))
        data = nbd
        clusters = Donkey(name=f"nb_{t}")
        clusters.data = data.T
        clusters.n_traj = data.shape[0]
        clusters.labels_ = np.load(f"donkey/nb_{t}.lab.npy")
        clusters.n_clus = np.max(clusters.labels_+1)
        Donkey.align_labels(clusters.labels_, ref)
        clusters.plot_clusters(axes=[0,1,2], angle=(-40,40,0), pts=False, axis_labels=["wing separation", "rhombicity", "wing length"],
                               save=f"/mnt/c/Users/karaj/Desktop/UoO/cl/paper/test/blank.png", axis_limits=[[1.2,3.2], [-0.9,0.9], [1.1,1.9]])
        plt.clf()
        ref = 1*clusters.labels_
        breakpoint()

    breakpoint()

    ref = np.zeros(639, dtype=int)
    for t in np.arange(17, dtype=int)*10:
        if 1:
            bonds = np.load("bonds.npy")[:,:,1:]
            veloc = bonds[:,1:,:] - bonds[:,:-1,:]
            bonds = bonds[:,:-1,:]
            nbd = np.hstack((bonds[:,t], veloc[:,t]))
        data = nbd
        clusters = Donkey(name=f"nb_{t}", log=True)
        pca = decomposition.PCA()
        pca.fit(data)
        print(pca.explained_variance_ratio_)
        for i, rat in enumerate(pca.explained_variance_ratio_): 
            if rat < 1e-2: break
        trans = pca.components_ @ data.T
        clusters.fit(trans[:i,:].T)
        clusters.data = data.T
        Donkey.align_labels(clusters.labels_, ref)
        clusters.plot_clusters(axes=[0,1,2], ticks=True, angle=(-40,40,0), title=f"{t//2} fs", axis_labels=["wing separation (Å)", "rhombicity (Å)", "wing length (Å)"],
                               save=f"/mnt/c/Users/karaj/Desktop/UoO/cl/paper/abs/nb_{t}.png", axis_limits=[[1.2,3.2], [-0.9,0.9], [1.1,1.9]])
        plt.clf()
        ref = 1*clusters.labels_
