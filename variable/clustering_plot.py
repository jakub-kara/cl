import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as skc
import pyqofta as p
import clustering as cl

fpaths = ["R_traj0" +"0"*(i<10) + str(i) + ".xyz" for i in range(100)]
ensemble = p.Ensemble(fpaths, "sh")
Ntraj = ensemble.ntrajs
Nts = ensemble.nts_max
Nat = ensemble.trajs[0].geometries[0].natoms
colours = ["#FF420E", "#579D1C", "#0084D1", "#7E0021", "#314004", "#83CAFF", "#AECF00", "#FF950E", "#004586"]
d16 = np.zeros((Nts, Ntraj))
for t in range(Nts):
    for i, traj in enumerate(ensemble):
        d16[t, i] = p.Molecule.bond_length(traj.geometries[t], [0,5])


def molecule_to_xyz(molecule, time):
    outstr = str(molecule.natoms) + "\n" + str(time) + "\n"
    for i, at in enumerate(molecule.coordinates):
         outstr += molecule.atom_labels[i] + " "
         outstr += str(at[0]) + " " + str(at[1]) + " " + str(at[2]) + "\n"
    return outstr

if __name__ == "__main__":
    clus_limits = [1,16]
    psfs = np.zeros(clus_limits[1]-clus_limits[0])
    dbis = np.zeros(clus_limits[1]-clus_limits[0])
    titles = ["Single", "Complete", "Average", "Ward"]
    fig, ax = plt.subplots(3, 4, sharex=True, sharey="row")
    ax[0,0].set_xlim(clus_limits)
    fig2, ax2 = plt.subplots(3, 4, sharex=True, sharey=True)
    for i in range (4):
        clusType = titles[i].lower()
        for Nclus in range(clus_limits[0], clus_limits[1]):
            clusters = skc.AgglomerativeClustering(n_clusters=Nclus, affinity="euclidean", linkage=clusType).fit(d16[-1,:].reshape(-1,1))
            #clusters = skc.KMeans(n_clusters=Nclus).fit(d16[-1,:].reshape(-1,1))
            positions = np.array([[i, 0] for i in d16[-1,:]])
            psfs[Nclus-clus_limits[0]] = cl.psf(clusters.labels_, positions)
            dbis[Nclus-clus_limits[0]] = cl.dbi(clusters.labels_, positions)
            if Nclus == clus_limits[0]:
                tsis = cl.tsi(clusters.children_, positions)
            if Nclus in [4,5,6]:
                bins = np.bincount(clusters.labels_)
                endpointdtype = [("d16", float), ("label", int)]
                endpoints = np.zeros((Nclus), dtype = endpointdtype)
                for lab in np.unique(clusters.labels_):
                    with open("avg_trajectories/avg_traj_" + titles[i].lower() + \
                        "_" + str(Nclus) + "_" + str(lab) + ".xyz", "w") as outfile:
                        ensemble.weights = np.where(clusters.labels_ == lab, 1/bins[lab], 0)
                        averaged_ensemble = np.zeros((ensemble.trajs[0].geometries[0].natoms, 3, ensemble.nts_max))
                        for ind, traj in enumerate(ensemble):
                            for ts, molc in enumerate(traj):
                                averaged_ensemble[:, :, ts] += molc.coordinates*ensemble.weights[ind]
                        avg_traj = []
                        for ts in range(ensemble.nts_max):
                            avg_traj.append(p.Molecule(ensemble.trajs[0].geometries[0].atom_labels, averaged_ensemble[:, :, ts]))
                        for ind, molc in enumerate(avg_traj):
                            outfile.write(molecule_to_xyz(molc, ensemble.trajs[0].time[ind]))
                    endpoints[lab] = (p.Molecule.bond_length(avg_traj[-1], [0,5]),  lab)
                for traj, col in enumerate(clusters.labels_):
                    #ax2[Nclus-4,i].plot(ensemble.trajs[0].time, d16[:,traj], colours[np.sort(endpoints, order="d16")[col][1]], \
                    #    linewidth="0.5")    
                    ax2[Nclus-4,i].plot(ensemble.trajs[0].time, d16[:,traj], colours[col], linewidth="0.5")    
        ax[0,i].set_title(titles[i])
        ax[0,i].plot(np.arange(clus_limits[0], clus_limits[1]), psfs, "-+",)
        ax[1,i].plot(np.arange(clus_limits[0], clus_limits[1]), dbis, "-+")
        ax[2,i].plot(np.arange(clus_limits[0], clus_limits[1]), np.concatenate(([None], tsis[clus_limits[0]-1:clus_limits[1]-2])), "-+")
        fig.text(0.5, 0.04, "Number of clusters", ha="center")
        ax2[0,i].set_title(titles[i])
        fig2.text(0.5, 0.04, "Time [fs]", ha="center")
        fig2.text(0.04, 0.5, "C1-C6 distance [Å]", va="center", rotation=90)
    ax[0,0].set_ylabel("psF")
    ax[1,0].set_ylabel("DBI")
    ax[2,0].set_ylabel("TSI")
    ax2[0,3].set_ylabel("4 clusters")
    ax2[0,3].yaxis.set_label_position("right")
    ax2[1,3].set_ylabel("5 clusters")
    ax2[1,3].yaxis.set_label_position("right")
    ax2[2,3].set_ylabel("6 clusters")
    ax2[2,3].yaxis.set_label_position("right")
    fig.savefig("indices2.png", dpi=300)
    fig2.savefig("trajectories2.png", dpi=300)
    plt.show()