import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
from matplotlib.image import imread
import pyqofta as p

plot_colours = [
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
        "#ffe4c4"]

def activate_tex():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Serif"]})

def figure1():

    points = np.array(
        [[0,0],
        [3,0],
        [4,0]]
    )

    lspace = np.linspace(-4,8,121)
    activate_tex()

    axs = plt.figure(figsize=(2*4.3/0.98, 2*3/0.98)).subplot_mosaic(
        """
        0
        1
        2
        3
        """,
        gridspec_kw = {
            "height_ratios": [1, 1, 1, 1]
        }
    )
    for i in range(4):
        axs[i] = axs.pop(str(i))
    
    plt.subplots_adjust(
        left=0.01,
        bottom=0.01, 
        right=0.99, 
        top=0.99, 
        wspace=0.1, 
        hspace=0.1)

    for ax in list(axs.values()):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-4,8])
        ax.set_ylim([-0.5,2.5])
        ax.set_aspect(1)
    
    gs = np.array(
        [np.exp(-(lspace-points[i,0])**2) for i in range(3)]
    )
    gsum = np.sum(gs, axis=0)

    #subplot1
    axs[0].text(-4+0.2, 2.5-0.2, "$(a)$", ha="left", va="top")
    for i in range(3):
        axs[0].plot(lspace, gs[i], c=plot_colours[i], zorder=0)
    axs[0].scatter(points[:,0], points[:,1], c=plot_colours[:3], zorder=1)

    #subplot2
    axs[1].text(-4+0.2, 2.5-0.2, "$(b)$", ha="left", va="top")
    for i in range(3):
        axs[1].plot(lspace, np.exp(-(lspace-points[i,0])**2), "--", c=plot_colours[i], zorder=0)
    axs[1].plot(lspace, gsum, c="k", zorder=1)
    axs[1].scatter(points[:,0], points[:,1], c=plot_colours[:3], zorder=1)

    #subplot3
    axs[2].text(-4+0.2, 2.5-0.2, "$(c)$", ha="left", va="top")
    axs[2].plot(lspace, gsum, c="k", zorder=1)
    axs[2].scatter(points[:,0], points[:,1], c=plot_colours[:3], zorder=1)
    axs[2].text(0, gsum[40]+0.5, "$C_1$", ha="center", va="center")
    axs[2].text(3.5, gsum[75]+0.5, "$C_2$", ha="center", va="center")

    #subplot4
    axs[3].text(-4+0.2, 2.5-0.2, "$(d)$", ha="left", va="top")
    axs[3].plot(lspace, gsum, c="k", zorder=1)
    axs[3].scatter(points[:,0], points[:,1], c=plot_colours[:3], zorder=1)
    axs[3].arrow(points[0,0], points[0,1], 0, gsum[40]-0.3, color="gray", head_width=0.1, zorder=0)
    axs[3].arrow(points[1,0], points[1,1], 0.3, gsum[75]-0.3, color="gray", head_width=0.1, zorder=0)
    axs[3].arrow(points[2,0], points[2,1], -0.3, gsum[75]-0.3, color="gray", head_width=0.1, zorder=0)
    axs[3].text(0, gsum[40]+0.5, "$C_1$", ha="center", va="center")
    axs[3].text(3.5, gsum[75]+0.5, "$C_2$", ha="center", va="center")

    plt.savefig("report/images/method_fig1.png", bbox_inches="tight", dpi=1200)
    plt.show()

def figure2():
    activate_tex()
    
    axs = plt.figure(figsize=(2*2.1/0.98, 2*2.1/0.98)).subplot_mosaic(
        """
        ab
        cd
        """,
        gridspec_kw = {
            "height_ratios": [1, 1]
        }
    )
    for i in range(4):
        axs[i] = axs.pop(str(i))
    
    plt.subplots_adjust(
        left=0.01,
        bottom=0.01, 
        right=0.99, 
        top=0.99, 
        wspace=0.1, 
        hspace=0.1)

    for ax in list(axs.values()):
        ax.set_xticks([])
        ax.set_yticks([])
    
    X, Y = np.linspace(-0.5,1,301), np.linspace(-1,0.5,301)
    px, py = np.meshgrid(np.linspace(0,300,10, endpoint=False)+15, np.linspace(0,300,10, endpoint=False)+15)
    Z = np.zeros((301,301))
    for ix, x in enumerate(X):
        for iy, y in enumerate(Y):
            Z[ix, iy] = np.exp(-x**2-y**2)

    #subplot00
    axs[0].scatter(px, py, c="gray", marker=".", s=1, zorder=1)
    axs[0].imshow(Z, cmap="binary", zorder=0, origin="lower")
    axs[0].text(9, 300-9, "$(a)$", ha="left", va="top", zorder=2, backgroundcolor="white")
    axs[0].scatter(52, 155, marker=".", c="magenta", s=30, zorder=2)
    axs[0].scatter(45, 165, marker="+", c="red", s=30, zorder=2)

    #subplot01
    axs[1].scatter(px, py, c="gray", marker=".", s=1, zorder=1)
    axs[1].imshow(Z, cmap="binary", zorder=0, origin="lower")
    axs[1].text(0+9, 300-9, "$(b)$", ha="left", va="top", zorder=2, backgroundcolor="white")
    axs[1].scatter(45, 165, marker="+", c="red", s=30, zorder=2)
    axs[1].scatter(15, 165, marker="+", c="orange", s=30, zorder=2)
    axs[1].scatter(75, 165, marker="+", c="orange", s=30, zorder=2)
    axs[1].scatter(45, 135, marker="+", c="orange", s=30, zorder=2)
    axs[1].scatter(45, 195, marker="+", c="orange", s=30, zorder=2)

    #subplot10
    axs[2].scatter(px, py, c="gray", marker=".", s=1, zorder=1)
    axs[2].imshow(Z, cmap="binary", zorder=0, origin="lower")
    axs[2].text(0+9, 300-9, "$(c)$", ha="left", va="top", zorder=2, backgroundcolor="white")
    axs[2].scatter(45, 165, marker="+", c="red", s=30, zorder=2)
    axs[2].arrow(45, 165, 20, 0, color="cyan", head_width=5, zorder=1)
    axs[2].scatter(75, 165, marker="+", c="red", s=30, zorder=2)
    axs[2].scatter(105, 165, marker="+", c="orange", s=30, zorder=2)
    axs[2].scatter(75, 135, marker="+", c="orange", s=30, zorder=2)
    axs[2].scatter(75, 195, marker="+", c="orange", s=30, zorder=2)
    
    #subplot11
    axs[3].scatter(px, py, c="gray", marker=".", s=1, zorder=1)
    axs[3].imshow(Z, cmap="binary", zorder=0, origin="lower")
    axs[3].text(0+9, 300-9, "$(d)$", ha="left", va="top", zorder=2, backgroundcolor="white")
    axs[3].scatter(45, 165, marker="+", c="red", s=30, zorder=2)
    axs[3].arrow(45, 165, 20, 0, color="cyan", head_width=5, zorder=1)
    axs[3].scatter(75, 165, marker="+", c="red", s=30, zorder=2)
    axs[3].arrow(75, 165, 20, 0, color="cyan", head_width=5, zorder=1)
    axs[3].scatter(105, 165, marker="+", c="red", s=30, zorder=2)
    axs[3].arrow(105, 165, 20, 0, color="cyan", head_width=5, zorder=1)
    axs[3].scatter(135, 165, marker="+", c="red", s=30, zorder=2)
    axs[3].arrow(135, 165, 20, 0, color="cyan", head_width=5, zorder=1)
    axs[3].scatter(165, 165, marker="+", c="red", s=30, zorder=2)
    axs[3].arrow(165, 165, 0, -20, color="cyan", head_width=5, zorder=1)
    axs[3].scatter(165, 135, marker="+", c="red", s=30, zorder=2)
    axs[3].arrow(165, 135, 20, 0, color="cyan", head_width=5, zorder=1)
    axs[3].scatter(195, 135, marker="+", c="red", s=30, zorder=2)
    axs[3].arrow(195, 135, 0, -20, color="cyan", head_width=5, zorder=1)
    axs[3].scatter(195, 105, marker="+", c="red", s=30, zorder=2)

    plt.savefig("report/images/method_fig2.png", bbox_inches="tight", dpi=1200)
    plt.show()

def figure3():
    import clustering as cl

    points = np.array(
        [[0,0],
        [3,0],
        [4.6,0]]
    )

    lspace = np.linspace(-4,8,121)

    activate_tex()
    axs = plt.figure(figsize=(2*2.1/0.98, 2*(2.1*2.5/12+2.2)/0.98)).subplot_mosaic(
        """
        aa
        bc
        de
        """,
        gridspec_kw = {
            "height_ratios": [2.1*2.5/12, 1, 1]
        }
    )
    for i in range(5):
        axs[i] = axs.pop(str(i))
    
    plt.subplots_adjust(
        left=0.01,
        bottom=0.01, 
        right=0.99, 
        top=0.99, 
        wspace=0.5/6.1, 
        hspace=0.5/(4+2.1*2.5/12))

    for ax in list(axs.values()):
        ax.set_xticks([])
        ax.set_yticks([])

    gs = np.array(
        [np.exp(-(lspace-points[i,0])**2) for i in range(3)]
    )
    gsum = np.sum(gs, axis=0)

    #subplot0
    axs[0].set_xlim([-4,8])
    axs[0].set_ylim([-0.5, 2])
    axs[0].set_aspect(1)
    axs[0].text(
        (axs[0].get_xlim()[1] - axs[0].get_xlim()[0])*0.03/2.1+axs[0].get_xlim()[0],
        -(axs[0].get_ylim()[1] - axs[0].get_ylim()[0])*0.03/2.1/2.5*12+axs[0].get_ylim()[1], 
        "$(a)$", ha="left", va="top")
    for i in range(3):
        axs[0].plot(lspace, gs[i], "--", c=plot_colours[i], zorder=0)
    axs[0].plot(lspace, gsum, c="k", zorder=1)
    axs[0].scatter(points[:,0], points[:,1], c=plot_colours[:3], zorder=1)
    axs[0].text(0, gsum[40]+0.5, "$C_1$", ha="center", va="center")
    axs[0].text(3.8, gsum[75]+0.5, "$C_2$", ha="center", va="center")

    #subplot10
    n_samples = 500
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    clus_obj = cl.Clustering(X_aniso, np.array([0,1]), 1)
    clus_obj.normalise_data()
    clus_obj.get_feature_vars()
    clus_obj.get_clusters()
    clus_obj.positions_to_labels()
    print(clus_obj.clusters)

    colours = ["#7f7f7f" if clus_obj.clusters[i] < 0 \
        else plot_colours[clus_obj.clusters[i]] for i in range(clus_obj.n_traj)]

    axs[1].set_aspect(1)
    axs[1].invert_yaxis()
    axs[1].scatter(
        clus_obj.data[:,clus_obj.use_features[0]],
        clus_obj.data[:,clus_obj.use_features[1]],
        c=colours,
        s=1)
    clus_obj.cluster_maxima = np.unique(clus_obj.cluster_positions, axis=0)
    axs[1].scatter(
        clus_obj.cluster_maxima[:,0],
        clus_obj.cluster_maxima[:,1],
        marker="+", c="k"
    )
    
    axs[1].text(
        (axs[1].get_xlim()[1] - axs[1].get_xlim()[0])*0.03+axs[1].get_xlim()[0],
        -(axs[1].get_ylim()[1] - axs[1].get_ylim()[0])*0.03+axs[1].get_ylim()[1], 
        "$(b)$", ha="left", va="top")

    #subplot11
    clus_obj.relabel_by_contributions()
    colours = ["#7f7f7f" if clus_obj.clusters[i] < 0 \
        else plot_colours[clus_obj.clusters[i]] for i in range(clus_obj.n_traj)]

    axs[2].set_aspect(1)
    axs[2].invert_yaxis()
    axs[2].scatter(
        clus_obj.data[:,clus_obj.use_features[0]],
        clus_obj.data[:,clus_obj.use_features[1]],
        c=colours,
        s=1)
    clus_obj.cluster_maxima = np.unique(clus_obj.cluster_positions, axis=0)
    
    axs[2].text(
        (axs[2].get_xlim()[1] - axs[2].get_xlim()[0])*0.03+axs[2].get_xlim()[0],
        -(axs[2].get_ylim()[1] - axs[2].get_ylim()[0])*0.03+axs[2].get_ylim()[1], 
        "$(c)$", ha="left", va="top")

    #subplot20
    p, q = 1, 0
    clus_obj.border_points = []
    cluster_avg1 = np.mean(
            clus_obj.abs_contributions[clus_obj.clusters==p][:,p]+clus_obj.abs_contributions[clus_obj.clusters==p][:,q])
    cluster_avg2 = np.mean(
        clus_obj.abs_contributions[clus_obj.clusters==q][:,p]+clus_obj.abs_contributions[clus_obj.clusters==q][:,q])
    cluster_avg = min(cluster_avg1, cluster_avg2)
    for i in range(clus_obj.n_traj):
        if clus_obj.abs_contributions[i,p]+clus_obj.abs_contributions[i,q] < cluster_avg:
            clus_obj.border_points.append(i)

    colours = ["#7f7f7f" if clus_obj.clusters[i] < 0 or i in clus_obj.border_points\
        else plot_colours[clus_obj.clusters[i]] for i in range(clus_obj.n_traj)]

    axs[3].set_aspect(1)
    axs[3].invert_yaxis()
    axs[3].scatter(
        clus_obj.data[:,clus_obj.use_features[0]],
        clus_obj.data[:,clus_obj.use_features[1]],
        c=colours,
        s=1)
    clus_obj.cluster_maxima = np.unique(clus_obj.cluster_positions, axis=0)
    
    axs[3].text(
        (axs[3].get_xlim()[1] - axs[3].get_xlim()[0])*0.03+axs[3].get_xlim()[0],
        -(axs[3].get_ylim()[1] - axs[3].get_ylim()[0])*0.03+axs[3].get_ylim()[1], 
        "$(d)$", ha="left", va="top")

    #subplot21
    clus_obj.merge_clusters()
    colours = ["#7f7f7f" if clus_obj.clusters[i] < 0 \
        else plot_colours[clus_obj.clusters[i]] for i in range(clus_obj.n_traj)]

    axs[4].set_aspect(1)
    axs[4].invert_yaxis()
    axs[4].scatter(
        clus_obj.data[:,clus_obj.use_features[0]],
        clus_obj.data[:,clus_obj.use_features[1]],
        c=colours,
        s=1)
    
    axs[4].text(
        (axs[4].get_xlim()[1] - axs[4].get_xlim()[0])*0.03+axs[4].get_xlim()[0],
        -(axs[4].get_ylim()[1] - axs[4].get_ylim()[0])*0.03+axs[4].get_ylim()[1], 
        "$(e)$", ha="left", va="top")

    plt.savefig("report/images/method_fig3.png", bbox_inches="tight", dpi=1200)
    plt.show()

def figure4():
    activate_tex()
    
    fig, axs = plt.subplots(2,2)
    plt.subplots_adjust(
        left=0.01,
        bottom=0.01, 
        right=0.99, 
        top=0.99, 
        wspace=0.1, 
        hspace=-0.2)

    #subplot00
    img = imread("./report/images/chd.png")
    axs[0,0].imshow(img, origin="upper")
    

    #subplot01
    img = imread("./report/images/cec_ht.png")
    axs[0,1].imshow(img, origin="upper")

    #subplot10
    img = imread("./report/images/czc_ht.png")
    axs[1,0].imshow(img, origin="upper")

    #subplot11
    img = imread("./report/images/tzt_ht.png")
    axs[1,1].imshow(img, origin="upper")


    xlim, ylim = 0, 0
    for i in range(4):
        if axs[i//2,i%2].get_xlim()[1] > xlim:
            xlim = axs[i//2,i%2].get_xlim()[1]
        if axs[i//2,i%2].get_ylim()[0] > ylim:
            ylim = axs[i//2,i%2].get_ylim()[0]        

    print(xlim, ylim)

    abcd = ["a", "b", "c", "d"]
    for i in range(4):
        axs[i//2,i%2].set_xticks([])
        axs[i//2,i%2].set_yticks([])
        diffx, diffy = xlim - axs[i//2,i%2].get_xlim()[1], ylim - axs[i//2,i%2].get_ylim()[0]
        axs[i//2,i%2].set_xlim([-diffx/2, xlim-diffx/2])
        axs[i//2,i%2].set_ylim([ylim-diffy/2, -diffy/2])
        axs[i//2,i%2].text(-diffx/2+10, -diffy/2+10, "$({})$".format(abcd[i]), ha="left", va="top")

    plt.show()

def swap_labels(labels, l1, l2):
    m = np.amax(labels) + 1
    labels[labels==l1] = m
    labels[labels==l2] = l1
    labels[labels==m] = l2

def figure5():
    import clustering as cl

    activate_tex()
    fig = plt.figure(figsize=(2*2.1/0.98, 2*4.3/0.98))
    axs = fig.subplot_mosaic(
        """
        01
        23
        45
        67
        """,
        gridspec_kw = {
            "height_ratios": [1, 1, 1, 1]
        }
    )
    for i in range(8):
        axs[i] = axs.pop(str(i))
    
    plt.subplots_adjust(
        left=0.01,
        bottom=0.01, 
        right=0.99, 
        top=0.99, 
        wspace=0.1, 
        hspace=0.1)

    for ax in list(axs.values()):
        ax.set_xticks([])
        ax.set_yticks([])  
    
    #load ensemble and reference structures
    fpaths = ["R_traj0" +"0"*(i<10) + str(i) + ".xyz" for i in range(100)]
    ensemble = p.Ensemble.load_ensemble(fpaths, "sh")

    #define some useful variables and create empty data arrays
    Ntraj, Nts, Nat, Nfeatures = ensemble.ntrajs, ensemble.nts_max, ensemble.trajs[0].geometries[0].natoms, 18
    data = np.zeros((Nts, Ntraj, Nfeatures))
    cl.populate_data(data, ensemble, Nts)

    for j in range(Nfeatures):
        data[:,:,j] -= np.amin(data[:,:,j])
        data[:,:,j] /= np.amax(data[:,:,j])

    for j, t in enumerate([0, 15, 25, 35, 45, 55, 65, 75]):
        clus_obj = cl.Clustering(data[t], np.array([13, 14]), 1)
        clus_obj.get_feature_vars()
        clus_obj.get_clusters()
        clus_obj.positions_to_labels()
        clus_obj.relabel_by_contributions()
        clus_obj.merge_clusters()

        print(clus_obj.clusters)

        if j==5:
            swap_labels(clus_obj.clusters, 0, 1)
        elif j==6:
            swap_labels(clus_obj.clusters, 0, 2)
            swap_labels(clus_obj.clusters, 2, 3)

        colours = ["#7f7f7f" if clus_obj.clusters[i] < 0 \
            else plot_colours[clus_obj.clusters[i]] for i in range(clus_obj.n_traj)]

        axs[j].set_aspect(1)
        axs[j].scatter(
            clus_obj.data[:,clus_obj.use_features[0]]/clus_obj.n_pts,
            clus_obj.data[:,clus_obj.use_features[1]]/clus_obj.n_pts,
            c=colours,
            s=1)
        axs[j].set_xlim([-0.1, 1.1])
        axs[j].set_ylim([-0.1, 1.1])

        axs[j].text(
            (axs[0].get_xlim()[1] - axs[0].get_xlim()[0])*0.03+axs[0].get_xlim()[0],
            -(axs[0].get_ylim()[1] - axs[0].get_ylim()[0])*0.03+axs[0].get_ylim()[1], 
            "$({})$".format(list(string.ascii_lowercase)[j]), ha="left", va="top")

    plt.savefig("report/images/method_fig5.png", bbox_inches="tight", dpi=1200)
    breakpoint()
    plt.show()

def figure6():
    import clustering as cl

    activate_tex()
    fig = plt.figure(figsize=(2*2.1/0.98, 2*4.3/0.98))
    plt.subplots_adjust(
        left=0.01,
        bottom=0.01, 
        right=0.99, 
        top=0.99, 
        wspace=0.1, 
        hspace=0.1)
    
    #load ensemble and reference structures
    fpaths = ["R_traj0" +"0"*(i<10) + str(i) + ".xyz" for i in range(100)]
    ensemble = p.Ensemble.load_ensemble(fpaths, "sh")

    #define some useful variables and create empty data arrays
    Ntraj, Nts, Nat, Nfeatures = ensemble.ntrajs, ensemble.nts_max, ensemble.trajs[0].geometries[0].natoms, 18
    data = np.zeros((Nts, Ntraj, Nfeatures))
    cl.populate_data(data, ensemble, Nts)

    for j in range(Nfeatures):
        data[:,:,j] -= np.amin(data[:,:,j])
        data[:,:,j] /= np.amax(data[:,:,j])

    for j, t in enumerate([0, 15, 25, 35, 45, 55, 65, 75]):
        clus_obj = cl.Clustering(data[t], np.array([5, 13, 14]), 1)
        clus_obj.get_feature_vars()
        clus_obj.get_clusters()
        clus_obj.positions_to_labels()
        clus_obj.relabel_by_contributions()
        clus_obj.merge_clusters()
        
        if j>=4:
            swap_labels(clus_obj.clusters, 0, 1)

        if j>=6:
            clus_obj.clusters[clus_obj.clusters == 3] += 1

        colours = ["#7f7f7f" if clus_obj.clusters[i] < 0 \
            else plot_colours[clus_obj.clusters[i]] for i in range(clus_obj.n_traj)]

        ax = fig.add_subplot(4, 2, j+1, projection="3d")

        #ax.set_aspect(1)
        ax.scatter(
            clus_obj.data[:,clus_obj.use_features[0]]/clus_obj.n_pts,
            clus_obj.data[:,clus_obj.use_features[1]]/clus_obj.n_pts,
            clus_obj.data[:,clus_obj.use_features[2]]/clus_obj.n_pts,
            c=colours,
            s=1)
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_zlim([-0.1, 1.1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.azim += 180

        ax.text(
            -(ax.get_xlim()[1] - ax.get_xlim()[0])*0.03+ax.get_xlim()[1], 
            -(ax.get_ylim()[1] - ax.get_ylim()[0])*0.03+ax.get_ylim()[1],
            -(ax.get_zlim()[1] - ax.get_zlim()[0])*0.03+ax.get_zlim()[1], 
            "$({})$".format(t*2))
        del clus_obj

    plt.savefig("report/images/method_fig6.png", bbox_inches="tight", dpi=1200)
    breakpoint()
    plt.show()

if __name__ == "__main__":
    breakpoint()