import numpy as np
from clustering import Donkey

from matplotlib.collections import LineCollection as lc
from mpl_toolkits.mplot3d.art3d import Line3DCollection as lc3d
import matplotlib.pyplot as plt
import matplotlib.image as pim
import matplotlib.colors as mcl
from matplotlib.gridspec import GridSpec

from scipy.interpolate import interp1d
from matplotlib.colors import colorConverter

def colored_line_segments(xs,ys,zs=None,color='k',mid_colors=False):
    if isinstance(color,str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(len(xs))])   
    segs = []
    seg_colors = []    
    lastColor = [color[0][0],color[0][1],color[0][2]]        
    start = [xs[0],ys[0]]
    end = [xs[0],ys[0]]        
    if not zs is None:
        start.append(zs[0])
        end.append(zs[0])     
    else:
        zs = [zs]*len(xs)            
    for x,y,z,c in zip(xs,ys,zs,color):
        if mid_colors:
            seg_colors.append([(chan+lastChan)*.5 for chan,lastChan in zip(c,lastColor)])        
        else:   
            seg_colors.append(c)        
        lastColor = c[:-1]           
        if not z is None:
            start = [end[0],end[1],end[2]]
            end = [x,y,z]
        else:
            start = [end[0],end[1]]
            end = [x,y]                 
        segs.append([start,end])               
    colors = [(*color,1) for color in seg_colors]    
    return segs, colors

def segmented_resample(xs,ys,zs=None,color='k',n_resample=100,mid_colors=False):    
    n_points = len(xs)
    if isinstance(color,str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(n_points)])   
    n_segs = (n_points-1)*(n_resample-1)        
    xsInterp = np.linspace(0,1,n_resample)    
    segs = []
    seg_colors = []
    hiResXs = [xs[0]]
    hiResYs = [ys[0]]    
    if not zs is None:
        hiResZs = [zs[0]]        
    RGB = color.swapaxes(0,1)
    for i in range(n_points-1):        
        fit_xHiRes = interp1d([0,1],xs[i:i+2])
        fit_yHiRes = interp1d([0,1],ys[i:i+2])        
        xHiRes = fit_xHiRes(xsInterp)
        yHiRes = fit_yHiRes(xsInterp)    
        hiResXs = hiResXs+list(xHiRes[1:])
        hiResYs = hiResYs+list(yHiRes[1:])   
        R_HiRes = interp1d([0,1],RGB[0][i:i+2])(xsInterp)        
        G_HiRes = interp1d([0,1],RGB[1][i:i+2])(xsInterp)      
        B_HiRes = interp1d([0,1],RGB[2][i:i+2])(xsInterp)                               
        lastColor = [R_HiRes[0],G_HiRes[0],B_HiRes[0]]                
        start = [xHiRes[0],yHiRes[0]]
        end = [xHiRes[0],yHiRes[0]]           
        if not zs is None:
            fit_zHiRes = interp1d([0,1],zs[i:i+2])             
            zHiRes = fit_zHiRes(xsInterp)             
            hiResZs = hiResZs+list(zHiRes[1:]) 
            start.append(zHiRes[0])
            end.append(zHiRes[0])                
        else:
            zHiRes = [zs]*len(xHiRes) 
            
        if mid_colors: seg_colors.append([R_HiRes[0],G_HiRes[0],B_HiRes[0]])        
        for x,y,z,r,g,b in zip(xHiRes[1:],yHiRes[1:],zHiRes[1:],R_HiRes[1:],G_HiRes[1:],B_HiRes[1:]):
            if mid_colors:
                seg_colors.append([(chan+lastChan)*.5 for chan,lastChan in zip((r,g,b),lastColor)])
            else:            
                seg_colors.append([r,g,b])            
            lastColor = [r,g,b]            
            if not z is None:
                start = [end[0],end[1],end[2]]
                end = [x,y,z]  
            else:
                start = [end[0],end[1]]
                end = [x,y]                
            segs.append([start,end])

    colors = [(*color,1) for color in seg_colors]    
    data = [hiResXs,hiResYs] 
    if not zs is None:
        data = [hiResXs,hiResYs,hiResZs] 
    return segs, colors, data      

def faded_segment_resample(xs,ys,zs=None,color='k',fade_len=20,n_resample=100,direction='Head'):      
    segs, colors, hiResData = segmented_resample(xs,ys,zs,color,n_resample)    
    n_segs = len(segs)   
    if fade_len>len(segs):
        fade_len=n_segs    
    if direction=='Head':
        #Head fade
        alphas = np.concatenate((np.zeros(n_segs-fade_len),np.linspace(0,1,fade_len)))
    else:        
        #Tail fade
        alphas = np.concatenate((np.linspace(1,0,fade_len),np.zeros(n_segs-fade_len)))
    colors = [(*color[:-1],alpha) for color,alpha in zip(colors,alphas)]
    return segs, colors, hiResData 

def main():
    default = np.array([
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
            "#984ea3",
            "#999999",
            "#e41a1c",
            "#dede00",
            "darkturquoise",
            "gray",
            "#000000"
    ])
    labels = np.zeros((17,639), dtype=int)
    ref = np.zeros(639, dtype=int)
    for i, t in enumerate(np.arange(17, dtype=int)*10):
        if 1:
            bonds = np.load("bonds.npy")[:,:,1:]
        labels_ = np.load(f"donkey/nb_{t}.lab.npy")
        Donkey.align_labels(labels_, ref)
        labels[i] = labels_
        ref = 1*labels_
    if 0:
        labs = [{
            8: [2],
            10: [4],
            16: [5],
        }, {
            8:  [0],
            10: [4],
            16: [5]
        }]
    else:
        labs = [{
            8:  [2],
            10: [4],
            16: [4],
        }]
    cols = [-2]
    save = f"/mnt/c/Users/karaj/Desktop/UoO/cl/paper/test/{labs[0][8][0]}{labs[0][10][0]}{labs[0][16][0]}.png"
    # save = f"/mnt/c/Users/karaj/Desktop/UoO/cl/paper/traj/all.png"
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    tot = 0
    pops = np.load("pops.npy")
    breakpoint()
    for col, lab in zip(cols, labs):
        trajs = np.argwhere(np.logical_and.reduce([np.logical_or.reduce([labels[key]==i for i in val]) for key, val in lab.items()]))
        # trajs = np.arange(labels.shape[1])[:,None]
        tot += trajs.shape[0]
        
        NPOINTS = bonds.shape[1]
        cmap = mcl.LinearSegmentedColormap.from_list("", ["black", default[col]])
        COLORS = np.array([cmap(i)[:-1] for i in np.linspace(0,1,NPOINTS)])

        # ax.set_xlabel("wing separation (Å)")
        # ax.set_ylabel("rhombicity (Å)")
        # ax.set_zlabel("wing length (Å)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim([1.2,3.2])
        ax.set_ylim([-0.9,0.9])
        ax.set_zlim([1.1,1.9])
        ax.view_init(azim=-40, elev=40, roll=0)
        for traj in trajs:
            traj = traj[0]
            segs, colors = colored_line_segments(bonds[traj,:,0],bonds[traj,:,1],bonds[traj,:,2],color=COLORS,mid_colors=False)
            ax.add_collection(lc3d(segs, colors=colors, alpha=0.25))
        ax.scatter(bonds[trajs,-1,0], bonds[trajs,-1,1], bonds[trajs,-1,2], edgecolor="black", facecolor=default[col], marker="o", zorder=10)
    plt.title(f"{tot} trajectories", size=24)
    # plt.tight_layout()
    # plt.subplots_adjust(left=0)
    plt.savefig(save, dpi=600, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    if 1:
        main()
        exit()
    else:
        fig, axs = plt.subplots(3,3, figsize=(5,5))

        plots = ["60", "70", "80", "90", "100", "110", "120", "130", "140"]
        for i in range(3):
            for j in range(3):
                idx = 3*i + j
                print(idx)
                img = pim.imread(f"/mnt/c/Users/karaj/Desktop/UoO/cl/paper/abs/nb_{plots[idx]}.png")
                axs[i,j].set_aspect("auto")
                axs[i,j].imshow(img)
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
                axs[i,j].text(100,100, chr(97+idx), va="top", ha="left")
                # print(axs[i,j].get_xlim())
        # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=1, bottom=0)
        plt.savefig(f"/mnt/c/Users/karaj/Desktop/UoO/cl/paper/nb_all.png", dpi=600, bbox_inches="tight")
