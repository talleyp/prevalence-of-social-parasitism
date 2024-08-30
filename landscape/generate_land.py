import numpy as np
import utils as m
from parameter import data
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl

np.random.seed()

pink = cm.get_cmap('YlGn')
newcolors = pink(np.linspace(0,1,256))
red = np.array([184/256, 0/256, 0/256, 1])
newcolors[-1, :] = red
newcmp = ListedColormap(newcolors)
mpl.rcParams["figure.autolayout"] = True
pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.
style_path = "paper.mplstyle"
plt.style.use("paper.mplstyle")
mpl.rcParams['text.usetex'] = True
jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
              "CQG": {"onecol": 374.*pt}, # CQG is only one column
              # Add more journals below. Can add more properties to each journal
             }

land_width = jour_sizes["PRD"]["onecol"]


def plot_v(z,data,filename='land.png'):
    if data['pre_p']:
        Zf = z[:data['M']*5].reshape((data['M'],5))
        F = Zf[:,-1]
    else:
        Zf = z[:data['M']*5].reshape((data['M'],5))
        Zp = z[data['M']*5:]
        F = np.hstack((Zf[:,-1],Zp[2]))

    wts = data['weights']*np.tile(F,(data['weights'].shape[0],1))
    grid = np.nanargmax(wts,axis=1)
    pre_wts = data['weights'][np.arange(grid.shape[0]),grid]
    pre_wts = pre_wts.reshape((2*data['landscape'],2*data['landscape'])).T
    grid = grid.reshape((2*data['landscape'],2*data['landscape'])).T
    unique, A = np.unique(grid[~np.isnan(grid)].flatten(), return_counts=True)
    fig,ax = plt.subplots(1,1,figsize=(land_width,land_width))
    ax.grid(False)
    offset = np.min((1,1.1*np.max(pre_wts)))-np.max(pre_wts)
    if data['pre_p']:
        # pc = ax.pcolormesh(grid,cmap=newcmp,vmin=0,vmax=data['M']-1,alpha=pre_wts+offset)
        pc = ax.pcolormesh(grid,cmap=newcmp,vmin=0,vmax=data['M']-1,alpha=0.8)
    else:
        # pc = ax.pcolormesh(grid,cmap=newcmp,vmin=0,vmax=data['M'],alpha=pre_wts+offset)
        pc = ax.pcolormesh(grid,cmap=newcmp,vmin=0,vmax=data['M'],alpha=0.8)
    X = np.arange(0,grid.shape[0])
    Y = np.arange(0,grid.shape[1])
    for cind in unique:
        # make mask for weights
        # extract weights
        # multiply weights by F for respective colony
        # plot contour
        grid_mask = np.zeros(grid.shape)
        grid_mask[grid==cind] = 1
        wts_c = wts[:,cind].reshape((2*data['landscape'],2*data['landscape'])).T
        Z = F[cind]*wts_c*grid_mask
        ax.contour(X,Y, Z, 4, colors='k',alpha=0.1)
    for i, ((x,y),) in enumerate(zip(data['D'])):
        if i==data['M']-int(data['pre_p']) :
            if data['pre_p']:
                plt.text(x,y,0, ha="center", va="center")
            else:
                plt.text(x,y,'P', ha="center", va="center")
        else:
            plt.text(x,y,i, ha="center", va="center")
    # plt.savefig(f'../figs/4-{filename}')



def run_sim(mdl,N,data,space_type='ring'):
    L = m.LAND()
    data['M'],data['D'] =L(space_type,data)
    data['pre_p']=True
    data['M'] = data['M']+1
    z0 = m.start_values(data['M'])
    data['weights'] = mdl(data).make_grid()
    plot_v(z0,data,'layout.pdf')
    np.savetxt(f'../data/landscape/lands/D-{N}.out',data['D'])
    plt.show()
    

data['M'] = 45
data['radius'] = 70 
data['alp'] = 0
mdl = m.ANT
N = 365*100
for i in range(N):
    run_sim(mdl,i,data.copy(),space_type='poisson')

