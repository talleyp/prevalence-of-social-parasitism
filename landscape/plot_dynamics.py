import numpy as np
import utils as m
from parameter import data
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl

pink = cm.get_cmap('YlGn')
newcolors = pink(np.linspace(0,1,256))
red = np.array([184/256, 0/256, 0/256, 1])
newcolors[-1, :] = red
newcmp = ListedColormap(newcolors)

pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.
style_path = "paper.mplstyle"
plt.style.use("paper.mplstyle")
mpl.rcParams['text.usetex'] = True
jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
              "CQG": {"onecol": 374.*pt}, # CQG is only one column
              # Add more journals below. Can add more properties to each journal
             }

land_width = jour_sizes["PRD"]["onecol"]
dyn_width = jour_sizes["PRD"]["twocol"]
ratio = 2#(1 + 5 ** 0.5) / 2

def plot_v(z,data,T_dead,filename='land.png'):
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
        pc = ax.pcolormesh(grid,cmap=newcmp,vmin=0,vmax=data['M']-1,alpha=0.6)
    else:
        # pc = ax.pcolormesh(grid,cmap=newcmp,vmin=0,vmax=data['M'],alpha=pre_wts+offset)
        pc = ax.pcolormesh(grid,cmap=newcmp,vmin=0,vmax=data['M'],alpha=0.6)
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
        Z = F[cind]*wts_c*grid_mask*(1-int(cind in T_dead))
        ax.contour(X,Y, Z, 4, colors='k',alpha=0.1)
    for i, ((x,y),) in enumerate(zip(data['D'])):
        if i==data['M']-int(data['pre_p']) :
            if data['pre_p']:
                plt.text(x,y,0, ha="center", va="center")
            else:
                plt.text(x,y,'P', ha="center", va="center")
        else:
            if i in T_dead:
                plt.text(x,y,i, ha="center", va="center",alpha=0.4)
            else:
                plt.text(x,y,i, ha="center", va="center")
    
    plt.savefig(f'../figs/{filename}')



k = 5
Ms = [47,47,46,45,45]

data['M'] = Ms[k-1]+1
data['D'] = np.loadtxt('data/lands/D-1.out')
data['weights'] = m.ANT(data).make_grid()
data['pre_p'] = False

Z = np.loadtxt(f"data/dynamics{k}/Z.txt")
T = np.loadtxt(f"data/dynamics{k}/T.txt")
M = Ms[k-1]
data['M'] = M
C = m.get_colony_size(Z,M,False)

T_dead = np.zeros(M)

for i in range(M):
    T_dead[i] = np.argmax(C[i,:]<100)
    

fig,ax = plt.subplots(1,1,figsize=(dyn_width,dyn_width/2))

ax.plot(T/365,Z[-2,:],label='Young Polyergus',linewidth=3)
ax.plot(T/365,Z[-1,:],label='Raiders',linewidth=3)
ax.plot(T/365,np.sum(Z[[-6,-7],:],axis=0),label='Slaves',linewidth=3)
first_ind = True
C_dead = []
for ind in range(M):
    Tind = T_dead[ind]
    if Tind>0:
        C_dead.append(ind)
        if first_ind:
            ax.axvline(x=T[int(Tind)]/365,c='red',ls='--',label='Host colony collapse')
            ax.text(T[int(Tind)]/365,2500,ind,ha="center", va="center",c='black')
            first_ind=False
        else:
            ax.axvline(x=T[int(Tind)]/365,c='red',ls='--')
            if ind in [43,36]:
                ax.text(T[int(Tind)]/365,3000+500,ind,ha="center", va="center",c='black')
            else:
                ax.text(T[int(Tind)]/365,3000,ind,ha="center", va="center",c='black')
ax.set_xlabel('Years')
ax.set_ylabel('Number of ants')
ax.legend(loc=1, bbox_to_anchor=[1, 0.5],prop={'size': 6})
plt.tight_layout()
plt.savefig(f'../figs/land-dynamics-{k}.pdf')

plot_v(Z[:,-1],data,C_dead,'5-end-dead.pdf')