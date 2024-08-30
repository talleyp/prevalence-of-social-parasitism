import numpy as np
import utils as m
from parameter import data
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl

np.random.seed()
# np.random.seed(0)


pink = cm.get_cmap('YlGn')
newcolors = pink(np.linspace(0,1,256))
red = np.array([184/256, 0/256, 0/256, 1])
newcolors[-1, :] = red
newcmp = ListedColormap(newcolors)
# mpl.rcParams["figure.autolayout"] = True
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
    plt.savefig(f'../figs/5-{filename}')



def run_sim(mdl,N,data,space_type='ring'):
    L = m.LAND()
    # data['M'],data['D'] =L(space_type,data)
    data['D'] = np.loadtxt('../data/landscape/lands/D-1.out')
    data['pre_p']=True
    data['M'] = data['M']+1
    z0 = m.start_values(data['M'])
    data['weights'] = mdl(data).make_grid()
    plot_v(z0,data,'layout.pdf')
    
    tpre,zpre = m.run_once(mdl(data),z0,t_end=365*5,method='LSODA')

    plot_v(z0,data,'before-poly.pdf')
    
    data['pre_p']=False
    data['M'] = data['M']-1

    z0 = m.poly_start(zpre[:,-1],data['M'])
    Z = np.zeros((z0.shape[0],N+2))
    T = np.zeros((N+2,1))
    Z[:,0] = z0

    z = np.array(z0+ mdl(data).full_de(z0)).reshape(data['M']*5+8,1)
    z = np.hstack((z0.reshape(data['M']*5+8,1),z))
    z[z<0]=0.001
    M = mdl(data)

    Z[:,1] = z[:,-1]
    T[1] = 1
    rfin=2*N
    str_out = ''
    for r in range(2,N+2):
        z_tmp = np.array(Z[:,r-1] +M.full_de(Z[:,r-1])).reshape(data['M']*5+8,1)
        z_tmp[z_tmp<0]=0.001
        T[r] = T[r-1]+1
        Z[:,r] = z_tmp.T
        Z[:,r],cind = M.discrete_raid(data,Z[:,r])
        rules = [
            rfin>=N+2,
            r>=365*10,
            np.max(np.abs(np.sum(Z[:,r-360:],axis=0)-np.sum(Z[:,r])))<200
            ]
        if np.sum(Z[-8:,r])<8 and rfin>=N+2:
            rfin = r
            str_out = 'lo'
        elif all(rules):
            rfin = r
            str_out = 'hi'
        if r>=rfin:
            T = T[:r+1]
            Z = Z[:,:r+1]
            break
    print(data['M'],str_out)
    return Z,T,data

data['M'] = 45
data['radius'] = 70 
data['alp'] = 0
mdl = m.ANT
N = 365*100
# Z,T,data = run_sim(mdl,N,data.copy(),space_type='uni-ring')
Z,T,data = run_sim(mdl,N,data.copy(),space_type='poisson')
np.savetxt('../data/landscape/dynamics5/Z.txt',Z)
np.savetxt('../data/landscape/dynamics5/T.txt',T)
z = Z[:,-1]

plot_v(z,data,'end.pdf')
