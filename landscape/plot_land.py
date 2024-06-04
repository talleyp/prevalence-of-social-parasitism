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
mpl.rcParams["figure.autolayout"] = True


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
    fig,ax = plt.subplots(1,1)
    offset = np.min((1,1.1*np.max(pre_wts)))-np.max(pre_wts)
    if data['pre_p']:
        pc = ax.pcolormesh(grid,cmap=pink,vmin=0,vmax=data['M'],alpha=pre_wts+offset)
    else:
        pc = ax.pcolormesh(grid,cmap=newcmp,vmin=0,vmax=data['M'],alpha=pre_wts+offset)
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
    # fig.colorbar(pc)
    for i, ((x,y),) in enumerate(zip(data['D'])):
        if i==data['M']-int(data['pre_p']) :
            plt.text(x,y,'P', ha="center", va="center")
        else:
            plt.text(x,y,i, ha="center", va="center")
    plt.savefig(filename)



def run_sim(mdl,N,data,space_type='ring'):
    L = m.LAND()
    data['M'],data['D'] =L(space_type,data)
    data['pre_p']=True
    data['M'] = data['M']+1
    z0 = m.start_values(data['M'])
    z0 = 0.5*z0
    data['weights'] = mdl(data).make_grid()
    plot_v(z0,data,'layout.png')

    tpre,zpre = m.run_once(mdl(data),z0,t_end=365*60)

    Zpre = m.get_colony_size(zpre,data['M']-1,data['pre_p'])

    plot_v(z0,data,'before-poly.png')
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
    z[:,-1],cind = M.discrete_raid(data,z[:,-1])
    Z[:,1] = z[:,-1]
    T[1] = 1
    rfin=2*N
    raid_ind = []
    str_out = ''
    for r in range(2,N+2):
        z_tmp = np.array(Z[:,r-1] +M.full_de(Z[:,r-1])).reshape(data['M']*5+8,1)
        z_tmp[z_tmp<0]=0.001
        T[r] = T[r-1]+1
        Z[:,r] = z_tmp.T

        Z[:,r],cind = M.discrete_raid(data,Z[:,r])
        raid_ind.append(cind[0])
        rules = [
            rfin>=N+2,
            r>=365*10,
            np.max(np.abs(np.sum(Z[:,r-360:],axis=0)-np.sum(Z[:,r])))<200
            ]
        if np.sum(Z[-8:,r])<8 and rfin>=N+2:
            rfin = r
            str_out = 'lo'
        elif all(rules):
            rfin = r+1
            str_out = 'hi'
        if r>=rfin:
            T = T[:r+1]
            Z = Z[:,:r+1]
            break
    print(data['M'],str_out)
    return Z,T,raid_ind,data

data['M'] = 17
data['radius'] = 40
mdl = m.ANT
N = 365*200
Z,T,raid,data = run_sim(mdl,N,data,space_type='poisson')
z = Z[:,-1]
raid = np.array(raid)
unique,counts = np.unique(raid,return_counts=True)
Z = m.get_colony_size(Z,data['M'],data['pre_p'])
fig,ax = plt.subplots(1,1)
ax.set_axisbelow(True)
plt.grid(color='k',axis='y',which='major')
ax.bar(unique,counts,0.5)
ax.set_title('Raid count by colony index')
fig,ax = plt.subplots(1,1)
labs = [f'{x}' for x in range(0,data['M'])]
labs.append("P")
ax.plot(T/365,Z.T)
ax.set_xlabel('Years',fontsize=18)
ax.set_ylabel('Colony size',fontsize=18)
ax.set_title('Landscape Model',fontsize=24)


plot_v(z,data,'end.png')
