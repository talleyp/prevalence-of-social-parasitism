import numpy as np
import matplotlib.pyplot as plt
import os.path
import matplotlib as mpl

mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["figure.figsize"] = (8,4)

folders = ['data/single','data/two','data/landscape']
colors = ['black','red','blue']
labs = 'Formica energetics','Mean-field','Ring of colonies'
markers = ['o','*',"3"]
M = np.loadtxt(f'{folders[0]}/M.out',delimiter=',')

fig,(ax0,ax1) = plt.subplots(1,2,sharex=True,sharey=True)
for i in range(len(folders)):
    F = np.loadtxt(f'{folders[i]}/F.out',delimiter=',')
    ax0.scatter(M,F,label=labs[i],c=colors[i],marker=markers[i])
    if os.path.exists(f'{folders[i]}/P.out'):
        P = np.loadtxt(f'{folders[i]}/P.out',delimiter=',')
        ax1.scatter(M,P,label=labs[i],c=colors[i],marker=markers[i])
ax0.legend(loc='upper left')
ax1.legend(loc='upper left')
ax0.set_xlabel('Number of colonies',fontsize=18)
ax1.set_xlabel('Number of colonies',fontsize=18)
ax0.set_ylabel('Colony size',fontsize=18)
ax1.set_ylabel('Colony size',fontsize=18)
ax0.set_title('Formica',fontsize=24)
ax1.set_title('Polyergus',fontsize=24)
plt.tight_layout()
plt.savefig('equilibrium.pdf',dpi=199)
# plt.show()