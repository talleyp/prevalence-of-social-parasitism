import numpy as np
import matplotlib.pyplot as plt
import os.path
import matplotlib as mpl

# mpl.rcParams["figure.autolayout"] = True
# mpl.rcParams["figure.figsize"] = (8,4)

pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.
style_path = "paper.mplstyle"
plt.style.use("paper.mplstyle")
mpl.rcParams['text.usetex'] = True

jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
              "CQG": {"onecol": 374.*pt}, # CQG is only one column
              # Add more journals below. Can add more properties to each journal
             }

my_width = jour_sizes["PRD"]["twocol"]
ratio = 2#(1 + 5 ** 0.5) / 2


rtype = ['neutral','positive','negative']
k=0

folders = [
    f'../data/compare/fitness/single/{rtype[k]}',
    f'../data/compare/fitness/two/{rtype[k]}',
    f'../data/compare/fitness/landscape/{rtype[k]}'
]
colors = ['black','red','blue']
labs = 'Formica energetics','Mean-Coupling','Ring of colonies'
markers = ['o','*',"3"]
M = np.loadtxt(f'{folders[0]}/M.out',delimiter=',')

fig,(ax0,ax1) = plt.subplots(1,2,sharex=True,sharey=False,figsize = (my_width, my_width/ratio))
for i in range(len(folders)):
    F = np.loadtxt(f'{folders[i]}/Ffitness.out',delimiter=',')
    ax0.scatter(M,F,label=labs[i],c=colors[i],marker=markers[i])
    if os.path.exists(f'{folders[i]}/Pfitness.out'):
        P = np.loadtxt(f'{folders[i]}/Pfitness.out',delimiter=',')
        ax1.scatter(M,P,label=labs[i],c=colors[i],marker=markers[i])
ax0.legend(loc='center left')
ax1.legend(loc='center left')
# ax0.set_xlabel('Number of colonies, M',fontsize=12)
# ax1.set_xlabel('Number of colonies, M',fontsize=12)
# ax0.set_ylabel('Colony size',fontsize=12)
# ax1.set_ylabel('Colony size',fontsize=12)
ax0.set_title('Formica',fontsize=16) #
ax1.set_title('Polyergus',fontsize=16)
fig.supylabel('Fitness')
fig.supxlabel('Number of colonies, M')
plt.tight_layout()
plt.savefig(f'../figs/equilibrium-fitness-{rtype[k]}.pdf')
# plt.show()