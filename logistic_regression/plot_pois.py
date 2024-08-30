import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl

# mpl.rcParams["figure.autolayout"] = True
# mpl.rcParams["figure.figsize"] = (8,4)

#### using tex style

pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.
style_path = "paper.mplstyle"
plt.style.use("paper.mplstyle")
mpl.rcParams['text.usetex'] = True

jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
              "CQG": {"onecol": 374.*pt}, # CQG is only one column
              # Add more journals below. Can add more properties to each journal
             }

my_width = jour_sizes["PRD"]["twocol"]
# Our figure's aspect ratio
golden = (1 + 5 ** 0.5) / 2

### end tex style


D = np.loadtxt('../data/logistic_regression/poisson-table-score-full.out',delimiter=' ')
jitter = (np.random.random(D.shape[0])-0.5)*0.05

Pred = np.loadtxt("data/pois-pred-score-full.out")

plt.scatter(D[:,1],D[:,-1]+jitter,label='Simulated data',alpha=.3)

plt.plot(Pred[:,0], Pred[:,1], 'r-',linewidth=3,label='Fitted logistic regression')
plt.xlabel('Colonies between 23m and 73m away',fontsize=18)
plt.ylabel('Polyergus surival',fontsize=18)
# plt.xticks(np.arange(20,60,5),fontsize=14)
plt.tight_layout()
plt.savefig('../figs/poisson.pdf')
# plt.show()
