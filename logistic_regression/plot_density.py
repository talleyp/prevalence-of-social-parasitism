import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.rcParams["figure.autolayout"] = True
# mpl.rcParams["figure.figsize"] = (8,4)

#### using tex style

pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.
style_path = "../paper.mplstyle"
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

T = np.loadtxt('../data/logistic_regression/sweep.out',delimiter=' ')

dat = np.loadtxt("../data/logistic_regression/sweep-pred.out")
jitter = (np.random.random(T.shape[0])-0.5)*0.05

plt.scatter(T[:,1],T[:,0]+jitter,label='Simulated data',alpha=.3)
plt.plot(dat[:,0], dat[:,1], 'r-',linewidth=3,label='Fitted logistic regression')
plt.xlabel('Number of Formica colonies',fontsize=18)
plt.ylabel('Polyergus survival',fontsize=18)
# plt.xticks(ticks= np.arange(10,31,2))
plt.tight_layout()
plt.savefig('../figs/density-sweep.pdf')

