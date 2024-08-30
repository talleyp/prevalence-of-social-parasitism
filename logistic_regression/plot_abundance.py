import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde

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


fh = '../data/formica_density/df.csv'
dat = np.genfromtxt(fh,delimiter=',',skip_header=1)

M = dat[(dat[:,3]==1),2]
A = 1./M

density = gaussian_kde(A)
xs = np.linspace(0, 0.2, 200)
xred = np.linspace(.03, 0.1, 200)

# plt.plot(xs,density(xs))
plt.fill_between( xs, density(xs)/np.sum(density(xs)), color="#69b3a2", alpha=0.4)
plt.fill_between( xred, density(xred)/np.sum(density(xs)), color="#e64539", alpha=0.6)
plt.xlabel('Local abundance of Polyergus')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('../figs/formica-abundance-highligh.pdf')


plt.fill_between( xs, density(xs)/np.sum(density(xs)), color="#69b3a2", alpha=0.4)
plt.xlabel('Local abundance of Polyergus')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('../figs/formica-abundance.pdf')
