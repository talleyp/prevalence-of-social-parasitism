import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl

mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["figure.figsize"] = (8,4)



D = np.loadtxt('table17-2.out',delimiter=' ')
jitter = (np.random.random(D.shape[0])-0.5)*0.05

Pred = np.loadtxt("pois-pred.out")

plt.scatter(D[:,1],D[:,-1]+jitter,label='Simulated data',alpha=.3)

plt.plot(Pred[:,0], Pred[:,1], 'r-',linewidth=3,label='Fitted logistic regression')
plt.xlabel('Colonies further than 17m away',fontsize=18)
plt.ylabel('Polyergus surival',fontsize=18)
plt.xticks(np.arange(5,27,2),fontsize=14)
plt.tight_layout()
plt.savefig('poisson.png')
# plt.show()
