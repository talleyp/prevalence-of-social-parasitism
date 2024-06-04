import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["figure.figsize"] = (8,4)


T = np.loadtxt('sweep.out',delimiter=' ')

dat = np.loadtxt("sweep-pred.out")
jitter = (np.random.random(T.shape[0])-0.5)*0.05

plt.scatter(T[:,1],T[:,0]+jitter,label='Simulated data',alpha=.3)
plt.plot(dat[:,0], dat[:,1], 'r-',linewidth=3,label='Fitted logistic regression')
plt.xlabel('Number of Formica colonies',fontsize=18)
plt.ylabel('Polyergus survival',fontsize=18)
plt.xticks(ticks= np.arange(10,31,2))
plt.tight_layout()
plt.savefig('density-sweep.png')

