import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
import utils as m
from parameter import data

if __name__=='__main__':
    nplots = 1

    mdl = m.ANT

    z0 = m.start_values()
    
    t,z = m.run_once(mdl(data),z0, t_end=365*250)
    print(z[:,-1],'F/L',z[-1,-1]/z[1,-1])
    print(np.sum(z[:,-1]))
    p1,nplots = m.plot_one_run(t/365,z,nplots)

    plt.show()

