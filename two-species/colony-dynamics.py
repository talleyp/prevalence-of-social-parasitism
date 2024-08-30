import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
import utils as m
from parameter import data

if __name__=='__main__':
    nplots = 1

    mdl = m.ANT

    z0 = m.start_values()

    data['M'] = 14
    data['alp'] = 1
    
    t,z1 = m.run_once(mdl(data),z0, t_end=365*250,method='Radau')
    nplots = m.plot_one_run(t/365,z1,nplots,'Two Species Model')
    # p2,nplots = m.plot_both_sum(t,z1,nplots,'Two Species Model')
    # plt.figure(nplots)
    # plt.plot(t/365,z1[-1,:])
    print(z1[-8:,-1])


    plt.show()
