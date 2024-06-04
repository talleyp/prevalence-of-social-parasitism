import numpy as np
import utils as m
from parameter import data
import matplotlib.pyplot as plt

if __name__=='__main__':
    mdl = m.ANT2
    rs0 = data['rs']
    z0 = m.start_values(mdl(data))
    N = 40
    Ms = np.arange(1,N)
    Zfin = np.zeros(Ms.shape[0])
    
    data['prd'] = 1
    data['cst'] = True

    for i in range(N-1):
        M = Ms[i]
        data['rs'] = rs0/M
        t,z1 = m.run_once(mdl(data),z0, t_end=365*250)
        Zfin[i] = np.sum(z1[:,-1])
    plt.scatter(Ms,Zfin)
    plt.show()
    np.savetxt('../compare/data/single/F.out',Zfin,delimiter=',')
    np.savetxt('../compare/data/single/M.out',Ms,delimiter=',')
