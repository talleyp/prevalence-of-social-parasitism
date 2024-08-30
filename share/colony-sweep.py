import numpy as np
import utils as m
from parameter import data
import matplotlib.pyplot as plt

if __name__=='__main__':
    mdl = m.ANT
    p = data.copy()
    rs0 = p['rs']
    z0 = m.start_values()
    N = 40
    Ms = np.arange(1,N)
    Zfin = np.zeros(Ms.shape[0])
    fitness = np.zeros(Ms.shape[0])
    k = 0
    rtypes = ['neutral','positive','negative']
    p['alp'] = 0
    for i in range(N-1):
        M = Ms[i]
        p['rs'] = rs0/M
        t,z1 = m.run_once(mdl(p),z0, t_end=365*250)
        Zfin[i] = np.sum(z1[:,-1])
        alph = m.RESPONSE(p['alp']).alpha(np.sum(z1[[3,4],-1]),p['rs'])
        fitness[i] = (1-alph)*p['rp']*z1[2,-1]
    # plt.scatter(Ms,Zfin)
    # plt.show()
    np.savetxt(f'../data/compare/colony-size/single/{rtypes[k]}/F.out',Zfin,delimiter=',')
    np.savetxt(f'../data/compare/colony-size/single/{rtypes[k]}/M.out',Ms,delimiter=',')
    np.savetxt(f'../data/compare/fitness/single/{rtypes[k]}/fitness.out',fitness,delimiter=',')
    np.savetxt(f'../data/compare/fitness/single/{rtypes[k]}/M.out',Ms,delimiter=',')

