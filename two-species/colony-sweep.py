import numpy as np
import utils as m
from parameter import data

if __name__=='__main__':
    mdl = m.ANT
    rs0 = data['rs']
    z0 = m.start_values()
    N = 40
    Ms = np.arange(1,N)
    Zfin = np.zeros(Ms.shape[0])
    Pfin = np.zeros(Ms.shape[0])

    for i in range(N-1):
        M = Ms[i]
        data['rs'] = rs0
        data['M'] = M
        t,z1 = m.run_once(mdl(data),z0, t_end=365*250)
        Zfin[i] = np.sum(z1[:5,-1])
        Pfin[i] = np.sum(z1[5:,-1])
    
    np.savetxt('data/F.out',Zfin,delimiter=',')
    np.savetxt('data/P.out',Pfin,delimiter=',')
    np.savetxt('data/M.out',Ms,delimiter=',')
