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
    Pfin = np.zeros(Ms.shape[0])
    Ffitness = np.zeros(Ms.shape[0])
    Pfitness = np.zeros(Ms.shape[0])
    k = 0
    p['alp'] = k
    rtypes = ['neutral','positive','negative']
    for i in range(N-1):
        M = Ms[i]
        p['rs'] = rs0
        p['M'] = M
        t,z1 = m.run_once(mdl(p),z0, t_end=365*250)
        Zfin[i] = np.mean(np.sum(z1[:5,-50:],axis=0))
        Pfin[i] = np.mean(np.sum(z1[5:,-50:],axis=0))
        ## calculate fitness
        Zadu = np.mean(np.sum(z1[[3,4],-50:],axis=0))
        Padu = np.mean(np.sum(z1[[6,7,11,12],-50:],axis=0))
        raiders = np.mean(z1[-1,-50:])
        fpupa = np.mean(z1[2,-50:])
        ppupa = np.mean(z1[10,-50:])
        alpha_f = m.RESPONSE(p['alp']).alpha(Zadu,rs0*raiders/M)
        
        alpha_p = m.PRESPONSE().alpha(Padu)
        Ffitness[i] = (1-alpha_f)*fpupa*data['rp']
        Pfitness[i] = (1-alpha_p)*ppupa*data['rp']

    np.savetxt(f'../data/compare/colony-size/two/{rtypes[k]}/F.out',Zfin,delimiter=',')
    np.savetxt(f'../data/compare/colony-size/two/{rtypes[k]}/P.out',Pfin,delimiter=',')
    np.savetxt(f'../data/compare/colony-size/two/{rtypes[k]}/M.out',Ms,delimiter=',')
    np.savetxt(f'../data/compare/fitness/two/{rtypes[k]}/Ffitness.out',Ffitness,delimiter=',')
    np.savetxt(f'../data/compare/fitness/two/{rtypes[k]}/Pfitness.out',Pfitness,delimiter=',')
    np.savetxt(f'../data/compare/fitness/two/{rtypes[k]}/M.out',Ms,delimiter=',')
