## Find the critical density of ants 
## running with two hectares of area

import numpy as np
import utils as m
from parameter import data
from pathlib import Path
import pickle

def run_sim(mdl,N,p,space_type='uni-ring'):
    L = m.LAND()
    p['M'],p['D'] =L(space_type,p)
    p['pre_p']=True
    p['M'] = p['M']+1
    z0 = m.start_values(p['M'])
    p['weights'] = mdl(p).make_grid()
    # tpre,zpre = m.run_once(mdl(p),z0,t_end=365*20)

    p['pre_p']=False
    p['M'] = p['M']-1
    # z0 = m.poly_start(zpre[:,-1],p['M'])
    z0 = m.poly_start(z0,p['M'])

    Z = np.zeros((z0.shape[0],N+2))
    T = np.zeros((N+2,1))

    Z[:,0] = z0

    z = np.array(z0+ mdl(p).full_de(z0)).reshape(p['M']*5+8,1)
    z = np.hstack((z0.reshape(p['M']*5+8,1),z))
    z[z<0]=0.001
    M = mdl(p)
    z[:,-1],cind = M.discrete_raid(p,z[:,-1])

    Z[:,1] = z[:,-1]
    T[1] = 1
    rfin=2*N
    raid_ind = []
    
    for r in range(2,N+2):
        z_tmp = np.array(Z[:,r-1] +M.full_de(Z[:,r-1])).reshape(p['M']*5+8,1)
        z_tmp[z_tmp<0]=0.001
        T[r] = T[r-1]+1
        Z[:,r] = z_tmp.T
        Z[:,r],cind = M.discrete_raid(p,Z[:,r])
        raid_ind.append(cind[0])
        rules = [
            rfin>=N+2,
            r>=365*10,
            np.max(np.abs(np.sum(Z[:,r-360:],axis=0)-np.sum(Z[:,r])))<300
            ]
        if np.sum(Z[-8:,r])<(5*p['M'] )and rfin>=N+2:
            rfin = r
            print('lo')
        elif all(rules):
            rfin = r+10
            print('hi')
        if r>=rfin:
            T = T[:r+1]
            Z = Z[:,:r+1]
            break
    
    return Z,T

def calc_fitness(z,p):
    M = p['M']
    Zf = z[:M*5].reshape((M,5))
    Zp = z[M*5:]
    alphaf = m.RESPONSE(data['alp']).alpha(Zf[:,3]+Zf[:,4],0)
    af = np.mean((1-alphaf)*data['rp']*Zf[:,2])
    ap = m.PRESPONSE().alpha(np.sum(Zp[[1,2,6,7]]))
    pf = (1-ap)*Zp[5]*data['rp']
    return af,pf

def biomass(mdl,Nraids,space_type,p,M_min=5,M_max=100,rt='neutral',preload=False):
    '''
    We choose between M_min and M_max formica colonies
    and get the biomass at the end of the simulation
    '''
    size_folder = f'../data/compare/colony-size/landscape/{rt}'
    fit_folder = f'../data/compare/fitness/landscape/{rt}'
    
    if preload:
        F = np.loadtxt(f'{size_folder}/F.out',delimiter=',')
        P = np.loadtxt(f'{size_folder}/P.out',delimiter=',')
        Ffit = np.loadtxt(f'{fit_folder}/Ffitness.out',delimiter=',')
        Pfit = np.loadtxt(f'{fit_folder}/Pfitness.out',delimiter=',')
        Ms = np.loadtxt(f'{size_folder}/M.out',delimiter=',')
        N = Ms.shape[0]
        Nstart = np.where(F==0.)[0][0]
        Nrange = range(Nstart,N)
    else:
        Ms = np.arange(M_min,M_max)
        F = np.zeros(Ms.shape)
        P = np.zeros(Ms.shape)
        Ffit = np.zeros(Ms.shape)
        Pfit = np.zeros(Ms.shape)
        Nrange = range(Ms.shape[0])
    for i in Nrange:
        M = Ms[i]
        p['M']=int(M)
        z,t = run_sim(mdl,Nraids,p,space_type)
        Z = m.get_colony_size(z,p['M'])
        F[i] = np.average(Z[:-1,-1])
        P[i] = Z[-1,-1]
        Ffit[i],Pfit[i] = calc_fitness(z[:,-1],p.copy())
        ## save colony size
        np.savetxt(f'{size_folder}/F.out',F,delimiter=',')
        np.savetxt(f'{size_folder}/P.out',P,delimiter=',')
        np.savetxt(f'{size_folder}/M.out',Ms,delimiter=',')
        ## save fitness
        np.savetxt(f'{fit_folder}/Ffitness.out',Ffit,delimiter=',')
        np.savetxt(f'{fit_folder}/Pfitness.out',Pfit,delimiter=',')
        np.savetxt(f'{fit_folder}/M.out',Ms,delimiter=',')
    return 0

mdl = m.ANT
Nraids = 365*80
space_type = 'uni-ring'

data['radius'] = 30
k = 0
rtypes = ['neutral','positive','negative']
data['alp'] = k

biomass(mdl,Nraids,space_type,data.copy(),M_min=1,M_max=40,rt = rtypes[k],preload=True)
