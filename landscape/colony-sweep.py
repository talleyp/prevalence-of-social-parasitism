## Find the critical density of ants 
## running with two hectares of area

import numpy as np
import utils as m
from parameter import data
from pathlib import Path
import pickle

def run_sim(mdl,N,data,space_type='uni-ring'):
    L = m.LAND()
    data['M'],data['D'] =L(space_type,data)
    data['pre_p']=True
    data['M'] = data['M']+1

    z0 = m.start_values(data['M'])
    data['weights'] = mdl(data).make_grid()
    tpre,zpre = m.run_once(mdl(data),z0,t_end=5000)

    data['pre_p']=False
    data['M'] = data['M']-1
    z0 = m.poly_start(zpre[:,-1],data['M'])

    Z = np.zeros((z0.shape[0],N+2))
    T = np.zeros((N+2,1))

    Z[:,0] = z0

    z = np.array(z0+ mdl(data).full_de(z0)).reshape(data['M']*5+8,1)
    z = np.hstack((z0.reshape(data['M']*5+8,1),z))
    z[z<0]=0.001
    M = mdl(data)
    z[:,-1],cind = M.discrete_raid(data,z[:,-1])

    Z[:,1] = z[:,-1]
    T[1] = 1
    rfin=2*N
    raid_ind = []
    for r in range(2,N+2):
        z_tmp = np.array(Z[:,r-1] +M.full_de(Z[:,r-1])).reshape(data['M']*5+8,1)
        z_tmp[z_tmp<0]=0.001
        T[r] = T[r-1]+1
        Z[:,r] = z_tmp.T
        Z[:,r],cind = M.discrete_raid(data,Z[:,r])
        raid_ind.append(cind[0])
        rules = [
            rfin>=N+2,
            r>=365*10,
            np.max(np.abs(np.sum(Z[:,r-360:],axis=0)-np.sum(Z[:,r])))<300
            ]
        if np.sum(Z[-8:,r])<8 and rfin>=N+2:
            rfin = r
            print('lo')
        elif all(rules):
            rfin = r+10
            print('hi')
        if r>=rfin:
            T = T[:r+1]
            Z = Z[:,:r+1]
            break
    
    return Z,T,raid_ind,data

def biomass(mdl,Nraids,space_type,data,M_min=5,M_max=100):
    '''
    We choose between M_min and M_max formica colonies
    and get the biomass at the end of the simulation
    '''
    Ms = np.arange(M_min,M_max)
    F = np.zeros(Ms.shape)
    P = np.zeros(Ms.shape)
    
    for i in range(Ms.shape[0]):
        M = Ms[i]
        data['M']=M
        z,t,raids,data = run_sim(mdl,Nraids,data,space_type)
        Z = m.get_colony_size(z,data['M'])
        F[i] = np.average(Z[:-1,-1])
        P[i] = Z[-1,-1]
        print(M)
    return F,P

def save_data(z,t,raids,datas,folder):
    path_name = f'data/{folder}'
    Path(path_name).mkdir(parents=True,exist_ok=True)

    with open(f"{path_name}/z.p",'wb') as filehandle:
        pickle.dump(z, filehandle)
    with open(f"{path_name}/t.p",'wb') as filehandle:
        pickle.dump(t, filehandle)
    with open(f"{path_name}/data.p",'wb') as filehandle:
        pickle.dump(datas, filehandle)
    with open(f"{path_name}/raids.p",'wb') as filehandle:
        pickle.dump(raids, filehandle)

mdl = m.ANT
Nraids = 365*250
space_type = 'uni-ring'

data['radius'] = 30

Zfin,Pfin = biomass(mdl,Nraids,space_type,data,M_min=1,M_max=40)
Ms = np.arange(1,40)
np.savetxt('data/F.out',Zfin,delimiter=',')
np.savetxt('data/P.out',Pfin,delimiter=',')
np.savetxt('data/M.out',Ms,delimiter=',')