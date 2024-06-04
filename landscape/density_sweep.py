## Find the critical density of ants 
## running with two hectares of area

import numpy as np
import utils as m
from parameter import data
from pathlib import Path
# import pickle
import multiprocess as mp
# import ctypes

def run_sim(mdl,N,data,Mcol,x,space_type='ring'):
    # Pre-run setup
    data['M']= Mcol
    L = m.LAND()
    _,data['D'] =L(space_type,data)
    data['pre_p']=True
    data['M'] = data['M']+1 # because P is F for now
    z0 = m.start_values(data['M'])
    z0 = 0.5*z0
    data['weights'] = mdl(data).make_grid()

    # competition effects
    tpre,zpre = m.run_once(mdl(data),z0,t_end=365*60)
    
    # invasion of polyergus starts
    data['pre_p']=False
    data['M'] = data['M']-1 # F back to P
    z0 = m.poly_start(zpre[:,-1],data['M'])

    # Set up polyergus dynamic saving
    Z = np.zeros((z0.shape[0],N+2))
    T = np.zeros((N+2,1))
    Z[:,0] = z0
    raid_ind = []
    rfin=2*N

    # first raid
    z = z0+ mdl(data).full_de(z0)
    
    z[z<0]=0.001 # kill off basically dead colonies
    M = mdl(data)
    Z[:,1],cind = M.discrete_raid(data,z)
    T[1] = 1 

    # repeat N times 
    str_out = '_'
    for r in range(2,N+2):
        z_tmp = Z[:,r-1] +M.full_de(Z[:,r-1])
        z_tmp[z_tmp<0]=0.001
        T[r] = T[r-1]+1
        Z[:,r] = z_tmp.T
        Z[:,r],cind = M.discrete_raid(data,Z[:,r])
        raid_ind.append(cind[0])
        # equilibrium rules
        rules = [
            rfin>=N+2,
            r>=365*10,
            np.max(np.abs(np.sum(Z[:,r-360:],axis=0)-np.sum(Z[:,r])))<500
            ]
        if np.sum(Z[-8:,r])<8 and rfin>=N+2: # dead polyergus end
            rfin = r
            str_out = 'lo'
        elif all(rules): # landscape at equilibrium
            rfin = r+1
            str_out = 'hi'
        if r>=rfin: # we are done
            T = T[:r+1]
            Z = Z[:,:r+1]
            break
    print(data['M'],str_out)
    C = m.get_colony_size(Z,data['M'],data['pre_p'])
    
    np.savetxt(f"data/land-dist/{data['M']}-{x}-Z.out",C[:,-1],delimiter=',')
    np.savetxt(f"data/land-dist/{data['M']}-{x}-D.out",data['D'],delimiter=',')
    return 0

def to_numpy_array(shared_array, shape):
    '''Create a numpy array backed by a shared memory Array.'''
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)

def init_worker():
    '''
    Initialize worker for processing:
    '''
    np.random.seed()


def worker_fun(x, y):
    '''worker function'''
    run_sim(mdl,Nraids,data,Ms[y],x,space_type='poisson')

mdl = m.ANT
Nraids = 365*200
Nsims = 100
space_type = 'poisson'

data['radius'] = 40
Mmin = 10
Mmax = 30
Ms = np.arange(Mmin,Mmax+1)

shape = (Nsims,Ms.shape[0])

x_y_values = [(x, y) for x in range(shape[0]) for y in range(shape[1])]

pool = mp.Pool(60,maxtasksperchild=1,initializer=init_worker)
pool.starmap(worker_fun, x_y_values)
