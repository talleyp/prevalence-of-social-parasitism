## Find the critical density of ants 
## running with two hectares of area

import numpy as np
import utils as m
from parameter import data
from pathlib import Path
import pickle
import multiprocess as mp
import ctypes

def run_sim(mdl,N,p,x,M,space_type='ring'):
    # Pre-run setup
    L = m.LAND()
    p['M'] = M
    p['M'],p['D'] =L(space_type,p)

    p['pre_p']=True
    p['M'] = p['M']+1 # because P is F for now
    z0 = m.start_values(p['M'])
    p['weights'] = mdl(p).make_grid()

    # competition effects
    tpre,zpre = m.run_once(mdl(p),z0,t_end=365*5)
    
    # invasion of polyergus starts
    p['pre_p']=False
    p['M'] = p['M']-1 # F back to P
    z0 = m.poly_start(zpre[:,-1],p['M'])

    # Set up polyergus dynamic saving
    Z = np.zeros((z0.shape[0],N+2))
    T = np.zeros((N+2,1))
    Z[:,0] = z0
    raid_ind = []
    rfin=2*N

    # first raid
    z = z0+ mdl(p).full_de(z0)
    
    z[z<0]=0.001 # kill off basically dead colonies
    M = mdl(p)
    Z[:,1],cind = M.discrete_raid(p,z)
    T[1] = 1 

    # repeat N times 
    str_out = '_'
    for r in range(2,N+2):
        z_tmp = Z[:,r-1] +M.full_de(Z[:,r-1])
        z_tmp[z_tmp<0]=0.001
        T[r] = T[r-1]+1
        Z[:,r] = z_tmp.T
        Z[:,r],cind = M.discrete_raid(p,Z[:,r])
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
            rfin = r
            str_out = 'hi'
        if r>=rfin: # we are done
            T = T[:r+1]
            Z = Z[:,:r+1]
            break
    print(p['M'],str_out)
    C = m.get_colony_size(Z,p['M'],p['pre_p'])

    np.savetxt(f'../data/landscape/poisson/{x}-Z.out',C[:,-1],delimiter=',')
    np.savetxt(f'../data/landscape/poisson/{x}-D.out',p['D'],delimiter=',')

    return 0

def to_numpy_array(shared_array, shape):
    '''Create a numpy array backed by a shared memory Array.'''
    A = np.ctypeslib.as_array(shared_array)
    return A.reshape(shape)

# def init_worker(shared_arrayZ,shared_arrayD,shapeZ,shapeD):
def init_worker(shared_arrayM, shapeM):
    '''
    Initialize worker for processing:
    Give access to shared M array
    '''
    global Marr
    Marr = to_numpy_array(shared_arrayM, shapeM)
    


def worker_fun(x):
    '''worker function'''
    run_sim(mdl,Nraids,data.copy(),x,Marr[x],space_type)

mdl = m.ANT
Nraids = 365*100
Nsims = 500
space_type = 'poisson'

data['radius'] = 70
M = 46
data['M'] = M

shapeM = (Nsims,)


shared_arrayM = mp.Array(ctypes.c_int, np.random.poisson(data['M'],Nsims), lock=False)
x_values = [x for x in range(Nsims)]

pool = mp.Pool(60,
               maxtasksperchild=1,
               initializer=init_worker, 
               initargs=(shared_arrayM, shapeM)
            )
pool.map(worker_fun, x_values)

