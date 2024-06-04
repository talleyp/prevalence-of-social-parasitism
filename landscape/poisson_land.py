## Find the critical density of ants 
## running with two hectares of area

import numpy as np
import utils as m
from parameter import data
from pathlib import Path
import multiprocess as mp

def run_sim(mdl,N,data,x,space_type='ring'):
    # Pre-run setup
    L = m.LAND()
    data['M'],data['D'] =L(space_type,data)
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

    # np.savetxt(f'data/poisson/{x}-Z.out',C[:,-1],delimiter=',')
    # np.savetxt(f'data/poisson/{x}-D.out',data['D'],delimiter=',')
    # np.savetxt(f'data/land/D-{M}.out',Darr[:,:,1],delimiter=',')

    # return C[:,-1],data['D']
    return 0

# def to_numpy_array(shared_array, shape):
#     '''Create a numpy array backed by a shared memory Array.'''
#     A = np.ctypeslib.as_array(shared_array)
#     return A.reshape(shape)

# def init_worker(shared_arrayZ,shared_arrayD,shapeZ,shapeD):
def init_worker():
    '''
    Initialize worker for processing:
    Generate random seed
    '''
    # global Zarr
    # global Darr
    # Zarr = to_numpy_array(shared_arrayZ, shapeZ)
    # Darr = to_numpy_array(shared_arrayD, shapeD)
    np.random.seed()


def worker_fun(x):
    '''worker function'''
    run_sim(mdl,Nraids,data,x,space_type)

mdl = m.ANT
Nraids = 1#365*200
Nsims = 100
space_type = 'poisson-point'

data['radius'] = 40
M = 17
data['M'] = M

# shapeZ = (Nsims,data['M']+1)
# shapeD = (Nsims,data['M']+1,2)

# shared_arrayZ = mp.Array(ctypes.c_double, int(np.product(shapeZ)), lock=False)
# shared_arrayD = mp.Array(ctypes.c_double, int(np.product(shapeD)), lock=False)
# Zarr = to_numpy_array(shared_arrayZ, shapeZ)
# Darr = to_numpy_array(shared_arrayD, shapeD)
x_values = [x for x in range(Nsims)]


pool = mp.Pool(2,
               maxtasksperchild=1,
               initializer=init_worker, 
            #    initargs=(shared_arrayZ, shared_arrayD, shapeZ, shapeD)
            )
pool.map(worker_fun, x_values)

# np.savetxt(f'data/land/Z-{M}.out',Zarr,delimiter=',')
# np.savetxt(f'data/land/D-x-{M}.out',Darr[:,:,0],delimiter=',')
# np.savetxt(f'data/land/D-y-{M}.out',Darr[:,:,1],delimiter=',')