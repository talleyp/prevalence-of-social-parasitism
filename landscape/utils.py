import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,dblquad
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from itertools import product
from scipy.special import expit
import time

class RESPONSE:
    def __init__(self,r):
        self.tbase = 1500
        self.aspan = 1
        self.Kr    = .43
        self.Ka    = 2**7
        if r==0:
            # print('Non-Responsive to Raiding')
            self.theta=self.t1
        elif r==1:
            # print("Raiding induces gyne production")
            self.theta=self.t2
        else:
            # print("Raiding inhibits gyne production")
            self.theta=self.t3

    def t1(self,x):
        return self.tbase

    def t2(self,x):
        return self.tbase/(1+(x/self.Kr)**2)

    def t3(self,x):
        return self.tbase*(1+(x/self.Kr)**2)

    def alpha(self,x,y):
        t = self.theta(y)
        a = expit((t-x)/self.Ka)
        return self.aspan*a + (1-self.aspan)

class PRESPONSE:
    def __init__(self):
        self.tbase = 8000
        self.aspan = 1
        self.Ka    = 2**9

    def alpha(self,x):
        t = self.tbase
        a = expit((t-x)/self.Ka)
        return self.aspan*a + (1-self.aspan)

class ANT:
    def __init__(self,
                    p,
                ) -> None:
        self.c  = p['c']
        self.rq  = p['rq']
        self.re  = p['re']
        self.rl  = p['rl']
        self.rp  = p['rp']
        self.A   = p['A']
        self.mun = p['mun']
        self.muf = p['muf']
        self.mur = p['mur']
        self.Km  = p['Km']
        self.Kf  = p['Kf']
        self.Kb  = p['Kb']
        self.M   = p['M']
        self.B   = p['B']
        self.D   = p['D']
        self.lamp = p['lamp']
        self.fradius = p['fradius']
        self.land = p['landscape']
        self.comp = p['competition']
        if 'weights' in p.keys():
            self.weights = p['weights']
        self.land_disk = self.boundaries()
        self.scale = p['scale']
        self.pre_p = p['pre_p']
        self.alp  = RESPONSE(p['alp']).alpha 
        self.alpp  = PRESPONSE().alpha 
        self.eta  = p['eta']
        self.rs   = np.zeros(self.M)
        pass

    def __call__(self, t, z) -> np.array:        
        return self.full_de(z)
    
    def full_de(self,z) -> np.array:
        if self.pre_p:
            Zf = z.reshape((self.M,5))
            self.alphaf = self.alp(Zf[:,3]+Zf[:,4],self.rs)
            Ft = Zf[:,-1]
            if self.comp:
                A = self.voronoi_areas(Ft)
                self.cf = self.c * A/(10+A)
            else:
                self.cf = self.c*np.ones(self.M)
            return self.f_de(Zf)
        else:

            Zf = z[:self.M*5].reshape((self.M,5))
            Zp = z[self.M*5:]

            self.alphaf = self.alp(Zf[:,3]+Zf[:,4],self.rs)
            Ft = np.hstack((Zf[:,-1],Zp[2]))
            if self.comp:
                A = self.voronoi_areas(Ft)
                self.cf = self.c *A/(10+A)
            else:
                self.cf = self.c * np.ones(self.M)
            out =  np.hstack((self.f_de(Zf),self.p_de(Zp)))
            return out
    
    def f_de(self,z,t=1) -> np.array:
        E = z[:,0]
        L = z[:,1]
        P = z[:,2]
        N = z[:,3]
        F = z[:,4]
        alpha = self.alphaf
        

        B = E+(P+L)*(alpha+self.eta*(1-alpha))
        phiT = self.phi(self.cf[:self.M]*F, L*(alpha+self.eta*(1-alpha))+1)

        f = lambda x,y: 1/(1+(x/y)**2) 
        g = lambda x: np.log(10/9)*(self.Kb/x)**2
        nb = g(N/B) 
        fn = f(F/N, self.Kf)
        
        dE = phiT*self.rq - self.re*E*(1+nb)
        dL = self.re*E - self.rl*L*(phiT+nb) 
        dP = phiT*self.rl*L - P*self.rp*(1+nb) 
        dN = self.rp*P*alpha - N*fn*(self.mun+self.A) 
        dF = N*fn*self.A - F*self.muf

        return np.vstack((dE,dL,dP,dN,dF)).T.flatten()

    def p_de(self,z,t=1) -> np.array:
        P = z[0]
        N = z[1]
        F = z[2]

        E = z[3]
        L = z[4]
        U = z[5]
        S = z[6]
        R = z[7]

        cfT = self.cf[-1]
        alpha = self.alpp(N+S+F+R)

        phiT = self.phi(cfT*F, L*(alpha+self.eta*(1-alpha))+1)
        B = E+P+(L+U)*(alpha+self.eta*(1-alpha))

        f = lambda x,y: 1/(1+(x/y)**2) 
        g = lambda x: np.log(10/9)*(8/x)**2
        nb = g(N/B) 
        fn = f(F/(N+S+R), self.Kf)

        dP = -P*self.rp*(1+nb) 
        dN = P*self.rp - N*fn*(self.mun+self.A) 
        dF = N*fn*self.A - F*self.muf
        
        dE = phiT*self.rq - self.re*E*(1+nb)
        dL = self.re*E - self.rl*L*(phiT+nb) 
        dU = phiT*self.rl*L - U*self.rp*(1+nb)
        dS = self.rp*U*alpha - S*fn*(self.mun+self.B)
        dR = S*fn*self.B - self.mur*R
        
        return np.hstack((dP,dN,dF,dE,dL,dU,dS,dR)) 

    def phi(self,x,y):
        return (x/y)/(self.Km+x/y)
    
    def voronoi_areas(self,F):
        self.wts = self.weights * F[np.newaxis,:]
        self.grid = np.argmax(self.wts,axis=1)
        self.wts = np.take_along_axis(self.wts,self.grid[:,None],axis=1)
        n_cols = self.M+1-int(self.pre_p)
        A = np.zeros(n_cols)
        for j in range(n_cols):
            A[j] = np.sum(self.wts[self.grid==j])/F[j]

        return A*self.scale**2
    
    def make_grid(self) -> np.array:
        alp = 1/self.fradius
        s = self.scale
        npoints = int(2*self.land/s)
        ## scale the points on grid for distance metrics
        x = (np.arange(0,npoints)+1/2.)*s
        y = (np.arange(0,npoints)+1/2.)*s
        # points = np.array(np.meshgrid(x,y)).T.reshape(-1,2)

        ## x distance minimum between that and torus boundary
        Dx = np.tile(self.D[:,0],(x.shape[0],1))
        Xx = np.tile(x,(self.M,1)).T
        deltaX = np.array([np.abs(Xx-Dx),2*self.land-np.abs(Xx-Dx)])
        xdist = np.square(np.min(deltaX,axis=0))

        Dy = np.tile(self.D[:,1],(y.shape[0],1))
        Yy = np.tile(x,(self.M,1)).T
        deltaY =  np.array([np.abs(Yy-Dy),2*self.land-np.abs(Yy-Dy)])
        ydist = np.square(np.min(deltaY,axis=0))

        distance = np.sqrt(np.sum(
                        np.array(
                            [np.array(
                                np.meshgrid(xdist[:,i],ydist[:,i])
                                ).T.reshape(-1,2) for i in range(self.M)
                            ]
                        )
                    ,axis=2).T)
  
        return np.exp(-distance*alp)*alp
    
    def discrete_raid(self,data,z,plotraid=False) -> np.array:
        forager_ind = [(i+1)*5-1 for i in range(self.M)]
        forager_ind.append(self.M*5+2)
        F = z[forager_ind]
        S = z[-1]
        if S<1:
            self.rs = np.zeros(self.M)
            return z,(-1,0,0)
        n_scouts = int(np.ceil(S*data['scout-p']))

        grid_square = self.grid.reshape((2*self.land,2*self.land)).T
        wts_square = self.wts.reshape((2*self.land,2*self.land)).T
        scout_dist = np.random.exponential(self.lamp,n_scouts)
        scout_angle = 2*np.pi*np.random.uniform(size=n_scouts)
        scout_x = scout_dist*np.cos(scout_angle)+self.land
        scout_y = scout_dist*np.sin(scout_angle)+self.land
        locs = np.arange(self.M)
        rates = np.zeros(self.M)
        for x,y in zip(scout_x.astype(int),scout_y.astype(int)):
            xpos = int(np.mod(x,2*self.land)/self.scale)
            ypos = int(np.mod(y,2*self.land)/self.scale)
            xs = np.arange(xpos-5,xpos+5)%(2*self.land)
            ys = np.arange(ypos-5,ypos+5)%(2*self.land)
            coords = list(product(xs,ys))
            for ij in coords:
                if grid_square[ij]<self.M:
                    rates[grid_square[ij]] += wts_square[ij]
        locs = np.array(locs)
        rates= np.array(rates)
        loc_ind = locs<self.M
        rates = rates[loc_ind]
        locs = locs[loc_ind]
        if len(locs) == 0:
            return z,(-1,0,0)
        
        R = np.sum(rates)
        tau = np.random.exponential(1/R)
        if tau<np.random.exponential(1/(3/24)):
            self.rs = rates/R
            c_ind = int(np.random.choice(locs,p=rates/R))
            p_ind = int(5*c_ind+2)
            stolen = np.min((z[p_ind],S*data['prob-grab']))
            z[p_ind] -= stolen
            z[-8] += stolen*self.alphaf[c_ind]
            return z,(c_ind,F[c_ind],stolen)
        self.rs = np.zeros(rates.shape)
        return z,(-1,0,0)

    def boundaries(self) -> np.array:
        x = np.arange(0,2*self.land)+1/2.
        y = np.arange(0,2*self.land)+1/2.
        points = np.array(np.meshgrid(x,y)).T.reshape(-1,2)
        midpoint = np.array([int(2*self.land/2),int(2*self.land/2)]).reshape(1,2)
        land_disk = cdist(points,midpoint)
        in_ind = np.where(land_disk<=self.land)
        out_ind = np.where(land_disk>self.land)
        land_disk[in_ind]=1
        land_disk[out_ind]=np.nan
        return land_disk
    
def plot_raid(z,data,scout_x,scout_y):
    if data['pre_p']:
        Zf = z[:data['M']*5].reshape((data['M'],5))
        F = Zf[:,-1]
    else:
        Zf = z[:data['M']*5].reshape((data['M'],5))
        Zp = z[data['M']*5:]
        F = np.hstack((Zf[:,-1],Zp[2]))
    
    wts = data['weights']*np.tile(F,(data['weights'].shape[0],1))
    grid = np.nanargmax(wts,axis=1)
    grid = grid.reshape((2*data['landscape'],2*data['landscape'])).T
    unique, A = np.unique(grid[~np.isnan(grid)].flatten(), return_counts=True)
    if len(unique)<data['M']:
        missing = list(set(np.arange(data['M'])).difference(set(unique)))
        A = list(A)
        for n in missing:
            A.insert(n,0)
        A = np.array(A)
    fig,ax = plt.subplots(1,1)
    pc = ax.pcolormesh(grid,cmap='gist_earth',vmin=0,vmax=data['M']-1)
    ax.scatter(scout_x,scout_y)
    fig.colorbar(pc)
    for i, ((x,y),) in enumerate(zip(data['D'])):
        plt.text(x,y,i, ha="center", va="center")
    plt.show()

class LAND:
    def __call__(self,type,data) -> tuple:
        if type=='poisson':
            return self.spatial_poisson2d(data)
        elif type=='poisson-point':
            return self.spatial_poisson_point(data)
        elif type=='ring':
            return self.ring_2d(data)
        elif type=='uni-disc':
            return self.uni_disc_2d(data)
        elif type=='uni-spiral':
            return self.uni_spiral_2d(data)
        elif type=='uni-ring':
            return self.uni_ring_2d(data)
        elif type=='offset':
            return self.one_close_ring_2d(data)
        elif type=='two-ring':
            return self.two_rings(data)
        elif type=='mattern':
            return self.mattern(data)

    def spatial_poisson2d(self,data):
        M = int(data['M'])
        D = (2*data['radius']*np.random.random((M,2)))+data['landscape']-data['radius']
        D = np.vstack((D,[0+data['landscape'],0+data['landscape']]))
        return M,D
    
    def spatial_poisson_point(self,data):
        if M:
            pass
        else:
            M = np.random.poisson(data['M'])
        D = (2*data['radius']*np.random.random((M,2)))+data['landscape']-data['radius']
        D = np.vstack((D,[0+data['landscape'],0+data['landscape']]))
        return M,D

    def ring_2d(self,data):
        M = data['M']
        r = data['radius']
        theta = 2*np.pi*np.random.random((M,1))
        D = np.zeros((M,2))
        D[:,0] = np.transpose(r*np.cos(theta))+data['landscape']
        D[:,1] = np.transpose(r*np.sin(theta))+data['landscape']
        D = np.vstack((D,[0+data['landscape'],0+data['landscape']]))
        return M,D

    def uni_disc_2d(self,data):
        M = data['M']
        r = np.random.uniform(data['small-r'],data['radius'],M)
        theta = 2*np.pi*np.linspace(0,1-1/M,M)
        D = np.zeros((M,2))
        D[:,0] = np.transpose(r*np.cos(theta))+data['landscape']
        D[:,1] = np.transpose(r*np.sin(theta))+data['landscape']
        D = np.vstack((D,[0+data['landscape'],0+data['landscape']]))
        return M,D
    
    def two_rings(self,data):
        M = data['M']-5#//2
        N = data['M'] - M
        r = np.zeros(data['M'])
        r[:M] = data['radius']
        r[M:] = data['small-r']
        theta = np.zeros(data['M'])
        theta[:M] = 2*np.pi*np.linspace(0,1-1/M,M)
        theta[M:] = 2*np.pi*np.linspace(0,1-1/N,N)
        D = np.zeros((data['M'],2))
        D[:,0] = np.transpose(r*np.cos(theta))+data['landscape']
        D[:,1] = np.transpose(r*np.sin(theta))+data['landscape']
        D = np.vstack((D,[0+data['landscape'],0+data['landscape']]))
        return data['M'],D

    def uni_spiral_2d(self,data):
        M = data['M']
        r = np.linspace(data['small-r'],data['radius'],M)
        theta = 2*np.pi*np.linspace(0,1-1/M,M)
        D = np.zeros((M,2))
        D[:,0] = np.transpose(r*np.cos(theta))+data['landscape']
        D[:,1] = np.transpose(r*np.sin(theta))+data['landscape']
        D = np.vstack((D,[0+data['landscape'],0+data['landscape']]))
        return M,D

    def uni_ring_2d(self,data):
        M = data['M']
        r = data['radius']
        theta = 2*np.pi*np.linspace(0,1-1/M,M)
        D = np.zeros((M,2))
        D[:,0] = r*np.cos(theta)+data['landscape']
        D[:,1] = r*np.sin(theta)+data['landscape']
        D = np.vstack((D,[0+data['landscape'],0+data['landscape']]))
        return M,D

    def one_close_ring_2d(self,data):
        M = data['M']
        R = data['radius']
        r = data['small-r']
        theta = 2*np.pi*np.linspace(0,1-1/M,M)
        D = np.zeros((M,2))
        D[:-1,0] = R*np.cos(theta[:-1])
        D[:-1,1] = R*np.sin(theta[:-1])
        D[-1,0] = r*np.cos(theta[-1])
        D[-1,1] = r*np.sin(theta[-1])
        D = np.vstack((D,[0+data['landscape'],0+data['landscape']]))
        return M,D

    
class PROB:
    def __init__(self) -> None:
        pass

    def norm_2d(self,data):
        X = data['D'][:,0]
        Y = data['D'][:,1]
        sf = data['sf']
        sp = data['sp']
        c = lambda z: np.exp(-z**2/(2*(sf**2+sp**2)))/(np.sqrt(2*np.pi*(sf**2+sp**2)))
        px = np.array(
            [c(x) for x in X]
        )
        py = np.array(
            [c(y) for y in Y]
        )
        return px*py

    def expo_2d(self,data):
        '''
        get the probability a scout can reach
        a territory of size r at location data['D']
        returns a numpy array for each colony
        '''
        X = data['D'][:data['M'],0]
        Y = data['D'][:data['M'],1]
        lam = data['lamp']
        r = data['fradius']
        p = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            f = lambda x,y: lam*np.exp(-lam*np.sqrt((x-X[i])**2+(y-Y[i])**2))
            h = lambda x: Y[i]-np.sqrt(r**2-(x-X[i])**2)
            g = lambda x: Y[i]+np.sqrt(r**2-(x-X[i])**2)
            p[i] = dblquad(f,X[i]-r,X[i]+r,h,g)[0]
        return p


def comp_vector(data):
    R = data['fradius']
    return np.repeat(np.pi*R**2,data['M']+1)

def competition_matrix(data):
    coords = data['D']
    # coords = np.vstack((coords,[0,0]))
    D = distance_matrix(coords,coords)
    R = data['fradius']
    D[D>R] = R
    A = lambda d: R**2*np.arccos(d/R) - d*np.sqrt(R**2-d**2) 
    Overlap = A(D)
    return Overlap
     
def start_values(M) -> np.array:
    E = 291
    L = 127
    P = 363
    N = 5293
    F = 648
    tmp = np.tile([E,L,P,N,F],M)
    # tmp = np.hstack((tmp,P,N,F,E,L,P,F,0))
    # tmp = np.hstack((tmp,np.zeros(8)))
    return tmp

def start_values_rnd(M) -> np.array:
    E = 172.98172512
    L = 56.14696041
    P = 94.97668089
    N = 1154.37465288
    F = 138.5249578
    scale = np.random.random((M,1))
    col = np.array([E,L,P,N,F])
    tmp = np.hstack((col*scale[0],col*scale[1]))
    for i in range(2,M):
        tmp = np.hstack((tmp,col*scale[i]))
        
    # tmp = np.tile([E,L,P,N,F],M)
    # tmp = np.hstack((tmp,P,N,F,E,L,P,F,0))
    tmp = np.hstack((tmp,np.zeros(8)))
    return tmp

def poly_start(z0,M) -> np.array:
    P = z0[-3]
    N = z0[-2]
    F = z0[-1]
    p0 = np.array([P,N,F,0,0,0,0,0])
    zout = np.zeros(M*5+8)
    zout[:5*M] = z0[:5*M]
    zout[5*M:] = p0
    return zout

def get_colony_size(z:np.ndarray,M:int,pre_p=True):
    C = np.zeros((M+1,z.shape[1]))
    if not pre_p:
        C[-1,:] = np.sum(z[-8:,:],axis=0)
        for i in range(M):
            C[i,:] = np.sum(z[i*5:(i+1)*5,:],axis=0)
    else:
        for i in range(M+1):
            C[i,:] = np.sum(z[i*5:(i+1)*5,:],axis=0)
    return C

def run_once(
        model,
        z0: np.array, 
        t_start: int = 0,
        t_end: int = 1000,
        method='Radau') -> np.array:
       
    sol = solve_ivp(model, [0,t_end], z0, method=method,dense_output=True)
    ## Dense output true
    t = np.linspace(t_start, t_end, 300)
    z = sol.sol(t)

    return t,z