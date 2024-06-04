import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from parameter import *

class ANT2:
    def __init__(self,
                    p,
                ) -> None:
        self.cst = p['cst']
        self.prd = p['prd']
        self.cf  = p['c']
        self.rq  = p['rq']
        self.re  = p['re']
        self.rl  = p['rl']
        self.rp  = p['rp']
        self.rn  = p['rn']
        self.rs  = p['rs']
        self.A   = p['A']
        self.mun = p['mun']
        self.muf = p['muf']
        self.Km  = p['Km']
        self.Kf  = p['Kf']
        pass

    def __call__(self, t, z) -> np.array:
        return self.de(z,t)

    def de(self,z,t=1) -> np.array:
        E = z[0]
        L = z[1]
        P = z[2]
        N = z[3]
        F = z[4]

        if N<0.01:
            return np.hstack((0,0,0,0,0)) 

        cfT = self.cf#*self.food(t)
        phiT = self.phi(cfT*F, L+1)
        B = E+L+P

        f = lambda x,y: 1/(1+(x/y)**2) 
        # f = lambda x,y: np.exp(-x**2/(y**2))
        g = lambda x: np.log(10/9)*(8/x)**2
        # h = lambda x,y: np.exp(-x/y)
        nb = g(N/B) 
        fn = f(F/N, self.Kf)
        # munT = self.mun*f(F/N,self.Kf)
        # print(mun)
        
        dE = phiT*self.rq - self.re*E*(1+nb)
        dL = self.re*E - self.rl*L*(phiT+nb) 
        dP = phiT*self.rl*L - P*(self.rp+self.rs+nb*self.rp)
        dN = self.rp*P - N*fn*(self.A+self.mun) 
        dF = N*fn*self.A - F*self.muf

        return np.hstack((dE,dL,dP,dN,dF)) 

    def phi(self,x,y):
        return (x/y)/(self.Km+x/y)
    
    def sigma(self,x,y):
        return 0.5*(1-np.tanh(x-y))
    
    def food(self,t):
        return 1 if self.cst else 1*(np.sin(t*np.pi/self.prd))+1
    
    def rho(self,x,y):
        return x/(x+y)
    
def death(t,y) -> np.array:
    return np.sum(y)-1
    
def start_values(m, t_end = 10000) -> np.array:
    tmp = [ 290.00085115,  126.02680938,  357.28317317, 5109.96031245,  701.0198493 ]
    # return tmp
    return [x*.5 for x in tmp]# fsolve(m.de,tmp)

def run_once(
        model,
        z0: np.array, 
        t_start: int = 0,
        t_end: int = 1000) -> np.array:
    
    # model = m(rs,c,cst,rnM)
    
    sol = solve_ivp(model, [0,t_end], z0,method="Radau", dense_output=True)
    ## Dense output true
    t = np.linspace(t_start, t_end, 300)
    z = sol.sol(t)

    return t,z

def plot_one_run(t:np.ndarray,z:np.ndarray,nplots:int = 0,title: str = 'Colony',xlab:str ='time'):
    fig = plt.figure(nplots)
    # g = plt.plot(t,z[0:-1,:].T,linewidth=2,alpha=0.75)
    g = plt.plot(t,z.T,linewidth=2,alpha=0.75)
    plt.ylim(bottom=0)    
    plt.legend(iter(g), ('Egg','Larva','Pupa','Nurse','Forager','P Pupa','P Worker'))
    plt.title(title)
    # plt.savefig('discrete-rs.png')
    return fig,nplots+1


def plot_sum(
        t:np.ndarray,
        z:np.ndarray,
        nplots:int = 0,
        title: str = 'Colony',
        xlab:str ='Years', 
        leg:str ='Formica'):
    fig = plt.figure(nplots)
    g = plt.plot(t/365,np.sum(z,axis=0),linewidth=2,alpha=0.75,label=leg)
    plt.ylim(bottom=0)    
    plt.legend()
    plt.ylabel('Colony size',fontsize=18)
    plt.xlabel(xlab,fontsize=18)
    plt.title(title,fontsize=24)
    plt.savefig('single-dynamic.png')
    return fig,nplots+1

def plot_2sum(t:np.ndarray,z1:np.ndarray,z2:np.ndarray,nplots:int = 0,title: str = 'Colony',xlab:str ='time'):
    fig = plt.figure(nplots)
    # g = plt.plot(t,z[0:-1,:].T,linewidth=2,alpha=0.75)
    plt.plot(t,np.sum(z1,axis=0),linewidth=3,alpha=0.5,label='constant')
    plt.plot(t,np.sum(z2,axis=0),linewidth=3,alpha=0.5,label='oscillating')
    plt.ylim(bottom=0)    
    plt.legend()
    plt.title(title)
    plt.xlabel(xlab)
    # plt.savefig('discrete-rs.png')
    return fig,nplots+1

def plot_2diff(t:np.ndarray,z1:np.ndarray,z2:np.ndarray,nplots:int = 0,title: str = 'Colony',xlab:str ='time'):
    fig = plt.figure(nplots)
    # g = plt.plot(t,z[0:-1,:].T,linewidth=2,alpha=0.75)
    T1 = np.sum(z1,axis=0)
    T2 = np.sum(z2,axis=0)
    A = np.vstack((T1,T2))
    plt.plot(t,np.diff(A,axis=0).T,linewidth=2,alpha=0.9,label='difference')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlab)
    # plt.savefig('discrete-rs.png')
    return fig,nplots+1

def plot_ratio(t:np.ndarray,z:np.ndarray,nplots:int = 0,title: str = 'Colony',xlab:str ='time'):
    fig = plt.figure(nplots)
    # g = plt.plot(t,z[0:-1,:].T,linewidth=2,alpha=0.75)
    g = plt.plot(t,np.divide(z[4,:],z[3,:]),linewidth=2,alpha=0.75,label='Foragers/Nurses')
    # plt.ylim(bottom=0)    
    plt.legend()
    plt.title(title)
    plt.xlabel(xlab)
    # plt.savefig('discrete-rs.png')
    return fig,nplots+1

def color_plot(cs,rss,u,title,xlab,ylab):
    u_min, u_max = np.abs(u).min(), np.abs(u).max()

    fig, ax = plt.subplots()

    c = ax.pcolormesh(cs, rss, u, cmap='RdBu', vmin=u_min, vmax=u_max)
    ax.set_title(title)
    ax.set_ylabel(xlab)
    ax.set_xlabel(ylab)
    # set the limits of the plot to the limits of the data
    ax.axis([cs.min(), cs.max(), rss.min(), rss.max()])
    fig.colorbar(c, ax=ax)
    return fig,ax


def plot_probs():
    pass