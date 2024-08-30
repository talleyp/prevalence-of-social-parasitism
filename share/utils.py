import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import expit
from parameter import *

class RESPONSE:
    def __init__(self,r):
        self.tbase = 1500
        self.aspan = 1
        self.Kr    = 0.03
        self.Ka    = 2**7
        if r==0:
            # print('Non-Responsive to Raiding')
            self.theta=self.t1
        elif r==1:
            print("Raiding induces gyne production")
            self.theta=self.t2
        else:
            print("Raiding inhibits gyne production")
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

class ANT:
    def __init__(self,
                    p,
                ) -> None:
        self.c    = p['c']
        self.rq   = p['rq']
        self.re   = p['re']
        self.rl   = p['rl']
        self.rp   = p['rp']
        self.rs   = p['rs']
        self.A    = p['A']
        self.mun  = p['mun']
        self.muf  = p['muf']
        self.Km   = p['Km']
        self.Kf   = p['Kf']
        self.alp  = RESPONSE(p['alp']).alpha 
        self.Kb   = p['Kb']
        self.eta  = p['eta']
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
        
        alpha = self.alp(N+F,self.rs)
        # print('Percent gynes',1-alpha)

        phiT = self.phi(self.c*F, L*(alpha+self.eta*(1-alpha))+1)
        B = E+(alpha+self.eta*(1-alpha))*(L+P)

        f = lambda x,y: 1/(1+(x/y)**2) 
        g = lambda x: np.log(10/9)*(self.Kb/x)**2
        nb = g(N/B) 
        fn = f(F/N, self.Kf)

        dE = phiT*self.rq - self.re*E*(1+nb)
        dL = self.re*E - self.rl*L*(phiT+nb) 
        dP = phiT*self.rl*L - P*(self.rp+self.rs+nb*self.rp)
        dN = self.rp*P*alpha - N*fn*(self.A+self.mun) 
        dF = N*fn*self.A - F*self.muf

        # print(1/(fn*(self.A+self.mun)) )

        return np.hstack((dE,dL,dP,dN,dF)) 

    def phi(self,x,y):
        return (x/y)/(self.Km+x/y)
    
    
def start_values() -> np.array:
    tmp = np.array([180, 57, 100, 1200, 150. ])
    # return tmp
    return tmp

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