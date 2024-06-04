import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

class ANT:
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
        self.rs  = p['rs']
        self.A   = p['A']
        self.mun = p['mun']
        self.muf = p['muf']
        self.mur = p['mur']
        self.Km  = p['Km']
        self.Kf  = p['Kf']
        self.M   = p['M']
        self.B   = p['B']
        pass

    def __call__(self, t, z) -> np.array:
        Zf = z[:5]
        Zp = z[5:]
        self.raidT = self.raid(Zf[2],Zp[-1]) 
        if np.sum(z)<1:
            return np.repeat(0,13)
        elif np.sum(z[5:])<1:
            return np.hstack((self.f_de(Zf,t),np.repeat(0,8)))

        return np.hstack((self.f_de(Zf,t),self.p_de(Zp,t)))
    
    def full_de(self,z) -> np.array:
        '''For implicit solving purposes'''
        Zf = z[:5]
        Zp = z[5:]
        self.raidT = self.raid(Zf[2],Zp[-1]) 

        return np.hstack((self.f_de(Zf),self.p_de(Zp)))
    
    
    def f_de(self,z,t=1) -> np.array:
        E = z[0]
        L = z[1]
        P = z[2]
        N = z[3]
        F = z[4]

        cfT = self.cf
        phiT = self.phi(cfT*F, L+1)
        B = E+L+P

        f = lambda x,y: 1/(1+(x/y)**2) 
        g = lambda x: np.log(10/9)*(8/x)**2
        nb = g(N/B) 
        fn = f(F/N, self.Kf)

        
        dE = phiT*self.rq - self.re*E*(1+nb)
        dL = self.re*E - self.rl*L*(phiT+nb) 
        dP = phiT*self.rl*L - P*self.rp*(1+nb) - self.raidT/self.M
        dN = self.rp*P - N*fn*(self.mun+self.A) 
        dF = N*fn*self.A - F*self.muf

        return np.hstack((dE,dL,dP,dN,dF)) 

    def p_de(self,z,t=1) -> np.array:
        P = z[0]
        N = z[1]
        F = z[2]

        E = z[3]
        L = z[4]
        U = z[5]
        S = z[6]
        R = z[7]

        cfT = self.cf
        phiT = self.phi(cfT*F, L+1)
        B = E+L+P+U

        f = lambda x,y: 1/(1+(x/y)**2) 
        g = lambda x: np.log(10/9)*(8/x)**2
        nb = g(N/B) 
        fn = f(F/(N+S+R), self.Kf)


        dP = self.raidT  - P*self.rp*(1+nb) 
        dN = P*self.rp - N*fn*(self.mun+self.A) 
        dF = N*fn*self.A - F*self.muf
        
        dE = phiT*self.rq - self.re*E*(1+nb)
        dL = self.re*E - self.rl*L*(phiT+nb) 
        dU = phiT*self.rl*L - U*self.rp*(1+nb)
        dS = self.rp*U - S*fn*(self.mun+self.B)
        dR = S*fn*self.B - self.mur*R
        
        return np.hstack((dP,dN,dF,dE,dL,dU,dS,dR)) 

    def phi(self,x,y):
        return (x/y)/(self.Km+x/y)
    
    def raid(self,s,p):
        return self.rs*s*p

    
def death(t,y) -> np.array:
    return np.sum(y)-1
    
def start_values() -> np.array:
    E = 3e2
    L = 1.3e2
    P = 4e2
    N = 5.3e3
    F = 6.5e2
    tmp = np.hstack((E,L,P,N,F,
                     P,N,F,
                     E,L,P,F,F)
                    ) 
    return tmp/2.

def run_once(
        model,
        z0: np.array, 
        t_start: int = 0,
        t_end: int = 1000) -> np.array:
        
    sol = solve_ivp(model, [0,t_end], z0, method='Radau',dense_output=True)
    ## Dense output true
    t = np.linspace(t_start, t_end, 300)
    z = sol.sol(t)

    return t,z

def plot_one_run(t:np.ndarray,z:np.ndarray,nplots:int = 0,title: str = 'Colony',xlab:str ='time'):
    fig = plt.figure(nplots)
    # g = plt.plot(t,z[0:-1,:].T,linewidth=2,alpha=0.75)
    g = plt.plot(t,z.T,linewidth=2,alpha=0.75)
    plt.ylim(bottom=0)    
    plt.legend(iter(g), ('FC Egg','FC Larva','FC Pupa','FC Nurse','FC Forager',
                         'PC F Pupa','PC F Nurse','PC F Forager',
                         'PC Egg','PC Larva','PC Pupa','PC Slaver', 'PC Raider'))
    plt.title(title)

    nplots+=1
    FC = z[0:5,:]
    fig2 = plt.figure(nplots)

    g = plt.plot(t,FC.T,linewidth=2,alpha=0.75)
    plt.ylim(bottom=0)    
    plt.legend(iter(g), ('Egg','Larva','Pupa','Nurse','Forager'))
    plt.title('Formica colony')

    nplots+=1
    PC = z[5:,:]
    fig2 = plt.figure(nplots)

    g = plt.plot(t,PC.T,linewidth=2,alpha=0.75)
    plt.ylim(bottom=0)    
    plt.legend(iter(g), ('F Pupa','F Nurse','F Forager',
                         'P Egg','P Larva','P Pupa','Slaver','Raider'))
    plt.title('Polyergus colony')

    # plt.savefig('discrete-rs.png')
    return fig,nplots+1


def plot_sum(
        t:np.ndarray,
        z:np.ndarray,
        nplots:int = 0,
        title: str = 'Colony',
        xlab:str ='time', 
        leg:str ='Total colony'):
    fig = plt.figure(nplots)
    # g = plt.plot(t,z[0:-1,:].T,linewidth=2,alpha=0.75)
    g = plt.plot(t,np.sum(z,axis=0),linewidth=4,alpha=1)
    # plt.ylim(bottom=0)    
    plt.ylabel('Colony size')
    plt.xlabel(xlab)
    plt.title(title)
    # plt.savefig('discrete-rs.png')
    return fig,nplots+1

def plot_both_sum(t:np.ndarray,
                  z:np.ndarray,
                  nplots:int = 0,
                  title: str = 'Colony',
                  xlab:str ='Years', 
                leg:str ='Total colony'):
    fig = plt.figure(nplots)
    # g = plt.plot(t,z[0:-1,:].T,linewidth=2,alpha=0.75)
    g = plt.plot(t/365,np.sum(z[0:5],axis=0),linewidth=2,alpha=1,label='Formica')
    plt.plot(t/365,np.sum(z[5:,:],axis=0),linewidth=2,alpha=1,label='Polyergus')
    plt.ylim(bottom=0)    
    plt.ylabel('Colony size',fontsize=18)
    plt.xlabel(xlab,fontsize=18)
    plt.title(title,fontsize=24)
    plt.legend()

    plt.savefig('two-species-dynamic.png')
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