import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import utils as m
from parameter import data
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

import warnings
warnings.filterwarnings("error")
# plt.rcParams['text.usetex'] = True


def func(z,p) -> np.array:
    E = z[0]
    L = z[1]
    P = z[2]
    N = z[3]
    F = z[4]

    cf  = p['cf']
    rq  = p['rq']
    re  = p['re']
    rl  = p['rl']
    rp  = p['rp']
    rn  = p['rn']
    rs  = p['rs']
    A   = p['A']
    mun = p['mun']
    muf = p['muf']
    Km  = p['Km']
    Kf  = p['Kf']
    

    phi = lambda x,y: (x/y)/(Km+x/y)
    f = lambda x,y: 1/(1+(x/y)**2) #
    # f = lambda x,y: np.exp(-x**2/(y**2))
    g = lambda x: np.log(10/9)*(8/x)**2
    # h = lambda x,y: np.exp(-x/y)

    cfT = cf#*self.food(t)
    phiT = phi(cfT*F, L+1)
    B = E+L+P

    nb = g(N/B) 
    fn = A*f(F/N, Kf)
    munT = mun*f(F/N,Kf)
    
    dE = phiT*rq - re*E*(1+nb)
    dL = re*E - rl*L*(phiT+nb) 
    dP = phiT*rl*L - P*(rp+rs) - P*nb*rp
    dN = rp*P - N*(munT+fn+rn) 
    dF = N*fn - F*muf

    return np.hstack((dE,dL,dP,dN,dF))

def run_param(k,vals):
    f = lambda x,y: 1/(1+(x/y)**2) #
    # z0 = [1.5e2,3e2,5e2,1.2e4,4e3]
    z0 = [ 290.00085115, 126.02680938, 357.28317317, 5109.96031245, 701.0198493 ]
    
    M = len(vals)
    E = np.zeros((M,1))
    L = np.zeros((M,1))
    P = np.zeros((M,1))
    N = np.zeros((M,1))
    F = np.zeros((M,1))
    col_max = 0
    fnmax = (0,0)
    for i in range(M):
        data[k] = vals[i]
        try:
            E[i],L[i],P[i],N[i],F[i] = fsolve(func, z0, args=data)
            if (
                i>0 and np.sum([E[i],L[i],P[i],N[i],F[i]]) > 5 and (
                np.sum([E[i],L[i],P[i],N[i],F[i]]) < 0.97*np.sum([E[i-1],L[i-1],P[i-1],N[i-1],F[i-1]]) or
                0.97* np.sum([E[i],L[i],P[i],N[i],F[i]])>np.sum([E[i-1],L[i-1],P[i-1],N[i-1],F[i-1]]) or 
                np.sum([E[i],L[i],P[i],N[i],F[i]]) < 0
                )
            ):
                mdl = m.ANT2(data)
                sol = solve_ivp(mdl, [0,365*50], z0, method="Radau", dense_output=True)
                t = np.linspace(0, 365*50, 300)
                z = sol.sol(t)
                E[i],L[i],P[i],N[i],F[i] = z[:,-1]
        except:
            mdl = m.ANT2(data)
            sol = solve_ivp(mdl, [0,365*50], z0, method="Radau", dense_output=True)
            t = np.linspace(0, 365*50, 300)
            z = sol.sol(t)
            E[i],L[i],P[i],N[i],F[i] = z[:,-1]
        
            
        # if np.sum([E[i],L[i],P[i],N[i],F[i]]) > col_max:
        #     col_max = np.sum([E[i],L[i],P[i],N[i],F[i]]) 
        #     fnmax = (data['A']*f(F[i]/N[i], data['Kf']),i)
        if data['rp']*P[i] > col_max:
            col_max = data['rp']*P[i]
            fnmax = (f(F[i]/N[i], data['Kf']),i)
    
    return np.hstack((E,L,P,N,F)).T, fnmax

@np.vectorize
def c_switch(key1,key2,val1,val2):
    # z0 = [1.5e2,3e2,5e2,1.2e4,4e3]
    z0 = [ 290.00085115, 126.02680938, 357.28317317, 5109.96031245, 701.0198493 ]
    data[key1] = val1
    data[key2] = val2
    # mdl = m.ANT2(data)
    # sol = solve_ivp(mdl, [0,100000], z0, dense_output=True)
    # t = np.linspace(0, 100000, 300)
    # z = sol.sol(t)
    # z0 = z[:,-1]
    

    try:
        E,L,P,N,F = fsolve(func, z0, args=data)
    except:
        # print(vals[i],i)
        # E,L,P,N,F = (np.nan,np.nan,np.nan,np.nan,np.nan)
        mdl = m.ANT2(data)
        sol = solve_ivp(mdl, [0,5000], z0, dense_output=True)
        t = np.linspace(0, 5000, 300)
        z = sol.sol(t)
        E,L,P,N,F  = z[:,-1]            
    
    return np.sum(np.hstack((E,L,P,N,F)))

def just_switch(key1,key2,vals1,vals2,title,nplots=0,xlab=''):
    phi = lambda x,y: np.divide(np.divide(x,y),data['Km']+np.divide(x,y))
    g = lambda x: np.log(10/9)*(8/x)**2
    fig, axs = plt.subplots()
    # fig1 = plt.figure(nplots+1)
    max_demo = np.zeros((5,len(vals2)))
    val_lab = []
    vind = 0
    for v in vals2:
        data[key2] = v
        if v!=0 and key2!='c':
            vleg = int(1/v)
        elif key2=='rs' and v==0:
            vleg = 'no raiding'
        else:
            vleg = v
        val_lab.append(f"{vleg}")
        z,fnmax = run_param(key1,vals1)
        c = np.sum(z,axis=0)
        
        axs.plot(vals1,c,linewidth=2,alpha=0.75,label=f'{key2}={vleg}')
        if c[fnmax[1]]>2:
            axs.scatter(vals1[fnmax[1]],c[fnmax[1]])
        else:
            axs.scatter(vals1[fnmax[1]],c[fnmax[1]],alpha=0)
        vind+=1
        print(fnmax,v,z[4][fnmax[1]]/z[3][fnmax[1]],vals1[fnmax[1]])
  
    axs.set_title(title,fontsize=24)
    axs.set_ylabel('Total colony size',fontsize=18)
    axs.legend() 
    axs.set_xlabel('Switch rate A',fontsize=18)
    # plt.savefig('single.png')
    plt.tight_layout()
    plt.savefig(f'single-{key2}.pdf',dpi=199)
    return 

def two_params(key1,key2,vals1,vals2,title,nplots=0,xlab=''):
    phi = lambda x,y: np.divide(np.divide(x,y),data['Km']+np.divide(x,y))
    g = lambda x: np.log(10/9)*(8/x)**2
    fig, axs = plt.subplots(2,2)
    # fig1 = plt.figure(nplots+1)
    max_demo = np.zeros((5,len(vals2)))
    val_lab = []
    vind = 0
    for v in vals2:
        data[key2] = v
        if v!=0 and key2!='c':
            vleg = int(1/v)
        elif key2=='rs' and v==0:
            vleg = 'no raiding'
        else:
            vleg = v
        val_lab.append(f"{vleg}")
        z,fnmax = run_param(key1,vals1)
        c = np.sum(z,axis=0)

        phiC = phi(z[4],z[1]+1)
        gC = g(np.divide(z[3],z[0]+z[1]+z[2]))
        brood_prob = (data['re']+data['rl']*phiC+data['rp'])/(data['re']+data['rl']*phiC+data['rp'] + gC*(data['re'] + data['rl']+data['rp']))
        
        # if Ad==np.max(Adenom) or Ad==np.min(Adenom):
        axs[0,0].plot(vals1,c,linewidth=2,alpha=0.75,label=f'{vleg}')
        # axs[0,1].plot(vals1,np.divide(z[4],z[3]),linewidth=2,alpha=0.5,label=f'{vleg}') # forage to nurse
        # axs[1,0].plot(vals1,np.divide(z[3],z[0]+z[1]+z[2]),linewidth=2,alpha=0.5,label=f'{vleg}') # nurse to brood
        axs[1,1].plot(vals1,vals1/(vals1+data['mun']),linewidth=2,alpha=0.75,label=f'{vleg}')
        # axs[1,1].plot(vals1,np.divide(z[4],z[1]+1),linewidth=2,alpha=0.5,label=f'{vleg}') # forage to larva+queen
        # axs[1,0].plot(vals1,phi(z[4],z[1]+1))
        axs[0,1].plot(vals1,brood_prob,alpha=0.75)
        if c[fnmax[1]]>2:
            axs[0,0].scatter(vals1[fnmax[1]],c[fnmax[1]])
            axs[1,1].scatter(vals1[fnmax[1]], vals1[fnmax[1]]/(vals1[fnmax[1]]+data['mun']))
            # axs[1,0].scatter(vals1[fnmax[1]],phi(z[4,fnmax[1]],z[1,fnmax[1]]+1))
            axs[0,1].scatter(vals1[fnmax[1]],brood_prob[fnmax[1]])
            # axs[0,1].scatter(vals1[fnmax[1]],z[4][fnmax[1]]/z[3][fnmax[1]] ) # forage to nurse
            # axs[1,0].scatter(vals1[fnmax[1]], z[3][fnmax[1]]/ ( z[0][fnmax[1]]+z[1][fnmax[1]]+z[2][fnmax[1]] )) # nurse to brood
            # axs[1,1].scatter(vals1[fnmax[1]], z[4][fnmax[1]]/ ( z[1][fnmax[1]]+1 )) # forage to larva+queen
            
        else:
            axs[0,0].scatter(vals1[fnmax[1]],c[fnmax[1]],alpha=0)
            axs[1,1].scatter(vals1[fnmax[1]], vals1[fnmax[1]]/(vals1[fnmax[1]]+data['mun']),alpha=0)
            # axs[1,0].scatter(vals1[fnmax[1]],phi(z[4,fnmax[1]],z[1,fnmax[1]]+1),alpha=0)
            axs[0,1].scatter(vals1[fnmax[1]],brood_prob[fnmax[1]],alpha=0)
            # axs[0,1].scatter(vals1[fnmax[1]],z[4][fnmax[1]]/z[3][fnmax[1]],alpha=0 )
            # axs[1,0].scatter(vals1[fnmax[1]], z[3][fnmax[1]]/ ( z[0][fnmax[1]]+z[1][fnmax[1]]+z[2][fnmax[1]] ),alpha=0)
            # axs[1,1].scatter(vals1[fnmax[1]], z[4][fnmax[1]]/ ( z[1][fnmax[1]]+1 ),alpha=0)
            
        if np.sum(z[:,fnmax[1]])>2:
            max_demo[:,vind] = z[:,fnmax[1]]
        else:
            max_demo[:,vind] = np.zeros((5,1))
        vind+=1
        print(fnmax,v,z[4][fnmax[1]]/z[3][fnmax[1]],vals1[fnmax[1]])

    bottom = 0
    colony_weights = {
        'Egg': max_demo[0,:]/np.sum(max_demo,axis=0),
        'Larva': max_demo[1,:]/np.sum(max_demo,axis=0),
        'Pupa': max_demo[2,:]/np.sum(max_demo,axis=0),
        'Nurse': max_demo[3,:]/np.sum(max_demo,axis=0),
        'Forager': max_demo[4,:]/np.sum(max_demo,axis=0)
    }
    
    for bar_lab, weight_count in colony_weights.items():
        axs[1,0].bar(val_lab,weight_count,label=bar_lab,width=0.5,bottom=bottom)
        bottom+=weight_count
  
    # ax0.set_xlabel('Switch rate A',fontsize=18)
    plt.suptitle(title,fontsize=24)
    axs[0,0].set_ylabel('total colony size',fontsize=18)
    # axs[0,1].set_ylabel('Forager to nurse ratio',fontsize=18)
    # axs[1,0].set_ylabel('Nurse to brood ratio',fontsize=18)
    # axs[1,1].set_ylabel('Forager to Larva and Queen ratio',fontsize=18)
    axs[1,1].set_ylabel('probability nurse to forager',fontsize=18)
    axs[1,0].set_ylabel('Relative abundance at optimum',fontsize=18)
    axs[0,1].set_ylabel('probability brood reaches adulthood',fontsize=18)
    # axs[1,0].set_ylabel(r'$\phi$',fontsize=18)
    axs[0,0].legend() 
    # axs[0,1].legend() 
    # axs[0,1].legend()
    axs[1,0].legend()
    # axs[1,1].legend()
    axs[0,0].set_xlabel('Switch rate A',fontsize=18)
    axs[1,1].set_xlabel('Switch rate A',fontsize=18)
    axs[1,0].set_xlabel(xlab,fontsize=18)
    
    # axs[1,2].set_xlabel('Switch rate A',fontsize=18)
    
    # ax0.set_title(title,fontsize=24)
    
    # ax0.tight_layout()
    # ax1.tight_layout()
    return nplots+1

def two_params_oneplot(key1,key2,vals1,vals2,title,nplots=0):
    fig, axs = plt.subplots()

    val_lab = []
    vind = 0

    X = []
    Y = []
    Z = []
    
    for v in vals2:
        data[key2] = v
        if v!=0 and key2!='cf':
            vleg = int(1/v)
        elif key2=='rs' and v==0:
            vleg = 'no raiding'
        else:
            vleg = v
        if vind == 0:
            cmin = vleg
            cmax = vleg
        val_lab.append(f"{vleg}")
        z,fnmax = run_param(key1,vals1)
        c = np.sum(z,axis=0)
        
        
        if c[fnmax[1]]>2:
            if vleg > cmax:
                cmax = vleg
            if vleg < cmin:
                cmin = vleg
            # print(v)
            # print(vleg)
            X.append(vals1[fnmax[1]])
            Y.append(vals1[fnmax[1]]/(vals1[fnmax[1]]+data['mun']))
            Z.append(vleg)
            
        # elif vind > 10:
        #     break
        # else:
        #     im =axs.scatter(vals1[fnmax[1]], vals1[fnmax[1]]/(vals1[fnmax[1]]+data['mun']),alpha=0)
        vind+=1
        print(fnmax,v,z[4][fnmax[1]]/z[3][fnmax[1]],vals1[fnmax[1]])

    im = axs.scatter(X, Y, c=Z, alpha=.8,cmap='viridis')
    # ax0.set_xlabel('Switch rate A',fontsize=18)
    plt.suptitle(title,fontsize=24)
    axs.set_ylabel('probability nurse becomes a forager',fontsize=18)
    axs.set_xlabel('Switch rate A',fontsize=18)
    im.set_clim([cmin,cmax])
    clb = fig.colorbar(im, ax=axs)
    clb.ax.set_title('$\mu_N$',fontsize=14)

    return nplots+1

def one_param(key,vals,xlab,nplots=0):
    z,fnmax = run_param(key,vals)
    
    tstring = f'Max colony size at {key}={vals[fnmax[1]]:.3f}.'# \
    #             Effective switch rate = {fnmax[0][0]:.5f}. \
    #             F/N = {z[4][fnmax[1]]/z[3][fnmax[1]]:.3f}'
    # tstring = 'test'

    # _,nplots = m.plot_one_run(vals,z, nplots=nplots,title=tstring,xlab=key)
    fig,nplots = m.plot_sum(vals,z, nplots=nplots,title=tstring,xlab=key)
    plt.savefig('sum.png')
    # _,nplots = m.plot_ratio(vals,z, nplots=nplots,title=tstring,xlab=key)
    
    return nplots+1

def main():
    nplots = 0
    
    key1 = 'A'
    key2 = 'c'
    vals1 = np.linspace(0.9,.001,500)
    # vals2 = np.hstack(1/np.arange(50,550,50))
    # vals2 = np.arange(50,501,50)
    vals2 = np.hstack((2,5,10,20,50,100))
    # vals2 = np.hstack((0,1/np.array((14,30,60,100,180,365))))
    # vals2 = np.hstack((1/np.array((7,14,30,60,100,180,365))))
    # vals2 = np.hstack((1/np.array((1,7,14,23,30))))
    # vals2 = [1/7]
    # vals2 = np.hstack((1/np.arange(20,500,10),np.linspace(1/3.,1/20.,45)))
    # vals2 = np.linspace(1,.001,500)
    
    title = 'Single Formica colony'
    nplots = just_switch(key1,key2,vals1,vals2,title,nplots,'Switch rate A')
    # nplots = two_params_oneplot(key1,key2,vals1,vals2,title,nplots)

    # one_param(key1,vals1,key1,nplots)


    plt.show()

if __name__=='__main__':
    main()