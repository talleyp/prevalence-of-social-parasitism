import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import utils as m
from parameter import data

#### using tex style

pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.
style_path = "paper.mplstyle"
plt.style.use("paper.mplstyle")
mpl.rcParams['text.usetex'] = True

jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
              "CQG": {"onecol": 374.*pt}, # CQG is only one column
              # Add more journals below. Can add more properties to each journal
             }

my_width = jour_sizes["PRD"]["twocol"]
# Our figure's aspect ratio
golden = (1 + 5 ** 0.5) / 2

### end tex style


def run_param(k,vals,p):
    f = lambda x,y: 1/(1+(x/y)**2) 
    z0 = [ 217.82156039, 120.44290952, 128.2751778, 1867.79642731, 226.44118306]
    
    M = len(vals)
    E = np.zeros((M,1))
    L = np.zeros((M,1))
    P = np.zeros((M,1))
    N = np.zeros((M,1))
    F = np.zeros((M,1))
    Fmax = 0
    max_ind = 0
    Fit = np.zeros(M)

    
    for i in range(M):
        p[k] = vals[i]
        mdl = m.ANT(p)

        mdl = m.ANT(p)
        sol = solve_ivp(mdl, [0,365*50], z0, method="Radau", dense_output=True)
        t = np.linspace(0, 365*50, 300)
        z = sol.sol(t)
        E[i],L[i],P[i],N[i],F[i] = z[:,-1]

        alp = m.RESPONSE(p['alp']).alpha
        alpha = 1-alp(N[i,0]+F[i,0],0)

        Fit[i] = p['rp']*P[i,0]*alpha
            
        if Fit[i] > Fmax:
            Fmax = Fit[i]
            max_ind = i
    
    return Fit,max_ind

def just_switch(key1,key2,vals1,vals2,title,xlab=''):
    fig, axs = plt.subplots()
    val_lab = []
    vind = 0
    p = data.copy()
    p['rs'] = 0
    for v in vals2:
        p[key2] = v
        if v!=0 and key2!='c' and key2!='rhoq':
            vleg = int(1/v)
        elif key2=='rs' and v==0:
            vleg = 'no raiding'
        else:
            vleg = v
        val_lab.append(f"{vleg}")
        F,max_ind = run_param(key1,vals1,p)
        
        axs.plot(vals1,F,linewidth=2,alpha=0.75,label=f'{key2}={vleg}')
        if F[max_ind]>1e-5:
            axs.scatter(vals1[max_ind],F[max_ind])
        else:
            axs.scatter(vals1[max_ind],F[max_ind],alpha=0)
        vind+=1
        print(v,vals1[max_ind])
  
    axs.set_ylabel('Fitness')
    axs.legend() 
    axs.set_xlabel('Switch rate A') 
    plt.tight_layout()
    plt.savefig(f'../figs/single-alpha.pdf',dpi=199)
    return fig


def main():
    key1 = 'A'
    key2 = 'c'
    vals1 = np.linspace(1,.001,500)
    vals2 = np.array([5,8,10,15,25,50,100])
    title = 'Single Formica colony'
    fig = just_switch(key1,key2,vals1,vals2,title,'Switch rate A')
    plt.show()

if __name__=='__main__':
    main()

# 5 0.30530460921843683
# 8 0.29129058116232465
# 10 0.2872865731462926
# 15 0.2812805611222444
# 25 0.2752745490981964
# 50 0.2712705410821643
# 100 0.2692685370741482