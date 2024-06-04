import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import utils as m
from parameter import data
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d



def run_param(k,vals):
    z0 = m.start_values()
    
    M = len(vals)

    C = np.zeros((M,len(z0)))

    col_max = [0,0]
    max_ind = [0,0]
    maxTime = 365*1000
    for i in range(M):
        data[k] = vals[i]
        mdl = m.ANT(data)
        try:
            C[i,:] = fsolve(mdl.full_de, z0, args=data)
            
            if (
                i>0 and np.sum(C[i,5:]) > 5 and (
                np.sum(C[i,:]) < 0.97*np.sum(C[i-1,5:]) or
                0.97* np.sum(C[i,5:])>np.sum(C[i-1,5:]) or 
                np.sum(C[i,5:]) < 0
                )
            ):
                sol = solve_ivp(mdl, [0,maxTime], z0, method="Radau", dense_output=True)
                t = np.linspace(0, maxTime, 300)
                z = sol.sol(t)
                C[i,:] = z[:,-1]
        except:
            sol = solve_ivp(mdl, [0,maxTime], z0, method="Radau", dense_output=True)
            t = np.linspace(0, maxTime, 300)
            z = sol.sol(t)
            C[i,:] = z[:,-1]

        if np.sum(C[i,0:5]) > col_max[0]:
            col_max[0] = np.sum(C[i,0:5]) 
            max_ind[0] = i
        if np.sum(C[i,5:]) > col_max[1]:
            col_max[1] = np.sum(C[i,5:]) 
            max_ind[1] = i

    return C.T, max_ind

def just_switch(key1,key2,vals1,vals2,title,nplots=0,xlab=''):
    fig, axs = plt.subplots()
    max_demo_f = np.zeros((5,len(vals2)))
    max_demo_p = np.zeros((8,len(vals2)))
    val_lab = []
    vind = 0
    for v in vals2:
        data[key2] = v
        if v!=0 and (key2!='c' and key2!='M'):
            vleg = int(1/v)
        elif key2=='rs' and v==0:
            vleg = 'no raiding'
        else:
            vleg = v
        val_lab.append(f"{vleg}")
        z,max_ind = run_param(key1,vals1)
        
        cf = np.sum(z[0:5],axis=0)
        cp = np.sum(z[5:],axis=0)

        axs.plot(vals1,cp,linewidth=2,alpha=.75,label=f'{key2}={vleg}')

        if cp[max_ind[1]]>2:
            axs.scatter(vals1[max_ind[1]],cp[max_ind[1]])
            max_demo_p[:,vind] = z[5:,max_ind[1]]
            max_demo_f[:,vind] = z[0:5,max_ind[1]]
        else:
            axs.scatter(vals1[max_ind[1]],cp[max_ind[1]],alpha=0)
            max_demo_p[:,vind] = z[5:,max_ind[1]]
            max_demo_f[:,vind] = z[0:5,max_ind[1]]
        vind+=1
        print(v, vals1[max_ind[1]])

  
    axs.set_title(title,fontsize=24)
    axs.set_ylabel('Total colony size',fontsize=18)
    axs.legend()
    axs.set_xlabel(xlab,fontsize=18)
    plt.tight_layout()
    plt.savefig('two-species.pdf',dpi=199)

    return nplots+1

def two_params(key1,key2,vals1,vals2,title,nplots=0,xlab=''):
    fig, axs = plt.subplots(1,2)
    max_demo_f = np.zeros((5,len(vals2)))
    max_demo_p = np.zeros((8,len(vals2)))
    val_lab = []
    vind = 0
    for v in vals2:
        data[key2] = v
        if v!=0 and (key2!='cf' and key2!='M'):
            vleg = int(1/v)
        elif key2=='rs' and v==0:
            vleg = 'no raiding'
        else:
            vleg = v
        val_lab.append(f"{vleg}")
        z,max_ind = run_param(key1,vals1)
        
        cf = np.sum(z[0:5],axis=0)
        cp = np.sum(z[5:],axis=0)

        axs[0].plot(vals1,cf,linewidth=2,alpha=.75,label=f'{vleg}')
        axs[1].plot(vals1,cp,linewidth=2,alpha=.75,label=f'{vleg}')

        if cp[max_ind[1]]>2:
            axs[0].scatter(vals1[max_ind[1]],cf[max_ind[1]])
            axs[1].scatter(vals1[max_ind[1]],cp[max_ind[1]])
            max_demo_p[:,vind] = z[5:,max_ind[1]]
            max_demo_f[:,vind] = z[0:5,max_ind[1]]
        else:
            axs[0].scatter(vals1[max_ind[1]],cf[max_ind[1]],alpha=0)
            axs[1].scatter(vals1[max_ind[1]],cp[max_ind[1]],alpha=0)
            max_demo_p[:,vind] = z[5:,max_ind[1]]
            max_demo_f[:,vind] = z[0:5,max_ind[1]]
        vind+=1
        print(v, vals1[max_ind[1]])

  
    plt.suptitle(title,fontsize=24)
    axs[0].set_ylabel('Total colony size',fontsize=18)
    axs[0].set_title('Formica colony',fontsize=18)
    axs[1].set_ylabel('Total colony size',fontsize=18)
    axs[0].legend()
    axs[1].set_title('Polyergus colony',fontsize=18)
    axs[0].set_xlabel(xlab,fontsize=18)
    axs[1].set_xlabel(xlab,fontsize=18)

    return nplots+1

def one_param(key,vals,nplots=0):

    z,fnmax = run_param(key,vals)

    _,nplots = m.plot_both_sum(vals,z, nplots=nplots,title=f"Effect of switch rate on landscape with {data['M']} colonies",xlab=key)
    return nplots+1

def main():
    nplots = 0
    
    key1 = 'B'
    key2 = 'c'
    vals1 = np.linspace(.9,.001,200)
    vals2 = np.hstack((2,5,10,20,50,100))

    title = 'Polyergus Colony'

    nplots = just_switch(key1,key2,vals1,vals2,title,nplots,'Switch rate B')

    plt.show()

if __name__=='__main__':
    main()