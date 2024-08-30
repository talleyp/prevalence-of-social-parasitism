import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import utils as m
from parameter import data
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.special import expit


def run_param(k,vals):
    z0 = m.start_values()
    
    M = len(vals)

    F = np.zeros(M)

    Fmax = 0
    max_ind = 0
    maxTime = 365*1000
    for i in range(M):
        data[k] = vals[i]
        mdl = m.ANT(data)
        sol = solve_ivp(mdl, [0,maxTime], z0, method="Radau", dense_output=True)
        t = np.linspace(0, maxTime, 1000)
        z = sol.sol(t)
        zavg = np.mean(z[:,-100:],axis=1)
        
        A = np.sum(zavg[[6,7,11,12]])
        a = 1-m.PRESPONSE().alpha(A)
        F[i] = data['rp']*zavg[5]*a

        if F[i] > Fmax:
            Fmax = F[i]
            max_ind = i
    return F,max_ind

def just_switch(key1,key2,vals1,vals2,title,nplots=0,xlab=''):
    fig, axs = plt.subplots()
    # fig1 = plt.figure(nplots+1)
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
        F,max_ind = run_param(key1,vals1)

        axs.plot(vals1,F,linewidth=2,alpha=.75,label=f'{key2}={vleg}')
        # plt.show()

        if F[max_ind]>1:
            axs.scatter(vals1[max_ind],F[max_ind])
        #     max_demo_p[:,vind] = F[max_ind]
        #     max_demo_f[:,vind] = F[max_ind]
        else:
            axs.scatter(vals1[max_ind],F[max_ind],alpha=0)
        #     max_demo_p[:,vind] = z[5:,max_ind[1]]
        #     max_demo_f[:,vind] = z[0:5,max_ind[1]]
        vind+=1
        print(v, vals1[max_ind])

  
    # ax0.set_xlabel('Switch rate A',fontsize=18)
    axs.set_title(title,fontsize=24)
    axs.set_ylabel('Fitness',fontsize=18)
    axs.legend()
    axs.set_xlabel(xlab,fontsize=18)
    plt.tight_layout()
    plt.savefig('test.pdf',dpi=199)

    return nplots+1


def main():
    nplots = 0
    
    key1 = 'B'
    # key2 = 'c'
    key2 = 'M'
    vals1 = np.linspace(1,.001,500)
    # vals2 = np.array([5,8,10,15])
    vals2 = np.array([10,11,12,13,15,20,30])

    title = 'Polyergus Colony'

    nplots = just_switch(key1,key2,vals1,vals2,title,nplots,'Switch rate B')

    plt.show()

if __name__=='__main__':
    main()
