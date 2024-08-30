import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.optimize import curve_fit
import json
from parameter import data

def deviance(X, y, model):
    return 2*metrics.log_loss(y, model.predict_proba(X), normalize=False)

def logifunc(x,x0,k):
    return 1 / (1 + np.exp(-(x-x0)/k))


def table_row(Z,Rs,R0,R1):
    out = np.zeros(4)
    out[-1]=Z[-1]
    n_inner = (Rs <= R0).sum()
    n_middle = (Rs<=R1).sum()-n_inner
    n_outer = len(Rs)-n_inner-n_middle
    out[0] = n_inner
    out[1] = n_middle
    out[2] = n_outer
    return out

mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["figure.figsize"] = (8,4)



def make_table(files,R0,R1):
    R_dict = {0:[],1:[]}
    counter = 0
    table = np.zeros((len(files),4))
    for file in files:
        D = np.loadtxt(file,delimiter=',')
        
        R = D-offset
        R = np.sqrt( np.sum(np.square(R),axis=1) )
        # R_list.append(R)
        Zfile = file.replace("D","Z")
        Z = np.loadtxt(Zfile,delimiter=',')
        Z_outcome = [1 if x>500 else 0 for x in Z[:-1]]
        Z_outcome.append(1 if Z[-1]>5000 else 0)
        outcome.append(Z_outcome)
        table[counter,:] = table_row(Z_outcome,R,R0,R1)
        counter+=1
        R_dict[Z_outcome[-1]].append(R)
    return table, R_dict

folder = '../data/landscape/poisson/'
folder2 = './data/landscape/poisson-2/'
outcome = []
offset = data['landscape']
files = glob.glob(f'{folder}*-D.out')
files2 = glob.glob(f'{folder2}*-D.out')
files.extend(files2)

score = .85
dev = 700
R0s = np.arange(1,60)
R0max_s = 0
R1max_s = 0
R0max_d = 0
R1max_d = 0
for R0 in R0s:
    R1s = np.arange(R0+1,101)
    for R1 in R1s:
        table,_ = make_table(files,R0,R1)
        x = table[:,1].reshape(-1, 1)
        y = table[:,-1]#.reshape(-1, 1)
        clf = LogisticRegression().fit(x,y)
        score_n = clf.score(x,y)
        dev_n = deviance(x,y,clf)
        if score_n>score:
            score = score_n
            print(R0,R1,score,dev_n)
            R0max_s = R0
            R1max_s = R1
        if dev_n<dev:
            dev = dev_n
            print(R0,R1,score_n,dev_n)
            R0max_d = R0
            R1max_d = R1


R0max_s = 23
R1max_s = 73


table,d = make_table(files,R0max_s,R1max_s)
print(R0max_s,R1max_s)
np.savetxt('../data/landscape/tables/poisson-table-score-full-test.out',table)
np.savetxt('../data/logistic_regression/poisson-table-score-full-test.out',table)

