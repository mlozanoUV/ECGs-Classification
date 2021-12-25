import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg.matfuncs import _solve_P_Q
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, average_precision_score
from scipy import signal
from sklearn.preprocessing import StandardScaler, normalize
from mat4py import loadmat
from sklearn.ensemble import IsolationForest
# Clasificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from ts_utils import *
import itertools

n_samples_chinos  = 333
n_samples_test    = 43
n_samples_sim     = 2496
show_pca          = False # Ver PCAs...
AllprecNames      = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
precNames         = ["V2"]
d1                = {'LV':0, 'RV':1}
d2                = {'Left':0, 'Right':1}

sample_size       = 10
  
path_sim          = 'QRS_Sims2.mat'
path_test         = 'QRS_CARTO2.mat'
path_chinos       = './QRS_Database2.mat'

def runRAW(precNames):
    global YMcn, YMtn, YMsn

    # Matriz de precordiales re-sampleadas
    Mc = get_MLeads(dChinos,precNames, sample_size = sample_size, dim=(n_samples_chinos,len(precNames)*sample_size))
    samples_ok = np.array(get_Attr(dChinos, 'OK'))
    oks = np.argwhere(samples_ok==1).flatten()
    Mc = Mc[oks]

    Mcn = normalize(Mc, norm='l1')
    # Clase L/R
    YMcn = YMcn[oks]
    """
    # identify and remove outliers 
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(Mcn)
    mask = np.where(yhat != -1)[0]
    Mcn, YMcn = Mcn[mask, :], YMcn[mask]
    """
    print("Chinos samples::", Mcn.shape)
    
    # Matriz de test
    Mt = get_MLeads(dTest,precNames, sample_size = sample_size, dim=(n_samples_test,len(precNames)*sample_size))
    samples_ok = np.array(get_Attr(dTest, 'OK'))
    oks = np.argwhere(samples_ok==1).flatten()
    Mt = Mt[oks]
    Mtn = normalize(Mt, norm='l1')
    YMtn = YMtn[oks]  
    print("Test(Clinic)-samples::", Mtn.shape)
    
    # Matriz de muestas simuladas
    Ms = get_MLeads(dSim, precNames, sample_size = sample_size, dim=(n_samples_sim,len(precNames)*sample_size))
    Msn = normalize(Ms, norm='l1')
    print("Simulated samples::", Msn.shape)
  
    color = ['red', 'blue']
    for i in range(200):
        plt.plot(Msn[i], c=color[YMsn[i]], alpha=0.2)
    #plt.show()
           
    clf = svm.NuSVC(nu=0.35, kernel='rbf')
    clf.fit(Mcn, YMcn)

    predicted_t = clf.predict(Mtn)
    #predicted_c = clf.predict(Mcn)
    predicted_s = clf.predict(Msn)
    
    print(classification_report(YMtn, predicted_t))
    print(classification_report(YMsn, predicted_s))

    ast = accuracy_score(YMtn, predicted_t)
    asc = accuracy_score(YMsn, predicted_s)
    print("Accuracies: ", ast, asc)
    #print(classification_report(YMsn, predicted_s))

def load_dict_from_file(filename):
    f = open(filename,'r')
    data=f.read()
    f.close()
    return eval(data)

##
# Cargamos las matrices: Chinos, test-Clinic y Simuladas
dChinos = loadmat(path_chinos)
dTest = loadmat(path_test)
dSim  = loadmat(path_sim)

dfChinos = pd.DataFrame(dChinos['QRS'])
dfTest = pd.DataFrame(dTest['QRS_CARTO'])
dfSim = pd.DataFrame(dSim['QRS_Sims'])

#samples_ok = dChinos['QRS']['OK']

YMcn = get_Attr(dChinos,'LeftRight')
YMcn = binarize(YMcn, d1)
YMtn = get_Attr(dTest,'LeftRigth')
YMtn = binarize(YMtn, d1)
YMsn = get_Attr(dSim,'LeftRigth')
YMsn = binarize(YMsn, d1)

Ys = [YMcn, YMtn, YMsn]

runRAW(precNames)
