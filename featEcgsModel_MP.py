import sys
import numpy as np
import matplotlib.pyplot as plt
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

from ts_utils import *
import itertools, math
import multiprocessing

n_samples_chinos  = 333
n_samples_test    = 43
n_samples_sim     = 1716
show_pca          = False # Ver PCAs...
AllprecNames      = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
precNames         = ["V1","V2","V3"]
d1                = {'LV':0, 'RV':1}
d2                = {'Left':0, 'Right':1}

sample_size       = 10 #300
  
path_sim          = 'QRS_Sims.mat'
path_test         = 'QRS_CARTO.mat'
path_chinos       = './QRS_Database.mat'


def runRAW(precNames):
    
    dChinos = Ldicts[0]
    dTest   = Ldicts[1]
    dSim    = Ldicts[2]
    dscores = Ldicts[3]

    # Matriz de precordiales re-sampleadas
    Mc = get_MLeads(dChinos,precNames, sample_size = sample_size, dim=(n_samples_chinos,len(precNames)*sample_size))
    samples_ok = np.array(get_Attr(dChinos, 'OK'))
    oks = np.argwhere(samples_ok==1).flatten()
    Mc = Mc[oks]
    Mcn = normalize(Mc, norm='l1')
    # Clase L/R
    YMcn = Ys[0]
    YMcn = YMcn[oks]
    # identify and remove outliers 
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(Mcn)
    mask = np.where(yhat != -1)[0]
    Mcn, YMcn = Mcn[mask, :], YMcn[mask]
    #print("Chinos samples::", Mcn.shape)

    # Matriz de test
    Mt = get_MLeads(dTest,precNames, sample_size = sample_size, dim=(n_samples_test,len(precNames)*sample_size))
    samples_ok = np.array(get_Attr(dTest, 'OK'))
    oks = np.argwhere(samples_ok==1).flatten()
    Mt = Mt[oks]
    Mtn = normalize(Mt, norm='l1')
    #print("Test(Clinic)-samples::", Mtn.shape)
    YMtn = Ys[1]  
    YMtn = YMtn[oks]

    # Matriz de muestas simuladas
    Ms = get_MLeads(dSim,precNames,sample_size = sample_size, dim=(n_samples_sim,len(precNames)*sample_size))
    Msn = normalize(Ms, norm='l1')
    #print("Simulated samples::", Msn.shape)
    YMsn = Ys[2]
           
    clf = svm.NuSVC(nu=0.75, kernel='rbf')
    clf.fit(Msn, YMsn)

    predicted_t = clf.predict(Mtn)
    predicted_c = clf.predict(Mcn)
    #predicted_s = clf.predict(Msn)
    
    #print(classification_report(YMtn, predicted_t))
    #print(classification_report(YMcn, predicted_c))

    ast = accuracy_score(YMtn, predicted_t)
    asc = accuracy_score(YMcn, predicted_c)
    
    dscores['_'.join(precNames)] = (ast, asc)
    print(precNames, ast, asc)
    return (ast, asc) #dscores
    #print("Simulated Set")
    #print(classification_report(YMsn, predicted_s))



def mp_run(AllPrecComb, nprocs):
    def worker(AllPrecComb, out_q):
        """ The worker function, invoked in a process. 'nums' is a
            list of numbers to factor. The results are placed in
            a dictionary that's pushed to a queue.
        """
        outdict = {}
        for n in AllPrecComb:
            scores = runRAW(n)
            outdict['_'.join(n)] = scores
        out_q.put(outdict)

    # Each process will get 'chunksize' nums and a queue to put his out
    # dict into
    out_q = multiprocessing.Queue()
    chunksize = int(math.ceil(len(AllPrecComb) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = multiprocessing.Process(
                target=worker,
                args=(AllPrecComb[chunksize * i:chunksize * (i + 1)],
                      out_q))
        procs.append(p)
        p.start()

    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    return resultdict



############################################################
# Cargamos las matrices: Chinos, test-Clinic y Simuladas
dChinos = loadmat(path_chinos)
dTest = loadmat(path_test)
dSim  = loadmat(path_sim)

YMcn = get_Attr(dChinos,'LeftRight')
YMcn = binarize(YMcn, d2)
YMtn = get_Attr(dTest,'LeftRigth')
YMtn = binarize(YMtn, d1)
YMsn = get_Attr(dSim,'LeftRigth')
YMsn = binarize(YMsn, d1)

Ys = [YMcn, YMtn, YMsn]

dscores =  {}
AllPrecComb = []
for L in range(0, len(AllprecNames)+1):
    for subset in itertools.combinations(AllprecNames, L):
      if len(subset) > 0:
        precNames = list(subset)
        AllPrecComb.append(precNames)

print("Ncombinaciones precordiales: ", len(AllPrecComb))
Ldicts = [dChinos, dTest, dSim, dscores]
dscores = mp_run(AllPrecComb, 6)

f = open('dictScores.1.txt','w')
f.write(str(dscores))
f.close()
