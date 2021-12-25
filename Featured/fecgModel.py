import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from scipy import signal
import scipy as sp
from sklearn.preprocessing import StandardScaler, normalize
from mat4py import loadmat

# Clasificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.model_selection import cross_val_score

from ts_utils import *
import itertools
import pandas as pd
import tqdm


n_samples_chinos  = 333
n_samples_test    = 43
n_samples_sim     = 2496
show_pca          = False # Ver PCAs...
select_features   = False
#AllprecNames      = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
#precNames         = ["V2","II"]
d1                = {'LV':0, 'RV':1}
d2                = {'Left':0, 'Right':1}

n_features        = 300
  
path_sim_f        = 'QRS_Sims2_features.mat'
path_test_f       = 'QRS_CARTO2_features.mat'
path_chinos_f     = 'QRS_Database2_features.mat'
path_sim          = 'QRS_Sims2.mat'
path_test         = 'QRS_CARTO2.mat'
path_chinos       = 'QRS_Database2.mat'

# Cargamos las matrices: Chinos, test-Clinic y Simuladas
dChinos = loadmat(path_chinos)
dTest   = loadmat(path_test)
dSim    = loadmat(path_sim)

dfc = loadmat(path_chinos_f) #pd.read_csv(path_chinos_f)
dft = loadmat(path_test_f) #pd.read_csv(path_test_f)
dfs = loadmat(path_sim_f) #pd.read_csv(path_sim_f)

# Load .mat
path = path_chinos

# Load .mat
def load_fmat(path, y_label):
    mat_contents_features = loadmat(path)
    file_key = list(mat_contents_features)[0]
    mat_contents_features = mat_contents_features[file_key]
    if not y_label in mat_contents_features.keys():
      print(path, mat_contents_features.keys())

    N_elements = len(mat_contents_features[y_label])
    
    # Get variable names
    variable_names = mat_contents_features.get("varnames",["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"])

    # If raw signal, resample to 10 samples
    if "varnames" not in mat_contents_features:
        for k in tqdm.tqdm(variable_names):
            for i in range(N_elements):
                sample = np.array(mat_contents_features[k][i]).squeeze()
                sample_interpolated = sp.interpolate.interp1d(np.linspace(0,1,sample.size),sample)(np.linspace(0,1,10))
                mat_contents_features[k][i] = sample_interpolated

    # Obtain X, y and ids matrices
    X = np.array([np.array([mat_contents_features[k][i] for k in variable_names]) for i in range(N_elements)])
    if "varnames" not in mat_contents_features:
        X = np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
    y = np.array([mat_contents_features[y_label][i] for i in range(N_elements)])
    ids = np.array(mat_contents_features['name'])

    return X,y

    # Do mixup
    #Xhat,yhat,weights,idhat = src.generate_augmentation.mixup(X,y,ids,N_max=2)



Mcn, YMcn = load_fmat(path_chinos_f, 'LeftRight')
Mtn, YMtn = load_fmat(path_test_f, 'LeftRigth') 
Msn, YMsn = load_fmat(path_sim_f, 'LeftRigth') 


print("Rango Matrices X (chinos, test, simuladas):", Mcn.shape, Mtn.shape, Msn.shape)
print("Rango Matrices Y (chinos, test, simuladas):", YMcn.shape, YMtn.shape, YMsn.shape)

if select_features:
  X, y = np.copy(Msn), np.copy(YMsn)
  clf = ExtraTreesClassifier(n_estimators=20)
  #clf = LinearSVC(C=0.01, penalty="l2", dual=False)
  clf = clf.fit(X, y)
  model = SelectFromModel(clf, prefit=True)
  Msn = model.transform(X)
  Mcn = model.transform(np.copy(Mcn))
  Mtn = model.transform(np.copy(Mtn))
  print("Nuevo Rango Matrices (chinos, test, simuladas):", Mcn.shape, Mtn.shape, Msn.shape)

X, y = Msn, YMsn
print(X.shape, y.shape)
#plot_features(X, y)

"""  
# identify and remove outliers 
iso   = IsolationForest(contamination=0.1)
yhat  = iso.fit_predict(Mcn)
mask  = np.where(yhat != -1)[0]
Mcn, YMcn = Mcn[mask, :], YMcn[mask]
"""

#describe(Mcn,Msn,Mtn)

if (show_pca):
  show_sets(Mcn, Msn, Mtn, YMcn, YMsn,YMtn)  

clf = svm.NuSVC(nu=0.2, kernel='rbf')

scores = cross_val_score(clf, Msn, YMsn, cv=5)
print("Msn CV_Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

X = np.concatenate( (Msn, Mtn) )
y = np.concatenate( (YMsn, YMtn) )
scores = cross_val_score(clf, X, y, cv=5)
print("Msn+Mtn CV_Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.NuSVC(nu=0.75, kernel='rbf')
clf.fit(X, y)
predicted_c = clf.predict(Mcn)
print("Tr: Msn + Mtn , Test: Chinos")
print(classification_report(YMcn, predicted_c))

clf = svm.NuSVC(nu=0.2, kernel='rbf')

X = np.concatenate( (Msn, Mcn) )
y = np.concatenate( (YMsn, YMcn) )
scores = cross_val_score(clf, X, y, cv=5)
print("Msn+Mcn CV_Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = svm.NuSVC(nu=0.75, kernel='rbf')
clf.fit(X, y)
predicted = clf.predict(Mtn)        
print("Tr: Msn + Mcn , Test: Clinic")
print(classification_report(YMtn, predicted))

##
print("Done!")
