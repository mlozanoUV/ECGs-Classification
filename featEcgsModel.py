import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from scipy import signal
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


n_samples_chinos  = 333
n_samples_test    = 43
n_samples_sim     = 1716
show_pca          = False # Ver PCAs...
select_features   = False
#AllprecNames      = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
#precNames         = ["V2","II"]
d1                = {'LV':0, 'RV':1}
d2                = {'Left':0, 'Right':1}

n_features        = 300
  
path_sim_f        = 'QRS_Sims_Features.csv'
path_test_f       = 'QRS_CARTO_Features.csv'
path_chinos_f     = 'QRS_Database_Features.csv'
path_sim          = 'QRS_Sims.mat'
path_test         = 'QRS_CARTO.mat'
path_chinos       = 'QRS_Database.mat'

# Cargamos las matrices: Chinos, test-Clinic y Simuladas
dChinos = loadmat(path_chinos)
dTest   = loadmat(path_test)
dSim    = loadmat(path_sim)

dfc = pd.read_csv(path_chinos_f)
dft = pd.read_csv(path_test_f)
dfs = pd.read_csv(path_sim_f)

mc = dfc.iloc[1:, 1:].to_numpy() # Fila 1 (cabecera) y col 1 (tiempo_total): las salto ...
mt = dft.iloc[1:, 1:].to_numpy() 
ms = dfs.iloc[1:, 1:].to_numpy() 

YMcn = get_Attr(dChinos,'LeftRight')
YMcn = binarize(YMcn, d2)
YMtn = get_Attr(dTest,'LeftRigth')
YMtn = binarize(YMtn, d1)
YMsn = get_Attr(dSim,'LeftRigth')
YMsn = binarize(YMsn, d1)

Mcn = normalize(mc.T, norm='l2')  # en este caso funciona un poco mejor que l1
Mtn = normalize(mt.T, norm='l2')
Msn = normalize(ms.T, norm='l2')

print("Rango Matrices (chinos, test, simuladas):", Mcn.shape, Mtn.shape, Msn.shape)

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
plot_features(X, y)

"""
X, y = Msn, YMsn
plot_features(X, y)
# plot features by class/color
fig = plt.figure()
color = ['r+','b+']
for k in range(100):
    i = np.random.choice(len(X))
#    plt.xticks(range(10))
    plt.plot(X[i], color[y[i]], alpha=.152)
plt.show()
"""
  
# identify and remove outliers 
iso   = IsolationForest(contamination=0.1)
yhat  = iso.fit_predict(Mcn)
mask  = np.where(yhat != -1)[0]
Mcn, YMcn = Mcn[mask, :], YMcn[mask]

#describe(Mcn,Msn,Mtn)

if (show_pca):
  show_sets(Mcn, Msn, Mtn, YMcn, YMsn,YMtn)  

#clf = KNeighborsClassifier(n_neighbors=3)
#clf = MLPClassifier(solver='sgd', alpha=1e-3, hidden_layer_sizes=(100, 2), random_state=1, max_iter=5000)
clf = RandomForestClassifier(n_estimators=100)
        
#clf = svm.NuSVC(nu=0.2, kernel='rbf')

#clf.fit(Msn, YMsn)

#predicted_t = clf.predict(Mtn)
#predicted_c = clf.predict(Mcn)
#predicted_s = clf.predict(Msn)

#print(classification_report(YMtn, predicted_t))
#print(classification_report(YMcn, predicted_c))
#print(classification_report(YMsn, predicted_s))

#
# Validacion Cruzada

#clf = svm.SVC(kernel='linear', C=1)
#clf = svm.NuSVC(nu=0.2, kernel='rbf')
X = np.concatenate( (Msn, Mcn) )
y = np.concatenate( (YMsn, YMcn) )
scores = cross_val_score(clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf.fit(X, y)

predicted_t = clf.predict(Mtn)
#predicted_c = clf.predict(Mcn)
#predicted_s = clf.predict(Msn)

print(classification_report(YMtn, predicted_t))
#print(classification_report(YMcn, predicted_c))
#print(classification_report(YMsn, predicted_s))

##
print("Done!")
