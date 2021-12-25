import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from scipy import signal
from sklearn.preprocessing import StandardScaler, normalize
from mat4py import loadmat
from sklearn.ensemble import IsolationForest
# Clasificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix

from ts_utils import *
import itertools

n_samples_chinos  = 333
n_samples_test    = 43
n_samples_sim     = 1716
show_pca          = True   # Ver PCAs...
AllprecNames      = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
precNames         = ["aVF","V2", "V3"]
d1                = {'LV':0, 'RV':1}
d2                = {'Left':0, 'Right':1}

sample_size       = 10
  
path_sim          = 'QRS_Sims.mat'
path_test         = 'QRS_CARTO.mat'
path_chinos       = 'QRS_Database.mat'

# Cargamos las matrices: Chinos, test-Clinic y Simuladas
dChinos = loadmat(path_chinos)
dTest   = loadmat(path_test)
dSim    = loadmat(path_sim)

# Matrices de matlab: Chinos, simuladas
Mc  = get_MLeads(dChinos,precNames, sample_size = sample_size, dim=(n_samples_chinos,len(precNames)*sample_size))
Mcn = normalize(Mc, norm='l1')
Ms  = get_MLeads(dSim,precNames,sample_size = sample_size, dim=(n_samples_sim,len(precNames)*sample_size))
Msn = normalize(Ms, norm='l1')

# Clases: Sublocations
nsamples = len(dSim['QRS_Sims']['Sublocation'])
labels_s, mask = get_Sublocations(dSim['QRS_Sims']['Sublocation'])
YMsn, labels2class = labelize(labels_s)
print("Original Simulated samples::", Msn.shape, YMsn.shape)

nsamples = len(dChinos['QRS']['Sublocation'])
labels_c, _ = get_Sublocations(dChinos['QRS']['Sublocation'])
YMcn, iY, dshared = filter_labels(labels_c, labels2class)
Mcn = Mcn[iY, :]
print("Chinos Shape: ", len(labels_c), Mcn.shape, YMcn.shape)

# Como los chinos solo tienen muestras con 4 clases comunes, elimino las no comunes
YMsn, iY, _ = filter_labels(labels_s, dshared)
Msn = Msn[iY, :]
print("Shared Chinos-Simulated samples::", Msn.shape, YMsn.shape)  


###################################################

if (show_pca):
  pca(Msn, YMsn, 'PCA_Sim_SubLocations', 2, 'PCA: Simulated (Sublocations)', target_names=dshared.keys())
  pca(Mcn, YMcn, 'PCA_Chinos_SubL', 2, 'PCA: Chinos (SubLocations)', target_names=dshared.keys())

###################################################

clf = KNeighborsClassifier(n_neighbors=1)
#clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1, max_iter=1000)
#clf = RandomForestClassifier(n_estimators=10)
#clf = svm.NuSVC(nu=0.05, kernel='rbf')

# Validacion cruzada con el propio conjunto
scores = cross_val_score(clf, Msn, YMsn, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Entrenamiento
clf.fit(Msn, YMsn)

#predicted_t = clf.predict(Mtn)
predicted_c = clf.predict(Mcn)
#predicted_s = clf.predict(Msn)

#print(classification_report(YMtn, predicted_t))
print(classification_report(YMcn, predicted_c))
#print(classification_report(YMsn, predicted_s))
plot_confusion_matrix(clf, Mcn, YMcn)  # doctest: +SKIP
plt.show()  # doctest: +SKIP

##
print("Done!")
