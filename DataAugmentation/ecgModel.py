import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ecgm_utils import get_Attr, get_MLeads, binarize, load_data_set, load_fmat, loadmat, eval_model

from generate_augmentation import mixup


n_samples_chinos  = 333
n_samples_test    = 43
n_samples_sim     = 2496
show_pca          = False # Ver PCAs...
AllprecNames      = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
precNames         = ["V2"]

sample_size       = 10

# Raw Signal Data Sets  
path_sim          = 'QRS_Sims2.mat'
path_test         = 'QRS_CARTO2.mat'
path_chinos       = './QRS_Database2.mat'

# Feature based Data Sets
path_sim_f        = 'QRS_Sims2_features.mat'
path_test_f       = 'QRS_CARTO2_features.mat'
path_chinos_f     = 'QRS_Database2_features.mat'


############################################################################################
############################################################################################
#
# Cargamos las matrices: Chinos, test-Clinic y Simuladas (Features)
#
print("Cargando feature based Data Sets .....")
Mcn, YMcn,_     = load_fmat(path_chinos, 'LeftRight') 
Mtn, YMtn,_     = load_fmat(path_test, 'LeftRigth') # Cambian las dos ultimas letras!
Msn, YMsn, ids  = load_fmat(path_sim, 'LeftRigth') 

print("Rango Matrices X (chinos, test, simuladas):", Mcn.shape, Mtn.shape, Msn.shape)
print("Rango Matrices Y (chinos, test, simuladas):", YMcn.shape, YMtn.shape, YMsn.shape)

print("Mix  up: ")
Xhat,yhat,weights,idhat = mixup(Msn,YMsn,ids,N_max=2)
print(Xhat.shape, yhat.shape)
Msn, YMsn = Xhat, yhat
   
## Entrenamiento de la SVM y Evaluacion
eval_model([Msn, YMsn, "Simulated"], [[Mtn, Mcn], [YMtn, YMcn], ['Clinic', 'Chinos']], _nu = 0.7)
print("========================================\n")
eval_model([Mcn, YMcn, "Chinos"], [[Mtn, Msn], [YMtn, YMsn], ['Clinic', 'Simulated']], _nu = 0.35)
print("========================================\n")
eval_model([Mtn, YMtn, "Clinic"], [[Msn, Mcn], [YMsn, YMcn], ['Simulated', 'Chinos']], _nu = 0.7)


print("\n\n -------------------------------------\n\n")
#
# Cargamos las matrices: Chinos, test-Clinic y Simuladas (Raw Signal)
#

dChinos = loadmat(path_chinos)
dTest   = loadmat(path_test)
dSim    = loadmat(path_sim)

#Mcn = Matriz Chinos normalizada; Msn = Matriz simuladas norm. ; Mtn = Matriz Clinic norm.
Mcn, YMcn = load_data_set(dChinos,  sample_size, n_samples_chinos,  precNames, 'LeftRight' )
Mtn, YMtn = load_data_set(dTest,    sample_size, n_samples_test,    precNames, 'LeftRigth')
Msn, YMsn = load_data_set(dSim,     sample_size, n_samples_sim,     precNames, 'LeftRigth')

print("Rango Matrices X (chinos, test, simuladas):", Mcn.shape, Mtn.shape, Msn.shape)
print("Rango Matrices Y (chinos, test, simuladas):", YMcn.shape, YMtn.shape, YMsn.shape)
    
## Entrenamiento de la SVM y Evaluacion
eval_model([Msn, YMsn, "Simulated"], [[Mtn, Mcn], [YMtn, YMcn], ['Clinic', 'Chinos']], _nu = 0.7)
print("========================================\n")
eval_model([Mcn, YMcn, "Chinos"], [[Mtn, Msn], [YMtn, YMsn], ['Clinic', 'Simulated']], _nu = 0.35)
print("========================================\n")
eval_model([Mtn, YMtn, "Clinic"], [[Msn, Mcn], [YMsn, YMcn], ['Simulated', 'Chinos']], _nu = 0.7)
print("========================================\n")
    

"""
# Plot samples
color = ['red', 'blue']
for i in range(200):
    plt.plot(Msn[i], c=color[YMsn[i]], alpha=0.2)
#plt.show()
        
"""

