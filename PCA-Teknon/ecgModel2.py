import sys, random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.measurements import label
from ecgm_utils import get_Attr, get_MLeads, binarize, load_data_set, load_fmat, loadmat, eval_model
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score,plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from biosppy.signals import ecg
from scipy import signal
from scipy.spatial.distance import euclidean
from ECG_PCA import ECG_PCA, smooth

n_samples_chinos  = 333
n_samples_test    = 43
n_samples_sim     = 2496

AllprecNames      = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
precNames         = ["V2"]

sample_size       = 100

# Sim Signal Data Sets  # Raw Signal Data Sets  
path_sim          = 'QRS_Sims2.mat'
path_test         = 'QRS_CARTO2.mat'
path_chinos       = 'QRS_Database2.mat'

colors_LR = ['blue', 'red']
mark_size = 10    
#######################################

def plot_leads(Xn, precNames, numsamples = 30, dfTest=[]):
   
    fig, ax = plt.subplots(4, 3, figsize=(8,10))
    fig.suptitle('Leads')
    for ix, x in enumerate(Xn[:numsamples]):
        for i, prec in enumerate(precNames):
            s = x[sample_size*i:sample_size*(1+i)]
            row, col = i//3, i%3
            ax[row,col].plot(s, c=colors_LR[YMsn[ix]], alpha=0.2)
            ax[row,col].set_title(prec)
            ax[row,col].set_aspect('auto')
            if len(dfTest) >0 :
                Xtn = preprocess(dfTest, prec)
                ax[row,col].plot(Xtn, c='green')
            
            

    plt.tight_layout()
    plt.show()

def preprocess(df, lead):
    print(df.columns)
    V2 = df[lead].to_numpy()
    plt.plot(V2)
    plt.show()
    # process it and plot
    rpeaks = np.array(ecg.hamilton_segmenter(V2)).flatten()
    #t_beat =  int(rpeaks[-1])
    print(lead, rpeaks) #
    t_beat = int(input("Tiempo del pico a recortar? ...."))
    
    #print(" |-> Tiempo de corte= ", t_beat)
    V2 = V2[t_beat-w_size:t_beat+w_size]
    V2r = signal.resample(V2,sample_size).flatten()
    V2s = smooth(V2r)
    V2s = V2s/abs(V2s).max()
    V2s-= V2s[0]

    return V2s 

def ecg_meanLR(Xp, Yp):
    # Senyales promedio para cada clase
    left = np.argwhere(Yp == 0)
    rigth = np.argwhere(Yp == 1)
    Xpl = Xp[left]
    Xpr = Xp[rigth]
    Xpml, Xpmr = Xpl.mean(axis=0), Xpr.mean(axis=0)
    return Xpml.flatten(), Xpmr.flatten()


# Carga de datos
#######################################

#for p in AllprecNames:
p           = sys.argv[2] # Lead

# Datos real-time (csv)
rt_test     = sys.argv[1]
w_size      = 150
df_ecg      = pd.read_csv(rt_test)
#df_ecg.plot(subplots=True, layout=(4,3))
#plt.show()

dSim        = loadmat(path_sim)      
Msn, YMsn   = load_data_set(dSim, sample_size, n_samples_sim, [p], 'LeftRigth')

dChinos     = loadmat(path_chinos)      
Mcn, YMcn   = load_data_set(dChinos, sample_size, n_samples_chinos, [p], 'LeftRight')

dTest     = loadmat(path_test)      
Mtn, YMtn   = load_data_set(dTest, sample_size, n_samples_test, [p], 'LeftRigth')

#Preproceso (Señales Teknon)
pp          = preprocess(df_ecg, p)

###########################################################
# Plot señal media de cada clase del cjto de entrenamiento
aux = list(range(Mtn.shape[0]))
signals = []
for n in range(n_samples_test):
    signals.append(random.choice(aux))
for i in signals:
    plt.plot(Mtn[i], c=colors_LR[YMtn[i]], alpha=0.2)

Lm, Rm = ecg_meanLR(Msn, YMsn)
#plt.plot(Lm, c=colors_LR[0], ls='--')
#plt.plot(Rm, c=colors_LR[1], ls='--')
# Plot senyal de test
plt.plot(pp, c='black')
print("Lead: ",  p)
plt.show()

# PCA de los conjuntos
nfeatures = 2
pca_sim  = ECG_PCA(X=Msn, y= YMsn, n_dim=nfeatures)
pca_chinos = ECG_PCA(X=Mcn, y= YMcn, n_dim=nfeatures)
pca_test = ECG_PCA(X=Mtn, y= YMtn, n_dim=nfeatures)
pcas = [pca_sim, pca_chinos, pca_test]

#pca_sim.pairplot(X_t = pca_chinos.X, y_t=YMcn+2)
pca_sim.plot_pca_variance()
#pca_chinos.plot_pca_variance()
#pca_test.plot_pca_variance()

#pca.ppal_dirs(show=True)

################################################
# Pruebas clasificador-PCA
PCA_clf  = svm.NuSVC(nu = 0.6, kernel='rbf')
scores = cross_val_score(PCA_clf, pca_sim.get_X_tr(nfeatures), YMsn)
print("PCA:: CV_Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

RAW_clf  = svm.NuSVC(nu = 0.6, kernel='rbf')
scores = cross_val_score(RAW_clf, Msn, YMsn)
print("Raw:: CV_Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#s = [phase_align(pca_sim.mean, beat, [10,90]) for beat in Mtn]
#Xd = [shift(beat,desp,mode='nearest') for beat, desp in zip(Mtn, s)] 

Xpca, Ypca = pca_sim.X_tr[:,0], pca_sim.X_tr[:,1]
i0, i1 = np.argwhere(YMsn == 0), np.argwhere(YMsn == 1)
plt.scatter(Xpca[i0], Ypca[i0], c='blue', label='L_Sim')
plt.scatter(Xpca[i1], Ypca[i1], c='red', label='R_Sim')

PCA_clf.fit(pca_sim.X_tr[:,:nfeatures], YMsn)
RAW_clf.fit(Msn, YMsn)


# Test Teknon
Xtpca = pca_sim.transfrom(pp.reshape(1,sample_size),nfeatures)
pred = PCA_clf.predict(  Xtpca.reshape(1,nfeatures) )
predr = RAW_clf.predict( pp.reshape(1,sample_size) )
print(pred, predr)

plt.scatter(Xtpca[0], Xtpca[1], c='yellow', label='TK_test')


# PCA Test
Xvt, Yvt = Mtn, YMtn
i0, i1 = np.argwhere(Yvt == 0), np.argwhere(Yvt == 1)
Xtpca = pca_sim.transfrom(Xvt,nfeatures)
plt.scatter(Xtpca[i0,0], Xtpca[i0,1], c='green', label='L_Test')
plt.scatter(Xtpca[i1,0], Xtpca[i1,1], c='black', label='R_Test')
plt.legend()

plt.show()


pred = PCA_clf.predict(  Xtpca )
predr = RAW_clf.predict( Xvt )

print(len(pred), len(predr))
print(classification_report(Yvt, pred))
print(classification_report(Yvt, predr))








