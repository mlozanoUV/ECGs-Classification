import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colors as mmarkers
import sys, os
import scipy.io
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy import signal



def pca2(r,c, fname, title, target_names, yr=[], yc=[]):
  
  mr = np.zeros(len(r)) 
  mc = np.ones(len(c)) 
  marks = np.concatenate((mr, mc)).flatten()  
  Xall = np.concatenate((r, c))
  Yall = np.concatenate((yr, yc)).flatten()  
  #print(Xall.shape, Yall.shape)
  pca(Xall, Yall, fname, 2, title, target_names, marks.astype(int))
  
def pca(X, y, save_as, ncomponents, title='', target_names = ["LV","RV"], marks=[]):
    
    #target_names = ["LV","RV"]

    pca = PCA(n_components=ncomponents)
    X_r = pca.fit(X).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
        % str(pca.explained_variance_ratio_))

    #plt.figure()
    fig = plt.figure()

    colors = ['turquoise', 'darkorange']
    lmarks = ['8','s']
    lw = 2
    if ncomponents == 3:
        ax = fig.add_subplot(111, projection='3d')
        for color, i, target_name in zip(colors, [0, 1], target_names):
            print(color, i, target_name)
            ax.scatter(X_r[y == i, 0], X_r[y == i, 1], X_r[y == i, 2], color=color, alpha=.18, lw=lw,
                        label=target_name)
    elif ncomponents == 2:
        #for color, i, target_name in zip(colors, [0, 1], target_names):
        #    print(color, i, target_name)
        #    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.18, lw=lw,
        #                label=target_name, marker=lmarks[i])
        plt.scatter(X_r[:, 0], X_r[:, 1], c=y, alpha=.18, lw=lw, label=target_names)
             
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.savefig(save_as) 
    plt.show()
    return X_r


import re
def decode(s):
    #Pacientes Ruben: JJC_e3_f23
    mo = re.match("(\w*)_e(\d*)_f(\d*)",s)
    return mo.group(1), mo.group(2), mo.group(3)

def mds_representation(attributes, classes = None, n_ect = None, hier_class = None, save_as=''):

    mds = MDS(n_components=2, metric=True, n_jobs=4, dissimilarity="euclidean")
    att_trans = mds.fit_transform(attributes)

    if hier_class is not None and classes is not None and n_ect is not None:
        fig, [ax0,ax1] = plt.subplots(1,2)
        tp = True
    else:
        fig, ax0 = plt.subplots(1,1)


    if classes is not None and n_ect is not None:

        colors = get_list_of_n_colors(n_ect)
        ect = np.array( [i for c in classes for i in range(n_ect) if eval(decode(c)[1]) == i] )
        
        for color, i in zip(colors, range(n_ect)):
            ax0.scatter(att_trans[ect == i, 0], att_trans[ect == i, 1], color=color, alpha=.18,
                            label=f"E {str(i)}", marker=i%12)
        ax0.legend()

    if hier_class is not None:

        ax1=ax0
        n_labels = hier_class.max() + 1
        colors = get_list_of_n_colors(n_labels)
        colors = ['turquoise', 'darkorange']
        for color, i in zip(colors,range(n_labels)):
            ax1.scatter(att_trans[hier_class == i, 0], att_trans[hier_class == i, 1], color=color, alpha=.18,
                            label=str(i))
        ax1.legend()

    plt.title(save_as[:-4])
    plt.savefig(save_as)
    plt.show()


def get_list_of_n_colors(n):
    import random

    colors = np.array(list(mcolors.XKCD_COLORS.keys()))
    x = random.sample(range(colors.shape[0]), n)
    c = colors[x]
    return c

# Plot some samples 
def describe(Mcn, Msn, Mtn):

  f,ax = plt.subplots(3,1)

  for i in range(25):
    ax[0].plot(Mcn[i])
    ax[1].plot(Msn[i])
    ax[2].plot(Mtn[i])

  plt.show()

# Extrae la matriz (caso, lead0+lead1+...lead11)
def get_MLeads(dTt, Lprec, sample_size = 300, dim=(0,0)):
  M = np.zeros([dim[0], dim[1]])
  for key in dTt:
    X12p = dTt[key]
    for ip, prec in enumerate(Lprec):
      for i, pat in enumerate(X12p[prec]):
        M[i, ip*sample_size:(ip+1)*sample_size] = signal.resample(pat,sample_size).flatten()

  return M

def get_Attr(dictM, att):
  for key in dictM:
      return dictM[key][att]


def binarize(Y, d):
  for i,lr in enumerate(Y):
    Y[i] = d[lr]
  return np.array(Y)

def show_sets(Mcn,Msn,Mtn, YMcn, YMsn, YMtn):
  
  describe(Mcn, Msn, Mtn)
  
  ## PCA Ruben-Training vs Chinos
  pca(Mcn, YMcn, 'PCA_Chinos_LR', 2, 'PCA: Chinos (LV vs RV)', ["LV", "RV"])
  pca(Msn, YMsn, 'PCA_Sim_LR', 2, 'PCA: Simulated (LV vs RV)', ["LV", "RV"])
  
  pca2(Mcn, Msn, 'PCA_ch-sim.png', 'PCA: Chinos vs Simulated',["Chi", "Sim"],YMcn, YMsn)
  pca2(Mcn, Mtn, 'PCA_ch-test.png', 'PCA: Chinos vs Test ',["Chi", "Test" ], YMcn, YMtn)

  # Chinos vs Test
  Xall = np.concatenate((Mcn,Mtn))
  Y1 = np.zeros(len(Mcn)) 
  Y2 = np.ones(len(Mtn)) 
  Yall = np.concatenate((Y1, Y2)).flatten()
  mds_representation(Xall, hier_class=Yall.astype(int), save_as="mds_chinos_vs_test.png")

  # Simulated vs Test
  Xall = np.concatenate((Msn, Mtn))
  Y1 = np.zeros(len(Msn)) 
  Y2 = np.ones(len(Mtn)) 
  Yall = np.concatenate((Y1, Y2)).flatten()
  mds_representation(Xall, hier_class=Yall.astype(int), save_as="mds_simulated_vs_test.png")

  # Chinos vs Simulated
  Xall = np.concatenate((Mcn, Msn))
  Y1 = np.zeros(len(Mcn)) 
  Y2 = np.ones(len(Msn)) 
  Yall = np.concatenate((Y1, Y2)).flatten()
  mds_representation(Xall, hier_class=Yall.astype(int),save_as="mds_chinos_vs_simulated.png")

def combinaciones_precordiales(AllprecNames):
  Lpruebas = []
  for L in range(1, len(AllprecNames)+1):
      for subset in itertools.combinations(AllprecNames, L):
        Lpruebas.append( list(subset) )
  return Lpruebas

# Cargar subLocations 
def get_Sublocations(Slocs):
  lids, labels = [], []
  for i, k in enumerate(Slocs):
    #if k != []:
      lids.append(i)
      labels.append(k)
      
  return labels, lids

# Numera las clases o labels
def labelize(labels):
  Y = np.zeros(len(labels))
  cont = 0
  labels2id = {}

  for i,label in enumerate(labels):
    if label not in labels2id:
      labels2id[label] = cont
      cont+=1
    
    #print(i, label, labels2id[label])
    Y[i] = labels2id[label]
      
  return Y, labels2id

# filtra labels dejando pasar solo las muestras cuya clase esta en dlabels
def filter_labels(labels, dlabels):
    Y, iY = [], []
    shared = {}
    for i,label in enumerate(labels):
      if label != [] and label in dlabels:
        Y.append(dlabels[label])
        iY.append(i)
        shared[label] = dlabels[label]

    return np.array(Y), np.array(iY), shared
  
def plot_features(X, y, nmax = 10):
    from sklearn.ensemble import ExtraTreesClassifier

    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importance")
    plt.bar(range(min(X.shape[1],nmax)), importances[indices[:nmax]],
            color="r", yerr=std[indices[:nmax]], align="center")
    plt.xticks(range(min(X.shape[1], nmax)), indices[:nmax])
    plt.xlim([-1, min(X.shape[1],nmax)])
    plt.savefig('feature_importance.png')
    plt.show()
