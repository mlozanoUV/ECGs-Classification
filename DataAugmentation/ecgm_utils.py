import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import itertools
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score,plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import  normalize
from mat4py import loadmat
import scipy as sp
import tqdm



# Load feature based files .mat (@Guillermo)
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

    return X,y, ids

# Train and Evaluation of a model: Test is a list of sets
def eval_model(Training, Test, _cv = 5,_nu = 0.35):
           
    X, Y, name = Training[0],Training[1],Training[2]
    Xtest, YTest, Tnames = Test[0], Test[1], Test[2]
    
    clf = svm.NuSVC(nu=0.1, kernel='rbf')
    scores = cross_val_score(clf, X, Y, cv=_cv)
    print(name+" CV_Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf = svm.NuSVC(nu=_nu, kernel='rbf')
    clf.fit(X, Y)

    for i, ts in enumerate(Xtest):
        predicted = clf.predict(ts)
        #print( classification_report(YTest[i], predicted) )
        #plot_confusion_matrix(clf, ts, YTest[i])  # doctest: +SKIP
        #plt.show()  # doctest: +SKIP
        acc_sc = accuracy_score(YTest[i], predicted)
        print("Test Set: "+Tnames[i]+":    Accuracy = ", acc_sc)
        

# Loads a data set
def load_data_set(dict_data_set, sample_size, n_samples, precNames, target):

    d1                = {'LV':0, 'RV':1}
    d2                = {'Left':0, 'Right':1}

    # Matriz de precordiales re-sampleadas
    X = get_MLeads(dict_data_set,precNames, sample_size = sample_size, dim=(n_samples,len(precNames)*sample_size))

    samples_ok = np.array(get_Attr(dict_data_set, 'OK'))
    if len(samples_ok) > 0:
        oks = np.argwhere(samples_ok==1).flatten()
        X = X[oks]

    X = normalize(X, norm='max')
    Y = get_Attr(dict_data_set, target)
   
    Y = binarize(Y, d1)
    if len(samples_ok) > 0:
        Y = np.array(Y)[oks]
    
    #print("Loaded Data Set ::", X.shape, Y.shape)

    return X, Y


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
      if att in dictM[key].keys(): 
        return dictM[key][att]
      else:
        return []

def binarize(Y, d):
  for i,lr in enumerate(Y):
     Y[i] = d[lr]
  return np.array(Y)


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
