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
import itertools

n_samples_chinos  = 333
n_samples_test    = 43
n_samples_sim     = 2496
show_pca          = False # Ver PCAs...
AllprecNames      = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
precNames         = ["V2"]
d1                = {'LV':0, 'RV':1}
d2                = {'Left':0, 'Right':1}

sample_size       = 10 #300
  
path_sim          = 'QRS_Sims2.mat'
path_test         = 'QRS_CARTO2.mat'
path_chinos       = './QRS_Database2.mat'

def runRAW(precNames, cv=False):
    global YMcn, YMtn, YMsn

    # Matriz de precordiales re-sampleadas
    Mc = get_MLeads(dChinos,precNames, sample_size = sample_size, dim=(n_samples_chinos,len(precNames)*sample_size))
    
    samples_ok = np.array(get_Attr(dChinos, 'OK'))
    oks = np.argwhere(samples_ok==1).flatten()
    Mc = Mc[oks]
    
    Mcn = normalize(Mc, norm='l1')
    # Clase L/R
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
    YMtn = YMtn[oks]  
    
    #print("Test(Clinic)-samples::", Mtn.shape)
    # Matriz de muestas simuladas
    Ms = get_MLeads(dSim,precNames,sample_size = sample_size, dim=(n_samples_sim,len(precNames)*sample_size))
    Msn = normalize(Ms, norm='l1')
    #print("Simulated samples::", Msn.shape)
  
    if cv == True:
        from sklearn.model_selection import cross_val_score
        #clf = svm.SVC(kernel='linear', C=1)
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
        #predicted_s = clf.predict(Msn)
        
        print(classification_report(YMcn, predicted_c))


        X = np.concatenate( (Msn, Mcn) )
        y = np.concatenate( (YMsn, YMcn) )
        scores = cross_val_score(clf, X, y, cv=5)
        print("Msn+Mcn CV_Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        clf.fit(X, y)
        predicted = clf.predict(Mtn)        
        print(classification_report(YMtn, predicted))


        if False:
            indices, importances = plot_features(X,y)
            xf,yf = [],[]
            for f in indices:
                xf.append(f)
                yf.append(importances[f])

            #fig, axs = plt.subplots(1,5)
            fig = plt.figure()
            color = ['r','b']
            plt.ylim((0,1))
            plt.xlim((-0.5,9.5))
            plt.bar(xf, yf, color='green', alpha=0.2)
            
            for k in range(100):
                i = np.random.choice(len(X))
                plt.xticks(range(10))
                plt.plot(X[i]+0.5, color[y[i]], alpha=.152)
            plt.show()
            """
            clf = svm.NuSVC(nu=0.7, kernel='rbf')
            clf.fit(X, y)

            predicted_t = clf.predict(Mcn)
            
            print(classification_report(YMtn, predicted_t))
            """
        

    else:       
        clf = svm.NuSVC(nu=0.75, kernel='rbf')
        clf.fit(Msn, YMsn)

        predicted_t = clf.predict(Mtn)
        predicted_c = clf.predict(Mcn)
        #predicted_s = clf.predict(Msn)
        
        print(classification_report(YMtn, predicted_t))
        print(classification_report(YMcn, predicted_c))

        ast = accuracy_score(YMtn, predicted_t)
        asc = accuracy_score(YMcn, predicted_c)
        print("Accuracies: ", ast, asc)
        #print(classification_report(YMsn, predicted_s))

def load_dict_from_file(filename):
    f = open(filename,'r')
    data=f.read()
    f.close()
    return eval(data)

def plot_features(X, y):
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
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.savefig('feature_importance.png')
    plt.show()
    return indices, importances
##
# Cargamos las matrices: Chinos, test-Clinic y Simuladas
dChinos = loadmat(path_chinos)
dTest = loadmat(path_test)
dSim  = loadmat(path_sim)


#samples_ok = dChinos['QRS']['OK']

YMcn = get_Attr(dChinos,'LeftRight')
YMcn = binarize(YMcn, d1)
YMtn = get_Attr(dTest,'LeftRigth')
YMtn = binarize(YMtn, d1)
YMsn = get_Attr(dSim,'LeftRigth')
YMsn = binarize(YMsn, d1)

Ys = [YMcn, YMtn, YMsn]

runRAW(precNames, cv=True)
