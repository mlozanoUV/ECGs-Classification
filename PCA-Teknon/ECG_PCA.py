#!/usr/bin/env python3
import os, sys
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class ECG_PCA:

    def __init__(self, X=None, y = None, n_dim = None):

        self.X : np.ndarray = None
        self.X_tr : np.ndarray = None
        self.y : np.ndarray = None

        self.n_dim : int = None
        self.mean : np.ndarray = None
        self.pca_mat : np.ndarray = None
        self.pca_var : np.ndarray = None
        self.pca_obj = None

        if X is not None:
            self.fit_X(X)
        if y is not None:
            self.y = y
        if n_dim is not None:
            self.n_dim = n_dim

    #

    def fit_X(self, X):

        self.pca_obj = PCA(svd_solver = 'full')
        self.pca_obj.fit(X)
        self.X = X
        self.mean = self.pca_obj.mean_
        self.pca_mat=self.pca_obj.components_
        self.pca_var=self.pca_obj.explained_variance_
        self.X_tr = self.transfrom(X, n_dim=self.pca_obj.n_components_)
    #

    def get_X(self):
        return self.X
    #
    def get_X_tr(self, n_dim=None):
        return self.X_tr[:, :n_dim]


    def set_y(self, y):
        self.y = y
        return
    #

    def get_y(self):
        return self.y
    #


    def set_n_dim(self, n):
        self.n_dim = n
        return
    #

    def get_n_dim(self):
        return self.n_dim
    #


    def transfrom(self, X, n_dim = None):
        """
        Transform dataset to PCA space
        Args:
        -----
            X : array (n_samples, n_features)

            n_dim : int (optional)
                Default to self.n_dim. The desired dimensionality in the PCA space.

        Returns:
        --------
            X_tr : list[array] or array
        """

        if n_dim is None:
            n_dim =  self.n_dim

        X_tr=[]

        for x in X:
            x_tr = self.pca_mat[:n_dim].dot(x)
            c = x_tr / np.sqrt(self.pca_var[:n_dim])
            X_tr.append(c)

        if len(X_tr)==1:
            return np.array(X_tr[0])
        else:
            return np.array(X_tr)
        #

    def inv_transform(self, X):
        """
        Transform dataset from PCA space
        Args:
        -----
            X : array (n_samples, n_coeffs)

        Returns:
        --------
            X_tr : list[array] or array
        """

        if X.shape[1] != self.n_dim:
            print(f"Warning: Dimensions do not match. X passed has {X.shape[0]} elements of dimension {X.shape[1]}. PCA n_dim has been set to {self.n_dim}.... ")
            n = X.shape[1]
        else:
            n = self.n_dim

        if len(X.shape) < 1:
            return False

        if len(X.shape) == 1:
            X = [X]

        X_tr = []

        for x in X:
            x_tr = x * np.sqrt(self.pca_var[:n]).flatten()
            hd = self.mean + (x_tr * self.pca_mat[:n].T).T.sum(axis=0)
            X_tr.append(hd)

        if len(X_tr)==1:
            return X_tr[0]
        else:
            return np.array(X_tr)
    #

    def ppal_dirs(self, n_comp = None, sigmas = [-1,1], show = False):

        if n_comp is None:
            n_comp = self.n_dim

        ppal_dirs = []
        for i in range(n_comp):
            for j in sigmas:
                d = np.zeros((1,n_comp))
                d[:,i] = j
                hd = self.inv_transform(d)
                ppal_dirs.append(hd)
        if show:
            fig, axes = plt.subplots(n_comp, 2)
            for ax, d in zip(axes.ravel(), ppal_dirs):
                ax.plot(range(len(d)), d)
            plt.show()

        return ppal_dirs
    #

    def plot_pca_variance(self, thrs=None):

        plt.rcParams.update({'font.size': 16})
        fig, ax1 = plt.subplots()

        color = 'b'
        #ax1.set_xticks(range(pca.n_components_))
        #ax1.set_xticklabels([f'{i}' for i in range(pca.n_components_)])
        ax1.set_xlabel('n components')
        ax1.set_ylabel('variance per component', color=color)
        ax1.bar(np.arange(0, self.pca_obj.n_components_), self.pca_obj.explained_variance_, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'g'
        ax2.set_ylabel('cummulative variance in %', color=color)  # we already handled the x-label with ax1
        var = [np.sum(self.pca_obj.explained_variance_ratio_[:n])*100 for n in range(self.pca_obj.n_components_)]
        ax2.plot(np.arange(0, self.pca_obj.n_components_), var, 'k-o', mec='g', mfc='w')
        if thrs:
            ax2.axhline(thrs, linestyle='-.', color='gray')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
    #

    
    def pairplot(self, show=True, n_dim = None, X_t = None, y_t = None):

        if n_dim is None:
            n_dim = self.n_dim

        X_ld = self.transfrom(self.X)
        if X_t is not None:
            X_tld = self.transfrom(X_t)
            X_ld = np.concatenate( (X_ld, X_tld), axis=0 )

        Y = 0
        if self.y is not None:
            Y = self.y
        if X_t is not None and y_t is not None:
            Y = np.concatenate( (Y, y_t), axis=0 )

        ld_X_df = pd.DataFrame(X_ld, columns = [f'c{i}' for i in range(self.n_dim)])
        ld_X_df['class'] = Y

        sns.pairplot(data=ld_X_df, hue='class', vars=[f'c{i}' for i in range(self.n_dim)], palette='tab10')
        if show:
            plt.show()
    #

#End ECG_PCA


def make_random_gaussian_coords(coords, n=5):

    m = coords.mean(axis=0)
    cov = np.cov(coords.T)
    coords = np.random.multivariate_normal(m, cov, size=n).T

    return coords
#

def make_random_bootstraped_coords(coords, n=5):

    btstrp_coords = np.array([resample(coords.T[i], n_samples=n) for i in range(coords.shape[1])]).T

    return btstrp_coords
#

def make_random_uniform_coords(coords, n=5):

    low = coords.min(axis=0)
    upp = coords.max(axis=0)

    unif_coords = np.array([np.random.uniform(low=l, high=u, size=n) for l, u in zip(upp, low)]).T

    return unif_coords
#

def ld_to_hd(coord, pca_mat, pca_var, mean, show=False):

    """
        Args:
        -----
            coord : array (n_samples, n_coeffs)
            pca_mat : array(n_coeff, N)
                The pca transformation matrix, where N is the high
                dimensionality of the original space.
            pca_var : array (n_coeff,)
                Variance of each principal component.
            mean : array (N,)

        Returns:
        --------
            HD : list[array] or array
    """

    if len(coord.shape) < 1:
        return False

    if len(coord.shape) == 1:
        coord = [coord]

    HD = []

    for c in coord:
        c_tr = c * np.sqrt(pca_var).flatten()
        hd = mean + (c_tr * pca_mat.T).T.sum(axis=0)
        HD.append(hd)

    if len(HD)==1:
        return HD[0]
    else:
        return HD
#

def hd_to_ld(X, pca_mat, pca_var, mean):

    coords=[]

    for x in X:
        c_tr = pca_mat.dot(x)
        coord = c_tr / np.sqrt(pca_var)
        coords.append(coord)

    if len(coords)==1:
        return np.array(coords[0])
    else:
        return np.array(coords)
#

def ppal_dirs(pca_mat, pca_var, mean, sigmas=[-2,2], show=False):

    n_comp = len(pca_mat)
    ppal_dirs = []
    for i in range(n_comp):
        for j in sigmas:
            d = np.zeros((n_comp,))
            d[i] = j
            hd = ld_to_hd(d,pca_mat, pca_var, mean)
            ppal_dirs.append(hd)
    if show:
        fig, axes = plt.subplots(n_comp, 2)
        for ax, d in zip(axes.ravel(), ppal_dirs):
            ax.plot(range(len(d)), d)
        plt.show()
    return ppal_dirs
#

def plot_pca_variance(pca, thrs=None):

    plt.rcParams.update({'font.size': 16})
    fig, ax1 = plt.subplots()

    color = 'b'
    #ax1.set_xticks(range(pca.n_components_))
    #ax1.set_xticklabels([f'{i}' for i in range(pca.n_components_)])
    ax1.set_xlabel('n components')
    ax1.set_ylabel('variance per component', color=color)
    ax1.bar(np.arange(0, pca.n_components_), pca.explained_variance_, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'g'
    ax2.set_ylabel('cummulative variance in %', color=color)  # we already handled the x-label with ax1
    var = [np.sum(pca.explained_variance_ratio_[:n])*100 for n in range(pca.n_components_)]
    ax2.plot(np.arange(0, pca.n_components_), var, 'k-o', mec='g', mfc='w')
    if thrs:
        ax2.axhline(thrs, linestyle='-.', color='gray')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def load_data(db_dir = None):

    db_dir = 'leads.LR'
    precs = {}
    precs['class'] = None
    for f in os.listdir(db_dir):
        name, t, _ = f.split('.')
        if t == 'X':
            precs[name] = np.loadtxt(db_dir+'/'+f)
        elif precs['class'] is None:
            precs['class'] = np.loadtxt(db_dir+'/'+f)

    return precs

def test_PCA():

    sim_ecgs = load_data()

    prec = 'V2'
    X_hd, y = sim_ecgs[prec], sim_ecgs['class']

    #PCA i plot variances
    pca = PCA(svd_solver = 'full')
    pca.fit(X_hd)
    mean = pca.mean_
    plot_pca_variance(pca, thrs=95)

    """
    ndim = 5
    pca_mat=pca.components_[:ndim]
    pca_var=pca.explained_variance_.ravel()[:ndim]
    ppal_dirs(pca_mat, pca_var, mean, show=True)


    X_ld = hd_to_ld(X_hd, pca_mat, pca_var, mean)
    ld_X_df = pd.DataFrame(X_ld, columns = [f'c{i}' for i in range(ndim)])
    ld_X_df['class'] = sim_ecgs['class']
    sns.pairplot(data=ld_X_df, hue='class', vars=[f'c{i}' for i in range(ndim)])
    plt.show()
    """

    XX_ld= hd_to_ld(X_hd, pca_mat=pca.components_, pca_var=pca.explained_variance_.ravel(), mean=mean)

    max_feat=15
    acc=[]
    for i in range(max_feat):
        X = XX_ld[:,:i]
        clf = SVC(C=0.2, kernel='linear')
        cv_res = cross_val_score(clf, X, y)
        acc.append(cv_res.mean())
        print(f"Nfeatures {i},\nSeparabilidad entre sub-poblaciones: {acc[-1]}")

    plt.plot(range(max_feat), acc)
    plt.xlabel('N dims')
    plt.ylabel('Accuracy')
    plt.title(f"SVM classifier")
    plt.show()

    return pca

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[:-window_len+1]

if __name__ == "__main__":

    #Demo
    sim_ecgs = load_data()

    prec = 'V2'
    X_v2, y = sim_ecgs[prec], sim_ecgs['class']

    ecg_pca = ECG_PCA(X=X_v2, y = y, n_dim=5)
    #ecg_pca.pairplot()
    #ecg_pca.set_n_dim(8)
    #ecg_pca.pairplot()
    ecg_pca.ppal_dirs()
