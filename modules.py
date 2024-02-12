import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import torch

def RankMe(embedding):
    s = np.linalg.svd(embedding, compute_uv=False)
    p = s / np.abs(s.sum())
    entropy = -(p*np.log(p)).sum()
    rankme = np.exp(entropy)
    return rankme
def get_eigenspectrum(activations_np,max_eigenvals=2048):
    feats = activations_np.reshape(activations_np.shape[0],-1)
    feats_center = feats - feats.mean(axis=0)
    pca = PCA(n_components=min(max_eigenvals, feats_center.shape[0], feats_center.shape[1]), svd_solver='full')
    pca.fit(feats_center)
    eigenspectrum = pca.explained_variance_ratio_
    return eigenspectrum

def fit_powerlaw(arr, start, end):
    x_range = np.arange(start, end + 1).astype(int)
    y_range = arr[x_range - 1]  # because the first eigenvalue is at index 0, so eigenval_{start} is at index (start-1)
    reg = LinearRegression().fit(np.log(x_range).reshape(-1, 1), np.log(y_range).reshape(-1, 1))
    y_pred = np.exp(reg.coef_ * np.log(x_range).reshape(-1, 1) + reg.intercept_)
    return -reg.coef_[0][0], x_range, y_pred

def stringer_get_powerlaw(ss, trange):
    # COPIED FROM Stringer+Pachitariu 2018b github repo! (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:, np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:, np.newaxis], np.ones((ss.size, 1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    max_range = 500 if len(ss) >= 512 else len(
        ss) - 10  # subtracting 10 here arbitrarily because we want to avoid the last tail!
    fit_R2 = r2_score(y_true=logss[trange[0]:max_range], y_pred=np.log(np.abs(ypred))[trange[0]:max_range])
    try:
        fit_R2_100 = r2_score(y_true=logss[trange[0]:100], y_pred=np.log(np.abs(ypred))[trange[0]:100])
    except:
        fit_R2_100 = None
    return alpha, ypred, fit_R2, fit_R2_100