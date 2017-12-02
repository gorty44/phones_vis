# This converts `fingerprints.npy` to `.tsv` formatted t-SNE embeddings and plots of those embeddings in the `tsne/` and `plot/` folders respectively. If you add multiple values to `perplexity` and `initial_dims` then all combinations will be computed (in parallel). Good perplexities are in the range 1-200 with the best range around 30-100. Good `initial_dims` are in the range 30 and higher, with the dimensionality of your input data being the highest possible value (e.g., a 32x32 fingerprint would have a highest possible `initial_dims` value of 32x32=1024).
# 
# Change the "mode" to try different t-SNE variations.
# * "fingerprints" will only use `fingerprints.npy`
# * "predicted_labels" will only use `predicted_labels.npy`
# * "predicted_encoding" will only use `predicted_encoding.npy`
# * "combined" will use all of the above data
data_root = 'data/'
initial_dims = [40]
perplexities = [50]
mode = 'fingerprints'

from matplotlib import pyplot as plt
from time import time
from os.path import join
import numpy as np
import itertools
from utils import *
from bhtsne import *

def save_tsv(data, fn):
    np.savetxt(fn, data, fmt='%.5f', delimiter='\t')
def tsne(data, data_root, prefix, initial_dims=30, perplexity=30):
    mkdir_p(data_root + 'tsne')
    mkdir_p(data_root + 'plot')
    
    figsize = (16,16)
    pointsize = 2

    X_2d = list(run_bh_tsne(data, initial_dims=initial_dims, perplexity=perplexity, no_dims=2))
    X_2d = normalize(np.array(X_2d))
    save_tsv(X_2d, join(data_root, 'tsne/{}.{}.{}.2d.tsv'.format(prefix, initial_dims, perplexity)))
    
    plt.figure(figsize=figsize)
    plt.scatter(X_2d[:,0], X_2d[:,1], edgecolor='', s=pointsize)
    plt.tight_layout()
    plt.savefig(join(data_root, 'plot/{}.{}.{}.png'.format(prefix, initial_dims, perplexity)))
    plt.close()
    
    X_3d = list(run_bh_tsne(data, initial_dims=initial_dims, perplexity=perplexity, no_dims=3))
    X_3d = normalize(np.array(X_3d))
    save_tsv(X_3d, join(data_root, 'tsne/{}.{}.{}.3d.tsv'.format(prefix, initial_dims, perplexity)))
    
    plt.figure(figsize=figsize)
    plt.scatter(X_2d[:,0], X_2d[:,1], edgecolor='', s=pointsize, c=X_3d)
    plt.tight_layout()
    plt.savefig(join(data_root, 'plot/{}.{}.{}.png'.format(prefix, initial_dims, perplexity)))
    plt.close()


# load and normalize any dataset we need
if mode == 'fingerprints' or mode == 'combined':
    fingerprints = np.load(join(data_root, 'fingerprints.npy'))
    fingerprints = fingerprints.reshape(len(fingerprints), -1)
if mode == 'predicted_labels' or mode == 'combined':
    predicted_labels = np.load(join(data_root, 'predicted_labels.npy'))
    predicted_labels -= predicted_labels.min()
    predicted_labels /= predicted_labels.max()
if mode == 'predicted_encoding' or mode == 'combined':
    predicted_encoding = np.load(join(data_root, 'predicted_encoding.npy'))
    std = predicted_encoding.std(axis=0)
    predicted_encoding = predicted_encoding[:, std > 0] / std[std > 0]
    
if mode == 'fingerprints':
    data = fingerprints
if mode == 'predicted_labels':
    data = predicted_labels
if mode == 'predicted_encoding':
    data = predicted_encoding
if mode == 'combined':
    data = np.hstack((fingerprints, predicted_labels, predicted_encoding))
    
print(data.shape)

data = data.astype(np.float64)
def job(params):
    start = time()
    tsne(data, data_root, mode, initial_dims=params[0], perplexity=params[1])
    print('initial_dims={}, perplexity={}, {} seconds'.format(params[0], params[1], time() - start))

params = list(itertools.product(initial_dims, perplexities))

for param in params:
    job(param)
