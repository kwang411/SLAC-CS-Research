from __future__ import print_function
from __future__ import division
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import Callback
import numpy as np
from numpy.random import seed
import os
import random
import tensorflow as tf
from keras.models import model_from_json
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

iterations = 100
epochs = 20000
n_neighbors = 30
weights = []
weight_layer = 0

xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

import re
from ast import literal_eval

X = """[[5.5044885 -5.5959344 -5.1676397  5.5634074]
 [-3.4730031  3.9971626  8.290578   7.6628895]
 [8.767298   7.899208  -3.9842615  4.1127563]
 [5.110225  -5.7416444  5.163678  -6.258786 ]
 [-8.4813595 -7.5828295  3.6933115 -3.990437 ]
 [5.0955997  8.369304  -4.0160613 -8.258075 ]
 [-7.7072396 -8.768881  -4.15464    3.935515 ]
 [-7.795831   8.791065  -3.415188  -3.5934584]
 [8.419221  -8.43345   -3.9453008 -3.9798753]
 [-5.6615243 -5.4694986  5.7470927  5.5409718]
 [3.5693712 -3.3332837  8.508117   8.445337 ]
 [4.6251583  7.27673    4.6185913  7.1218743]
 [-3.762027   3.8699229 -8.04816   -8.459708 ]
 [-5.0923595 -5.398147  -5.1660466 -5.806125 ]
 [-4.973972   6.3447237 -5.0441914  6.4706287]
 [5.9103556 -5.519969  -5.7469864  5.8850093]
 [-4.31322    4.098057  -8.193453  -8.959643 ]
 [-8.015529   8.402092   3.1256425  3.6962323]
 [5.529783   5.9632664 -5.4024553 -6.3968344]
 [-8.859722  -8.922084  -3.4903512  3.8241081]
 [-6.032752   5.8889894  5.3572907 -6.2617173]
 [-6.319627   6.356305   6.3688283 -5.6847153]
 [6.462197   5.4316483 -6.837786  -5.7868395]
 [-8.04072    7.9967184  3.1619697  3.3380308]
 [-3.796314   3.9632125 -8.408154  -8.676466 ]
 [3.6010363  3.7929564  8.167711  -8.193304 ]
 [6.1185136 -5.9193225 -6.204758   5.381818 ]
 [5.700419   5.0376563 -5.7349415 -5.4699697]
 [8.231711  -8.51803    3.9535148  3.7917962]
 [-3.3112118  2.9841988 -8.281447  -7.844318 ]
 [-8.613477  -8.10344   -3.6525595  3.5339496]
 [8.269363   8.5610895  3.8013415 -3.96688  ]
 [6.0355177 -6.136958  -6.2129664  6.255385 ]
 [6.618305   4.621661   6.7907887  4.62509  ]
 [-8.3910885 -8.79463   -4.5148864  4.270094 ]
 [6.2737236 -6.45752   -5.9995074  6.265124 ]
 [3.5436113  3.2981427  7.8506093 -8.166137 ]
 [6.0966883  5.4729733 -5.7448745 -4.9144   ]
 [-6.417684  -6.0049114  6.4961014  5.759174 ]
 [5.331264  -5.314103  -4.9778504  5.0518126]
 [-3.9613545 -3.845105  -9.275012   8.761464 ]
 [-4.913122  -4.7732887 -6.2447505 -5.0280724]
 [8.220534  -8.528026  -3.3999088 -3.4939442]
 [-4.416417  -3.6994376 -8.002445   8.313104 ]
 [5.6854815  5.9519677 -5.500941  -6.360398 ]
 [-4.5862117  4.9169703 -4.795343   5.837531 ]
 [6.5777736  5.5786943 -6.1691694 -6.1321387]
 [-6.409338   4.7770762 -6.38792    4.808336 ]
 [-5.860679   5.5913157  6.0082464 -5.866915 ]
 [-5.504088   5.995935   5.833399  -6.4064927]
 [-6.1433163  4.510293  -5.4933414  4.4894943]
 [-5.0341434  4.6899095 -5.74695    4.790284 ]
 [-3.3297653 -3.0081844  7.5700464 -8.3471365]
 [7.8196507 -8.588441   3.4738486  3.5123086]
 [-8.329799   8.335902  -3.9050326 -3.892882 ]
 [-8.205322   8.777051  -3.8853738 -4.096609 ]
 [5.9641542 -6.552076  -6.3238645  6.2572093]
 [-6.5137496  5.985477   6.6144958 -6.292779 ]
 [-4.7030144  7.1812124 -4.6989384  6.512058 ]
 [4.680476  -5.748434   4.6924    -5.528359 ]
 [6.0337024 -5.0334024  5.2740993 -4.9149666]
 [7.7177014 -8.127328   3.6199381  3.127018 ]
 [-6.329111  -5.521577   5.649657   5.7722464]
 [-5.7770925  5.4050813  5.821326  -5.7063313]
 [4.416485  -3.7957878  8.36816    7.8797674]
 [7.666128   8.005173   4.0593905 -3.2289395]
 [7.9898505  8.338073   3.6445112 -3.7734673]
 [-4.7338543 -6.481072  -4.7541394 -5.7904305]
 [-6.1629386  5.3423443  6.147208  -5.5121007]
 [-6.302548  -4.8750362 -6.672848  -4.8648896]
 [6.410969   4.702738   6.1861734  4.693011 ]
 [6.069324  -5.100219   5.3501964 -5.067703 ]
 [5.673997  -5.9800973 -5.436837   5.8965783]
 [5.250573  -4.7260566  5.6754956 -4.776253 ]
 [4.6113973  5.86878    4.561856   5.0508137]
 [-3.6018374 -3.3899782 -8.36454    8.356848 ]
 [8.284373   7.8970456 -3.6119542  4.483752 ]
 [5.5875397 -5.8594236 -5.8344316  5.2063804]
 [-3.059961  -3.063348   7.9200187 -8.09606  ]
 [-6.253227  -5.2531147  6.203957   5.903812 ]
 [-5.328562  -5.954135   5.8565674  5.5524135]
 [7.6214614  7.918478   1.8846462 -3.297655 ]
 [6.0044765 -5.417742  -5.7116327  5.8292303]
 [-5.9702945  5.8315153  6.029764  -5.9244576]
 [6.306002  -5.6215277 -6.0952206  5.9682603]
 [3.5652356 -3.8764098 -8.169312  -8.8103075]
 [5.188962   4.982534   5.1583996  5.008035 ]
 [6.184995   5.7391467 -6.2215915 -5.506561 ]
 [5.8367543 -4.611109   5.79678   -4.6158133]
 [-5.1822767 -5.9809866  5.4785767  6.035393 ]
 [7.8294992  8.056298  -3.2011719  3.5632763]
 [6.5492787 -6.004353  -6.3637066  5.4082174]
 [-5.894731  -5.8027806  5.763825   5.503325 ]
 [7.8908553 -8.40016   -3.6565444 -3.8555915]
 [-8.031691  -8.133458   3.3193173 -3.9210784]
 [-5.821414   6.456529   5.5193343 -6.4347754]
 [4.9728866 -5.614537   4.9728136 -6.1289597]
 [-5.964271  -5.093229  -5.3571525 -5.0688105]
 [-7.790229  -8.177528   2.7467763 -3.8497283]
 [5.9156303  6.279135  -6.1226764 -6.679048 ]]"""
X = re.sub('\s+', ',', X)


X = np.array(literal_eval(X))
y = [str(i) for i in range(iterations)]
#y = [j+str(i) for i in range(iterations) for j in ['a']]
n_samples, n_features = X.shape

print(X)
print(y)
print(n_samples)
print(n_features)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(Z, title=None):
    z_min, z_max = np.min(Z, 0), np.max(Z, 0)
    Z = (Z - z_min) / (z_max - z_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(Z.shape[0]):
        plt.text(Z[i, 0], Z[i, 1], y[i],
                 # color=plt.cm.Set1(int(y[i][1:]) / 10.),
                 fontdict={'weight': 'bold', 'size':9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# #----------------------------------------------------------------------
# # Plot images of the digits
# n_img_per_row = 20
# img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')


#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the weights")


#----------------------------------------------------------------------
# Projection on to the first 2 principal components

print("Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
plot_embedding(X_pca,
               "Principal Components projection of the weights (time %.2fs)" %
               (time() - t0))

# #----------------------------------------------------------------------
# # Projection on to the first 2 linear discriminant components

# print("Computing Linear Discriminant Analysis projection")
# X2 = X.copy()
# X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
# t0 = time()
# X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
# plot_embedding(X_lda,
#                "Linear Discriminant projection of the digits (time %.2fs)" %
#                (time() - t0))


#----------------------------------------------------------------------
# Isomap projection of the digits dataset
print("Computing Isomap embedding")
t0 = time()
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
print("Done.")
plot_embedding(X_iso,
               "Isomap projection of the weights (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
t0 = time()
X_lle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_lle,
               "Locally Linear Embedding of the weights (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='modified')
t0 = time()
X_mlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_mlle,
               "Modified Locally Linear Embedding of the weights (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# HLLE embedding of the digits dataset
print("Computing Hessian LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='hessian')
t0 = time()
X_hlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_hlle,
               "Hessian Locally Linear Embedding of the weights (time %.2fs)" %
               (time() - t0))


#----------------------------------------------------------------------
# LTSA embedding of the digits dataset
print("Computing LTSA embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='ltsa')
t0 = time()
X_ltsa = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_ltsa,
               "Local Tangent Space Alignment of the weights (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# MDS  embedding of the digits dataset
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
X_mds = clf.fit_transform(X)
print("Done. Stress: %f" % clf.stress_)
plot_embedding(X_mds,
               "MDS embedding of the weights (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# Random Trees embedding of the digits dataset
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
t0 = time()
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_transformed)

plot_embedding(X_reduced,
               "Random forest embedding of the weights (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# Spectral embedding of the digits dataset
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
t0 = time()
X_se = embedder.fit_transform(X)

plot_embedding(X_se,
               "Spectral embedding of the weights (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=2, method='exact')
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the weights (time %.2fs)" %
               (time() - t0))

plt.show()
print(tsne.kl_divergence_)

