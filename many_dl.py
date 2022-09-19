import numpy as np
import healpy as hp
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import os
from my_pca import pca_subtraction
from unet import unet_3d
from sklearn.decomposition import PCA
from many_package import *
from datetime import datetime
from rearr_best import *

smooth_beam = 'cosine'
cosmo_obs = np.load('input/HI_freq64_seed5.npy')
obs = np.load('input/obs_freq64_seed5_%s.npy'%smooth_beam)

pca_nu = 3; splt = 5
pca3 = []; cosmo3 = []
index_rearr = np.load('input/rearr_nside4.npy')
for i,sky in enumerate(np.split(obs, splt, axis=0)):
        print('PCA mode:  %d'%(i+1))
        cosmo_temp = cosmo_obs[i, :, :]
        pca3.append(pca_subtraction(sky, pca_nu, index_rearr, n_nu=64, nu_start=0, n_nu_out=64, n_nu_avg=3))
        cosmo3.append(cosmo_subtraction(cosmo_temp, n_nu_out=64))

cosmo = np.concatenate(cosmo3, axis=0)
pca_test = np.concatenate(pca3, axis=0)

params = {
    'nu_dim'        : 64,
    'x_dim'         : 64,
    'n_filters'     : 40,
    'conv_width'    : 3,
    'network_depth' : 3,
    'batch_size'    : 16,
    'num_epochs'    : 5,
    'dropout'       : 0.2,
    'act'           : 'relu',
    'lr'            : 0.0001,
    'wd'            : 1e-5,
    'batchnorm_in'  : True,
    'batchnorm_out' : True,
    'batchnorm_up'  : True,
    'batchnorm_down': True,
    'maxpool'       : True,
    'momentum'      :  0.02,
    'model_num'     : 1,
    'load_model'    : False,
}

net = unet_3d.unet3D(n_filters=params['n_filters'],
                      conv_width=params['conv_width'],
                      nu_dim=params['nu_dim'],
                      x_dim=params['x_dim'],
                      network_depth=params['network_depth'],
                      dropout=params['dropout'],
                      batchnorm_down=params['batchnorm_down'],
                      batchnorm_in=params['batchnorm_in'],
                      batchnorm_out=params['batchnorm_out'],
                      batchnorm_up=params['batchnorm_up'],
                      maxpool=params['maxpool'],
                      momentum=params['momentum']
                      )
net = net.build_model()
 
net.compile(optimizer=tf.optimizers.Adam(learning_rate=params['lr'],
                                                 beta_1=0.9, beta_2=0.999, amsgrad=False),
                                                 loss="logcosh",metrics=["mse", "logcosh"])


x = np.expand_dims(pca_test, axis=-1)
y = np.expand_dims(cosmo, axis=-1)

N_TRAIN = int(3*192)
N_VAL  = int(1*192)

x_train = x[:N_TRAIN]
x_val   = x[N_TRAIN:-N_VAL]
x_test  = x[-N_VAL:]

y_train = y[:N_TRAIN]
y_val   = y[N_TRAIN:-N_VAL]
y_test  = y[-N_VAL:]



history = net.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['num_epochs'],validation_data=(x_val, y_val))

y_pred = net.predict(x_test)

cosmo_cut = y_test[:,:,:,12,0].reshape(-1)[index_rearr]
unet_cut = y_pred[:,:,:,12,0].reshape(-1)[index_rearr]
obs_cut = x_test[:,:,:,12,0].reshape(-1)[index_rearr]


lmax = 80
import matplotlib.pyplot as plt
obs_cl = hp.anafast(obs_cut, lmax=lmax)
unet_cl = hp.anafast(unet_cut, lmax=lmax)
cosmo_cl = hp.anafast(cosmo_cut, lmax=lmax)
ell = np.arange(0, len(obs_cl))
plt.loglog(ell, ell*(ell+1)*cosmo_cl, linewidth=2, label='REAL')
plt.loglog(ell, ell*(ell+1)*obs_cl, linewidth=2, label='PCA')
plt.loglog(ell, ell*(ell+1)*unet_cl, linewidth=2, label='PCA + UNET')
plt.xlabel('Multipole $\ell$')
plt.ylabel('$\ell(\ell+1)C_l/2\pi$')
plt.title('%s smooth'%smooth_beam)
plt.legend()
plt.savefig('%s.png'%smooth_beam, dpi=500)
