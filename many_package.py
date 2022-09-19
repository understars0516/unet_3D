import numpy as np
import healpy as hp
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import os
from my_pca import pca_subtraction

from unet import unet_3d

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def gen_rearr(nside):
    if (nside==1):
        return np.array([0,1,2,3])
    else:
        smaller = np.reshape(gen_rearr(nside-1),(2**(nside-1),2**(nside-1)))
        npixsmaller = 2**(2*(nside-1))
        top = np.concatenate((smaller,smaller+npixsmaller),axis=1)
        bot = np.concatenate((smaller+2*npixsmaller,smaller+3*npixsmaller),axis=1)
        whole = np.concatenate((top,bot))
        return whole.flatten()

def pca_subtraction(input_map, n_comp, index_array, n_nu=None, 
                    nu_start=0, n_nu_out=64, n_nu_avg=3):
  
    # "GLOBAL" parameters ----------------------------------------------------
    MAP_NSIDE = 256
    WINDOW_NSIDE = 4
    # resolution of the outgoing window
    NPIX_WINDOW = (MAP_NSIDE/WINDOW_NSIDE)**2
    # actual side length of window
    WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))
    # ------------------------------------------------------------------------
    rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))
    nwinds = int(hp.nside2npix(WINDOW_NSIDE))

    # initialize the PCA algorithm
    pca = PCA()
    # allocate the output array
    pca_reduced_out = np.zeros(input_map.shape).reshape(192, 64, 64, 64)
    
    # flatten input map into full-sky maps stacked in frequency
    # input_map = input_map.reshape((-1, 64))
    input_map = input_map.reshape((-1, 64))
  
    # do PCA removal of n_comp components
    pca.fit(input_map)
    obs_pca = pca.transform(input_map)
    ind_arr = np.reshape(np.arange(np.prod(obs_pca.shape)),obs_pca.shape)

  
    mask = np.ones(obs_pca.shape)
    for i in range(n_comp,obs_pca.shape[1]):
        mask[ind_arr%obs_pca.shape[1]==i] = 0
    obs_pca = obs_pca*mask
    obs_pca_red = pca.inverse_transform(obs_pca)
    print("Now I'm doing the minimum subtraction...")
    print("...removing the first %d principal components"%(n_comp))
    obs_pca_red = input_map - obs_pca_red

    inds = np.arange(hp.nside2npix(MAP_NSIDE))
    inds_nest = hp.ring2nest(MAP_NSIDE,inds)

    for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
        inds_in = np.where((inds_nest//NPIX_WINDOW)==PIX_SELEC)
        to_rearr_inds = inds_nest[inds_in] - PIX_SELEC*NPIX_WINDOW
        to_rearr = obs_pca_red[inds_in]
        to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
        to_rearr = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH,n_nu_out))
        ind = (0)*nwinds + PIX_SELEC
        pca_reduced_out[ind] = to_rearr
                  
    return pca_reduced_out

def cosmo_subtraction(input_map, n_nu_out=64):
  
    # "GLOBAL" parameters ----------------------------------------------------
    MAP_NSIDE = 256
    # SIM_NSIDE = MAP_NSIDE
    WINDOW_NSIDE = 4
    # NUM_SIMS = 1
    # resolution of the outgoing window
    NPIX_WINDOW = (MAP_NSIDE/WINDOW_NSIDE)**2
    # actual side length of window
    WINDOW_LENGTH = int(np.sqrt(NPIX_WINDOW))
    # ------------------------------------------------------------------------

    # rearrange indices
    rearr = gen_rearr(int(np.log2(MAP_NSIDE/WINDOW_NSIDE)))
    nwinds = int(hp.nside2npix(WINDOW_NSIDE))
  
    cosmo_out = np.zeros(input_map.shape).reshape(192, 64, 64, 64)

    input_map = input_map.reshape((-1, 64))
  
    inds = np.arange(hp.nside2npix(MAP_NSIDE))
    inds_nest = hp.ring2nest(MAP_NSIDE,inds)

    for PIX_SELEC in np.arange(hp.nside2npix(WINDOW_NSIDE)):
        # get the indices of the pixels which actually are in the larger pixel
        inds_in = np.where((inds_nest//NPIX_WINDOW)==PIX_SELEC)
        to_rearr_inds = inds_nest[inds_in] - PIX_SELEC*NPIX_WINDOW
        to_rearr = input_map[inds_in]
        to_rearr = (to_rearr[np.argsort(to_rearr_inds)])[rearr]
        to_rearr = np.reshape(to_rearr,(WINDOW_LENGTH,WINDOW_LENGTH,n_nu_out))
        ind = (0)*nwinds + PIX_SELEC
        cosmo_out[ind] = to_rearr
                  
    return cosmo_out
