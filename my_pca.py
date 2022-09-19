import numpy as np
import healpy as hp
from sklearn.decomposition import PCA

def pca_subtraction(input_map, n_comp):
  


    pca = PCA()
  
    pca.fit(input_map)
    obs_pca = pca.transform(input_map)
    # ind_arr = np.reshape(np.arange(np.prod(obs_pca.shape)),obs_pca.shape)

    # mask = np.ones(obs_pca.shape)
    # for i in range(n_comp,obs_pca.shape[1]):
        # mask[ind_arr%obs_pca.shape[1]==i] = 0
    # obs_pca = obs_pca*mask
    obs_pca_red = pca.inverse_transform(obs_pca)
    obs_pca_red = input_map - obs_pca_red

                  
    return obs_pca_red

# pca3 = []
# for i in range(1, 6):
#     skymap = np.load('obss/obs%d.npy'%i)
#     print('running No. %d:'%i)
#     pca3.append(pca_subtraction(skymap, 3))
    

# #%%
# pca4 = np.array(pca3)
# pca5 = []
# for i in range(1, 6):
#     skymap5 = pca4[i-1, :, :]
#     print('running No. %d:'%i)
#     pca5.append(pca_subtraction(skymap5, 3))