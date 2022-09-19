import healpy as hp
import numpy as np

def diagonalOrder(matrix):
    order_list = []
    ROW, COL = matrix.shape
    for line in range(1, (ROW + COL)):
        start_col = max(0, line - ROW)
        count = min(line, (COL - start_col), ROW)
        for j in range(0, count):
            order_list.append(matrix[min(ROW, line) - j - 1, start_col + j])
    return np.array(order_list)

def my_rearr(image_reorder):
    lines_rearr = np.arange(1-len(image_reorder), len(image_reorder))*-1
    ss1_rearr = np.arange(1, int(len(image_reorder)/2)+1, 1).repeat(2)
    ss2_rearr = np.arange(int(len(image_reorder)/2), 0, -1).repeat(2)
    nums_rearr = np.hstack((ss1_rearr, ss2_rearr))
    nums_rearr = np.delete(nums_rearr, int(len(nums_rearr)/2))
    mm_rearr = []
    for i in range(len(lines_rearr)):
        array_rearr = image_reorder[:, ::-1].diagonal(offset=lines_rearr[i])
        temp1_rearr = np.roll(array_rearr, nums_rearr[i])
        for j in range(len(temp1_rearr)):
            mm_rearr.append(temp1_rearr[j])
    mm_rearr = np.array(mm_rearr)
    result_rearr =(mm_rearr[(image_reorder.reshape(-1)).astype(int)].reshape(len(image_reorder),len(image_reorder))).astype(int)
    return result_rearr

def my_resquare(side):
    ROW = COL = side
    M = np.arange(ROW*COL).reshape(ROW, COL)
    M2 = diagonalOrder(M)
    observation = np.arange(ROW*COL)
    reorder = np.zeros(ROW*COL)
    reorder[M2] = observation
    image = reorder.reshape(ROW, COL).astype(int)
    reimage = my_rearr(image)
    
    return reimage


def my_rearray(side):
    ROW = COL = side
    M = np.arange(ROW*COL).reshape(ROW, COL)
    M2 = diagonalOrder(M)
    observation = np.arange(ROW*COL)
    reorder = np.zeros(ROW*COL)
    reorder[M2] = observation
    image = reorder.reshape(ROW, COL).astype(int)
    rearray = my_rearr(image)

    return np.array(rearray.astype(int))


def my_square(side):
    ROW = COL = side
    M = np.arange(ROW*COL).reshape(ROW, COL)
    M2 = diagonalOrder(M)
    observation = np.arange(ROW*COL)
    reorder = np.zeros(ROW*COL)
    reorder[M2] = observation
    image = reorder.reshape(ROW, COL).astype(int)
    
    return image


def my_array(side):
    ROW = COL = side
    M = np.arange(ROW*COL).reshape(ROW, COL)
    M2 = diagonalOrder(M)
    observation = np.arange(ROW*COL)
    reorder = np.zeros(ROW*COL)
    reorder[M2] = observation

    return np.array(reorder.astype(int))



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

def arr_to_map(rearr_map):
    mmap = rearr_map.astype(float)
    nside = hp.npix2nside(rearr_map.size)
    rearr = np.load('obss/rearr_%d.npy'%nside)
    result = np.zeros_like(mmap.reshape(-1)).astype(float)
    for i in range(192):
        for idx, num in enumerate(mmap[i, :, :].reshape(-1)):
            result[rearr[i, :,:].reshape(-1)[idx]] = num
    return result

def nside_to_remap(nside1, nside2):
    mmap1 = np.arange(hp.nside2npix(nside1))
    mmap_temp = []
    for i in mmap1:
        zero_map = np.zeros_like(mmap1)
        zero_map[i] = 1
        zero_udmap = hp.ud_grade(zero_map, nside2)
        one_udmap = np.where(zero_udmap == 1)[0]
        one_tempmap = one_udmap[my_resquare(int(nside2/nside1))]
        mmap_temp.append(one_tempmap)
    return np.array(mmap_temp)