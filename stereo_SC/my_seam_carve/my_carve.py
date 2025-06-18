# USAGE:
# python seam_carving.py (-resize | -remove) -im IM -out OUT [-mask MASK]
#                        [-rmask RMASK] [-dy DY] [-dx DX] [-vis] [-hremove] [-backward_energy]
# Examples:
# python seam_carving.py -resize -im demos/ratatouille.jpg -out ratatouille_resize.jpg 
#        -mask demos/ratatouille_mask.jpg -dy 20 -dx -200 -vis
# python seam_carving.py -remove -im demos/eiffel.jpg -out eiffel_remove.jpg 
#        -rmask demos/eiffel_mask.jpg -vis

import numpy as np
import cv2
import argparse
from numba import jit
import numba as nb
from scipy import ndimage as ndi

from matplotlib import pyplot as plt

SEAM_COLOR = np.array([0, 0, 255])        # seam visualization color (BGR)
SHOULD_DOWNSIZE = True                    # if True, downsize image for faster carving
DOWNSIZE_WIDTH = 500                      # resized image width if SHOULD_DOWNSIZE is True
ENERGY_MASK_CONST = 100000.0              # large energy value for protective masking
MASK_THRESHOLD = 10                       # minimum pixel intensity for binary mask
USE_FORWARD_ENERGY = True                 # if True, use forward energy algorithm

########################################
# UTILITY CODE
########################################

def visualize(im, boolmask=None):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        # vis[np.where(boolmask == False)] = SEAM_COLOR
        vis[~boolmask] = SEAM_COLOR
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis

def visualize_seam(im, boolmask=None):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        # vis[np.where(boolmask == False)] = SEAM_COLOR
        vis[~boolmask] = SEAM_COLOR
    cv2.imwrite("C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/results/sc_img/visual_seam.png", vis)
    return vis

def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)

def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)    

########################################
# ENERGY FUNCTIONS
########################################

@nb.jit(cache=True, nopython=True)
def roll_2d(im, axis, way):
    """Roll the array by the specified amount"""
    arr = np.copy(im)  # 元の配列を変更しないようにコピーを作成
    if axis == 0:   # 縦
        if way == 1:   # 上
            arr[:-1] = im[1:]
        elif way == -1:  # 下
            arr[1:] = im[:-1]
    elif axis == 1: # 横
        if way == 1:   # 右
            arr[:,:-1] = im[:,1:]
        elif way == -1:  # 左
            arr[:,1:] = im[:,:-1]
    return arr


@nb.jit(cache=True, nopython=True)
def roll_1d(im, way):
    """Roll the array by the specified amount"""
    arr = np.copy(im)  # 元の配列を変更しないようにコピーを作成
    if way == 1:   # 右
        arr[:-1] = im[1:]
    elif way == -1:  # 左
        arr[1:] = im[:-1]
    return arr

@nb.jit(cache=True, nopython=True)
def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.

    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = im.shape[:2]

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    # -----------------------------------------------------------
    U = roll_2d(im, 0, 1)   # 上にシフト
    L = roll_2d(im, 1, -1)  # 左にシフト
    R = roll_2d(im, 1, 1)   # 右にシフト
    # -----------------------------------------------------------
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = roll_1d(mU, -1)
        mR = roll_1d(mU, 1)
        
        mULR = np.stack((mU, mL, mR))
        cULR = np.stack((cU[i], cL[i], cR[i]))
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)

        for col in range(w):
            m[i, col] = mULR[argmins[col], col]
            energy[i, col] = cULR[argmins[col], col]

    return energy

########################################
# SEAM HELPER FUNCTIONS
######################################## 

def remove_seam(im, boolmask_3c):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask_3c] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))

def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))

@nb.jit(cache=True, nopython=True)
def dp_forward(M, backtrack):
    h, w = M.shape
    for i in range(1, h):
        prev_row = M[i - 1]
        for j in range(w):
            # 左端
            if j == 0:
                idx = np.argmin(prev_row[j:j+2])
                min_energy = prev_row[j + idx]
                backtrack[i, j] = j + idx

            # 右端
            elif j == w - 1:
                idx = np.argmin(prev_row[j-1:j+1])
                min_energy = prev_row[j - 1 + idx]
                backtrack[i, j] = j - 1 + idx

            # 中央部
            else:
                idx = np.argmin(prev_row[j-1:j+2])
                min_energy = prev_row[j - 1 + idx]
                backtrack[i, j] = j - 1 + idx

            M[i, j] += min_energy
    return M, backtrack


@nb.jit(cache=True, nopython=True)
def get_minimum_seam(im, mask=None, remove_mask=None):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h, w = im.shape[:2]
    
    M = forward_energy(im)

    if mask is not None:
        M[np.where(mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST

    backtrack = np.zeros_like(M, dtype=np.int32)

    # populate DP matrix
    # バックトラック配列を作成
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            M[i, j] += min_energy

    # backtrack to find path
    boolmask = np.ones((h, w), dtype = np.bool_)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        j = backtrack[i, j]

    return boolmask, M

########################################
# MAIN ALGORITHM
######################################## 

def seams_removal(im, num_remove, mask=None, vis=False, rot=False):
    gray = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    for seam in range(num_remove):
        boolmask, M = get_minimum_seam(gray, mask)
        if seam == 0:
            cv2.imwrite("C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/images/sample/img/forward_energy_roll.jpg", M.astype(np.uint8))
        if vis:
            if seam == 0:
                visualize_seam(im, boolmask)
            visualize(im, boolmask)
        gray = remove_seam_grayscale(gray, boolmask)
        im = remove_seam(im, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    return im, mask

########################################
# MAIN DRIVER FUNCTIONS
########################################

def seam_carve(im, dy, dx, mask=None, vis=False):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    if mask is not None:
        mask = mask.astype(np.float64)

    output = im

    if dx < 0:
        output, mask = seams_removal(output, -dx, mask, vis)

    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_removal(output, -dy, mask, vis, rot=True)
        output = rotate_image(output, False)

    return output

if __name__ == '__main__':
    print("seam_carve.py")
    im = cv2.imread("../images/report/Broadway_tower_edit.jpg")

    mask = None
    rmask = None

    # downsize image for faster processing
    h, w = im.shape[:2]

    dy, dx = 0, -100
    output = seam_carve(im, dy, dx, mask, True)
    cv2.imwrite("C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/images/sample/img/sample_sc.png", output)
