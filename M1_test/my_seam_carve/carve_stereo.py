# import warnings
from enum import Enum
from typing import Optional, Tuple
import cv2
import numba as nb

import numpy as np
from scipy.ndimage import sobel
import pandas as pd

from matplotlib import pyplot as plt

from tqdm import tqdm
import time

DROP_MASK_ENERGY = 1000000
KEEP_MASK_ENERGY = 1000000

FLAG = -1

class OrderMode(str, Enum):
    WIDTxy_FIRST = "width-first"
    HEIGHT_FIRST = "height-first"

class EnergyMode(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"

def visualize(im, boolmask=None):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == True)] = 255
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis

def _arr_enum(enum_class) -> Tuple:
    return tuple(x.value for x in enum_class)

def _check_src(src: np.ndarray) -> np.ndarray:
    """Ensure the source to be RGB or grayscale"""
    src = np.asarray(src)
    if src.size == 0 or src.ndim not in (2, 3):
        raise ValueError(
            f"expect a 3d rgb image or a 2d grayscale image, got image in shape {src.shape}"
        )
    return src

def _check_mask(mask: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Ensure the mask to be a 2D grayscale map of specific shape"""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expect mask to be a 2d binary map, got shape {mask.shape}")
    if mask.shape != shape:
        raise ValueError(
            f"expect the shape of mask to match the image, got {mask.shape} vs {shape}"
        )
    return mask

@nb.njit
def _get_seam_mask(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w = src.shape
    mask = np.zeros((h, w), dtype=np.bool_)
    for i in range(h):
        mask[i, seam[i]] = True
    return mask

@nb.njit
def _clip(value: int, min_value: float, max_value: float) -> np.ndarray:
    """
    配列の値が指定した範囲内に収まるように切り取る。
    引数
        value : クリッピングする入力配列。
        min_value : クリッピングする最小値。
        max_value : クリッピングする最大値。
    戻り値
        指定された範囲内の値でクリッピングされた配列。
    """
    return max(min_value, min(value, max_value))

# ペア画像の一行のコストを計算
@nb.jit(cache=True, nopython=True)
def _calc_pair_energy(
    gray: np.ndarray,  # すでに両端は拡張済み
    disp: np.ndarray, 
    height: int,
) :
    _, ex_width = gray.shape
    width = ex_width - 2  # 両端を拡張しているので、実際の幅は-2する必要がある
    inf = np.float32(np.inf)

    choices = np.zeros((3, width), dtype=np.float32)

    H_UPPER = height - 1  # 上の行のインデックス
    H_LOWER = height      # 下の行のインデックス

    for idx in range(0, width):
        # idx   : 注目ピクセルのx座標
        # d_idx : 注目ピクセルのx座標から視差値を引いた値

        d_idx = _clip(idx - disp[H_UPPER, idx], 0, width-1)  # ペア画像の注目ピクセル
        
        disp_ul = disp[H_LOWER, _clip(idx-1, 0, width-1)]   # ulから引く視差値
        ul      = _clip(idx-1 - disp_ul, 0, width-1)  # ペア画像の左上のピクセル(idx-1がdispの範囲を超えないようにclipしている)
        disp_un = disp[H_UPPER, idx]    # unから引く視差値
        un      = _clip(idx - disp_un, 0, width-1)  # ペア画像の左上のピクセル(idx-1がdispの範囲を超えないようにclipしている)
        disp_ur = disp[H_LOWER, _clip(idx+1, 0, width-1)]   # urから引く視差値
        ur      = _clip(idx+1 - disp_ur, 0, width-1)  # ペア画像の右上のピクセル(idx+1がdispの範囲を超えないようにclipしている)

        # 注目ピクセルが削除されたときに発生するエネルギー
        if d_idx-1 < 0 or d_idx+1 >= width:
            mid = inf  # 画像の端を超える場合は無限大
        else:
            mid = np.abs(gray[H_LOWER, d_idx-1] - gray[H_LOWER, d_idx+1])

        for way, under_idx in enumerate([ul, un, ur]):
            current_cost = mid

            # スライス操作の開始点と終了点を直接定義
            if under_idx < d_idx:
                s1_start, s1_end = under_idx, d_idx
                s2_start, s2_end = under_idx + 1, d_idx + 1
            elif under_idx == d_idx:
                # このケースではスライス計算は不要
                choices[way, idx] = current_cost
                continue # スライス計算をスキップして次の way へ
            else: # under_idx > d_idx
                s1_start, s1_end = d_idx + 1, under_idx + 1
                s2_start, s2_end = d_idx, under_idx
            
            # 各スライスの長さを計算
            len1 = s1_end - s1_start
            len2 = s2_end - s2_start
            
            # スライスが有効かどうかの最終チェックと、物理的な終端を超える場合のinf付加
            # 「空リストは発生しない」という保証に基づき、len1 > 0 のチェックを削除
            if (len1 == len2 and
                s1_start >= 0 and s1_end <= width and 
                s2_start >= 0 and s2_end <= width):
                
                current_cost += np.sum(np.abs(gray[H_UPPER, s1_start:s1_end] - gray[H_LOWER, s2_start:s2_end]))
            else:
                current_cost = np.inf
            
            choices[way, idx] = current_cost

    return choices

# # ペア画像の一行のコストを計算
# @nb.jit(cache=True, nopython=True)
# def _calc_pair_energy(
#     gray: np.ndarray,  # すでに両端は拡張済み
#     disp: np.ndarray, 
#     height: int,
# ) :
#     _, ex_width = gray.shape
#     width = ex_width - 2  # 両端を拡張しているので、実際の幅は-2する必要がある

#     cost_left   = np.zeros(width, dtype=np.float32)  # 予めサイズを確保する
#     cost_mid    = np.zeros(width, dtype=np.float32)
#     cost_right  = np.zeros(width, dtype=np.float32)

#     for idx in range(0, width):
#         # idx   : 注目ピクセルのx座標
#         # d_idx : 注目ピクセルのx座標から視差値を引いた値

#         d_idx = idx - disp[height-1, idx]  # ペア画像の注目ピクセル
#         if d_idx < 0:  # 画像の左端を超えないように制限
#             d_idx = 0

#         if idx-1 < 0:   # 座標が画像の左端を超えないように制限
#             d_ul = 0
#         else:
#             if idx-1 - disp[height, idx-1] < 0:  # 画像の左端を超えないように制限
#                 d_ul = 0
#             else:
#                 d_ul = (idx-1) - disp[height, idx-1]

#         if idx - disp[height-1, idx] < 0:  # 画像の左端を超えないように制限
#             d_un = 0
#         else:
#             d_un = idx - disp[height, idx]

#         if idx+1 > width-1: 
#             d_ur = width-1
#         else:
#             if idx+1 - disp[height-1, idx+1] < 0:  # 画像の左端を超えないように制限
#                 d_ur = 0
#             else:
#                 d_ur = idx+1 - disp[height-1, idx+1]

#         # 注目ピクセルが削除されたときに発生するエネルギー
#         mid = np.abs(gray[height, d_idx-1] - gray[height, d_idx+1])

#         for way, u_idx in enumerate([d_ul, d_un, d_ur]):

#             if u_idx < d_idx: 
#                 if len(gray[height-1, u_idx:d_idx]) == len(gray[height, u_idx+1:d_idx+1]):
#                     cost = mid + np.sum(np.abs(gray[height-1, u_idx:d_idx] - gray[height, u_idx+1:d_idx+1]))
#                 else:
#                     cost = np.inf
#             elif u_idx == d_idx:
#                 cost = mid
#             else:
#                 if len(gray[height-1, d_idx+1:u_idx+1]) == len(gray[height, d_idx:u_idx]):
#                     cost = mid + np.sum(np.abs(gray[height-1, d_idx+1:u_idx+1] - gray[height, d_idx:u_idx]))
#                 else:
#                     cost = np.inf

#             if way == 0:
#                 cost_left[idx]  = cost
#             elif way == 1:
#                 cost_mid[idx]   = cost
#             elif way == 2:
#                 cost_right[idx] = cost

#     choices = np.vstack((cost_left, cost_mid, cost_right))

#     return choices


@nb.jit(cache=True, nopython=True)
def roll_2d(im:np.ndarray, axis: int, way: int):
    """
    入力された二次元配列を指定された軸に沿ってずらす(空いたところは隣の値で埋める)
    Args:
        im   : 入力配列
        axis : ずらす軸(0:縦, 1:横)
        way  : ずらす方向(1:上/右, -1:下/左)
    Returns:
        arr  : ずらした配列
    """
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
def roll(im : np.ndarray, way : int):
    """
    入力配列を左右にずらす(空いたところは隣の値で埋める)
    Args:
        im  : 入力配列
        way : ずらす方向(1:右, -1:左)
    Returns:
        arr : ずらした配列
    """
    arr = np.copy(im)  # 元の配列を変更しないようにコピーを作成

    if way == 1:   # 上
        arr[:-1] = im[1:]
    elif way == -1:  # 下
        arr[1:] = im[:-1]

    return arr

# @nb.jit(cache=True, nopython=True)
def _get_forward_seam(
    gray_arr       : np.ndarray, 
    disp_arr       : np.ndarray,
    aux_energy_arr : Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    順方向エネルギーを使って最小垂直シームを計算する
    Args:
        gray_arr       : グレースケール画像の配列
        disp_arr       : 深度マップの配列
        aux_energy_arr : マスク画像の配列（オプション）
    Returns:
        seam           : 計算されたシームの配列
        pair_seam      : ペア画像の計算されたシームの配列
    """

    frames, h, w = gray_arr.shape

    expanded_gray_arr = np.zeros((frames, h, w + 2), dtype=gray_arr.dtype)  # 1枚目のグレースケール画像を左右に1ピクセル拡張する（DP[時間平面]の初期化に使用）
    expanded_gray_arr[:, :, 1:-1] = gray_arr
    expanded_gray_arr[:, :, 0] = gray_arr[:, :, 0]
    expanded_gray_arr[:, :, -1] = gray_arr[:, :, -1]

    if aux_energy_arr is None:
        processed_aux_energy_arr = np.zeros((frames, h, w), dtype=np.float32)   # マスク画像がない場合はゼロで初期化
    else:
        processed_aux_energy_arr = aux_energy_arr.copy()

    # 動的計画法コストの初期化
    inf = np.array([np.inf], dtype=np.float32)  # 画像の外側に設定するための無限値
    dp = np.zeros(w + 2, dtype=np.float32)      # 画像平面の動的計画法のコストを格納する配列
    dp[0] = inf
    dp[1:-1] = np.abs(expanded_gray_arr[0, 0, 2:] - expanded_gray_arr[0, 0, :-2])
    dp[-1] = inf

    dp_time_arr = np.zeros((frames,w + 2), dtype=np.float32) # 時間平面の動的計画法のコストを格納する配列
    dp_time_arr[:, 0] = inf
    dp_time_arr[:, -1] = inf

    base_idx = np.arange(-1, w - 1, dtype=np.int32)     # base_idx  : 動的計画法のコストを格納する配列のインデックスを指定するための配列
    seam = np.zeros((frames, h), dtype=np.int32)        # seam      : 計算されたシームの配列
    parent = np.empty((frames, h, w), dtype=np.int32)   # parent    : すべての画素においてコストの最小値がある方向[0:右下,1:下,2:左下]をインデックスで指定できるようにした配列
    pair_seam = np.zeros((frames, h), dtype=np.int32)   # pair_seam : ペア画像の計算されたシームの配列

    energy_arr = np.zeros((frames, h, w), dtype=np.float32)  # エネルギーマップの初期化

    choices = np.zeros((3, w), dtype=np.float32) 

    for frame in range (0, frames): # 画像の枚数分だけループ
        energy = energy_arr[frame]
        aux_energy = processed_aux_energy_arr[frame]

        # 対象フレームのグレースケール画像を取得
        gray_curr = expanded_gray_arr[frame]

        # 対象フレームの視差マップを取得
        disp = disp_arr[frame]

        # 左右にずらした画像
        # 左右を1ピクセル拡張しているので2ピクセルずらしている
        shl_curr = gray_curr[:, 2:]     # shift left current
        shr_curr = gray_curr[:, :-2]    # shift right current
        shu_curr = roll_2d(gray_curr, 0, -1)[:, 1:-1]   # shift up current 

        base_cost = np.abs(shl_curr - shr_curr)

        # 対象ピクセルが削除された後，左右のピクセルがくっつくことにより発生するエネルギー
        cost_mid = base_cost
        cost_left = base_cost + np.abs(shu_curr - shr_curr)
        cost_right = base_cost + np.abs(shu_curr - shl_curr)

        # マスク部分にコストを追加
        if aux_energy is not None:
            base_cost += aux_energy

        # 時間(t-w)方向に１つ前の列
        # 対象フレームの前フレームを取得
        if frame != 0:
            gray_back = expanded_gray_arr[frame-1]
            shift_time = gray_back[:, 1:-1]  

            # 時間方向の中央コストを計算
            xt_cost_mid      = base_cost
            xt_cost_left     = base_cost + np.abs(shift_time - shr_curr)   # 画像(h-w)方向に１つ前の列
            xt_cost_right    = base_cost + np.abs(shift_time - shl_curr)   # 画像(h-w)方向に１つ前の列

        for height in range(0, h):
            if frame == 0:
                if height != 0:
                    # 1枚目の場合
                    # 最上段のエネルギー
                    dp_mid = dp[1:-1]
                    dp_left = dp[:-2]
                    dp_right = dp[2:]

                    choices[0, :] = cost_left[height]   + dp_left       # 左下方向のコスト
                    choices[1, :] = cost_mid[height]    + dp_mid        # 中央方向のコスト
                    choices[2, :] = cost_right[height]  + dp_right      # 右下方向のコスト

                    # ペア画像の動的計画法のコストを加算
                    # stereo_cost = _calc_pair_energy(gray_curr, disp, height)
                    # choices = choices + stereo_cost

                    min_idx = np.argmin(choices, axis=0)    # コストの最小値がある方向[0:右下,1:下,2:左下]を格納

                    # 最小コストをdp_midに累積
                    dp[1:-1] = choices[min_idx, np.arange(w)]

                    # エネルギーを格納
                    energy[height, :] = choices[min_idx, np.arange(w)]

                    # 各インデックス位置の値を加算
                    parent[frame][height] = min_idx + base_idx
            else:
                dp_time = dp_time_arr[frame]  # 時間平面の動的計画法のコストを格納する配列

                # 画像平面の動的計画法コスト（DP）
                dp_mid   = dp_time[1:-1]
                dp_left  = dp_time[:-2]
                dp_right = dp_time[2:]

                if height == 0:  # 最上段の動的計画法の計算
                    choices[0, :] = xt_cost_left[height]       # 左下方向のコスト
                    choices[1, :] = xt_cost_mid[height]        # 中央方向のコスト
                    choices[2, :] = xt_cost_right[height]      # 右下方向のコスト
                else:   # 2行目以降の動的計画法の計算
                    choices[0, :] = cost_left[height]   + dp_left     + xt_cost_left[height]    # 左下方向のコスト
                    choices[1, :] = cost_mid[height]    + dp_mid      + xt_cost_mid[height]     # 中央方向のコスト
                    choices[2, :] = cost_right[height]  + dp_right    + xt_cost_right[height]   # 右下方向のコスト

                # # ペア画像の動的計画法のコストを加算
                # if height > 0:
                #     stereo_cost = _calc_pair_energy(gray_curr, disp, height)
                #     choices += stereo_cost

                # 上からコストの最小値がある方向[0:右下,1:下,2:左下]を格納
                min_idx = np.argmin(choices, axis=0)

                # 最小コストをdp_midに累積
                dp_time[1:-1] = choices[min_idx, np.arange(w)]
                energy[height, :] = choices[min_idx, np.arange(w)]

                # 各インデックス位置の値を加算
                parent[frame][height] = min_idx + base_idx

    # 動的計画法のコストの最小値インデックスを取得
    min_dp_idx = np.argmin(dp[1:-1])

    # 
    for frame in range (0, frames):
        disp = disp_arr[frame]
        for height in range(h-1, -1, -1):
            seam[frame][height] = min_dp_idx
            pair_min_dp_idx = min_dp_idx - disp[height, min_dp_idx]
            
            if 0 > pair_min_dp_idx:
                pair_seam[frame][height] = 0
            elif 0 <= pair_min_dp_idx < w:
                pair_seam[frame][height] = pair_min_dp_idx
            elif pair_min_dp_idx >= w:
                pair_seam[frame][height] = w-1

            min_dp_idx = parent[frame][height, min_dp_idx]

    return seam, pair_seam

# @nb.jit(cache=True, nopython=True)
# def _get_forward_seam(
#     gray_arr       : np.ndarray, 
#     disp_arr       : np.ndarray,
#     aux_energy_arr : Optional[np.ndarray],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     順方向エネルギーを使って最小垂直シームを計算する
#     Args:
#         gray_arr       : グレースケール画像の配列
#         disp_arr       : 深度マップの配列
#         aux_energy_arr : マスク画像の配列（オプション）
#     Returns:
#         seam           : 計算されたシームの配列
#         pair_seam      : ペア画像の計算されたシームの配列
#     """

#     frames, h, w = gray_arr.shape

#     gray = gray_arr[0]                                                      # 1枚目のグレースケール画像を取得(DP[画像平面]の初期化に使用)
#     gray = np.concatenate((gray[:, :1], gray, gray[:, -1:]), axis=1)        # 1枚目のグレースケール画像を左右に1ピクセル拡張する（DP[時間平面]の初期化に使用）
#     inf = np.array([np.inf], dtype=np.float32)                              # 画像の外側に設定するための無限値
#     dp = np.concatenate((inf, np.abs(gray[0, 2:] - gray[0, :-2]), inf))     # 1枚目, 1列目のエネルギーのみ絶対値差分を取ってエネルギーを計算する
#     dp_time_arr = np.concatenate((inf, np.zeros(w, dtype=np.float32), inf))     # 動的計画法のコストを格納する配列

#     parent = np.empty((frames, h, w), dtype=np.int32)   # parent    : すべての画素においてコストの最小値がある方向[0:右下,1:下,2:左下]をインデックスで指定できるようにした配列
#     base_idx = np.arange(-1, w - 1, dtype=np.int32)     # base_idx  : 動的計画法のコストを格納する配列のインデックスを指定するための配列
#     seam = np.zeros((frames, h), dtype=np.int32)        # seam      : 計算されたシームの配列
#     pair_seam = np.zeros((frames, h), dtype=np.int32)   # pair_seam : ペア画像の計算されたシームの配列

#     if aux_energy_arr is None:
#         processed_aux_energy_arr = np.zeros((frames, h, w), dtype=np.float32)   # マスク画像がない場合はゼロで初期化
#     else:
#         processed_aux_energy_arr = aux_energy_arr.copy()

#     energy_arr = np.zeros((frames, h, w), dtype=np.float32)  # エネルギーマップの初期化

#     for frame in range (0, frames): # 画像の枚数分だけループ
#         energy = energy_arr[frame]
#         aux_energy = processed_aux_energy_arr[frame]

#         # 対象フレームのグレースケール画像を取得
#         gray_curr = gray_arr[frame]
#         gray_curr = np.concatenate((gray_curr[:, :1], gray_curr, gray_curr[:, -1:]), axis=1)    # hstackよりconcatenateの方が高速なのでこちらを利用する

#         # 対象フレームの前フレームを取得
#         if frame != 0:
#             gray_back = gray_arr[frame-1]
#             gray_back = np.concatenate((gray_back[:, :1], gray_back, gray_back[:, -1:]), axis=1)

#         # 対象フレームの視差マップを取得
#         disp = disp_arr[frame]

#         # 左右にずらした画像
#         # 左右を1ピクセル拡張しているので2ピクセルずらしている
#         L = gray_curr[:, 2:]
#         R = gray_curr[:, :-2]
#         T = roll_2d(gray_curr, 0, -1)[:, 1:-1]

#         if frame == 0:  # 1枚目の場合
#             shl_curr = L
#             shr_curr = R
#             prev_mid = T  # 画像(h-w)方向に１つ上の列

#             # 対象ピクセルが削除された後，左右のピクセルがくっつくことにより発生するエネルギー
#             cost_mid = np.abs(shl_curr - shr_curr)
#             cost_left = cost_mid + np.abs(prev_mid - shr_curr)
#             cost_right = cost_mid + np.abs(prev_mid - shl_curr)

#             # マスク部分にコストを追加
#             if aux_energy is not None:
#                 cost_mid += aux_energy
        
#             for height in range(1, h):
#                 # print(f"height = {height} / {h}")  # 進捗表示
#                 # 最上段のエネルギー
#                 dp_mid = dp[1:-1]
#                 dp_left = dp[:-2]
#                 dp_right = dp[2:]

#                 choices = np.vstack((
#                     cost_left[height]   + dp_left ,     # 左下方向のコスト
#                     cost_mid[height]    + dp_mid ,      # 中央方向のコスト
#                     cost_right[height]  + dp_right      # 右下方向のコスト
#                 ))

#                 # ペア画像の動的計画法のコストを加算
#                 stereo_cost = _calc_pair_energy(gray_curr, disp, height)
#                 choices = choices + stereo_cost

#                 min_idx = np.argmin(choices, axis=0)    # コストの最小値がある方向[0:右下,1:下,2:左下]を格納
#                 parent[frame][height] = min_idx + base_idx

#                 # 最小コストをdp_midに累積
#                 for j, i in enumerate(min_idx):
#                     dp[j+1] = choices[i, j]

#                 # choisesからmin_idx方向に対応した最小方向の画素を取得
#                 for j in range(w):
#                     energy[height, j] = choices[min_idx[j], j]

#         else:   # 1枚目以降の場合
#             dp_t = dp_time_arr.copy()
#             shl_curr = L  # 左にずらした画像
#             shr_curr = R  # 右にずらした画像

#             shu_curr = T   # 画像(h-w)方向に１つ上の列
#             shift_time = gray_back[:, 1:-1]   # 時間(t-w)方向に１つ前の列

#             # 画像平面の中央コストを計算
#             xy_cost_mid      = np.abs(shl_curr - shr_curr)    # ピクセルを消したときに発生するコスト
#             cost_left     = cost_mid + np.abs(shu_curr - shr_curr)   # 画像(h-w)方向に１つ上の列
#             cost_right    = cost_mid + np.abs(shu_curr - shl_curr)   # 画像(h-w)方向に１つ上の列

#             # 時間方向の中央コストを計算
#             xt_cost_mid      = np.abs(shl_curr - shr_curr)    # ピクセルを消したときに発生するコスト
#             xt_cost_left     = cost_mid + np.abs(shift_time - shr_curr)   # 画像(h-w)方向に１つ前の列
#             xt_cost_right    = cost_mid + np.abs(shift_time - shl_curr)   # 画像(h-w)方向に１つ前の列

#             for height in range (0, h):    # 画像の高さ分だけループ
#                 # 画像平面の動的計画法コスト（DP）
#                 dp_mid   = dp_t[1:-1]
#                 dp_left  = dp_t[:-2]
#                 dp_right = dp_t[2:]

#                 if height == 0:  # 最上段の動的計画法の計算
#                     choices = np.vstack((
#                         xt_cost_left[height],   # 左下方向のコスト
#                         cost_mid[height],       # 中央方向のコスト
#                         xt_cost_right[height]   # 右下方向のコスト
#                     ))
#                 else:   # 2行目以降の動的計画法の計算
#                     choices = np.vstack((
#                         cost_left[height]     + dp_left     + xt_cost_left[height]   ,
#                         xy_cost_mid[height]      + dp_mid      + xt_cost_mid[height]       , 
#                         cost_right[height]    + dp_right    + xt_cost_right[height]  
#                     ))

#                 # ペア画像の動的計画法のコストを加算
#                 if height > 0:
#                     stereo_cost = _calc_pair_energy(gray_curr, disp, height)
#                     choices = choices + stereo_cost

#                 # 上からコストの最小値がある方向[0:右下,1:下,2:左下]を格納
#                 min_idx = np.argmin(choices, axis=0)

#                 # choisesからmin_idx方向に対応した最小方向の画素を取得
#                 for j in range(w):
#                     energy[height, j] = choices[min_idx[j], j]

#                 # 動的計画法のコストの更新
#                 for j, i in enumerate(min_idx):
#                     dp_t[j+1] = choices[i, j]

#                 # 各インデックス位置の値を加算
#                 parent[frame][height] = min_idx + base_idx

#     # 動的計画法のコストの最小値インデックスを取得
#     min_dp_idx = np.argmin(dp[1:-1])
#     for frame in range (0, frames):
#         disp = disp_arr[frame]
#         for height in range(h-1, -1, -1):
#             seam[frame][height] = min_dp_idx
#             pair_min_dp_idx = min_dp_idx - disp[height, min_dp_idx]
            
#             if 0 > pair_min_dp_idx:
#                 pair_seam[frame][height] = 0
#             elif 0 <= pair_min_dp_idx < w:
#                 pair_seam[frame][height] = pair_min_dp_idx
#             elif pair_min_dp_idx >= w:
#                 pair_seam[frame][height] = w-1

#             min_dp_idx = parent[frame][height, min_dp_idx]

#     # print(f"_get_forward_seam終了")
#     return seam, pair_seam

@nb.jit(cache=True, nopython=True)
def _make_idx_map(frames: int, h: int, w: int) -> np.ndarray:
    """
    画像のインデックスを格納する配列を作成します
    args:
        frames  : 画像の枚数
        h       : 画像の高さ
        w       : 画像の幅
    returns:
        idx_map : 画像のインデックスを格納する配列
    """
    # 要素が w の配列を作成し、高さ h まで拡張する
    idx_map = np.empty((frames, h, w), dtype=np.int32)
    for t in range(frames):
        for i in range(h):
            for j in range(w):
                idx_map[t, i, j] = j
    return idx_map

@nb.jit(cache=True, nopython=True)
def _create_seam_masks(gray_arr: np.ndarray, seam_arr: np.ndarray) -> np.ndarray:
    """
    削除するシームマスクを作成します
    args:
        gray_arr: グレースケール画像の配列
        seam_arr: シームの配列
    returns:
        mask_arr: 削除するシームのマスク配列
    """
    frames, h, w = gray_arr.shape
    mask_arr = np.zeros((frames, h, w), dtype=np.bool_)
    for t in range(frames):
        mask_arr[t] = _get_seam_mask(gray_arr[t], seam_arr[t])
    return mask_arr

@nb.jit(cache=True, nopython=True)
def _remove_seams(gray_arr: np.ndarray, seam_mask_arr: np.ndarray) -> np.ndarray:
    frames, h, w = gray_arr.shape
    output = np.zeros((frames, h, w - 1), dtype=gray_arr.dtype)

    for t in range(frames):
        for i in range(h):
            col_idx = 0
            for j in range(w):
                if not seam_mask_arr[t, i, j]:
                    output[t, i, col_idx] = gray_arr[t, i, j]
                    col_idx += 1
    return output

def _get_forward_seams(
    gray_arr       : np.ndarray,
    gray_pair_arr  : np.ndarray,      # 追加：　グレースケール化したペア画像(右画像の予定)
    disp_arr       : np.ndarray,      # 追加：　深度マップ 
    num_seams       : int, 
    aux_energy_arr : Optional[np.ndarray]
) -> np.ndarray:
    
    """順方向エネルギーを使って最小N本の垂直シームを計算する"""
    """Compute the minimum N vertical seams using forward energy"""
    frames, h, w = gray_arr.shape

    # seamsとseamの違い
    # 　seams：削除するpxの数だけ計算したシームを累積して最終的に出力する画像サイズと同じ配列（最終的に使うもの）
    # 　seam ：削除するシームを格納する配列（毎回計算されて更新されるシーム）

    # シームを格納するFalseで初期化されたリストを作成
    seams       = np.zeros((frames, h, w), dtype=np.bool_)
    pair_seams  = np.zeros((frames, h, w), dtype=np.bool_)

    # 要素が入力画像の高さだけある配列
    rows = np.arange(h, dtype=np.int32)

    # ブロードキャストとは配列をshape(h, w)で指定した形状に変換する機能
    # 要素が w の配列を作成し、高さ h まで拡張する
    idx_map_arr = _make_idx_map(frames, h, w)  # 各画像のインデックスを格納する配列
    pair_idx_map_arr = idx_map_arr.copy()  # ペア画像のインデックスを格納する配列

    print("calculating seams...")
    for i in tqdm(range(num_seams)):
        # 削除シームを決定
        seam, pair_seam = _get_forward_seam(gray_arr, disp_arr, aux_energy_arr)

        # 最終的な計算結果になる計算されたシームを削除するためのマスクを作成
        # 削除するシームのマスクを作成
        seam_mask_arr = _create_seam_masks(gray_arr, seam)
        pair_seam_mask_arr = _create_seam_masks(gray_pair_arr, pair_seam)

        for t in range(len(gray_arr)): 
            # 最終的に出力する配列に選択されたシームを格納         
            seams[t][rows, idx_map_arr[t][rows, seam[t]]] = True
            pair_seams[t][rows, pair_idx_map_arr[t][rows, pair_seam[t]]] = True
            visualize(gray_arr[t],seam_mask_arr[t])
        

        # 画像のシーム部分を削除
        h,w = gray_arr[t].shape

        # 画像のシーム部分を削除
        gray_arr = _remove_seams(gray_arr, seam_mask_arr)
        gray_pair_arr = _remove_seams(gray_pair_arr, pair_seam_mask_arr)
        idx_map_arr = _remove_seams(idx_map_arr, seam_mask_arr)
        pair_idx_map_arr = _remove_seams(pair_idx_map_arr, pair_seam_mask_arr)

        disp_arr = _remove_seams(disp_arr, seam_mask_arr)
        # マスク画像もシーム部分を削る
        if aux_energy_arr is not None:
            aux_energy_arr = _remove_seams(aux_energy_arr, seam_mask_arr)
        
    return seams, pair_seams

def _get_seams(
    gray_arr        : np.ndarray, 
    gray_pair_arr   : np.ndarray,      # 追加：　グレースケール化したペア画像(右画像の予定)
    disp_arr        : np.ndarray,      # 追加：　深度マップ
    num_seams       : int, 
    energy_mode     : str, 
    aux_energy_arr  : Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """グレースケール画像から最小N個の継ぎ目を得る"""
    gray_arr = np.asarray(gray_arr, dtype=np.float32)
    gray_pair_arr = np.asarray(gray_pair_arr, dtype=np.float32)

    # 今回はforward のみ利用する
    if energy_mode == EnergyMode.FORWARD:
        return _get_forward_seams(gray_arr, gray_pair_arr, disp_arr, num_seams, aux_energy_arr)
    else:
        raise ValueError(
            f"expect energy_mode to be one of {_arr_enum(EnergyMode)}, got {energy_mode}"
        )

def to_gray(arr):
    """Convert images to grayscale"""
    frames, h, w, _ = arr.shape
    gray_arr = np.zeros((frames,h,w))  # グレースケール画像の初期化
    for frame in range(frames):
        gray_arr[frame] = cv2.cvtColor(arr[frame], cv2.COLOR_BGR2GRAY)
    return gray_arr

# 画像幅の縮小
def _reduce_width(
    sbs_arr        : np.ndarray,
    disp_arr       : np.ndarray,           # 追加：　深度マップ
    delta_width    : int,
    energy_mode    : str,
    aux_energy_arr : Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    
    """Reduce the width of image by delta_width pixels"""
    frames, _, w, _         = sbs_arr.shape
    src_arr        = sbs_arr[:, :, :w//2]   # 左側画像
    src_pair_arr   = sbs_arr[:, :, w//2:]   # 右側画像

    src = src_arr[0]

    assert src.ndim in (2, 3) and delta_width >= 0

    # シーム計算のため入力画像がグレースケールか判断
    if src.ndim == 2:
        # 元々がグレースケール画像
        gray_arr        = src_arr
        gray_pair_arr   = src_pair_arr
        src_h, src_w    = src.shape

        # dst_shape : 幅を縮小させた後の画像サイズ
        dst_shape: Tuple[int, ...] = (src_h, src_w - delta_width)   
    else:
        gray_arr = to_gray(src_arr)
        gray_pair_arr = to_gray(src_pair_arr)
        src_h, src_w, src_c = src.shape

        # dst_shape : 幅を縮小させた後の画像サイズ
        dst_shape = (src_h, src_w - delta_width, src_c)

    to_keep_arr: np.ndarray
    to_keep_pair_arr: np.ndarray

    to_keep_arr, to_keep_pair_arr = _get_seams(gray_arr, gray_pair_arr, disp_arr, delta_width, energy_mode, aux_energy_arr)

    # シーム部分の削除
    dst_arr      = np.array([src_arr[i][~to_keep_arr[i]].reshape(dst_shape) for i in range(frames)])
    dst_pair_arr = np.array([src_pair_arr[i][~to_keep_pair_arr[i]].reshape(dst_shape) for i in range(frames)])

    if aux_energy_arr is not None:
        aux_energy_arr = np.array([aux_energy_arr[i][~to_keep_arr[i]].reshape(dst_shape[:2]) for i in range(frames)])   # 最終的には必要ないのでペアの分はつくらない

    return dst_arr, dst_pair_arr, aux_energy_arr

# 画像のリサイズ(今回は縮小のみ扱う)
def _resize_width(
    sbs_arr    : np.ndarray,
    disp_arr   : np.ndarray,           # 追加：　深度マップ（多分カラー画像
    width       : int,
    energy_mode : str,
    aux_energy_arr  : Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """縦方向の継ぎ目を取り除いて画像の幅をリサイズする"""
    src = sbs_arr[0]
    assert src.size > 0 and src.ndim in (2, 3)
    assert width > 0

    src_w = int(src.shape[1]//2)

    # 画像幅と入力された数値を比較し、結果によって縮小・拡張を判断（今回は縮小のみ考える）
    dst, dst_pair, aux_energy_arr = _reduce_width(sbs_arr, disp_arr, src_w - width, energy_mode, aux_energy_arr)
    
    return dst, dst_pair, aux_energy_arr

def stereo_video_resize(  
    sbs_arr        : np.ndarray,          # 追加：　SBS画像リスト
    disp_arr       : np.ndarray,          # 追加：　深度マップ（gray）
    size            : Optional[Tuple[int, int]] = None,
    energy_mode     : str = "forward",
    order           : str = "width-first",
    mask_arr       : Optional[np.ndarray] = None,
) -> np.ndarray:
    
    """
    Function:
        内容を考慮したシームカービングアルゴリズムを用いて，画像のサイズを変更します.
    
    Args:
        src:
            RGB またはグレースケール形式の入力画像
        size : 
            2 タプル（w, h）
        energy_mode :  
            入力画像に対するエネルギー計算の方針
            "backward" または "forward" のいずれか.
            "backward" の場合，各ピクセルの勾配としてエネルギーを計算する.
            "forward" の場合，各ピクセルを削除した後の隣接ピクセル間の距離としてエネルギーを計算する.
        order : 
            水平方向と垂直方向の継ぎ目を除去する順番.
            width-first" あるいは "height-first" のいずれかを指定する
            width-first-"モードでは、まず垂直方向の継ぎ目を削除または挿入し、次に水平方向の継ぎ目を削除します。
            一方 "height-first" はその逆です。
        keep_mask  : 
            マスク画像
            指定しない場合は、どの領域も保護されない。
        drop_mask  : 
            削除するバイナリオブジェクトマスク。
            与えられた場合、画像をターゲットサイズにリサイズする前にオブジェクトが除去されます。
        step_ratio :  
            1回のシームカービングステップにおける最大サイズ拡大率.
            ターゲットサイズが大きすぎる場合、画像は複数のステップに分割されます。

    Return : 
        リサイズされた結果画像リスト？動画化する？
    """

    sbs = sbs_arr[0]
    src = sbs[:, :sbs.shape[1]//2]
    frames = len(sbs_arr)

    # 入力画像の形式が間違っていないか確認
    _check_src(src)

    if order not in _arr_enum(OrderMode):
        raise ValueError(
            f"expect order to be one of {_arr_enum(OrderMode)}, got {order}"
        )

    # マスク関連
    if mask_arr is not None:
        keep_mask = mask_arr[0]
        _check_mask(keep_mask, src.shape[:2])
        aux_energy_arr = np.array([np.zeros(src.shape[:2], dtype=np.float32) for _ in range(frames)])

        for i in range(frames):
            aux_energy_arr[i][mask_arr[i]] += KEEP_MASK_ENERGY

    width, height = size
    width = round(width)
    height = round(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"expect target size to be positive, got {size}")

    # 縮小処理
    src_arr, pair_src_arr, aux_energy_arr = _resize_width(
        sbs_arr, disp_arr, width, energy_mode, aux_energy_arr
    )

    sbs_arr = np.array([cv2.hconcat([src, pair_src]) for src,pair_src in zip(src_arr, pair_src_arr)])

    return np.array(sbs_arr)