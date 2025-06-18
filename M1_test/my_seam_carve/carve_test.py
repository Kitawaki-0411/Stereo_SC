import warnings
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

KEEP_MASK_ENERGY = 10000000


class OrderMode(str, Enum):
    WIDTH_FIRST = "width-first"
    HEIGHT_FIRST = "height-first"


class EnergyMode(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"

def visualize(im, boolmask=None):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[boolmask] = 255
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis

def _list_enum(enum_class) -> Tuple:
    return tuple(x.value for x in enum_class)


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a grayscale image"""
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    return (rgb @ coeffs).astype(rgb.dtype)


def _get_seam_mask(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Convert a list of seam column indices to a mask"""
    # 一辺の要素数が w の単位行列(要素がすべて１)を作成
    return np.eye(src.shape[1], dtype=bool)[seam]


def _remove_seam_mask(src: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    """Remove a seam from the source image according to the given seam_mask"""
    if src.ndim == 3:
        h, w, c = src.shape
        seam_mask = np.broadcast_to(seam_mask[:, :, None], src.shape)
        dst = src[~seam_mask].reshape((h, w - 1, c))
    else:
        h, w = src.shape
        dst = src[~seam_mask].reshape((h, w - 1))
    return dst


def _get_energy(gray: np.ndarray) -> np.ndarray:
    """Get backward energy map from the source image"""
    assert gray.ndim == 2

    gray = gray.astype(np.float32)
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


# ◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆ #
# 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 #
# 　　　　　　　　　　　　　　　　　　　　　　　　ここからステレオ対応用のプログラム　　　　　　　　　　　　　　　　　　　　　　　　　 #
# 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 #
# ◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆ #

# ペア画像の一行のコストを計算
# 視差値を引くやり方
@nb.jit(cache=True, nopython=True)
def _calc_pair_energy(
    gray: np.ndarray,  # すでに両端は拡張済み
    disp: np.ndarray, 
    h   : int, 
    w   : int
) :
    cost_left = np.zeros(w)  # 予めサイズを確保する
    cost_mid = np.zeros(w)
    cost_right = np.zeros(w)

    for idx in range(0,w):
        # disp  : 視差マップ
        # idx   : 注目ピクセルのx座標
        # d_idx : 注目ピクセルのx座標から視差値を引いた値

        # ペア画像の注目座標の計算　＝　主画像から視差値を引いた値
        # if idx - disp[h, idx] < 0:  # 画像の左端を超えないように制限
        #     d_idx = 0
        # else:
        #     d_idx = idx - disp[h, idx]
        d_idx = max(0,idx - disp[h, idx])

        # 注目ピクセルの左下の座標の計算
        # if idx-1 < 0:   # 座標が画像の左端を超えないように制限
        #     ul = 0
        # else:
        #     ul = idx-1 - disp[h-1, idx-1]
        #     if ul < 0:  # 画像の左端を超えないように制限
        #         ul = 0

        ul = max(0, idx-1 - disp[h-1, idx-1]) if idx-1 > 0 else 0

        # 注目ピクセルの下の座標の計算
        un = idx - disp[h-1, idx]
        if un < 0:  # 画像の左端を超えないように制限
            un = 0

        # 注目ピクセルの右下の座標の計算
        if idx+1 > w - 1:
            ur = w - 1
        else:
            ur = idx+1 - disp[h-1, idx+1]
            if ur < 0:  # 画像の左端を超えないように制限
                ur = 0

        # 注目ピクセルが削除されたときに発生するエネルギー
        mid = np.abs(gray[h, d_idx-1] - gray[h, d_idx+1])

        for way, u_idx in enumerate([ul,un,ur]):   
            if d_idx == 0:    # ペア画像の注目ピクセルの座標が画像の左端(0)のとき
                cost = np.inf
                if way == 0:
                    cost_left[idx] = cost
                elif way == 1:
                    cost_mid[idx] = cost
                elif way == 2:
                    cost_right[idx] = cost
            else:
                if u_idx < d_idx: 
                    cost = mid + np.sum(np.abs(gray[h-1, u_idx:d_idx-1] - gray[h, u_idx+1:d_idx]))
                elif u_idx == d_idx:
                    cost = mid
                else:
                    cost = mid + np.sum(np.abs(gray[h-1, d_idx+1:u_idx] - gray[h, d_idx:u_idx-1]))

                if way == 0:
                    cost_left[idx]  = cost
                elif way == 1:
                    cost_mid[idx]   = cost
                elif way == 2:
                    cost_right[idx] = cost

    choices = np.vstack((cost_left, cost_mid, cost_right))

    return choices

# @nb.jit(cache=True, nopython=True)
# def _calc_pair_energy(gray: np.ndarray, disp: np.ndarray, h: int, w: int):
#     cost_left = np.zeros(w)
#     cost_mid = np.zeros(w)
#     cost_right = np.zeros(w)

#     for idx in range(w):
#         d_idx = max(0, idx - disp[h, idx])
#         ul = max(0, idx - 1 - disp[h-1, idx-1]) if idx > 0 else 0
#         un = max(0, idx - disp[h-1, idx])
#         ur = max(0, idx + 1 - disp[h-1, idx+1]) if idx < w - 1 else w - 1

#         if d_idx <= 0 or d_idx >= w - 1:
#             mid = 255  # 境界値
#         else:
#             mid = abs(gray[h, d_idx - 1] - gray[h, d_idx + 1])

#         for way, u_idx in enumerate([ul, un, ur]):
#             if d_idx == 0:
#                 cost = 1e10
#             else:
#                 cost = mid
#                 if u_idx < d_idx:
#                     for i in range(d_idx - u_idx - 1):
#                         cost += abs(gray[h-1, u_idx + i] - gray[h, u_idx + i + 1])
#                 elif u_idx > d_idx:
#                     for i in range(u_idx - d_idx - 1):
#                         cost += abs(gray[h-1, d_idx + i + 1] - gray[h, d_idx + i])

#             if way == 0:
#                 cost_left[idx] = cost
#             elif way == 1:
#                 cost_mid[idx] = cost
#             else:
#                 cost_right[idx] = cost
#     return np.vstack((cost_left, cost_mid, cost_right))

@nb.jit(cache=True, nopython=True)
def roll_2d(im, axis, way):
    """
    2次元配列を指定の方向にシフトする関数です
    ====================================================================
        axis = 0    : 縦方向にシフト
        axis = 1    : 横方向にシフト
        way = 1     : 上または右にシフト
        way = -1    : 下または左にシフト
    ====================================================================

    引数:
        im (np.ndarray) : 入力画像
        axis (int)      : シフトする方向(0:縦, 1:横)
        way (int)       : シフトの方向(1:上または右, -1:下または左)

    戻り値:
        arr (np.ndarray): シフト後の画像
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
def roll_1d(im, way):
    """
    1次元配列を指定の方向にシフトする関数です
    ====================================================================
        way = 1     : 右にシフト
        way = -1    : 左にシフト
    ====================================================================

    引数:
        im (np.ndarray) : 入力画像
        way (int)       : シフトの方向(1:右, -1:左)

    戻り値:
        arr (np.ndarray): シフト後の画像
    """
    arr = np.copy(im)  # 元の配列を変更しないようにコピーを作成
    if way == 1:   # 右
        arr[:-1] = im[1:]
    elif way == -1:  # 左
        arr[1:] = im[:-1]
    return arr

@nb.jit(cache=True, nopython=True)
def clip(x, x_min, x_max):
    """
    Numba で高速動作する np.clip 相当の関数。
    """
    result = np.empty_like(x)
    for i in range(x.shape[0]):
        val = x[i]
        if val < x_min:
            val = x_min
        elif val > x_max:
            val = x_max
        result[i] = val
    return result

@nb.jit(cache=True, nopython=True)
def _get_stereo_forward_seam(
    gray: np.ndarray, 
    disp: np.ndarray,
    aux_energy: Optional[np.ndarray]
) -> np.ndarray:
    """
    Forward energy を用いてシームを計算する関数
    
    引数:
        gray (np.ndarray)       : グレースケール化した画像
        disp (np.ndarray)       : 深度マップ
        aux_energy (np.ndarray) : 補助エネルギー（オプション）

    戻り値:
        seam (np.ndarray)       : シームのインデックス
        pair_seam (np.ndarray)  : ペア画像のシームのインデックス
        energy (np.ndarray)     : エネルギーマップ
    """
    h, w = gray.shape

    # 水平方向に「画像の最前列」＋「画像」＋「画像の最後列」をくっつけてる
    # 多分オリジナル画像のままだと一番端だけ動的計画法できないから両端を拡張しているんだと思う
    gray = np.hstack((gray[:, :1], gray, gray[:, -1:]))
    # 無限配列を作成してる
    inf = np.array([np.inf], dtype=np.float32)
    # 1列目のエネルギーのみ絶対値差分を取ってエネルギーを計算する。両端に無限配列をくっつけているのは謎
    dp = np.concatenate((inf, np.abs(gray[0, 2:] - gray[0, :-2]), inf))
    # 多分、もとになる初期化されていない空配列を作成している
    parent = np.empty((h, w), dtype=np.int32)
    # 注目画素の左上(0), 上(1), 右上(2)で表記される min_idx を、
    # 左端からのインデックスに変換するための配列
    base_idx = np.arange(-1, w - 1, dtype=np.int32)
    # エネルギーマップの確認用
    energy = np.zeros((h, w))  # エネルギーマップの初期化

    # 対象が含まれる行を左右にずらしたものを用意
    # 左右を1ピクセル拡張しているので2ピクセルずらしている
    shl_curr = gray[:, 2:]
    shr_curr = gray[:, :-2]
    prev_mid = roll_2d(gray, 0, -1)[:, 1:-1]  # 画像(h-w)方向に１つ上の列

    # 最上段のエネルギー
    dp_mid = dp[1:-1]
    dp_left = dp[:-2]
    dp_right = dp[2:]

    # 対象ピクセルが削除された後，左右のピクセルがくっつくことにより発生するエネルギー
    cost_mid = np.abs(shl_curr - shr_curr)

    # マスク部分にコストを追加
    if aux_energy is not None:
        cost_mid += aux_energy

    cost_left = cost_mid + np.abs(prev_mid - shr_curr)
    cost_right = cost_mid + np.abs(prev_mid - shl_curr)

    for r in range(1, h):
        # コストを以下の形で記録
        choices = np.vstack((
            cost_left[r]   + dp_left , 
            cost_mid[r]    + dp_mid , 
            cost_right[r]  + dp_right
        ))

        # 右視点側のコスト計算結果を加算
        stereo_cost = _calc_pair_energy(gray,disp,r,w)
        choices = choices + stereo_cost

        # 上からコストの最小値がある方向[0:右下,1:下,2:左下]を格納
        min_idx = np.argmin(choices, axis=0)

        # エネルギーマップの計算
        for j in range(w):
            energy[r, j] = choices[min_idx[j], j]

        # 各ピクセルからの最小コスト方向のインデックスをベースインデックスに加算
        parent[r] = min_idx + base_idx

        # 最小値を記録
        for j, i in enumerate(min_idx):
            dp_mid[j] = choices[i, j]

    # argmin は index を出力
    c = np.argmin(dp[1:-1])
    seam = np.empty(h, dtype=np.int32)
    pair_seam = np.empty(h, dtype=np.int32)

    for r in range(h - 1, -1, -1):
        seam[r] = c
        cd = c - disp[r, c]
        pair_seam[r] = cd        
        c = parent[r, c]

    seam = clip(seam, 0, w - 1)
    pair_seam = clip(pair_seam, 0, w - 1)

    return seam, pair_seam, energy

def _get_stereo_forward_seams(
    gray: np.ndarray,
    pair_gray: np.ndarray,      # 追加：　グレースケール化したペア画像(右画像の予定)
    disp: np.ndarray,           # 追加：　深度マップ 
    num_seams: int, 
    aux_energy: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    入力画像の幅を num_seams の数だけ削減するためのシームを取得する関数

    引数:
        gray (np.ndarray)       : グレースケール化した画像
        pair_gray (np.ndarray)  : グレースケール化したペア画像(右画像の予定)
        disp (np.ndarray)       : 深度マップ
        num_seams (int)        : 削減するシームの数
        aux_energy (np.ndarray) : 補助エネルギー（オプション）
        
    戻り値:
        seams (np.ndarray)      : シームのインデックス
        pair_seams (np.ndarray) : ペア画像のシームのインデックス
    """
    h, w = gray.shape

    # シームを格納するFalseで初期化されたリスト
    seams = np.zeros((h, w), dtype=bool)
    pair_seams = np.zeros((h, w), dtype=bool)

    # 要素が入力画像の高さだけある配列
    rows = np.arange(h, dtype=np.int32)
    pair_rows = np.arange(h, dtype=np.int32)

    # ブロードキャストとは配列をshape(h, w)で指定した形状に変換する機能
    # 要素が w の配列を作成し、高さ h まで拡張する
    idx_map = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))
    pair_idx_map = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))

    # for _ in range(): は _ の変数を使用しないときに使う
    # tqdmを使用してプログレスバーを表示
    for i in tqdm(range(num_seams)):
        # 入力画像に対するシームを作成
        seam, pair_seam, energy = _get_stereo_forward_seam(gray, disp,  aux_energy)

        if i == 0:  # 最初のシームをエネルギーマップを表示
            plt.imshow(energy)
            plt.show()

        # 最終的に出力する配列に選択されたシームを格納
        seams[rows, idx_map[rows, seam]] = True
        pair_seams[pair_rows, pair_idx_map[pair_rows, pair_seam]] = True

        # 以降は内部で処理するためのシームカービング
        # 削除するシームのマスクを作成
        seam_mask = _get_seam_mask(gray, seam)
        pair_seam_mask = _get_seam_mask(pair_gray, pair_seam)

        #　シームを可視化（いづれか片方のみ） 
        visualize(gray, seam_mask)
        # visualize(pair_gray, pair_seam_mask)

        # 画像のシーム部分を削除
        gray = _remove_seam_mask(gray, seam_mask)
        pair_gray = _remove_seam_mask(pair_gray, pair_seam_mask)

        # 画像のインデックスも同じようにシーム部分を削除
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        pair_idx_map = _remove_seam_mask(pair_idx_map, pair_seam_mask)

        # 視差マップも同じ用にシーム部分を削除？
        disp = _remove_seam_mask(disp, seam_mask)

        # マスク画像もシーム部分を削る
        if aux_energy is not None:
            aux_energy = _remove_seam_mask(aux_energy, seam_mask)
        
        time.sleep(0.1)  # 処理にかかる時間をシミュレートするための待機時間

    return seams, pair_seams

# 画像幅の縮小
def _reduce_stereo_width(
    src: np.ndarray,
    src_pair: np.ndarray,       # 追加：　ペア画像(右画像の予定)
    disp: np.ndarray,           # 追加：　深度マップ
    delta_width: int,
    aux_energy: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    画像の幅を delta_width ピクセルだけ縮小する関数

    引数:
        src (np.ndarray)       : 入力画像
        src_pair (np.ndarray)  : ペア画像(右画像の予定)
        disp (np.ndarray)      : 深度マップ
        delta_width (int)     : 縮小する幅
        aux_energy (np.ndarray): 補助エネルギー（オプション）

    戻り値:
        dst (np.ndarray)       : 縮小後の画像
        dst_pair (np.ndarray)  : 縮小後のペア画像(右画像の予定)
        aux_energy (np.ndarray): 縮小後の補助エネルギー（オプション）
    """

    assert src.ndim in (2, 3) and delta_width >= 0

    # 入力画像がグレースケールか判断
    if src.ndim == 2:
        gray = src
        pair_gray = src_pair
        src_h, src_w = src.shape

        # dst_shape = 幅を縮小させた後の画像サイズ
        dst_shape: Tuple[int, ...] = (src_h, src_w - delta_width)   
    else:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        pair_gray = cv2.cvtColor(src_pair, cv2.COLOR_RGB2GRAY)
        src_h, src_w, src_c = src.shape

        # dst_shape = 幅を縮小させた後の画像サイズ
        dst_shape = (src_h, src_w - delta_width, src_c)

    to_keep: np.ndarray
    to_keep_pair: np.ndarray

    # numba で高速に動作させるためにfloat32型に変換(宣言しないとfloat64型になり遅くなる)
    gray = np.asarray(gray, dtype=np.float32)
    pair_gray = np.asarray(pair_gray, dtype=np.float32)

    to_keep, to_keep_pair =  _get_stereo_forward_seams(gray, pair_gray, disp, delta_width, aux_energy)

    dst = src[~to_keep].reshape(dst_shape)
    dst_pair = src_pair[~to_keep_pair].reshape(dst_shape)

    if aux_energy is not None:
        aux_energy = aux_energy[~to_keep].reshape(dst_shape[:2])     # 最終的には必要ないのでペアの分はつくらない

    return dst, dst_pair, aux_energy

def stereo_resize(
    src         : np.ndarray,
    src_pair    : np.ndarray,
    disp        : np.ndarray,
    size        : Optional[Tuple[int, int]] = None,
    keep_mask   : Optional[np.ndarray] = None,
) -> np.ndarray:
    
    """
    内容を考慮したシームカービングアルゴリズムを用いて，画像のサイズを変更します.
    """

    if disp.ndim == 3:
        disp_gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
    else:
        disp_gray = disp

    aux_energy = None
    if keep_mask is not None:
        keep_mask = keep_mask.astype(bool)
        aux_energy = np.zeros(src.shape[:2], dtype=np.float32)
        aux_energy[keep_mask] += KEEP_MASK_ENERGY

    width, height = size
    width = round(width)
    height = round(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"expect target size to be positive, got {size}")
    src, src_pair, aux_energy = _reduce_stereo_width(src, src_pair, disp_gray,  src.shape[1] - width, aux_energy)

    return src, src_pair


# ◆◆ 通常画像のリサイズ ◆◆

@nb.jit(cache=True, nopython=True)
def _get_forward_seam(gray: np.ndarray, aux_energy: Optional[np.ndarray]) -> np.ndarray:
    """Compute the minimum vertical seam using forward energy"""
    """順方向エネルギーを使って最小垂直シームを計算する"""
    h, w = gray.shape

    # 水平方向に「画像の最前列」＋「画像」＋「画像の最後列」をくっつけてる
    # 多分オリジナル画像のままだと一番端だけ動的計画法できないから両端を拡張しているんだと思う
    gray = np.hstack((gray[:, :1], gray, gray[:, -1:]))
    # 無限配列を作成してる
    inf = np.array([np.inf], dtype=np.float32)
    # 1列目のエネルギーのみ絶対値差分を取ってエネルギーを計算する。両端に無限配列をくっつけているのは謎
    dp = np.concatenate((inf, np.abs(gray[0, 2:] - gray[0, :-2]), inf))

    # 多分、もとになる初期化されていない空配列を作成している
    parent = np.empty((h, w), dtype=np.int32)

    # 注目画素の左上(0), 上(1), 右上(2)で表記される min_idx を、
    # 左端からのインデックスに変換するための配列
    base_idx = np.arange(-1, w - 1, dtype=np.int32)
    
    # エネルギーマップの確認用
    energy = np.zeros((h, w))  # エネルギーマップの初期化

    # 対象が含まれる行を左右にずらしたものを用意
    # 左右を1ピクセル拡張しているので2ピクセルずらしている
    shl_curr = gray[:, 2:]
    shr_curr = gray[:, :-2]
    prev_mid = roll_2d(gray, 0, -1)[:, 1:-1]  # 画像(h-w)方向に１つ上の列

    # 最上段のエネルギー
    dp_mid = dp[1:-1]
    dp_left = dp[:-2]
    dp_right = dp[2:]

    # 対象ピクセルが削除された後，左右のピクセルがくっつくことにより発生するエネルギー
    cost_mid = np.abs(shl_curr - shr_curr)
    cost_left = cost_mid + np.abs(prev_mid - shr_curr)
    cost_right = cost_mid + np.abs(prev_mid - shl_curr)

    # マスク部分にコストを追加
    if aux_energy is not None:
        cost_mid += aux_energy
    
    for r in range(1, h):
        # 各コストの選択肢を表示
        choices = np.vstack((
            cost_left[r]    + dp_left, 
            cost_mid[r]     + dp_mid, 
            cost_right[r]   + dp_right
        ))

        # シームとして選択される各ピクセルから、コストが最小の次のピクセルへの方向を記録
        min_idx = np.argmin(choices, axis=0)

        # シームとして選択される各ピクセルから、コストが最小の次のピクセルへのインデックスを記録
        parent[r] = min_idx + base_idx

        # numba does not support specifying axis in np.min, below loop is equivalent to:
        # `dp_mid[:] = np.min(choices, axis=0)` or `dp_mid[:] = choices[min_idx, np.arange(w)]`
        for j, i in enumerate(min_idx):
            dp_mid[j] = choices[i, j]

        # choisesからmin_idx方向に対応した最小方向の画素を取得
        for j in range(w):
            energy[r, j] = choices[min_idx[j], j]

    c = np.argmin(dp)
    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam, energy

def _get_forward_seams(
    gray: np.ndarray, num_seams: int, aux_energy: Optional[np.ndarray]
) -> np.ndarray:
    """Compute minimum N vertical seams using forward energy"""
    h, w = gray.shape
    seams = np.zeros((h, w), dtype=bool)
    rows = np.arange(h, dtype=np.int32)
    idx_map = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))
    for i in tqdm(range(num_seams)):
        seam, energy = _get_forward_seam(gray, aux_energy)
        if i == 0:
            plt.imshow(energy)
            plt.title("show Energy Map")
            plt.savefig("../results/test_img/energy_map.png")
            plt.show()
        seams[rows, idx_map[rows, seam]] = True
        seam_mask = _get_seam_mask(gray, seam)
        visualize(gray, seam_mask)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if aux_energy is not None:
            aux_energy = _remove_seam_mask(aux_energy, seam_mask)

    return seams


def _get_seams(
    gray: np.ndarray, 
    num_seams: int, 
    energy_mode: str, 
    aux_energy: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the minimum N seams from the grayscale image"""
    gray = np.asarray(gray, dtype=np.float32)
    
    # 今回はforward のみ利用する
    if energy_mode == EnergyMode.FORWARD:
        return _get_forward_seams(gray, num_seams, aux_energy)
    else:
        raise ValueError(
            f"expect energy_mode to be one of {_list_enum(EnergyMode)}, got {energy_mode}"
        )

# 画像幅の縮小
def _reduce_width(
    src: np.ndarray,
    delta_width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Reduce the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0

    # 入力画像がグレースケールか判断
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape

        # dst_shape = 幅を縮小させた後の画像サイズ
        dst_shape: Tuple[int, ...] = (src_h, src_w - delta_width)   
    else:
        # gray = _rgb2gray(src)
        # pair_gray = _rgb2gray(src_pair)
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        src_h, src_w, src_c = src.shape

        # dst_shape = 幅を縮小させた後の画像サイズ
        dst_shape = (src_h, src_w - delta_width, src_c)

    to_keep: np.ndarray

    to_keep = _get_seams(gray, delta_width, energy_mode, aux_energy)

    dst = src[~to_keep].reshape(dst_shape)

    if aux_energy is not None:
        aux_energy = aux_energy[~to_keep].reshape(dst_shape[:2])     # 最終的には必要ないのでペアの分はつくらない

    return dst, aux_energy

# 画像のリサイズ(今回は縮小のみ扱う)
def _resize_width(
    src: np.ndarray,
    width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    
    """Resize the width of image by removing vertical seams"""
    assert src.size > 0 and src.ndim in (2, 3)
    assert width > 0

    src_w = src.shape[1]

    # 画像幅と入力された数値を比較し、結果によって縮小・拡張を判断（今回は縮小のみ考える）
    dst, aux_energy = _reduce_width(src, src_w - width, energy_mode, aux_energy)
    
    return dst, aux_energy


# 試験的にステレオコストの重みも引数にしています
def resize(
    src         : np.ndarray,
    size        : Optional[Tuple[int, int]] = None,
    energy_mode : str = "forward",
    order       : str = "width-first",
    keep_mask   : Optional[np.ndarray] = None,
) -> np.ndarray:


    if order not in _list_enum(OrderMode):
        raise ValueError(
            f"expect order to be one of {_list_enum(OrderMode)}, got {order}"
        )

    aux_energy = None

    if keep_mask is not None:
        keep_mask = keep_mask.astype(bool)
        aux_energy = np.zeros(src.shape[:2], dtype=np.float32)
        aux_energy[keep_mask] += KEEP_MASK_ENERGY

    cv2.waitKey(0)

    width, height = size
    width = round(width)
    height = round(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"expect target size to be positive, got {size}")

    src, aux_energy = _resize_width(
        src, width, energy_mode, aux_energy
    )

    return src
