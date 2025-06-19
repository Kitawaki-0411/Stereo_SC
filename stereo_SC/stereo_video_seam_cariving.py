import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from tqdm.contrib import tenumerate

from matplotlib import pyplot as plt

from my_seam_carve import carve_stereo

from function import m1_rectification as rect
from function import m1_depth_estimation as vdisp
from function import m1_yolo_seg as seg

import pickle

def save_img(img, save_name):
    file_path = "C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/results/sc_img/" + save_name
    cv2.imwrite(file_path, img)

def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def resize_arr(list, h, w):
    # listの各要素をh,wにリサイズして返す
    resize_arr = []

    for img in list:
        resize_img = cv2.resize(img, (h,w))
        resize_arr.append(resize_img)

    return np.array(resize_arr)

def main():
    # 扱う動画のパス
    video_path = "video/people_rect.mp4"

    # 読み込むキャッシュファイルのパス
    read_disp_cache = "sample_data/full_video_disp_cache[resize_x0.5].pkl"
    read_mask_cache = "sample_data/full_video_mask_cache[resize_x0.5].pkl"

    # 保存するキャッシュファイルのパス
    save_disp_cache = "full_video_disp_cache[resize_x0.5].pkl"
    save_mask_cache = "full_video_mask_cache[resize_x0.5].pkl"

    # 保存する動画の名前(拡張子[.mp4]不要)とディレクトリ
    save_name       = "stereo_SC_stereo_cost"   # 保存する動画名
    save_disp_name  = f"disp_{save_name}"       # 保存する視差動画名
    save_dir        = "results/video"           # 保存するディレクトリ

    # 処理方法の選択 : 動画の全フレームを縮小(0), 動画のフレームを指定して縮小(1)
    # process_type = 0
    process_type = 1

    # 動画を分割して処理する場合の開始フレームと終了フレーム
    s = 200
    e = 260

    # 実行モード：データ[全フレームの深度マップとマスク]作成(0), キャッシュ利用(1)
    exe_mode = 0
    # exe_mode = 1

    # 縮小させるpxサイズ
    dist = 100
    
    # 等倍だとデータ量が大きすぎてセグメント(マスク作成）が機能しないのでリサイズ
    resize = 0.5

    # 動画の読み込み(容量でかすぎてピクル化できない)
    print("cutting video to frame")
    sbs_arr = rect.video2img(video_path)   # 動作確認OK（フレーム切り出しできていた）
    sbs_arr = np.array([cv2.resize(sbs, None, fx = resize, fy = resize) for sbs in sbs_arr])

    if process_type == 0:
        start = 0
        end = len(sbs_arr)
    else:
        start = s
        end = e

    if exe_mode == 0:   # データ作成モード
            left_arr = np.array([sbs[:, :sbs_arr[0].shape[1]//2] for sbs in sbs_arr])
            print(f"left_arr shape = {left_arr.shape}")

            h = left_arr[0].shape[1]
            w = left_arr[0].shape[0]

            print("Estimating video disparity")
            disp_arr = vdisp.video_depth_estimate(sbs_arr)    # 動作確認OK（しっかり深度推定できていた）
            disp_arr = resize_arr(disp_arr, h, w)
            print(disp_arr.shape)
            # 推定結果を保存
            save_results(disp_arr, save_disp_cache)

            print("making mask")
            mask_arr = seg.make_mask_arr(left_arr)
            mask_arr = (mask_arr > 0).astype(np.int32)  # 0,1のint型に変換
            # => 入力と同じ画像サイズでマスクが出力される
            print(mask_arr.shape)

            # 推定結果を保存
            save_results(mask_arr, save_mask_cache)
            print("Saved results to cache.")

    elif exe_mode == 1: # ピクル利用モード
        # キャッシュから読み込み(マスク・視差推定結果)
        print("Loading cached results...")
        disp_arr= load_results(read_disp_cache)
        mask_arr = load_results(read_mask_cache)
        mask_arr = mask_arr.astype(bool)  # データ型をbooleanに変換
        print("Loaded cached results!")


    print(f"sbs_arr  shape = {sbs_arr.shape}")
    print(f"disp_arr shape = {disp_arr.shape}")
    print(f"mask_arr shape = {mask_arr.shape}")

    h = sbs_arr[0].shape[0]
    w = sbs_arr[0].shape[1]//2

    # SCによる縮小処理
    SC_arr =  carve_stereo.stereo_video_resize(sbs_arr[start:end], disp_arr[start:end],  (w - dist, h), mask_arr = mask_arr[start:end])

    print(f"SC_arr check")
    print(f"SC_arr.shape : {SC_arr.shape}")
    cv2.waitKey(0)

    for i in range(e-s):
        cv2.imshow("sbs", SC_arr[i])
        cv2.waitKey(80)
    cv2.destroyAllWindows()

    # SCによる縮小結果の保存
    rect.imglist2video(SC_arr, save_name, save_dir)

    # SCによる縮小結果の深度推定
    SC_disp_arr = resize_arr(vdisp.video_depth_estimate(SC_arr), w - dist, h)
    SC_disp_map = np.empty((e-s, h, w - dist, 3), dtype=np.uint8)

    print(f"SC_disp_arr shape = {SC_disp_arr.shape}")
    print(f"SC_disp_map shape = {SC_disp_map.shape}")

    for i in range(e-s):
        SC_disp_map[i] = cv2.applyColorMap(SC_disp_arr[i], cv2.COLORMAP_JET)
    
    print(f"SC_disp_map shape = {SC_disp_map.shape}")

    rect.imglist2video(SC_disp_map, save_disp_name, save_dir)

if __name__ == "__main__":
    main()