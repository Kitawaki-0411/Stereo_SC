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

def resize_list(list, h, w):
    # listの各要素をh,wにリサイズして返す
    resize_list = []

    for img in list:
        resize_img = cv2.resize(img, (h,w))
        resize_list.append(resize_img)

    return np.array(resize_list)

def main():
    # 動画をフルで処理するかどうか
    # separate = 0: 動画をフルで処理
    # separate = 1: 動画をフレーム指定して処理
    separate = 1

    # 実行：データ作成モード = 0, ピクル利用モード = 1
    # exe_mode = 0
    exe_mode = 1

    # 動画を分割して処理する場合の開始フレームと終了フレーム
    # s = 200
    # e = 400

    s = 200
    e = 260

    # 縮小させるpxサイズ
    dist = 100
    
    # 動画の縮小
    resize = 0.5

    # 扱う動画のパス
    video_path = "C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/video/people_rect.mp4"

    # 必要なキャッシュファイルのパス
    read_disp_cache = "sample/full_disp_cache.pkl"
    read_mask_cache = "sample/full_mask_cache.pkl"
    save_disp_cache = "disp_cache.pkl"
    save_mask_cache = "mask_cache.pkl"

    # 保存する動画の名前(拡張子[.mp4]不要)とディレクトリ
    save_name = "stereo_SC_stereo_cost"
    save_disp_name = f"disp_{save_name}"
    save_dir = "C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/results/sc_video"

    # 動画の読み込み(容量でかすぎてピクル化できない)
    print("cutting video to frame")
    sbs_list = rect.video2img(video_path)   # 動作確認OK（フレーム切り出しできていた）
    sbs_list = np.array([cv2.resize(sbs, None, fx = resize, fy = resize) for sbs in sbs_list])

    if separate == 0:
        start = 0
        end = len(sbs_list)
    else:
        start = s
        end = e

    if exe_mode == 0:   # データ作成モード
            print(f"frames = {end - start}")
            left_list = np.array([sbs[:, :sbs_list[0].shape[1]//2] for sbs in sbs_list])
            print(f"left_list shape = {left_list[start:end+1].shape}")

            h = left_list[0].shape[1]
            w = left_list[0].shape[0]

            print("Estimating video disparity")
            disp_list = vdisp.video_depth_estimate(sbs_list[start:end])    # 動作確認OK（しっかり深度推定できていた）
            disp_list = resize_list(disp_list, h, w)
            print(disp_list.shape)
            # 推定結果を保存
            save_results(disp_list, save_disp_cache)

            print("making mask")
            mask_list = seg.make_mask_list(left_list[start:end])
            mask_list = (mask_list > 0).astype(np.int32)  # 0,1のint型に変換
            # => 入力と同じ画像サイズでマスクが出力される
            print(mask_list.shape)

            # 推定結果を保存
            save_results(mask_list, save_mask_cache)
            print("Saved results to cache.")

    elif exe_mode == 1: # ピクル利用モード
        # キャッシュから読み込み(マスク・視差推定結果)
        print("Loading cached results...")
        disp_list= load_results(read_disp_cache)
        mask_list = load_results(read_mask_cache)
        mask_list = mask_list.astype(bool)  # データ型をbooleanに変換
        print("Loaded cached results!")


    print(f"sbs_list  shape = {sbs_list.shape}")
    print(f"disp_list shape = {disp_list.shape}")
    print(f"mask_list shape = {mask_list.shape}")

    h = sbs_list[0].shape[0]
    w = sbs_list[0].shape[1]//2

    # SCによる縮小処理
    SC_list =  carve_stereo.stereo_video_resize(sbs_list[start:end], disp_list[start:end],  (w - dist, h), mask_arr=mask_list[start:end])

    print(f"SC_list check")
    print(f"SC_list.shape : {SC_list.shape}")
    cv2.waitKey(0)

    for i in range(e-s):
        cv2.imshow("sbs", SC_list[i])
        cv2.waitKey(80)
    cv2.destroyAllWindows()
    rect.imglist2video(SC_list, save_name, save_dir)

    # SCによる縮小結果の深度推定
    SC_disp_list = resize_list(vdisp.video_depth_estimate(SC_list), w - dist, h)
    SC_disp_map = np.empty((e-s, h, w - dist, 3), dtype=np.uint8)

    print(f"SC_disp_list shape = {SC_disp_list.shape}")
    print(f"SC_disp_map shape = {SC_disp_map.shape}")

    for i in range(e-s):
        SC_disp_map[i] = cv2.applyColorMap(SC_disp_list[i], cv2.COLORMAP_JET)
    
    print(f"SC_disp_map shape = {SC_disp_map.shape}")

    rect.imglist2video(SC_disp_map, save_disp_name, save_dir)

if __name__ == "__main__":
    main()