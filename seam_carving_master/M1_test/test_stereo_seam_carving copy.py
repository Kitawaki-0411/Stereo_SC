import numpy as np
from PIL import Image
import cv2

from matplotlib import pyplot as plt

from my_seam_carve import carve_test as ct
from ultralytics import YOLO
from function import m1_rectification as rect
from function import m1_depth_estimation as vdisp
from function import m1_yolo_seg as seg

import pickle

# の読み込み
def save_results(results, filename):
    """
    入力されたデータをピクル化して保存します

    引数:
        results(pikle): pikleデータ（深度推定結果やマスク画像など） 
        filename(str): 保存先のファイル名
    """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results(filepass):
    """
    入力されたpikleデータを読み込みます

    引数: 
        filename(str): 読み込むpikleデータのあるファイルパス
    
    戻り値:
        results(np.ndarray): pikleデータ（深度推定結果やマスク画像など）
    """
    with open(filepass, 'rb') as f:
        return pickle.load(f)

def make_mask(img): 
    """
    入力された画像から人の領域のマスクを作成します

    引数:
        img(np.ndarray): 入力画像
        filename(str): 保存先のファイル名

    戻り値:
        mask(np.ndarray): 人の領域のマスク画像
    """
    model = YOLO("yolov8n-seg.pt")
    results = model(img, save=True, retina_masks=True, task='segment') 
    for i, result in enumerate(results):
        if result == None:
            print("There is no person in the image.")
            return None
        masks = result.masks.cpu().numpy().data         # マスクデータを取得
        mask_class = result.boxes.cpu().numpy().data    # クラスデータを取得
        mask = np.zeros((masks.shape[1], masks.shape[2]))
        for i in range(masks.shape[0]):
            if mask_class[i][-1] == 0 : # person
                tmp_mask = masks[i]
                mask = np.logical_or(mask, tmp_mask)
    mask = convert_bool_arr(mask)   # 配列自体をbool型に変換（これしないとエラー出る）
    return mask

def convert_bool_arr(array):
    """
    入力された配列全体ををbool型に変換します.
    ※ numbaでは配列内の型ではなく配列自体の型が認識されるためこの関数を使用する必要があります.
    ※ np.array.astypeを使用すると, 配列の数値自体は変わりますが, 配列自体の方はbool型に変わらないためこの関数を使用しています.

    引数:
        array(np.ndarray): 入力配列

    戻り値:
        np.ndarray: bool型に変換された配列
    """

    return np.array([[bool(elem) for elem in row] for row in array])

def main():
    # pikleのデータパス
    sbs_cache_filename = "test_data/sbs.pkl"
    disp_cache_filename = "test_data/disp.pkl"
    mask_cache_filename = "test_data/mask.pkl"

    exe_mode = 1    # 実行する際に深度マップとマスク画像を作成するかどうか [作成する = 1, 作成しない(pikleを利用) = 0]
    check_disp = 1  # SC結果画像の深度マップを確認するかどうか [確認する = 1, 確認しない = 0]
    dist = 100      # 縮小させるpxサイズ
    resize = 1      # 縮小率

    img_path = "test_data/sbs.png"  # exe_mode = 0 の時に使用する画像

    if exe_mode == 0:   # データ作成モード
        sbs = cv2.imread(img_path)
        sbs = cv2.resize(sbs, None, fx = resize, fy = resize)
        save_results(sbs,sbs_cache_filename)

        h = sbs.shape[0]
        w = sbs.shape[1]//2

        print("Estimating video disparity")
        disp = vdisp.depth_estimation(sbs)
        disp = cv2.resize(disp, (w,h))
        print(disp.shape)
        save_results(disp, disp_cache_filename)

        print("making mask")
        mask = make_mask(sbs[:, :w])    #　入力と同じ画像サイズでマスクが出力される
        save_results(mask, mask_cache_filename)
        print("Saved results to cache.")

    elif exe_mode == 1: # ピクル利用モード
        print("Loading cached results...")
        sbs  = load_results(sbs_cache_filename)
        disp = load_results(disp_cache_filename)
        mask = load_results(mask_cache_filename)
        print("Loaded cached results!")

        h = sbs.shape[0]
        w = sbs.shape[1]//2

    print(f"sbs shape   :{sbs.shape}")
    print(f"disp shape  :{disp.shape}")
    print(f"mask shape  :{mask.shape}")

    src = sbs[:, :w]
    pair = sbs[:, w:]

    scale_down, scale_down_pair = ct.stereo_resize(src, pair, disp, (w - dist, h), keep_mask = mask) # 画像幅(w)から dist だけ増減させる

    scale_down      = cv2.cvtColor(scale_down      ,cv2.COLOR_BGR2RGB) # PIL は RGB 前提, imread は BGR で読み込むため RGB 変換が必要
    scale_down_pair = cv2.cvtColor(scale_down_pair ,cv2.COLOR_BGR2RGB)

    result_sbs = np.hstack((scale_down, scale_down_pair))

    
    # ==========================================================================================================
    # # サンプルで実行するためのデータ作成
    # save_results(sbs, 'C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/test_data/sbs.pkl')
    # save_results(disp, 'C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/test_data/disp.pkl')
    # save_results(mask, 'C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/test_data/mask.pkl')
    # cv2.imwrite("C:/oit/py23/SourceCode/m-research/seam_carving_master/M1_test/test_data/result_sbs.png", result_sbs)
    # ==========================================================================================================

    plt.imshow(result_sbs)
    plt.title('result_sbs')
    plt.show()

    if check_disp == 1:

        result_disp = vdisp.depth_estimation(result_sbs)
        result_disp = cv2.resize(result_disp, (w,h))
        compare_disp = np.hstack((disp, result_disp))
        plt.imshow(compare_disp)
        plt.title('compare_disp')
        plt.show()

    print(f'finish')
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()