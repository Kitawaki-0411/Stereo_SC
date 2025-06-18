# 計算したパラメータを読み込んで補正する
import cv2
import numpy as np
import glob
import os
import shutil
import sys
from tqdm import tqdm
from tqdm.contrib import tenumerate

# キャリブレーションデータ
CALIB_DATA    =  'stereo_calib2.npz'

def video2img(video_path):
    """
    動画を画像化してディレクトリに保存

    Args:
        (str)   video_path  : 対象の動画のパス

    Results:
        (list)  sbs_list    : 動画をフレームごとに切り出した
    
    """

    sbs_list = []

    cap = cv2.VideoCapture(f'{video_path}')

    if not cap.isOpened():
        print("not exist file")
        return

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True:
        ret, frame = cap.read()
        if ret:
            sbs_list.append(frame)
            pbar.update()
        else:
            pbar.close()
            return np.array(sbs_list)

def img2video(img_list : np.ndarray, save_name : str, save_dir : str):
    """
    画像リストを mp4 動画化する

    Args:
        (str)   save_name   : 動画元のファイル名
        (str)   iv_dir      : 動画化したい画像群が存在するディレクトリのパス
        (int)   start       : 先頭のフレーム数
        (int)   end         : 末尾のフレーム数

    Results:
        None
    """
    # 画像リスト読み込み
    # 画像のサイズを取得
    im = cv2.imread(img_list[0])
    h, w, _ = im.shape

    print(f"\n\nTurning images into videos")
    print(f'frames          : {len(img_list)}')
    print(f"Height, Width   : {h}, {w}")

    save_pass = f"{save_dir}/{save_name}.mp4"

    print(f"save pass   : {save_pass}")

    # encoder(for mp4)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    # 動画のコーデックを指定してcv2.VideoWriterクラスのインスタンスを作成
    video = cv2.VideoWriter(f"{save_pass}.mp4",fourcc, 30.0, (w, h))

    if not video.isOpened():
        print("can't be opened")
        sys.exit()
    
    for path in tqdm(img_list):
        img = cv2.imread(path)
        video.write(img)

    video.release()
    print("finish")

def imglist2video(img_list, save_name, save_dir):
    """
    画像リストを mp4 動画化する

    Args:
        (np.ndarray) img_list   : 動画化したい画像のリスト
        (str)   save_name       : 動画元のファイル名
        (str)   save_dir        : 動画化したい画像群が存在するディレクトリのパス

    Results:
        None
    """

    # 画像リスト読み込み
    # 画像のサイズを取得
    if img_list[0].ndim == 2:
        h, w = img_list[0].shape
    else:
        h, w, _ = img_list[0].shape

    save_pass = f"{save_dir}/{save_name}"

    # encoder(for mp4)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    # 動画のコーデックを指定してcv2.VideoWriterクラスのインスタンスを作成
    video = cv2.VideoWriter(f"{save_pass}.mp4",fourcc, 30.0, (w, h))
    
    for img in tqdm(img_list):
        video.write(img)

    video.release()


def _adjust_disparity(sbs, disp_num):
    """
    視差の調整
    
    Args:
        (np.ndarray)    sbs         : 視差調整するSBS画像 
        (int)           disp_num    : 調整する数値. disp_num < 0 のときは視差を減らし, disp_num > 0 のときは視差を増やす.
    
    Return:
        (np.ndarray)    adjust_sbs  : 視差を調整したSBS画像
    """

    left  =  sbs[:, :sbs.shape[1]//2]
    right =  sbs[:, sbs.shape[1]//2:]

    if disp_num < 0:
        # 視差を減らす
        disp_left  =  np.hstack((np.fliplr(left[:,:-disp_num]), left[:,:disp_num]))
        disp_right =  np.hstack((right[:,-disp_num:], np.fliplr(right[:,disp_num:])))
    elif disp_num > 0:
        # 視差を増やす
        disp_left  =  np.hstack((left[:,disp_num:], np.fliplr(left[:,-disp_num:])))
        disp_right =  np.hstack((np.fliplr(right[:,:disp_num]), right[:,:-disp_num]))
    
    adjust_sbs = np.hstack((disp_left, disp_right))

    return adjust_sbs


def rectification(img_path, adjust_disp = 0):
    """
    ステレオカメラのキャリブレーション結果から入力画像のレクティフィケーションを行う

    Args:
        (str)  img_path     : 入力画像のパス
    
    Results:
        (np.ndarray)    sbs : レクティフィケーション結果の画像
    """

    # キャリブレーション結果を読み込み
    calib_data = np.load(CALIB_DATA)
    mtx_l = calib_data['mtx_l']
    dist_l = calib_data['dist_l']
    mtx_r = calib_data['mtx_r']
    dist_r = calib_data['dist_r']
    R = calib_data['R']
    T = calib_data['T']

    # 画像の用意
    img_sbs   = cv2.imread(img_path)
    img_left  = img_sbs[:, :img_sbs.shape[1]//2]
    img_right = img_sbs[:, img_sbs.shape[1]//2:]

    # 画像のサイズに合わせて設定
    image_size  = (img_left.shape[1], img_left.shape[0])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, alpha=0
    )

    # リマップの設定
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2)

    # リマップを適用して画像を補正
    rectified_left = cv2.remap(img_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, map1_r, map2_r, cv2.INTER_LINEAR)

    # レクティフィケーション結果をSBSに変換
    sbs = cv2.hconcat([rectified_left, rectified_right])

    # adjust_disp が 0 以外なら視差調整する
    if adjust_disp != 0:
        sbs = _adjust_disparity(sbs, adjust_disp)

    return sbs


def video_rectification(path_list, base_name, save_dir, adjust_disp = 0, start = 0, ext ='jpg'):
    """
    動画のレクティフィケーション

        Args:
            (list)  path_list : 動画像のパスを格納したリスト
            (str)   base_name   : 動画元のファイル名
            (int)   start       : 先頭のフレーム数
            (int)   end         : 末尾のフレーム数
            (int)   adjust_disp : 視差を調整する場合の数値. デフォルト値は 0, adjust_dispが 0 以上で視差調整処理を行う．

        Return:
            (None)
    """
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    
    # レクティフィケーション処理
    cv2.namedWindow("result",cv2.WINDOW_NORMAL)

    rect_list = [rectification(img_path, adjust_disp) for i, img_path in tenumerate(path_list)]
    return rect_list

    # for i, img_path in tenumerate(path_list):
    #     num = str(i + start)

    #     sbs = rectification(img_path, adjust_disp)
    #     save_name = f'{base_name}{num.zfill(4)}'
        
    #     resize = 0.3
    #     cv2.imshow("result", cv2.resize(sbs, None, fx = resize, fy = resize))
    #     cv2.imwrite(f'{save_dir}/{save_name}.{ext}', sbs)

    #     if cv2.waitKey(1) == ord('q'):
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #         break
    # cv2.destroyAllWindows()


def rect_video(base_name, sbs_read_dir, save_dir, ad = 0, start = 0, end = 0):
    # 動画を画像へ切り出し
    if "n" == input(f"skip convert video to image?  y/n : "):
        print("convert video to image")
        video2img(base_name)
    else:
        pass

    # 画像を読み込む
    sbs_list = glob.glob(f'{sbs_read_dir}/*.JPG')

    # 指定範囲のみ動画化したい
    if end != 0:
        sbs_list = sbs_list[start : end+1]

    # ディレクトリがなければ作る
    os.makedirs(f'{save_dir}', exist_ok = True)

    # 動画をレクティフィケーション
    print(f'\nMake the image list rectificate')
    print(f'    save directory : {save_dir}')
    rect_list = video_rectification(sbs_list, base_name, save_dir, adjust_disp = ad, start = start)

    rect_sbs_read_dir = save_dir

    print(f'base_name           : {base_name}')
    print(f'sbs_read_dir        : {sbs_read_dir}')
    print(f'save_dir            : {save_dir}')
    print(f'rect_sbs_read_dir   : {rect_sbs_read_dir}')

    # 動画化
    # def img2video(save_name, read_dir, save_dir, start = 0, end = 0):
    if end != 0: 
        img2video(base_name, rect_sbs_read_dir, save_dir, start = start, end = end)
    else:
        # img2video(base_name, rect_sbs_read_dir, save_dir)
        imglist2video(rect_list, base_name, rect_sbs_read_dir, save_dir)

def rect_img(path, disp):
    # 画像をレクティフィケーション
    rect_sbs = rectification(path, disp)
    cv2.imwrite(f"C:/oit/py23/SourceCode/m-research/ZED2_calibration/rectified/adjust_disp/disp_{disp}.jpg",rect_sbs)

def main():
    # 動画のファイル名
    base_name = "people"

    # レクティフィケーションに関する定数
    read_dir  =  f'video/{base_name}/img'
    save_dir  =  f'rectified/{base_name}/SBS'

    rect_video(base_name, read_dir, save_dir)

if __name__ == "__main__":
    main()