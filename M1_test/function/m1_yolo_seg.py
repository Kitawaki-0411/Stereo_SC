from ultralytics import YOLO
import cv2
import numpy as np
from matplotlib import pyplot as plt

def make_mask(img): 
    # モデル読み込み
    model = YOLO("yolov8n-seg.pt")

    # 入力画像
    results = model(img, save=True, retina_masks=True, task='segment') 

    # セグメント結果からマスク画像を作る
    for i, result in enumerate(results):
        # マスクデータを取得(物体検出できなかったときはNoneが返る)
        if result.masks is not None:
            masks = result.masks.cpu().numpy().data
            mask_class = result.boxes.cpu().numpy().data

            mask = np.zeros((img.shape[0], img.shape[1]))

            for i in range(masks.shape[0]):
                if mask_class[i][-1] == 0 :
                    tmp_mask = masks[i]
                    mask = np.logical_or(mask, tmp_mask)
        else :
            mask = np.zeros((img.shape[0], img.shape[1]))

        

    return mask

def make_mask_list(img_list):
    if None in img_list:
        print("img_list is None")
    else:
        mask_list = np.array([make_mask(img) for img in img_list])
        
    return mask_list

def main():
    img = cv2.imread("sample_l.png")
    mask = make_mask(img)
    cv2.imwrite("mask.png", mask*255)

    # 推定結果
    fig = plt.figure(figsize=(15, 12))
    a1 = fig.add_subplot(1,1,1)
    a1.set_title("left_img",fontsize=11)
    plt.imshow(mask)
    plt.show()

if __name__ == "__main__":
    main()
