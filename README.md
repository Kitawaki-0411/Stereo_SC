# Stereo_seam_carving

## 概要

このリポジトリは、修士研究の一環として、**ステレオ動画像に対する立体視を維持したシームカービング**を実装するためのコードおよび関連資料を記録しています。

## フォルダ構成
stereo_SCのディレクトリやファイルの役割を簡単に記述します。

### ディレクトリ
* `crestereo`    　: 深度推定に必要なモデルや関数など
* `function`     　: stereo_video_seam_carvingに使用する関数など
* `my_seam_carve`　: シームカービングの関数など
* `sample_data`  　: 深度マップとマスク画像の推定結果など（このデータが有ればシームカービングの実行は可能）
* `results`      　: シームカービング結果を記録

## 必要なデータ
* [サンプルデータ](https://drive.google.com/drive/folders/1C_mDworgYfj2DWSmuJFlmqyC5soYeF59)

people_rect.mp4 の全フレームに対する深度マップ(full_disp_cache.pkl)とマスク画像(full_mask_cache.pkl)のピクルデータです．
実行の際に必要なので，ダウンロードして`sample`フォルダに配置してください

## 実行方法
```bash
python stereo_video_seam_carving.py
