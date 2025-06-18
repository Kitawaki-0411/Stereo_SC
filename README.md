# Stereo_seam_carving

## 概要

このリポジトリは、修士研究の一環として、**ステレオ動画像に対する立体視を維持したシームカービング**を実装するためのコードおよび関連資料を記録しています。

## 目的

* [具体的な研究目的や課題をここに記述（例: 動画像における立体視の破綻を低減するシームカービング手法の確立と検証）]
* [コードの進捗状況や、試行錯誤のプロセスを記録する]

## プロジェクト構成 (任意)

プロジェクト内の主要なディレクトリやファイルの役割を簡単に記述します。

* `src/`: メインのソースコード
* `data/`: 入力データや出力データ
* `notebooks/`: 実験用のJupyter Notebookなど
* `results/`: 実験結果やグラフなど

## 必要なデータ
* [サンプルデータ](https://drive.google.com/drive/folders/1C_mDworgYfj2DWSmuJFlmqyC5soYeF59)

people_rect.mp4 の全フレームに対する深度マップ(full_disp_cache.pkl)とマスク画像(full_mask_cache.pkl)のピクルデータです．
実行の際に必要なので，ダウンロードして`sample`フォルダに配置してください

## 実行方法
```bash
python stereo_video_seam_carving.py
