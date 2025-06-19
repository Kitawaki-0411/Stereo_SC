import cv2
import numpy as np
import glob
from tqdm import tqdm
from crestereo import CREStereo, CameraConfig
import os

def convert_int_num(array):
    # 各要素を整数型に変換（リスト内包表記を使用）
    return np.array([[int(elem) for elem in row] for row in array])

def depth_estimation(sbs_img):

	# Model options (not all options supported together)
	iters = 5            	# Lower iterations are faster, but will lower detail. 
							# Options: 2, 5, 10, 20 

	input_shape = (720, 1280)	# Input resolution. 
								# Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

	version = "combined"	# The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
							# Options: "init", "combined"

	# Camera options: baseline (m), focal length (pixel) and max distance
	# TODO: Fix with the values witht the correct configuration for YOUR CAMERA
	# camera_config = CameraConfig(0.12, 0.5*input_shape[1]/0.72) 
	# max_distance = 50

	# Initialize model
	# model_path = f'C:/oit/py23/SourceCode/m-research/ONNX-CREStereo-Depth-Estimation/models/crestereo_{version}_iter{iters}_{input_shape[0]}x{input_shape[1]}.onnx'
	model_path = f'crestereo/models/crestereo_{version}_iter{iters}_{input_shape[0]}x{input_shape[1]}.onnx'
	depth_estimator = CREStereo(model_path)
	disp = depth_estimator(sbs_img[:,:sbs_img.shape[1]//2], sbs_img[:,sbs_img.shape[1]//2:])
	disp = convert_int_num(disp).astype(np.uint8) 

	return disp

def video_depth_estimate(img_list):
	# Model options (not all options supported together)
	iters = 5            	# Lower iterations are faster, but will lower detail. 
							# Options: 2, 5, 10, 20 

	input_shape = (720, 1280)	# Input resolution. 
								# Options: (240,320), (320,480), (380, 480), (360, 640), (480,640), (720, 1280)

	version = "combined"	# The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
							# Options: "init", "combined"

	# Camera options: baseline (m), focal length (pixel) and max distance
	# TODO: Fix with the values witht the correct configuration for YOUR CAMERA
	# camera_config = CameraConfig(0.12, 0.5*input_shape[1]/0.72) 
	# max_distance = 50

	# Initialize model
	model_path = f'C:/oit/py23/SourceCode/m-research/ONNX-CREStereo-Depth-Estimation/models/crestereo_{version}_iter{iters}_{input_shape[0]}x{input_shape[1]}.onnx'
	depth_estimator = CREStereo(model_path)

	disp_list = [depth_estimator(img[:,:img.shape[1]//2], img[:,img.shape[1]//2:]) for img in tqdm(img_list)]

	# 各要素を整数型に変換
	disp_list =np.array( [np.array(convert_int_num(disp)).astype(np.uint8) for disp in disp_list])

	return disp_list