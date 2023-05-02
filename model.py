import argparse
import random

import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from torch.backends import cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device


class Model(QThread):
	is_connect_signal = pyqtSignal(bool)
	pic_signal = pyqtSignal(object)
	num_sum_signal = pyqtSignal(int)

	def __init__(self):
		super(QThread, self).__init__()
		self.timer_video = QtCore.QTimer()
		self.cap = cv2.VideoCapture()
		parser = argparse.ArgumentParser()
		parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
		parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
		parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
		parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
		parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
		parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
		parser.add_argument('--view-img', action='store_true', help='display results', default=True)
		parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
		parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
		parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
		parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
		parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
		parser.add_argument('--augment', action='store_true', help='augmented inference')
		parser.add_argument('--update', action='store_true', help='update all models')
		parser.add_argument('--project', default='runs/detect', help='save results to project/name')
		parser.add_argument('--name', default='exp', help='save results to project/name')
		parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
		self.opt = parser.parse_args()
		print(self.opt)

		source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

		self.device = select_device(self.opt.device)
		self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

		cudnn.benchmark = True

		# Load model
		self.model = attempt_load(
			weights, map_location=self.device)  # load FP32 model
		stride = int(self.model.stride.max())  # model stride
		self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
		if self.half:
			self.model.half()  # to FP16

		# Get names and colors
		self.names = self.model.module.names if hasattr(
			self.model, 'module') else self.model.names
		self.colors = [[random.randint(0, 255)
						for _ in range(3)] for _ in self.names]

	def connect_camera(self, video_file):
		if not self.timer_video.isActive():
			if video_file == "":
				# 0:默认使用第一个本地camera
				# rtmp: // localhost: 1935 / live / test
				flag = self.cap.open(0)
			else:
				flag = self.cap.open(filename=video_file)
			# 发送打开摄像头是否成功信号
			self.is_connect_signal.emit(flag)
		else:
			self.timer_video.stop()
			self.cap.release()

	def make_pic(self):
		flag, img = self.cap.read()
		if img is not None:
			showimg = img
			with torch.no_grad():
				img = letterbox(img, new_shape=self.opt.img_size)[0]
				# Convert
				# BGR to RGB, to 3x416x416
				img = img[:, :, ::-1].transpose(2, 0, 1)
				img = np.ascontiguousarray(img)
				img = torch.from_numpy(img).to(self.device)
				img = img.half() if self.half else img.float()  # uint8 to fp16/32
				img /= 255.0  # 0 - 255 to 0.0 - 1.0
				if img.ndimension() == 3:
					img = img.unsqueeze(0)
				# Inference
				pred = self.model(img, augment=self.opt.augment)[0]

				# Apply NMS
				pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
										   agnostic=self.opt.agnostic_nms)
				# Process detections
				for i, det in enumerate(pred):  # detections per image
					if det is not None and len(det):
						# Rescale boxes from img_size to im0 size
						det[:, :4] = scale_coords(
							img.shape[2:], det[:, :4], showimg.shape).round()
						# Write results
						for *xyxy, conf, cls in reversed(det):
							label = '%s %.2f' % (self.names[int(cls)], conf)
							plot_one_box(
								xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
					self.num_sum_signal.emit(len(det) if det is not None else 0)

			show = cv2.resize(showimg, (640, 480))
			result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
			self.pic_signal.emit(result)
		else:
			self.timer_video.stop()
			self.cap.release()
