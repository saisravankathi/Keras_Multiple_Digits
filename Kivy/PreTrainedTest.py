# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:08:20 2018

@author: S795641
"""


import cv2
import os
from imageai.Detection import ObjectDetection
import math

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "./RetinaNet/resnet50_coco_best_v2.0.1.h5"))
"""using YOLOv3 TINY to detect objects, so the model type and model path differs."""
#detector.setModelTypeAsTinyYOLOv3()
#detector.setModelPath(os.path.join(execution_path, "./YOLO/yolo-tiny.h5"))
detector.loadModel(detection_speed="flash")
"""detection_speed has values; normal, fast, faster, fastest, flash """

print(detector.modelPath)
