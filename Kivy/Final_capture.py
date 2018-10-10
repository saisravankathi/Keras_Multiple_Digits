# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:07:53 2018

@author: S795641
"""

import cv2
import os
import math
from imageai.Detection import ObjectDetection

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "./RetinaNet/resnet50_coco_best_v2.0.1.h5"))
"""using YOLOv3 TINY to detect objects, so the model type and model path differs."""
#detector.setModelTypeAsTinyYOLOv3()
#detector.setModelPath(os.path.join(execution_path, "./YOLO/yolo-tiny.h5"))
detector.loadModel(detection_speed="flash")
"""detection_speed has values; normal, fast, faster, fastest, flash """

cap = cv2.VideoCapture(0)
#frameRate = cap.get(cv2.CAP_PROP_FPS)
#fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
#out = cv2.VideoWriter('output.mp4', fourcc, 50, (640,480))

def detectObject(pathImg, value):
    #detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , "./images/cool2.jpg"), output_image_path=os.path.join(execution_path , "image_new.png"), custom_objects=custom_objects, minimum_percentage_probability=65)
    detections = detector.detectObjectsFromImage(input_image=pathImg, output_image_path=os.path.join(execution_path , "pimage"+str(value)+".png"), minimum_percentage_probability=70)
    for eachObject in detections:
       print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
       print("--------------------------------")
       
count=0     
#print(frameRate)
while(True):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    #frameRem = (frameId % math.floor(frameRate))
    if (ret != True):
        break
    if (True):
        filename ="frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
        detectObject(filename,count-1)
        os.remove(filename)
        img = cv2.imread("pimage"+str(count-1)+".png")
        cv2.imshow('prediction',img)
        os.remove("pimage"+str(count-1)+".png")
        #cv2.imshow('predict', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #if cv2.waitKey(1) & 0xFF == ord('d'):
        #detectObject(pathImg)

cap.release()
#out.release()
cv2.destroyAllWindows()