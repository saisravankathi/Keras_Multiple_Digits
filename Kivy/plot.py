# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 14:42:35 2018

@author: S795641
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/S795641/Desktop/Kivy/bot.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([])
plt.yticks([])
plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
plt.show()
