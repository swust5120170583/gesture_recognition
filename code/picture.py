import cv2
import numpy as np
import math
import fourierDescriptor as fd
import efd as efd
from scipy.ndimage.interpolation import geometric_transform

MIN_DESCRIPTOR = 500
NUMBER_OF_HARMONICS = 20

#二值化处理
def binaryMask(frame, x0, y0, width, height):
	cv2.rectangle(frame,(x0,y0),(x0+width, y0+height),(0,255,0))
	roi = frame[y0:y0+height, x0:x0+width]

	res = skinMask(roi) #肤色检测



	ret, fourier_result = fd.fourierDesciptor(res)
	efd_result, K, T = efd.elliptic_fourier_descriptors(res)

	return roi, res, ret, fourier_result, efd_result




def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res


