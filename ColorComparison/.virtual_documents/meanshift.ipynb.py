import ipywidgets as wd
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import hed


def applyBlur(img, sp=20, sr=50):
    meanshift = cv2.pyrMeanShiftFiltering(img,sp=sp,sr=sr)    
    return meanshift



img = cv2.imread("../data/imgs/olivino.jpg",1)
final = applyBlur(img, 40, 70)
hed.graphImgs([img[:,:,::-1], final[:,:,::-1]], ["Imagen Original", "Imagen con Filtro Mean Shift"])




imgray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = np.zeros(final.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
cv2.imwrite("contornos1.jpg",img_contours)

hed.graphImgs([final, img_contours])



