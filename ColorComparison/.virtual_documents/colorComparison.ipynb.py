import caer
import ipywidgets as wd
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import hed


imgFolder= "../data/imgs/"


def colorSpaces(img):
    img = cv2.imread(img)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    hsl = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    YBR = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    return rgb,hsv,lab,hsl,YBR

def edges(img):
    canny = cv2.Canny(img,100,200)
    return canny

@wd.interact(A=(0,4,1), cannyEdges=False, img=[('olivino', imgFolder+"olivino.jpg"), ('estilolitos', imgFolder+"estilolito.jpg"),\
                                          ("olivino serpentinizado",imgFolder+"olivino_serpentina.jpg")])

def actualice(A=1, cannyEdges=False,img=imgFolder+"olivino.jpg"):
    plt.figure(figsize=(9,5))
    colors = ["RGB", "HSV", "LAB", "HSL", "YBR"]
    if cannyEdges:
        plt.title(f"Bordes Canny Modelo Color: {colors[A]}")
        return plt.imshow(edges(colorSpaces(img)[A]))
    else:
        plt.title(f"Sección {img} Modelo Color: {colors[A]}")
        return plt.imshow(colorSpaces(img)[A])


olivino = imgFolder + "result.jpg"
groundtruth = imgFolder+"groundtruth.jpg"



gt = cv2.imread(groundtruth, 0)
gtCol = cv2.imread(groundtruth,1)
plt.imshow(gt)


olrgb,olhsv,ollab,olhsl,olYBR = colorSpaces(olivino)
images = [olrgb,olhsv,ollab,olhsl,olYBR]




@wd.interact(imagen=[('RGB', 0), ('HSV', 1),("LAB",2),\
                  ('HSL', 3), ('olYBR', 4)], cannyEdges=False, overlayTruth=False)



def actualice(imagen="RGB", cannyEdges=False, overlayTruth=False):
    plt.figure(figsize=(12,9))
    if overlayTruth:
        if cannyEdges:
            final = cv2.addWeighted(gt, 0.6, edges(images[imagen]), 0.4, 0.0)
            return plt.imshow(final)
        else:
            final = cv2.addWeighted(gtCol, 0.6, images[imagen], 0.4, 0.0)
            return plt.imshow(final)
    elif cannyEdges:
        #plt.title(f"Bordes Canny Modelo Color: {colors[A]}")
        return plt.imshow(edges(images[imagen]))
    else:
        #plt.title(f"Sección {img} Modelo Color: {colors[A]}")
        return plt.imshow(images[imagen])


def applyBlur(img, sp=20, sr=50):
    meanshift = cv2.pyrMeanShiftFiltering(img,sp=sp,sr=sr)
    
    return meanshift

@wd.interact(sp=(0,100,5), sr=(0,100,5), save=False)

def actualice(img=imgFolder+"olivino.jpg", sp=20, sr=50, save=False):
    img = cv2.imread(img,1)
    plt.figure(figsize=(12,9))
    final = applyBlur(img, sp, sr)
    if save:
        plt.imsave("result.jpg", final)
    return plt.imshow(final)


meanShift = cv2.imread(imgFolder + "result.jpg")
edgesMeanShift = edges(meanShift)

imgray1 = cv2.cvtColor(meanShift, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(imgray1, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = np.zeros(meanShift.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)


hed.graphImgs([edgesMeanShift, img_contours])



