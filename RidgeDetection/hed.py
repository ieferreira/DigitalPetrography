import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import cv2

def graphImgsColor(imgs, title=None, cmap="Greys_r", figsize=(12,6)):
    fig, axes = plt.subplots(nrows=1, ncols=len(imgs), figsize=figsize)
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], cmap=cmap)
        if title is not None:
            try:
                ax.set_title(title[i])
            except:
                pass
    fig.tight_layout()
    
def graphImgs(imgs, title=None, cmap="Greys_r", figsize=(12,6)):
    """ imgs ->  a list of images
        title ->  a list of strings        
        returns mpl subplots
    """
    fig, axes = plt.subplots(nrows=1, ncols=len(imgs), figsize=figsize)
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], cmap=cmap)
        if title is not None:
            try:
                ax.set_title(title[i])
            except:
                pass
    fig.tight_layout()
    

def hed(image, apply_canny=False, canny_inf=100, canny_sup=200):
    class CropLayer(object):
        # TAKEN FROM: https://github.com/opencv/opencv/blob/master/samples/dnn/edge_detection.py
        def __init__(self, params, blobs):
            self.xstart = 0
            self.xend = 0
            self.ystart = 0
            self.yend = 0

        # Our layer receives two inputs. We need to crop the first input blob
        # to match a shape of the second one (keeping batch size and number of channels)
        def getMemoryShapes(self, inputs):
            inputShape, targetShape = inputs[0], inputs[1]
            batchSize, numChannels = inputShape[0], inputShape[1]
            height, width = targetShape[2], targetShape[3]

            self.ystart = int((inputShape[2] - targetShape[2]) / 2)
            self.xstart = int((inputShape[3] - targetShape[3]) / 2)
            self.yend = self.ystart + height
            self.xend = self.xstart + width

            return [[batchSize, numChannels, height, width]]

        def forward(self, inputs):
            return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]


    # Load the model.
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")
    cv2.dnn_registerLayer('Crop', CropLayer)

    inp = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=image.shape[:2],
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv2.resize(out, (image.shape[1], image.shape[0]))
    out=  cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)
    if apply_canny: # if apply_canny is true

        edges = cv2.Canny(out,canny_inf,canny_sup)
        return edges
    else:         
        return out