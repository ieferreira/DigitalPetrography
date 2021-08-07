import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.cluster import KMeans


def load_img(fname):

    img = cv2.imread("sandLaminated.jpg", cv2.COLOR_BGR2RGB)
    return img


def plot_rgb(img):
    """Plots given image in RGB Color space using a 20,10 size on canvas

    Args:
        fname (string): path to image to be plotted
    """
    plt.rcParams["figure.figsize"] = (20, 10)

    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imgrgb)
    plt.title(imgrgb.shape)


def quantize_image(img, n_clusters):
    """Clusters image by color and returns a quantized image

    Args:
        img (np.array): Image to be quantized 
        n_clusters (int): Number of clusters to be used to quantize

    Returns:
        np.array: Quantized image
    """
    img = np.array(img, dtype=np.float64)/255  # se normaliza
    # redimesionando los datos
    m, l, k = img.shape[0], img.shape[1], img.shape[2]
    df = pd.DataFrame(img.reshape(m*l, k))
    df.columns = ["R", "G", "B"]

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df.sample(5000))
    df["clusters"] = kmeans.predict(df)

    centers = pd.DataFrame(kmeans.cluster_centers_)
    centers["clusters"] = range(n_clusters)
    df["ind"] = df.index
    df = df.merge(centers)
    df = df.sort_values("ind")
    df = df.drop("ind", 1)

    quant_img = df.iloc[:, 4:7].values

    quant_img = quant_img.reshape(img.shape[0], img.shape[1], img.shape[2])
    quant_img = np.array(quant_img, dtype=np.float64)

    return quant_img
