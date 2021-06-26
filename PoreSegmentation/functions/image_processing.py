from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


def import_image(file):
    """Imports PIL image for display and processing inside streamlit application

    :param file: File as given by streamlit file selector
    :type file: Streamlit file upload
    :return: Image to use
    :rtype: PIL.Image
    """
    img = Image.open(file)
    return_img = img.convert("RGB")
    return return_img


def quantize_image(img_num: np.ndarray, n_clusters: int) -> np.ndarray:
    """quantizes image in n_clusters of colors and returns color-segmented image

    :param img_num: Image to quantize as numpy array
    :type img_num: np.ndarray
    :param n_clusters: Number of clusters (colors) to quantize image into 
    :type n_clusters: int
    :return: Quantized image as numpy array
    :rtype: np.ndarray
    """

    m, l, k = img_num.shape[0], img_num.shape[1], img_num.shape[2]

    df = pd.DataFrame(img_num.reshape(m*l, k))

    df.columns = ["R", "G", "B"]
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df)
    df["clusters"] = kmeans.labels_

    centers = pd.DataFrame(kmeans.cluster_centers_)
    centers["clusters"] = range(n_clusters)
    s = df.clusters.value_counts(normalize=True, sort=False).mul(
        100)  # giving percentage by clustered color

    # finding index with "bluest" color for porosity
    colors = df.copy().groupby(["clusters"]).mean()

    colors["percentage"] = df.clusters.value_counts(
        normalize=True, sort=False).mul(100)  # mul(100) is == *100
    index_larger = colors.sort_values(by=["B"], ascending=False)
    por = index_larger.iloc[0].name
    colors['R'] = colors['R'].apply(lambda x: x*255)
    colors['G'] = colors['G'].apply(lambda x: x*255)
    colors['B'] = colors['B'].apply(lambda x: x*255)

    df["ind"] = df.index
    df = df.merge(centers)
    df = df.sort_values("ind")
    df = df.drop("ind", 1)
    quant_img = df.iloc[:, 4:7].values

    quant_img = quant_img.reshape(
        img_num.shape[0], img_num.shape[1], img_num.shape[2])

    return quant_img, colors, por
