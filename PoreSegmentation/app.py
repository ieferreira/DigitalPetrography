import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from pylab import rcParams
from functions.utils import *
from functions.image_processing import *
import streamlit as st


clean_folder()
clean_folder("responses/*")

st.markdown(
    """## Porosity identification in (blue) tinted sedimentary sections (petrography)""")
st.sidebar.title("Algorithms and parameters")
st.markdown(
    "An online app for petrographers (work in progress) please leave any comments to iveferreirach[at]unal.edu.co")
st.sidebar.markdown("Algorithms to use and parameters")
st.write("#### Please, Upload an Image")


file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.text("Upload an image..")
    st.sidebar.markdown("#### Waiting for an image...")

if file:
    img = import_image(file)
    img_org = img.copy()

    img_num = np.asarray(img_org)
    img_num = img_num/255

    st.sidebar.markdown("### Preprocessing")

    st.image(img, use_column_width=True)
    st.sidebar.markdown("""Select number of color clusters""")
    n_clusters = st.sidebar.slider("# of clusters", 1, 5, 3)

    if st.sidebar.checkbox("Generar imagen segmentada por colores", value=False):

        m, l, k = img_num.shape[0], img_num.shape[1], img_num.shape[2]

        df = pd.DataFrame(img_num.reshape(m*l, k))

        df.columns = ["R", "G", "B"]
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(df)
        df["clusters"] = kmeans.labels_

        centers = pd.DataFrame(kmeans.cluster_centers_)
        centers["clusters"] = range(n_clusters)
        df["ind"] = df.index
        df = df.merge(centers)
        df = df.sort_values("ind")
        df = df.drop("ind", 1)
        quant_img = df.iloc[:, 4:7].values

        quant_img = quant_img.reshape(
            img_num.shape[0], img_num.shape[1], img_num.shape[2])
        st.image(quant_img)
