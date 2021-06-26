import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from functions.utils import *
from functions.image_processing import *
import streamlit as st


clean_folder()

st.markdown(
    """## Porosity identification in (blue) tinted sedimentary sections (petrography)""")
st.sidebar.title("Algorithms and parameters")
st.markdown(
    "An online app for petrographers (work in progress) please leave any comments to iveferreirach[at]unal.edu.co")
st.sidebar.markdown("Algorithms to use and parameters")
st.write("#### Please, Upload an Image")


# file = st.file_uploader("", type=["jpg", "png"])
file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.text("Upload an image..")
    st.sidebar.markdown("#### Waiting for an image...")

if file:
    img = import_image(file)
    img_org = img.copy()
    img_num = np.asarray(img_org)
    img_num = img_num/255
    # img_num = img_num[:, :, -1]

    st.sidebar.markdown("### Preprocessing")

    st.image(img, use_column_width=True)
    st.sidebar.markdown("""Select number of color clusters""")
    n_clusters = st.sidebar.slider("# of clusters", 1, 5, 3)

    if st.sidebar.checkbox("Generar imagen segmentada por colores", value=False):

        quant_img, color_percentages, por = quantize_image(img_num, n_clusters)
        st.image(quant_img, clamp=True, channels="RGB")
        clustered_img = quant_img*255

        # m, l, k = clustered_img.shape[0], clustered_img.shape[1], clustered_img.shape[2]
        # df2 = pd.DataFrame(clustered_img.reshape(m*l, k))
        # df2.columns = ["R", "G", "B"]
        # contados = df2.apply(pd.value_counts).sum(axis=1)

        # total = pd.unique(contados)

        # suma = total.sum()
        # porcentaje = (total/suma)*100

        st.write("______________________ \n")
        st.write("POROSIDAD = ", round(
            color_percentages.iloc[por].percentage, 2))

        st.write("______________________ \n")
        st.write("Other color percentages: ")
        st.table(color_percentages)
