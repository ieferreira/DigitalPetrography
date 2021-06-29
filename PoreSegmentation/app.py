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

        st.write("______________________ \n")
        st.write("POROSIDAD = ", round(
            color_percentages.iloc[por].percentage, 2))

        # rgb_colors = color_percentages["RGB"].to_list()
        # rgb = []
        # for color in rgb_colors:
        #     color = color.split(",")
        #     rgb.append(color)
        # rgb_colors =
        # print(rgb)
        # hex_colors = [rgb_to_hex(rgb[i])
        #               for i in range(len(rgb))]
        # print(hex_colors)
        # plt.pie(color_percentages["percentage"], colors=hex_colors)
        # plt.savefig("colors_pie.png")

        st.write("______________________ \n")
        st.write("Other color percentages: ")
        st.table(color_percentages[["R", "G", "B", "percentage"]])
