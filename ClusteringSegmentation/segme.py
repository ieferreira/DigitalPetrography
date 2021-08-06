import numpy as np
from PIL import Image 
import string
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import base64
from sklearn.cluster import KMeans
import multiprocessing


img=Image.open('olivino.jpg')
pix=img.load()
print(img.mode, img.size)
palet_len=30

kmean=KMeans(palet_len, n_jobs=20)

# copy pixel data
data = np.asarray(img.getdata())

# fix the kmean
kmean.fit(data)

# assig each pixel to cluster
imgclust=kmean.predict(data)

# reshape data for easier access in copy process


def treed__pixel_chart(data, text, alpha=0.1):
    print(text)
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], alpha=alpha, color=data/255.0)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()
    
treed__pixel_chart(np.take(data, imgclust, axis=0), 'Pixels 3D chart compressed image', alpha=0.1)
