from skimage import io # se importa input-output
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 




qtz = "imagenes/cuarcita.jpg" # camino a la imagen a usar (cuarcita)
tap = "imagenes/tapete.jpg" # un tapete de mi cada
carb = "imagenes/carbonato.jpg" # carbonatos
oliv = "imagenes/olivino.jpg" # olivinos


img = io.imread(qtz)
io.imshow(img)
print(img.shape)


# reescalar la imagen, se dan el factor con la división
img_reescalada = rescale(img, 1.0/5.0, anti_aliasing=True) # 5 veces
io.imshow(img_reescalada)


# cambiar tamaño de la imagen

resize_img = resize(img, (200,200))
io.imshow(resize_img)


# reducir escala de la imagen

downscaled = downscale_local_mean(img,(4,3,3))
io.imshow(downscaled)


img = io.imread(qtz, as_gray=True)
io.imshow(img)



from skimage.filters import gaussian, sobel, scharr, roberts, prewitt


img = io.imread(qtz, as_gray=True)
gaussianImg = gaussian(img)
io.imshow(sobelImg)
io.imsave("resultados/gaussianCuarcita.jpg", gaussianImg)


img = io.imread(qtz, as_gray=True)
sobelImg = sobel(img)
io.imshow(sobelImg)
io.imsave("resultados/sobelCuarcita.jpg", sobelImg)


img = io.imread(qtz, as_gray=True)
scharrImg = scharr(img)
io.imshow(robertsImg)
io.imsave("resultados/scharrCuarcita.jpg", scharrImg)


img = io.imread(qtz, as_gray=True)
robertsImg = roberts(img)
io.imshow(robertsImg)
io.imsave("resultados/robertsCuarcita.jpg", robertsImg)


img = io.imread(qtz, as_gray=True)
prewittImg = prewitt(img)
io.imshow(prewittImg)
io.imsave("resultados/prewittCuarcita.jpg", prewittImg)


fig, axes =  plt.subplots(nrows=3,ncols=2, sharex=True, sharey=True, figsize=(8,8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title("Imagen Original")

ax[1].imshow(gaussianImg, cmap=plt.cm.gray)
ax[1].set_title("Imagen Filtro Gausiano")

ax[2].imshow(scharrImg, cmap=plt.cm.gray)
ax[2].set_title("Imagen Filtro Scharr")

ax[3].imshow(robertsImg, cmap=plt.cm.gray)
ax[3].set_title("Imagen Filtro Roberts")

ax[4].imshow(sobelImg, cmap=plt.cm.gray)
ax[4].set_title("Imagen Filtro Sobel")

ax[5].imshow(prewittImg, cmap=plt.cm.gray)
ax[5].set_title("Imagen Filtro Prewitt")

plt.savefig("resultados/comparacionFiltros.jpg")
