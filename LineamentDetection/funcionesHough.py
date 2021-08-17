# Creado por Ivan F. https://github.com/ieferreira, finales de Octubre de 2020
# funciones para facilitar uso del notebook HOUGH Google Earth

import cv2
import numpy as np
import matplotlib.pyplot as plt 

def load(filename):
    """
    loads image, takes string 
    input ->  file path as strig
    returns -> img loaded in cv2
    """
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    return img

def grey(img):
    """
    loads an image in greyscale
    input ->  file path as string
    output ->  image in cv2 in greyscale
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey

def graph(img, paleta="Greys"):
    """
    quickly plot an image
    input -> img file in cv2 format
    output -> matplotlib figure as 
    """
    plt.figure(figsize=(15,9))
    return plt.imshow(img, cmap=paleta)

def loadGrey(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey

def canny(grey, A, B):
    edges = cv2.Canny(grey, A,B)
    return edges


def plotCanny(grey, A, B):
    plt.Figure(figsize=(15,9))
    gf = cv2.Canny(grey, A,B)
    plt.title(f"Detector de bordes - Canny")
    return plt.imshow(gf)
    
def findLines(bordes, rho, theta, thr, mll, mlg):
    lineas = cv2.HoughLinesP(bordes, rho, theta, thr, minLineLength=mll, maxLineGap=mlg)
    return lineas

def drawLines(lineas, imagen):  
    if lineas is not None:
        for linea in lineas:
            x1,y1,x2,y2 = linea[0]
            # dibuja las lineas una a una en imagen, con color (255,0,0) y grosor de linea 1
            cv2.line(imagen, (x1,y1), (x2,y2), (255, 0,0), 1)
    else:
        # por si no se encuentra lineas o el algoritmo no funciona con los params dados
        raise ValueError("No se encontraron lineas con los parámetros dados")
    return imagen

def findCircles(bordes,n, pm1, pm2, mnDis, mnRad,  mxRad):
    circles = cv2.HoughCircles(bordes, cv2.HOUGH_GRADIENT, n,  param1=pm1,  param2=pm2,
                            minDist=mnDis,  minRadius=mnRad, maxRadius=mxRad)
    return circles

#circulos = findCircles(imagen2,1,50,30,100,0,100)

def drawCircles(circulos, img, escala=None):

    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for punto in circulos[0, :]:
            x, y, r = punto[0], punto[1], punto[2]

            # circunferencia del circulo
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            # pone el dato del valor como un texto adjunto
            if escala is not None:
                pixsUm = escala/100
                um = r*pixsUm
                cv2.putText(img,f"Tamano={round(um,1)}um\n Area = {round(np.pi*um**2,1)} um^2", (x+10,y+10), cv2.FONT_ITALIC, 0.5, (200,50,0,255),2)
                 
            else: 
                cv2.putText(img,f"RADIO={r}px", (x+10,y+10), cv2.FONT_ITALIC, 0.5, (200,50,0,255),2)

            # circulo para dibujar el radio, con color (0,122,255) y grosor de linea 3
            cv2.circle(img, (x, y), 1, (0, 122, 255), 3)
        return img
    else:
        raise ValueError("No se encontraron Círculos en la imagen que me pasaste con los parámetros dados")

