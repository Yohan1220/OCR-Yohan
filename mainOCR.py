import modOCR
import cv2

def main(ruta):

    image=cv2.imread(ruta)

    texto=modOCR.texto(image)
    return texto
