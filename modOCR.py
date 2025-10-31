from pytesseract import image_to_string
import renderizar, modFiltro
''''Este modulo proporciona la funcionalidad de extraer t retornar el
texto de una imagen utilizando OCR (Reconocimiento Óptico de Caracteres) a 
través de la biblioteca pytesseract.'''

def texto(image):
     
    # Preprocesar la imagen utilizando la función imagenFiltradaBinaria de modFiltro
    #imgPytesseract=modCorrecOrientacion.limpiarImagen(image) 
    #imgPytesseract=modFiltro.imagenFiltradaBinaria(image)

    imgRenderizada=renderizar.mainRenderizar(image)
    imgPytesseract=modFiltro.imagenFiltradaBinaria(imgRenderizada)

    # Extraer texto de la imagen procesada utilizando pytesseract
    text = image_to_string(imgPytesseract, lang='spa')
    
    # Retornar el texto extraído
    return text
