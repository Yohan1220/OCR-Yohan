import cv2
'''Este modulo proporciona funciones para filtrar imágenes,'''

# Función para filtrar la imagen: redimensionar y convertir a escala de grises
def imagenFiltrada(image):
    # ampliar la imagen
    img_resized = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # Convertir a escala de grises
    mejorar_iluminacion_contraste(image)
    img_resized=gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    return gray

# Aplicar un umbral binario a la imagen filtrada

def mejorar_iluminacion_contraste(imagen_original):
    """
    Aplica la ecualización adaptativa del histograma (CLAHE) para mejorar 
    la iluminación y el contraste local de la imagen.
    """
    if imagen_original is None:
        print("Error: La imagen de entrada es None.")
        return None

    # 1. Convertir al espacio de color LAB
    # LAB separa la Luminosidad (L) de los canales de color (A y B)
    lab = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2LAB)
    
    # 2. Separar los canales
    l, a, b = cv2.split(lab)
    
    # 3. Crear el objeto CLAHE
    # clipLimit es el umbral para limitar el contraste (40 es un buen valor inicial)
    # tileSize es el tamaño de la cuadrícula en la que se ecualiza (8x8 es el valor por defecto)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    
    # 4. Aplicar CLAHE solo al canal de luminosidad (L)
    cl = clahe.apply(l)
    
    # 5. Fusionar los canales A y B originales con el canal L mejorado
    limg = cv2.merge((cl, a, b))
    
    # 6. Convertir de nuevo a BGR
    imagen_mejorada = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return imagen_mejorada

def imagenFiltradaBinaria(image):
    # Obtener la imagen filtrada en escala de grises
    contraste=mejorar_iluminacion_contraste(image)
    gray = imagenFiltrada(contraste)

    # Aplicar umbral binario
    thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
    
    return thresh


            