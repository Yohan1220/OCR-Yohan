import cv2
import numpy as np

def visualizar_renglones(img):
    
    #Procesa una imagen de texto para resaltar los renglones.
    try:
        if img is None:
            print(f"Error: No se pudo cargar la ")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Binarización (Umbralización)
        # THRESH_OTSU automáticamente encuentra el mejor umbral
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        # 3. Definir el Kernel de Dilatación
        # Un kernel horizontal ancho (40) y bajo (5) para conectar caracteres
        # en el mismo renglón sin fusionar renglones adyacentes.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))

        # 4. Aplicar Dilatación
        # Esto 'une' los caracteres de cada renglón en una sola línea gruesa.
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        return dilated # Devuelve la imagen con los renglones resaltados

    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento: {e}")


def calculate_skew_angle(dilated_image: np.ndarray) -> float:
    """
    Calcula el ángulo de inclinación (sesgo) de una imagen de texto 
    preprocesada (dilatada).

    Args:
        dilated_image: La imagen binaria con los renglones unidos y resaltados.

    Returns:
        float: El ángulo de sesgo en grados.
    """
    # 1. Detección de Líneas con HoughLinesP
    # Ajusta los parámetros (ej. umbral) si detectas demasiadas líneas o muy pocas
    lines = cv2.HoughLinesP(
        dilated_image, 
        rho=1,                # Resolución de distancia
        theta=np.pi / 180,    # Resolución angular (1 grado)
        threshold=100,        # Mínimo de votos
        minLineLength=100,    # Longitud mínima de la línea
        maxLineGap=20         # Máxima brecha para unir segmentos
    )

    angles = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Evitamos divisiones por cero para líneas perfectamente verticales
            if x2 != x1:
                # Calcular la tangente (pendiente)
                m = (y2 - y1) / (x2 - x1)
                
                # Calcular el ángulo en radianes y convertir a grados
                # np.arctan devuelve el ángulo entre -90 y 90 grados
                angle_rad = np.arctan(m)
                angle_deg = np.degrees(angle_rad)
                
                angles.append(angle_deg)

    if angles:
        # Usamos la mediana para que sea más robusta a los valores atípicos (ruido)
        median_angle = np.median(angles)
        return median_angle
    else:
        # Si no se detectan líneas, asumimos que no hay rotación
        return 0.0
    
def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
   
    # Obtener las dimensiones de la imagen
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(
        image, 
        M, 
        (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated_image

def mainRenderizar(image):

    renglones=visualizar_renglones(image)
    angulo=calculate_skew_angle(renglones)
    return deskew_image(image,angulo)
