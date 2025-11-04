import cv2
import numpy as np

def findContornos(path="h.png"):
    imagen = cv2.imread(path)
    if imagen is None:
        print("No se pudo cargar la imagen.")
        return

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    img_suave = cv2.medianBlur(gris, 5)
    v = np.median(gris)

    th1 = max(0, v * 0.66)
    th2 = max(0, v * 1.33)

    kernel = np.ones((2, 2), np.uint8)
    binarizada = cv2.Canny(img_suave, 10, 150)
    binarizada = cv2.dilate(binarizada, kernel, iterations=1)
    binarizada = cv2.erode(binarizada, kernel, iterations=1)

    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Contornos encontrados:", len(contornos))
    if not contornos:
        print("No se encontraron contornos.")
        return

    mayor = max(contornos, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(mayor, True)
    curva = cv2.approxPolyDP(mayor, epsilon, True)

    if len(curva) != 4:
        print("El contorno principal no tiene 4 puntos, tiene:", len(curva))
        return

    # Asegurar orden correcto (puedes ajustar según tu imagen)
    pts = np.array([p[0] for p in curva], dtype="float32")

    # Ordena los puntos (superior izq, sup der, inf izq, inf der)
    suma = pts.sum(axis=1)
    resta = np.diff(pts, axis=1)

    iz_sup = pts[np.argmin(suma)]
    de_inf = pts[np.argmax(suma)]
    de_sup = pts[np.argmin(resta)]
    iz_inf = pts[np.argmax(resta)]

    puntosEntrada = np.float32([iz_sup, de_sup, iz_inf, de_inf])
    puntosSalida = np.float32([[0, 0], [270, 0], [0, 310], [270, 310]])

    matriz = cv2.getPerspectiveTransform(puntosEntrada, puntosSalida)
    newImg = cv2.warpPerspective(imagen, matriz, (270, 310))

    cv2.imshow("Salida", newImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

findContornos()
