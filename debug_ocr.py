# debug_ocr.py - Para depurar problemas de OCR con fondos de color
import cv2
import numpy as np
from PIL import Image
import easyocr

def procesar_celda_metodo1(celda_original):
    """Método 1: Convertir a gris normalmente"""
    if len(celda_original.shape) == 3:
        gris = cv2.cvtColor(celda_original, cv2.COLOR_BGR2GRAY)
    else:
        gris = celda_original
    
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(gris, (3, 3), 0)
    umbral = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return umbral

def procesar_celda_metodo2(celda_original):
    """Método 2: Usar CLAHE"""
    if len(celda_original.shape) == 3:
        gris = cv2.cvtColor(celda_original, cv2.COLOR_BGR2GRAY)
    else:
        gris = celda_original
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gris = clahe.apply(gris)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(gris, (3, 3), 0)
    umbral = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return umbral

def procesar_celda_metodo3(celda_original):
    """Método 3: Eliminar rosa primero, luego procesar"""
    if len(celda_original.shape) == 3:
        # Convertir a HSV
        hsv = cv2.cvtColor(celda_original, cv2.COLOR_BGR2HSV)
        
        # Crear máscara para colores claros/pasteles (rosa, celeste, etc)
        # Baja saturación + alta luminosidad = color pastel
        lower = np.array([0, 0, 200])  # Cualquier tono, baja saturación, alta luminosidad
        upper = np.array([180, 100, 255])
        mask_fondo = cv2.inRange(hsv, lower, upper)
        
        # Convertir a gris
        gris = cv2.cvtColor(celda_original, cv2.COLOR_BGR2GRAY)
        
        # Donde hay fondo claro, poner blanco
        gris[mask_fondo > 0] = 255
    else:
        gris = celda_original
    
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(gris, (3, 3), 0)
    umbral = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return umbral

def procesar_celda_metodo4(celda_original):
    """Método 4: Usar solo el canal más oscuro (donde está el número)"""
    if len(celda_original.shape) == 3:
        # Tomar el canal que tiene más contraste (generalmente el más oscuro)
        b, g, r = cv2.split(celda_original)
        
        # El canal con menor promedio tiene el número más visible
        promedio_b = np.mean(b)
        promedio_g = np.mean(g)
        promedio_r = np.mean(r)
        
        if promedio_b <= promedio_g and promedio_b <= promedio_r:
            gris = b
        elif promedio_g <= promedio_r:
            gris = g
        else:
            gris = r
    else:
        gris = celda_original
    
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(gris, (3, 3), 0)
    umbral = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return umbral

# Probar con imagen
imagen_path = input("Ingresá la ruta de la imagen del sudoku: ").strip('"')
imagen = cv2.imread(imagen_path)

# Procesar para encontrar el sudoku (código simplificado)
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gris, (5, 5), 0)
umbral = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)

contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contorno_sudoku = max(contornos, key=cv2.contourArea)

perimetro = cv2.arcLength(contorno_sudoku, True)
aproximacion = cv2.approxPolyDP(contorno_sudoku, 0.02 * perimetro, True)

# Transformar perspectiva
esquinas = aproximacion.reshape(4, 2)
suma = esquinas.sum(axis=1)
diff = np.diff(esquinas, axis=1)

arriba_izq = esquinas[np.argmin(suma)]
abajo_der = esquinas[np.argmax(suma)]
arriba_der = esquinas[np.argmin(diff)]
abajo_izq = esquinas[np.argmax(diff)]

pts_origen = np.float32([arriba_izq, arriba_der, abajo_der, abajo_izq])
lado = 450
pts_destino = np.float32([[0, 0], [lado, 0], [lado, lado], [0, lado]])

matriz = cv2.getPerspectiveTransform(pts_origen, pts_destino)
sudoku_transformado = cv2.warpPerspective(imagen, matriz, (lado, lado))

# Guardar el sudoku transformado
cv2.imwrite("debug_sudoku_transformado.png", sudoku_transformado)
print("✓ Sudoku transformado guardado en debug_sudoku_transformado.png")

# Extraer varias celdas para probar
tamaño_celda = lado // 9

# Vamos a probar con una celda que tiene "1" rosa
# Según tu imagen: fila 6, columna 0 tiene un "1" con fondo rosa
fila_problema = 6  
col_problema = 0

y1 = fila_problema * tamaño_celda
y2 = (fila_problema + 1) * tamaño_celda
x1 = col_problema * tamaño_celda
x2 = (col_problema + 1) * tamaño_celda

celda_original = sudoku_transformado[y1:y2, x1:x2]

# Probar los 4 métodos
print("\nProbando 4 métodos diferentes...\n")

metodo1 = procesar_celda_metodo1(celda_original)
cv2.imwrite("debug_metodo1_normal.png", metodo1)
print("Método 1 (Normal): debug_metodo1_normal.png")

metodo2 = procesar_celda_metodo2(celda_original)
cv2.imwrite("debug_metodo2_clahe.png", metodo2)
print("Método 2 (CLAHE): debug_metodo2_clahe.png")

metodo3 = procesar_celda_metodo3(celda_original)
cv2.imwrite("debug_metodo3_eliminar_rosa.png", metodo3)
print("Método 3 (Eliminar rosa): debug_metodo3_eliminar_rosa.png")

metodo4 = procesar_celda_metodo4(celda_original)
cv2.imwrite("debug_metodo4_canal_oscuro.png", metodo4)
print("Método 4 (Canal más oscuro): debug_metodo4_canal_oscuro.png")

# Probar OCR en cada método
print("\n--- Resultados de OCR ---")
reader = easyocr.Reader(['en'], gpu=False)

for nombre, imagen_metodo in [("Método 1", metodo1), ("Método 2", metodo2), 
                               ("Método 3", metodo3), ("Método 4", metodo4)]:
    resultado = reader.readtext(imagen_metodo, allowlist='123456789', detail=1)
    if resultado:
        texto = resultado[0][1]
        confianza = resultado[0][2]
        print(f"{nombre}: '{texto}' (confianza: {confianza:.2f})")
    else:
        print(f"{nombre}: No detectó nada")

print("\n✓ Revisá las imágenes generadas para ver cuál funciona mejor")
