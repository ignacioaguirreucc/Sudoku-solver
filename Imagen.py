# image_processor.py
import cv2
import numpy as np
import easyocr

class SudokuImageProcessor:
    def __init__(self):
        """Inicializa el lector de OCR una sola vez"""
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def extraer_sudoku(self, imagen):
        """
        Recibe una imagen y devuelve un array 9x9 con el sudoku detectado
        
        Args:
            imagen: numpy array (BGR) de la imagen
            
        Returns:
            list: matriz 9x9 con el sudoku (0 para celdas vacías)
        """
        # Preprocesamiento
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gris, (5, 5), 0)
        umbral = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        
        # Encontrar contornos y el sudoku
        contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno_sudoku = max(contornos, key=cv2.contourArea)
        
        # Detectar esquinas
        perimetro = cv2.arcLength(contorno_sudoku, True)
        aproximacion = cv2.approxPolyDP(contorno_sudoku, 0.02 * perimetro, True)
        
        if len(aproximacion) != 4:
            raise ValueError(f"No se pudo detectar el sudoku correctamente. Se encontraron {len(aproximacion)} esquinas en vez de 4")
        
        # Transformar perspectiva
        sudoku_transformado = self._transformar_perspectiva(imagen, aproximacion)
        
        # Extraer números
        sudoku_array = self._extraer_numeros(sudoku_transformado)
        
        return sudoku_array
    
    def _transformar_perspectiva(self, imagen, aproximacion):
        """Corrige la perspectiva del sudoku"""
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
        
        return sudoku_transformado
    
    def _extraer_numeros(self, sudoku_transformado):
        """Extrae los números de cada celda usando OCR"""
        gris = cv2.cvtColor(sudoku_transformado, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gris, (3, 3), 0)
        umbral = cv2.adaptiveThreshold(blur, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        sudoku_array = []
        lado = 450
        tamaño_celda = lado // 9
        
        for fila in range(9):
            fila_sudoku = []
            for columna in range(9):
                y1 = fila * tamaño_celda
                y2 = (fila + 1) * tamaño_celda
                x1 = columna * tamaño_celda
                x2 = (columna + 1) * tamaño_celda
                
                celda = umbral[y1:y2, x1:x2]
                margen = 5
                celda_limpia = celda[margen:-margen, margen:-margen]
                
                # Detectar si hay contenido
                pixeles_blancos = cv2.countNonZero(celda_limpia)
                area_celda = celda_limpia.shape[0] * celda_limpia.shape[1]
                porcentaje = (pixeles_blancos / area_celda) * 100
                
                if porcentaje > 3:
                    numero = self._leer_numero(celda_limpia)
                    fila_sudoku.append(numero)
                else:
                    fila_sudoku.append(0)
            
            sudoku_array.append(fila_sudoku)
        
        return sudoku_array
    
    def _leer_numero(self, celda):
        """Lee un número de una celda usando OCR"""
        # Mejorar contraste antes de OCR
        celda_mejorada = cv2.threshold(celda, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Intentar con configuración más estricta
        resultado = self.reader.readtext(celda_mejorada, allowlist='123456789', detail=1, paragraph=False)
        
        if resultado and len(resultado) > 0:
            texto = resultado[0][1].strip()
            confianza = resultado[0][2]
            
            # Solo aceptar si tiene confianza razonable
            if texto.isdigit() and 1 <= int(texto) <= 9 and confianza > 0.4:
                numero = int(texto)
                
                # Análisis de forma solo para 1 y 7 con baja confianza
                if (numero == 1 or numero == 7) and confianza < 0.7:
                    altura, ancho = celda_mejorada.shape
                    
                    # Analizar ancho del dígito
                    # El 1 es muy delgado (ancho < 40% del ancho total)
                    # El 7 ocupa más ancho (> 50% del ancho total)
                    columnas_con_pixeles = []
                    for col in range(ancho):
                        if cv2.countNonZero(celda_mejorada[:, col]) > 0:
                            columnas_con_pixeles.append(col)
                    
                    if len(columnas_con_pixeles) > 0:
                        ancho_digito = max(columnas_con_pixeles) - min(columnas_con_pixeles)
                        ratio_ancho = ancho_digito / ancho
                        
                        # Analizar distribución vertical
                        tercio_superior = celda_mejorada[0:altura//3, :]
                        tercio_medio = celda_mejorada[altura//3:2*altura//3, :]
                        
                        pixeles_arriba = cv2.countNonZero(tercio_superior)
                        pixeles_medio = cv2.countNonZero(tercio_medio)
                        pixeles_totales = cv2.countNonZero(celda_mejorada)
                        
                        # Heurísticas mejoradas:
                        # - El 1 es delgado (ratio_ancho < 0.35) y uniforme verticalmente
                        # - El 7 es más ancho (ratio_ancho > 0.45) y tiene mucho peso arriba
                        
                        if ratio_ancho < 0.35 and pixeles_medio > pixeles_arriba * 0.8:
                            # Muy delgado y uniforme = probablemente 1
                            return 1
                        elif ratio_ancho > 0.45 and pixeles_arriba > pixeles_medio * 1.2:
                            # Más ancho y con peso arriba = probablemente 7
                            return 7
                
                return numero
        
        return 0  # No se pudo leer