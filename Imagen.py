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
        # Normalizar colores primero (eliminar fondos de color)
        # Convertir a escala de grises de manera más robusta
        if len(imagen.shape) == 3:
            # Si tiene colores, usar conversión ponderada que ignora colores de fondo
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
            # Normalizar intensidades para eliminar el efecto de fondos coloreados
            gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
        else:
            gris = imagen
        
        # Preprocesamiento
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
        
        # PASO 1: Convertir a escala de grises
        if len(sudoku_transformado.shape) == 3:
            gris = cv2.cvtColor(sudoku_transformado, cv2.COLOR_BGR2GRAY)
        else:
            gris = sudoku_transformado
        
        # PASO 2: Usar CLAHE para mejorar contraste local
        # Esto funciona MUCHO mejor con fondos de color
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gris = clahe.apply(gris)
        
        # PASO 3: Preprocesamiento GLOBAL (antes de dividir en celdas)
        # Blur suave
        blur = cv2.GaussianBlur(gris, (5, 5), 0)
        
        # Umbral adaptativo en toda la imagen (no por celda)
        # Esto mantiene mejor el contraste en áreas con fondo de color
        umbral = cv2.adaptiveThreshold(blur, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # PASO 4: Guardar también la versión en escala de grises procesada
        # para usar como backup en OCR
        gris_procesado = gris.copy()
        
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
                
                # Extraer celda de ambas versiones
                celda_umbral = umbral[y1:y2, x1:x2]
                celda_gris = gris_procesado[y1:y2, x1:x2]
                
                margen = 5
                celda_umbral_limpia = celda_umbral[margen:-margen, margen:-margen]
                celda_gris_limpia = celda_gris[margen:-margen, margen:-margen]
                
                # Detectar si hay contenido
                pixeles_blancos = cv2.countNonZero(celda_umbral_limpia)
                area_celda = celda_umbral_limpia.shape[0] * celda_umbral_limpia.shape[1]
                porcentaje = (pixeles_blancos / area_celda) * 100
                
                # Rango más flexible para detectar números
                if porcentaje > 3 and porcentaje < 65:
                    # ESTRATEGIA 1: Probar con versión umbralizada
                    numero = self._leer_numero(celda_umbral_limpia)
                    
                    # ESTRATEGIA 2: Si falla, probar con versión en escala de grises
                    if numero == 0:
                        numero = self._leer_numero(celda_gris_limpia, usar_umbral=False)
                    
                    # ESTRATEGIA 3: Si todavía falla y hay bastante contenido, sin margen
                    if numero == 0 and porcentaje > 8:
                        celda_sin_margen = celda_gris[2:-2, 2:-2]
                        numero = self._leer_numero(celda_sin_margen, usar_umbral=False)
                    
                    fila_sudoku.append(numero)
                else:
                    fila_sudoku.append(0)
            
            sudoku_array.append(fila_sudoku)
        
        return sudoku_array
    
    def _leer_numero(self, celda, usar_umbral=True):
        """Lee un número de una celda usando OCR"""
        # Si se pide, aplicar umbral OTSU
        if usar_umbral:
            celda_mejorada = cv2.threshold(celda, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        else:
            # Para escala de grises, invertir para que números sean blancos
            celda_mejorada = cv2.threshold(celda, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Intentar OCR
        resultado = self.reader.readtext(celda_mejorada, allowlist='123456789', detail=1, paragraph=False)
        
        if resultado and len(resultado) > 0:
            texto = resultado[0][1].strip()
            confianza = resultado[0][2]
            
            # Ser más tolerante con la confianza (bajar umbral de 0.4 a 0.25)
            if texto.isdigit() and 1 <= int(texto) <= 9 and confianza > 0.25:
                numero = int(texto)
                
                # Análisis de forma para números problemáticos
                # Siempre verificar 1, y los demás solo con baja confianza
                if numero == 1 or (confianza < 0.75 and numero in [4, 7, 9]):
                    numero_corregido = self._verificar_forma(celda_mejorada, numero)
                    if numero_corregido is not None:
                        return numero_corregido
                
                return numero
        
        return 0  # No se pudo leer
    
    def _verificar_forma(self, celda, numero_detectado):
        """Verifica la forma del dígito para corregir errores comunes"""
        altura, ancho = celda.shape
        
        # Analizar ancho del dígito
        columnas_con_pixeles = []
        filas_con_pixeles = []
        
        for col in range(ancho):
            if cv2.countNonZero(celda[:, col]) > 0:
                columnas_con_pixeles.append(col)
        
        for row in range(altura):
            if cv2.countNonZero(celda[row, :]) > 0:
                filas_con_pixeles.append(row)
        
        if len(columnas_con_pixeles) == 0 or len(filas_con_pixeles) == 0:
            return None
            
        ancho_digito = max(columnas_con_pixeles) - min(columnas_con_pixeles)
        altura_digito = max(filas_con_pixeles) - min(filas_con_pixeles)
        ratio_ancho = ancho_digito / ancho
        ratio_altura = altura_digito / altura
        
        # Dividir en tercios verticales
        tercio_sup = celda[0:altura//3, :]
        tercio_med = celda[altura//3:2*altura//3, :]
        tercio_inf = celda[2*altura//3:altura, :]
        
        pix_sup = cv2.countNonZero(tercio_sup)
        pix_med = cv2.countNonZero(tercio_med)
        pix_inf = cv2.countNonZero(tercio_inf)
        pix_tot = cv2.countNonZero(celda)
        
        # Heurísticas específicas
        if numero_detectado in [1, 7]:
            # 1: muy delgado, vertical, distribución uniforme, alto
            # 7: más ancho, peso arriba, línea horizontal
            
            # El 1 es MUY delgado y ocupa mucha altura
            if ratio_ancho < 0.4 and ratio_altura > 0.65 and abs(pix_med - pix_sup) < pix_tot * 0.25:
                return 1
            # El 7 es más ancho y tiene más peso arriba
            elif ratio_ancho > 0.45 and (pix_sup > pix_med * 1.2 or pix_sup > pix_inf * 1.3):
                return 7
        
        elif numero_detectado in [4, 9]:
            # 4: tiene hueco abajo-izquierda, línea vertical derecha
            # 9: círculo arriba, línea abajo
            
            # Analizar mitad izquierda vs derecha
            mitad_izq = celda[:, :ancho//2]
            mitad_der = celda[:, ancho//2:]
            
            pix_izq = cv2.countNonZero(mitad_izq)
            pix_der = cv2.countNonZero(mitad_der)
            
            # El 9 tiene más peso arriba y es más circular
            # El 4 tiene peso más uniforme y línea vertical prominente
            if pix_sup > pix_inf * 1.5 and pix_izq > pix_der * 0.7:
                return 9
            elif pix_der > pix_izq * 1.2 and abs(pix_sup - pix_inf) < pix_tot * 0.3:
                return 4
        
        return None  # No se pudo determinar, usar detección original