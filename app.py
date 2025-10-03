# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from Imagen import SudokuImageProcessor
from Solver import resolver_completo

st.set_page_config(page_title="Sudoku Solver", page_icon="🔢", layout="centered")

st.title("🔢 Sudoku Solver")
st.write("Subí una foto de un sudoku y lo resuelvo al instante")

# Inicializar procesador (se cachea para no recrearlo cada vez)
@st.cache_resource
def get_processor():
    return SudokuImageProcessor()

processor = get_processor()

# Upload de imagen
archivo = st.file_uploader("Elegí una imagen del sudoku", type=['png', 'jpg', 'jpeg'])

if archivo:
    # Leer y mostrar imagen original
    imagen_pil = Image.open(archivo)
    imagen = np.array(imagen_pil)
    
    # Si es RGB, convertir a BGR para OpenCV
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sudoku Original")
        st.image(imagen_pil, use_container_width=True)
    
    # Botón para resolver
    if st.button("🚀 Resolver Sudoku", type="primary"):
        with st.spinner("Procesando imagen..."):
            try:
                # Extraer sudoku de la imagen
                sudoku_extraido = processor.extraer_sudoku(imagen)
                
                st.success("✅ Sudoku detectado correctamente")
                
                # Mostrar sudoku extraído
                with st.expander("Ver sudoku extraído"):
                    for fila in sudoku_extraido:
                        st.text(" ".join(str(n) if n != 0 else "·" for n in fila))
                
                # Resolver
                with st.spinner("Resolviendo..."):
                    sudoku_resuelto = resolver_completo(sudoku_extraido)
                
                # Mostrar resultado
                with col2:
                    st.subheader("Sudoku Resuelto")
                    
                    # Crear visualización bonita del resultado
                    resultado_html = "<div style='font-family: monospace; font-size: 20px;'>"
                    for i, fila in enumerate(sudoku_resuelto):
                        if i % 3 == 0 and i != 0:
                            resultado_html += "<hr style='margin: 5px 0;'>"
                        fila_str = ""
                        for j, num in enumerate(fila):
                            if j % 3 == 0 and j != 0:
                                fila_str += " | "
                            fila_str += f" {num} "
                        resultado_html += f"<p style='margin: 2px;'>{fila_str}</p>"
                    resultado_html += "</div>"
                    
                    st.markdown(resultado_html, unsafe_allow_html=True)
                    st.success("🎉 ¡Sudoku resuelto!")
                
            except ValueError as e:
                st.error(f"❌ Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Ocurrió un error inesperado: {str(e)}")
                st.write("Asegurate de que la imagen tenga un sudoku claro y bien iluminado")

else:
    st.info("👆 Subí una imagen para empezar")
    
    with st.expander("💡 Consejos para mejores resultados"):
        st.write("""
        - Sacá la foto desde arriba (vista cenital)
        - Asegurate de que haya buena iluminación
        - El sudoku debe estar completo en la imagen
        - Evitá sombras y reflejos
        """)