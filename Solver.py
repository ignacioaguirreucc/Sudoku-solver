sudoku = [
    [8,0,0, 0,0,0, 0,0,0],
    [0,0,3, 6,0,0, 0,0,0],
    [0,7,0, 0,9,0, 2,0,0],

    [0,5,0, 0,0,7, 0,0,0],
    [0,0,0, 0,4,5, 7,0,0],
    [0,0,0, 1,0,0, 0,3,0],

    [0,0,1, 0,0,0, 0,6,8],
    [0,0,8, 5,0,0, 0,1,0],
    [0,9,0, 0,0,0, 4,0,0]
]

def comparar_fila(numero, fila, tablero):
    for i in range(9):
        if tablero[fila][i]==numero:
            return 0
    return numero
            
    
def comparar_columna(numero, columna, tablero):
    for i in range(9):
        if tablero[i][columna]==numero:
            return 0
    return numero


def comparar_caja(numero, fila, columna, tablero):
    fila_caja=(fila//3)*3
    columna_caja=(columna//3)*3
    for i in range(3):
        for j in range(3):
            if tablero[fila_caja+i][columna_caja+j]==numero:
                return 0
    return numero
    


    

def candidato(sudoku):
    for i in range(9):
        for j in range(9):
            celda=sudoku[i][j]
            if celda==0:
                contador_candidatos=0
                for n in range(1,10):
                    candidato_fila=comparar_fila(n, i, sudoku)
                    candidato_columna=comparar_columna(n, j, sudoku)
                    candidato_caja=comparar_caja(n, i, j, sudoku)
                    if candidato_fila==candidato_columna==candidato_caja and candidato_fila!=0:
                        contador_candidatos+=1
                        candidato_unico=candidato_fila
                if contador_candidatos==1:
                    sudoku[i][j]=candidato_unico


def resolver_sudoku(sudoku):
    cambios = True
    while cambios:
        sudoku_anterior = [fila[:] for fila in sudoku]
        candidato(sudoku)
        cambios = (sudoku_anterior != sudoku)
    return sudoku



"""def encontrar_ceros(sudokus):
    mejor_celda=None
    minimo_candidatos=10
    for i in range(9):
        for j in range(9):
            if sudokus[i][j]==0:
                cont_candidatos=0
                for n in range(1,10):
                    candidato_fila=comparar_fila(n,i)
                    candidato_columna=comparar_columna(n,j)
                    candidato_caja=comparar_caja(n,i,j)
                    if candidato_fila==candidato_columna==candidato_caja and candidato_fila!=0:
                        cont_candidatos+=1
                if cont_candidatos==1:
                    return (i,j)
                if cont_candidatos<minimo_candidatos:
                    minimo_candidatos=cont_candidatos
                    mejor_celda=(i,j)
    return mejor_celda
"""
def encontrar_ceros(sudokus):
    for i in range(9):
        for j in range(9):
            if sudokus[i][j]==0:
                return (i,j)
    return None

def Backtracking(sudokus):
    if not encontrar_ceros(sudokus):
        return True
    coordenadas=encontrar_ceros(sudokus)
    fila,columna=coordenadas
    for n in range(1,10):
        candidato_fila=comparar_fila(n, fila, sudokus)
        candidato_columna=comparar_columna(n, columna, sudokus)
        candidato_caja=comparar_caja(n, fila, columna, sudokus)
        if candidato_fila==candidato_columna==candidato_caja and candidato_fila!=0:
           sudokus[fila][columna]=n
           if Backtracking(sudokus):
               return True
           sudokus[fila][columna]=0
    return False



def resolver_completo(sudoku):
    resolver_sudoku(sudoku)
    Backtracking(sudoku)
    return sudoku


# Solo ejecutar si se corre directamente (no cuando se importa)
if __name__ == "__main__":
    sudoku=resolver_completo(sudoku)
    print("\nSudoku resuelto:")
    for fila in sudoku:
        print(fila)

                        
                    
                    
                