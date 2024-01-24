####################### TIPOS DE DATOS ####################### 
# Texto 
text = "Buenos dias"
print(text)

# Lista
mList = [1,2,3]
print(mList)

# Diccionario 
dic = {
    "clave 1": 1,
    "clave 2": 100
}
print(dic)

# Vector Entero y Flotante
vec_ent = [10]*5
vec_flo = [3.14]*5

print(vec_ent)
print(vec_flo)

# Diccionario 
diccionario = {"entero": vec_ent, "flotante": vec_flo}
diccionario

# Cadeina doble
cadena_doble = ["Hola", "Buenas"]
print(cadena_doble)

####################### IMPORTAR DATOS #######################

# Librerias 
import pandas as pd

# Importar datos 
imp_SRI = pd.read_excel("data/tables/ventas_SRI.xlsx") 
print(imp_SRI)
