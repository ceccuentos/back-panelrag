import os
import pickle
import time

# Variable global que contendrá el diccionario
ALL_NODES_DICTIONARY = None
LAST_ACCESS_TIME = None
CACHE_EXPIRY = 60  # Segundos de inactividad antes de liberar el caché

def load_data_dict():
    global ALL_NODES_DICTIONARY
    if ALL_NODES_DICTIONARY is None:
        STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
        # Leer el diccionario desde el archivo
        with open(f"{STORAGE_DIR}/dicnodes.pkl", 'rb') as archivo:
            ALL_NODES_DICTIONARY = pickle.load(archivo)

# Llama a load_data_dict() al importar el módulo para cargar los datos inmediatamente
load_data_dict()

