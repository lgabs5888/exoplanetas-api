"""Rutas y parámetros del servicio. Se pueden cambiar con variables de entorno."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ALMACENAMIENTO = Path(os.environ.get("EXO_ALMACENAMIENTO", RAIZ / "almacenamiento"))
DB_PATH = Path(os.environ.get("EXO_DB", ALMACENAMIENTO / "exoplanetas.db"))
DIR_ARCHIVOS = ALMACENAMIENTO / "archivos"
DIR_MODELOS = ALMACENAMIENTO / "modelos"
SCHEMA = RAIZ / "db" / "schema.sql"

# Modelo inicial entrenado en el repositorio exoplanetas-ia (notebook 03)
MODELO_INICIAL = RAIZ / "modelos" / "modelo_v1.ubj"   # formato nativo de XGBoost
METADATOS_INICIALES = RAIZ / "modelos" / "modelo_v1.json"

CORS_ORIGENES = os.environ.get("EXO_CORS", "*").split(",")
MAX_FILAS_PREDICCION = 500
