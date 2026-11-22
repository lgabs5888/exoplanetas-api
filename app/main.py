"""Servicio de IA: detección de exoplanetas a partir de curvas de luz de Kepler.

Documentación interactiva (Swagger): http://localhost:8000/docs
"""
import json
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config, db, modelo
from .preprocesamiento import N_MEDICIONES


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    db.iniciar()
    modelo.registrar_modelo_inicial()
    yield


DESCRIPCION = """
API REST que predice si una estrella tiene un **exoplaneta** a partir de su curva de luz
(3.197 mediciones de brillo del telescopio Kepler de la NASA).

* **Predicciones:** enviar datos al modelo y obtener la clasificación.
* **Historial:** consultar las predicciones previas, con los datos enviados y el resultado.
* **Modelo:** métricas de rendimiento y versiones.
* **Entrenamiento:** enviar un archivo de entrenamiento y solicitar el reentrenamiento.
"""

TAGS = [
    {"name": "Predicciones", "description": "Clasificación de curvas de luz e historial."},
    {"name": "Modelo", "description": "Métricas de rendimiento y versiones del modelo."},
    {"name": "Entrenamiento", "description": "Archivos de entrenamiento y reentrenamiento."},
    {"name": "Sistema", "description": "Estado del servicio."},
]

app = FastAPI(title="Exoplanetas IA - Servicio de predicción", version="1.0.0",
              description=DESCRIPCION, openapi_tags=TAGS, lifespan=ciclo_de_vida)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGENES, allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- esquemas
class CurvaDeLuz(BaseModel):
    flux: List[float] = Field(..., min_length=N_MEDICIONES, max_length=N_MEDICIONES,
                              description=f"Las {N_MEDICIONES} mediciones de brillo (FLUX.1 a FLUX.{N_MEDICIONES}), en orden.")
    identificador: Optional[str] = Field(None, max_length=100, description="Nombre opcional de la estrella.")


class ResultadoPrediccion(BaseModel):
    id: int
    identificador: Optional[str]
    probabilidad: float = Field(..., description="Probabilidad de que la estrella tenga un exoplaneta (0 a 1).")
    prediccion: int = Field(..., description="1 = con planeta, 0 = sin planeta.")
    resultado: str
    umbral: float
    modelo_version: str
    etiqueta_real: Optional[int] = None


class ResultadoLote(BaseModel):
    total: int
    con_planeta: int
    resultados: List[ResultadoPrediccion]


class PrediccionHistorial(BaseModel):
    id: int
    identificador: Optional[str]
    origen: str
    probabilidad: float
    prediccion: int
    resultado: str
    etiqueta_real: Optional[int]
    modelo_version: str
    creado_en: str


class PrediccionDetalle(PrediccionHistorial):
    datos: List[float] = Field(..., description="Datos enviados: las 3.197 mediciones de brillo.")


class PaginaHistorial(BaseModel):
    total: int
    limite: int
    desplazamiento: int
    items: List[PrediccionHistorial]


class VersionModelo(BaseModel):
    id: int
    version: str
    algoritmo: str
    estado: str
    activo: bool
    umbral: Optional[float]
    hiperparametros: dict
    metricas: Optional[dict]
    archivo_entrenamiento_id: Optional[int]
    mensaje_error: Optional[str]
    creado_en: str
    finalizado_en: Optional[str]


class ArchivoEntrenamiento(BaseModel):
    id: int
    nombre_original: str
    filas: int
    positivos: int
    tamano_bytes: int
    subido_en: str


class SolicitudReentrenamiento(BaseModel):
    archivo_id: int = Field(..., description="ID del archivo de entrenamiento enviado antes.")
    activar: bool = Field(True, description="Si es verdadero, la nueva versión pasa a ser la activa al terminar.")


class RespuestaReentrenamiento(BaseModel):
    modelo_id: int
    estado: str
    mensaje: str


# ---------------------------------------------------------------- utilidades
def _texto(pred: int) -> str:
    return "Con planeta" if pred == 1 else "Sin planeta"


def _version(con, modelo_id):
    return con.execute("SELECT version FROM modelos WHERE id = ?", (modelo_id,)).fetchone()["version"]


def _clasificar_y_guardar(X, ids, origen, etiquetas=None):
    try:
        modelo_id, prob, pred = modelo.predecir(X)
    except RuntimeError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))
    umbral = modelo.modelo_activo()["umbral"]
    salida = []
    with db.conectar() as con:
        version = _version(con, modelo_id)
        for i in range(len(X)):
            real = None if etiquetas is None else int(etiquetas[i])
            cur = con.execute(
                "INSERT INTO predicciones (modelo_id, identificador, origen, datos, probabilidad, prediccion, etiqueta_real)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (modelo_id, ids[i], origen, json.dumps(np.round(X[i], 4).tolist()), float(prob[i]), int(pred[i]), real))
            salida.append(ResultadoPrediccion(id=cur.lastrowid, identificador=ids[i], probabilidad=float(prob[i]),
                                              prediccion=int(pred[i]), resultado=_texto(int(pred[i])),
                                              umbral=umbral, modelo_version=version, etiqueta_real=real))
    return salida


def _historial(fila, detalle=False):
    d = dict(fila)
    d["resultado"] = _texto(d["prediccion"])
    if detalle:
        d["datos"] = json.loads(d["datos"])
        return PrediccionDetalle(**d)
    d.pop("datos", None)
    return PrediccionHistorial(**d)


def _version_dict(fila):
    d = db.fila_a_dict(fila, ("hiperparametros", "metricas"))
    d["activo"] = bool(d["activo"])
    d.pop("ruta", None)
    return VersionModelo(**d)


# ---------------------------------------------------------------- endpoints
@app.get("/salud", tags=["Sistema"], summary="Estado del servicio")
def salud():
    activo = modelo.modelo_activo()
    return {"estado": "ok", "modelo_activo": activo["version"] if activo else None}


@app.post("/predicciones", response_model=ResultadoPrediccion, tags=["Predicciones"],
          summary="Clasificar una curva de luz (JSON)")
def predecir(curva: CurvaDeLuz):
    """Envía las 3.197 mediciones de brillo de una estrella y devuelve si tiene un exoplaneta."""
    X = np.asarray(curva.flux, dtype=np.float64)[None, :]
    if not np.isfinite(X).all():
        raise HTTPException(422, "La curva contiene valores no numéricos.")
    return _clasificar_y_guardar(X, [curva.identificador], "json")[0]


@app.post("/predicciones/archivo", response_model=ResultadoLote, tags=["Predicciones"],
          summary="Clasificar varias curvas desde un CSV")
async def predecir_archivo(archivo: UploadFile = File(..., description="CSV con columnas FLUX.1 a FLUX.3197 (LABEL opcional).")):
    """Clasifica cada fila del CSV. Si el archivo trae la columna LABEL, se guarda como etiqueta real
    para poder comparar después las predicciones con la realidad."""
    try:
        X, y = modelo.leer_csv(await archivo.read())
    except modelo.ErrorDatos as e:
        raise HTTPException(422, str(e))
    if len(X) > config.MAX_FILAS_PREDICCION:
        raise HTTPException(413, f"Máximo {config.MAX_FILAS_PREDICCION} filas por archivo.")
    base = (archivo.filename or "archivo").rsplit(".", 1)[0]
    ids = [f"{base} - fila {i + 1}" for i in range(len(X))]
    res = _clasificar_y_guardar(X, ids, "csv", y)
    return ResultadoLote(total=len(res), con_planeta=sum(r.prediccion for r in res), resultados=res)


@app.get("/predicciones", response_model=PaginaHistorial, tags=["Predicciones"],
         summary="Historial de predicciones")
def historial(limite: int = Query(20, ge=1, le=200), desplazamiento: int = Query(0, ge=0),
              solo_con_planeta: bool = False):
    """Lista las predicciones previas, de la más reciente a la más antigua.
    Para ver los datos enviados de una predicción, usa `GET /predicciones/{id}`."""
    filtro = "WHERE p.prediccion = 1" if solo_con_planeta else ""
    with db.conectar() as con:
        total = con.execute(f"SELECT COUNT(*) FROM predicciones p {filtro}").fetchone()[0]
        filas = con.execute(
            f"SELECT p.*, m.version AS modelo_version FROM predicciones p JOIN modelos m ON m.id = p.modelo_id {filtro}"
            " ORDER BY p.id DESC LIMIT ? OFFSET ?", (limite, desplazamiento)).fetchall()
    return PaginaHistorial(total=total, limite=limite, desplazamiento=desplazamiento,
                           items=[_historial(f) for f in filas])


@app.get("/predicciones/{prediccion_id}", response_model=PrediccionDetalle, tags=["Predicciones"],
         summary="Detalle de una predicción, con los datos enviados")
def detalle_prediccion(prediccion_id: int):
    with db.conectar() as con:
        fila = con.execute("SELECT p.*, m.version AS modelo_version FROM predicciones p"
                           " JOIN modelos m ON m.id = p.modelo_id WHERE p.id = ?", (prediccion_id,)).fetchone()
    if fila is None:
        raise HTTPException(404, "Predicción no encontrada.")
    return _historial(fila, detalle=True)


@app.get("/modelo/metricas", tags=["Modelo"], summary="Métricas de rendimiento del modelo activo")
def metricas():
    """Devuelve las métricas del modelo activo: validación cruzada y, si existen, las del conjunto de prueba.
    También incluye el desempeño observado en las predicciones que llegaron con etiqueta real."""
    activo = modelo.modelo_activo()
    if activo is None:
        raise HTTPException(503, "No hay un modelo activo.")
    with db.conectar() as con:
        filas = con.execute("SELECT prediccion, etiqueta_real FROM predicciones"
                            " WHERE modelo_id = ? AND etiqueta_real IS NOT NULL", (activo["id"],)).fetchall()
    produccion = None
    if filas:
        y = np.array([f["etiqueta_real"] for f in filas]); p = np.array([f["prediccion"] for f in filas])
        produccion = {"predicciones_con_etiqueta": len(filas),
                      "vp": int(((p == 1) & (y == 1)).sum()), "fp": int(((p == 1) & (y == 0)).sum()),
                      "fn": int(((p == 0) & (y == 1)).sum()), "vn": int(((p == 0) & (y == 0)).sum())}
    return {"version": activo["version"], "algoritmo": activo["algoritmo"], "umbral": activo["umbral"],
            "hiperparametros": activo["hiperparametros"], "metricas": activo["metricas"],
            "monitoreo_produccion": produccion}


@app.get("/modelo/versiones", response_model=List[VersionModelo], tags=["Modelo"],
         summary="Listar versiones del modelo")
def versiones():
    with db.conectar() as con:
        filas = con.execute("SELECT * FROM modelos ORDER BY id DESC").fetchall()
    return [_version_dict(f) for f in filas]


@app.get("/modelo/versiones/{modelo_id}", response_model=VersionModelo, tags=["Modelo"],
         summary="Estado de una versión (útil para seguir un reentrenamiento)")
def version(modelo_id: int):
    with db.conectar() as con:
        fila = con.execute("SELECT * FROM modelos WHERE id = ?", (modelo_id,)).fetchone()
    if fila is None:
        raise HTTPException(404, "Versión no encontrada.")
    return _version_dict(fila)


@app.put("/modelo/versiones/{modelo_id}/activar", response_model=VersionModelo, tags=["Modelo"],
         summary="Activar una versión anterior (rollback)")
def activar(modelo_id: int):
    with db.conectar() as con:
        fila = con.execute("SELECT * FROM modelos WHERE id = ?", (modelo_id,)).fetchone()
        if fila is None:
            raise HTTPException(404, "Versión no encontrada.")
        if fila["estado"] != "listo":
            raise HTTPException(409, "Solo se puede activar una versión en estado 'listo'.")
        con.execute("UPDATE modelos SET activo = 0 WHERE activo = 1")
        con.execute("UPDATE modelos SET activo = 1 WHERE id = ?", (modelo_id,))
        fila = con.execute("SELECT * FROM modelos WHERE id = ?", (modelo_id,)).fetchone()
    return _version_dict(fila)


@app.post("/entrenamiento/archivos", response_model=ArchivoEntrenamiento, status_code=201, tags=["Entrenamiento"],
          summary="Enviar un archivo de entrenamiento")
async def subir_archivo(archivo: UploadFile = File(..., description="CSV con LABEL y FLUX.1 a FLUX.3197 (formato de exoTrain.csv).")):
    """Guarda el archivo y lo valida. Después se puede usar en `POST /modelo/reentrenar`."""
    contenido = await archivo.read()
    try:
        X, y = modelo.leer_csv(contenido, requiere_etiqueta=True)
    except modelo.ErrorDatos as e:
        raise HTTPException(422, str(e))
    if y.sum() < 5:
        raise HTTPException(422, "El archivo necesita al menos 5 estrellas con planeta (LABEL = 2) para entrenar.")
    destino = config.DIR_ARCHIVOS / f"{uuid.uuid4().hex}.csv"
    destino.write_bytes(contenido)
    with db.conectar() as con:
        cur = con.execute("INSERT INTO archivos_entrenamiento (nombre_original, ruta, filas, positivos, tamano_bytes)"
                          " VALUES (?, ?, ?, ?, ?)",
                          (archivo.filename or "entrenamiento.csv", str(destino), len(X), int(y.sum()), len(contenido)))
        fila = con.execute("SELECT * FROM archivos_entrenamiento WHERE id = ?", (cur.lastrowid,)).fetchone()
    return ArchivoEntrenamiento(**dict(fila))


@app.get("/entrenamiento/archivos", response_model=List[ArchivoEntrenamiento], tags=["Entrenamiento"],
         summary="Listar archivos de entrenamiento")
def listar_archivos():
    with db.conectar() as con:
        filas = con.execute("SELECT * FROM archivos_entrenamiento ORDER BY id DESC").fetchall()
    return [ArchivoEntrenamiento(**dict(f)) for f in filas]


@app.post("/modelo/reentrenar", response_model=RespuestaReentrenamiento, status_code=202, tags=["Entrenamiento"],
          summary="Solicitar el reentrenamiento del modelo")
def reentrenar(solicitud: SolicitudReentrenamiento):
    """Entrena una nueva versión con el archivo indicado, en segundo plano.
    Usa validación cruzada estratificada de 5 pliegues para calcular las métricas y el umbral.
    Consulta el avance con `GET /modelo/versiones/{modelo_id}`."""
    try:
        modelo_id = modelo.iniciar_reentrenamiento(solicitud.archivo_id, solicitud.activar)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return RespuestaReentrenamiento(modelo_id=modelo_id, estado="entrenando",
                                    mensaje="Reentrenamiento iniciado. Consulta GET /modelo/versiones/{id} para ver el avance.")
