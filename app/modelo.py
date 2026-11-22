"""Carga, predicción y reentrenamiento del modelo de detección de exoplanetas."""
import io
import json
import threading
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from . import config, db
from .preprocesamiento import N_MEDICIONES, espectro_fourier, limpiar_curvas

COLUMNAS_FLUX = [f"FLUX.{i}" for i in range(1, N_MEDICIONES + 1)]
_cache = {"id": None, "artefacto": None}
_lock_entrenamiento = threading.Lock()


class ErrorDatos(ValueError):
    """Datos de entrada con formato inválido."""


# ---------------------------------------------------------------- lectura de datos
def leer_csv(contenido: bytes, requiere_etiqueta=False):
    """Lee un CSV con el formato de Kepler: columnas FLUX.1 ... FLUX.3197 y, opcionalmente, LABEL.

    Devuelve (X, y o None). LABEL se recodifica: 2 -> 1 (con planeta), 1 -> 0.
    """
    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception as e:  # noqa: BLE001
        raise ErrorDatos(f"No se pudo leer el CSV: {e}")
    if df.empty:
        raise ErrorDatos("El CSV no tiene filas.")
    faltan = [c for c in COLUMNAS_FLUX if c not in df.columns]
    if faltan:
        raise ErrorDatos(f"Faltan {len(faltan)} columnas de flujo (por ejemplo {faltan[0]}). "
                         f"Se esperan FLUX.1 a FLUX.{N_MEDICIONES}.")
    X = df[COLUMNAS_FLUX].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ErrorDatos("Hay valores vacíos o no numéricos en las columnas de flujo.")
    y = None
    if "LABEL" in df.columns:
        if not set(df["LABEL"].unique()) <= {1, 2}:
            raise ErrorDatos("La columna LABEL solo puede tener los valores 1 (sin planeta) o 2 (con planeta).")
        y = (df["LABEL"].to_numpy() == 2).astype(int)
    elif requiere_etiqueta:
        raise ErrorDatos("El archivo de entrenamiento necesita la columna LABEL.")
    return X, y


def caracteristicas(X, sigma=2, ventana=101):
    return espectro_fourier(limpiar_curvas(X, ventana=ventana, sigma=sigma))


# ---------------------------------------------------------------- modelo activo
def modelo_activo():
    with db.conectar() as con:
        fila = con.execute("SELECT * FROM modelos WHERE activo = 1").fetchone()
    return db.fila_a_dict(fila, ("hiperparametros", "metricas"))


def _cargar_xgb(ruta):
    m = XGBClassifier()
    m.load_model(ruta)
    return m


def _artefacto(modelo):
    """Carga (y guarda en memoria) el modelo de una versión. El modelo se guarda en el formato
    nativo de XGBoost (.ubj), que no depende de la versión de Python ni de scikit-learn."""
    if _cache["id"] != modelo["id"]:
        _cache["artefacto"] = {"modelo": _cargar_xgb(modelo["ruta"]), "umbral": modelo["umbral"],
                               "preprocesamiento": {"sigma": modelo["hiperparametros"].get("sigma", 2), "ventana": 101}}
        _cache["id"] = modelo["id"]
    return _cache["artefacto"]


def predecir(X):
    """Devuelve (modelo_id, probabilidades, predicciones) con el modelo activo."""
    modelo = modelo_activo()
    if modelo is None:
        raise RuntimeError("No hay un modelo activo.")
    art = _artefacto(modelo)
    prob = art["modelo"].predict_proba(caracteristicas(X, **art["preprocesamiento"]))[:, 1]
    return modelo["id"], prob, (prob >= art["umbral"]).astype(int)


def registrar_modelo_inicial():
    """Si la base está vacía, registra el modelo entrenado en el notebook como versión activa."""
    with db.conectar() as con:
        if con.execute("SELECT COUNT(*) FROM modelos").fetchone()[0]:
            return
        meta = json.loads(config.METADATOS_INICIALES.read_text(encoding="utf-8"))
        con.execute(
            "INSERT INTO modelos (version, algoritmo, ruta, hiperparametros, umbral, metricas, estado, activo, finalizado_en)"
            " VALUES (?, ?, ?, ?, ?, ?, 'listo', 1, datetime('now'))",
            (meta["version"], meta["algoritmo"], str(config.MODELO_INICIAL), json.dumps(meta["hiperparametros"]),
             meta["umbral"], json.dumps(meta["metricas"])))


# ---------------------------------------------------------------- entrenamiento
def metricas_en_umbral(y, prob, umbral):
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"umbral": float(umbral), "recall": float(recall_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "matriz_confusion": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)}}


def entrenar(X, y, hiperparametros, sigma=2, ventana=101, pliegues=5, semilla=42):
    """Valida con CV estratificada, elige el umbral que maximiza F1 y entrena con todos los datos."""
    if y.sum() < pliegues:
        raise ErrorDatos(f"Se necesitan al menos {pliegues} estrellas con planeta para reentrenar.")
    F = caracteristicas(X, sigma=sigma, ventana=ventana)
    peso = float((y == 0).sum() / y.sum())

    def nuevo():
        return XGBClassifier(**hiperparametros, scale_pos_weight=peso, eval_metric="aucpr",
                             n_jobs=-1, random_state=semilla)

    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(pliegues, shuffle=True, random_state=semilla).split(F, y):
        oof[va] = nuevo().fit(F[tr], y[tr]).predict_proba(F[va])[:, 1]
    prec, rec, umb = precision_recall_curve(y, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    umbral = float(umb[int(np.argmax(f1s))])
    metricas = {"validacion_cruzada": {"ap": float(average_precision_score(y, oof)),
                                       **metricas_en_umbral(y, oof, umbral),
                                       "estrellas": int(len(y)), "con_planeta": int(y.sum())}}
    final = nuevo().fit(F, y)
    artefacto = {"modelo": final, "umbral": umbral, "preprocesamiento": {"sigma": sigma, "ventana": ventana}}
    return artefacto, metricas


def iniciar_reentrenamiento(archivo_id: int, activar: bool):
    """Crea la versión en estado 'entrenando' y lanza el entrenamiento en segundo plano."""
    if not _lock_entrenamiento.acquire(blocking=False):
        raise RuntimeError("Ya hay un reentrenamiento en curso. Espera a que termine.")
    try:
        with db.conectar() as con:
            archivo = con.execute("SELECT * FROM archivos_entrenamiento WHERE id = ?", (archivo_id,)).fetchone()
            if archivo is None:
                raise LookupError(f"No existe el archivo de entrenamiento {archivo_id}.")
            base = con.execute("SELECT hiperparametros FROM modelos WHERE activo = 1").fetchone()
            hp = json.loads(base["hiperparametros"])
            version = "v" + datetime.now().strftime("%Y%m%d-%H%M%S")
            cur = con.execute(
                "INSERT INTO modelos (version, algoritmo, hiperparametros, estado, archivo_entrenamiento_id)"
                " VALUES (?, 'XGBoost', ?, 'entrenando', ?)", (version, json.dumps(hp), archivo_id))
            modelo_id = cur.lastrowid
    except Exception:
        _lock_entrenamiento.release()
        raise
    threading.Thread(target=_entrenar_en_segundo_plano,
                     args=(modelo_id, archivo["ruta"], hp, activar), daemon=True).start()
    return modelo_id


def _entrenar_en_segundo_plano(modelo_id, ruta, hp, activar):
    try:
        with open(ruta, "rb") as f:
            X, y = leer_csv(f.read(), requiere_etiqueta=True)
        hp_xgb = {k: v for k, v in hp.items() if k != "sigma"}
        artefacto, metricas = entrenar(X, y, hp_xgb, sigma=hp.get("sigma", 2))
        destino = config.DIR_MODELOS / f"modelo_{modelo_id}.ubj"
        artefacto["modelo"].save_model(destino)
        with db.conectar() as con:
            con.execute("UPDATE modelos SET estado = 'listo', ruta = ?, umbral = ?, metricas = ?,"
                        " finalizado_en = datetime('now') WHERE id = ?",
                        (str(destino), artefacto["umbral"], json.dumps(metricas), modelo_id))
            if activar:
                con.execute("UPDATE modelos SET activo = 0 WHERE activo = 1")
                con.execute("UPDATE modelos SET activo = 1 WHERE id = ?", (modelo_id,))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        with db.conectar() as con:
            con.execute("UPDATE modelos SET estado = 'error', mensaje_error = ?, finalizado_en = datetime('now')"
                        " WHERE id = ?", (str(e), modelo_id))
    finally:
        _lock_entrenamiento.release()
