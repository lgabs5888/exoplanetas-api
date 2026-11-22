"""Pruebas de la API. Usan una base de datos temporal y curvas sintéticas."""
import io
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

os.environ["EXO_ALMACENAMIENTO"] = tempfile.mkdtemp()
os.environ["EXO_DB"] = os.path.join(os.environ["EXO_ALMACENAMIENTO"], "prueba.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

N = 3197
rng = np.random.default_rng(0)


def curva(con_planeta):
    t = np.arange(N)
    x = 1000 + 20 * np.sin(t / 400) + rng.normal(0, 3, N)
    if con_planeta:
        x[(t % rng.integers(150, 400)) < 6] -= 60
    return x


def csv_bytes(n_pos, n_neg, con_label=True):
    filas = [curva(True) for _ in range(n_pos)] + [curva(False) for _ in range(n_neg)]
    df = pd.DataFrame(filas, columns=[f"FLUX.{i}" for i in range(1, N + 1)])
    if con_label:
        df.insert(0, "LABEL", [2] * n_pos + [1] * n_neg)
    buf = io.StringIO(); df.to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:
        yield c


def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200 and r.json()["modelo_activo"] == "v1"


def test_prediccion_json_y_historial(cliente):
    r = cliente.post("/predicciones", json={"flux": curva(True).tolist(), "identificador": "estrella-prueba"})
    assert r.status_code == 200
    cuerpo = r.json()
    assert 0 <= cuerpo["probabilidad"] <= 1 and cuerpo["prediccion"] in (0, 1)
    detalle = cliente.get(f"/predicciones/{cuerpo['id']}").json()
    assert len(detalle["datos"]) == N and detalle["identificador"] == "estrella-prueba"
    hist = cliente.get("/predicciones").json()
    assert hist["total"] >= 1 and hist["items"][0]["id"] == cuerpo["id"]


def test_prediccion_longitud_incorrecta(cliente):
    assert cliente.post("/predicciones", json={"flux": [1.0] * 10}).status_code == 422


def test_prediccion_csv_guarda_etiqueta_real(cliente):
    r = cliente.post("/predicciones/archivo", files={"archivo": ("lote.csv", csv_bytes(2, 3), "text/csv")})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["total"] == 5 and cuerpo["resultados"][0]["etiqueta_real"] == 1
    m = cliente.get("/modelo/metricas").json()
    assert m["monitoreo_produccion"]["predicciones_con_etiqueta"] == 5


def test_csv_invalido(cliente):
    r = cliente.post("/predicciones/archivo", files={"archivo": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422


def test_metricas(cliente):
    m = cliente.get("/modelo/metricas").json()
    assert m["version"] == "v1" and "validacion_cruzada" in m["metricas"] and "prueba" in m["metricas"]


def test_archivo_sin_suficientes_positivos(cliente):
    r = cliente.post("/entrenamiento/archivos", files={"archivo": ("e.csv", csv_bytes(2, 20), "text/csv")})
    assert r.status_code == 422


def test_reentrenamiento_completo(cliente):
    r = cliente.post("/entrenamiento/archivos", files={"archivo": ("nuevo.csv", csv_bytes(12, 60), "text/csv")})
    assert r.status_code == 201
    archivo_id = r.json()["id"]
    assert any(a["id"] == archivo_id for a in cliente.get("/entrenamiento/archivos").json())

    r = cliente.post("/modelo/reentrenar", json={"archivo_id": archivo_id, "activar": True})
    assert r.status_code == 202
    modelo_id = r.json()["modelo_id"]
    for _ in range(240):
        v = cliente.get(f"/modelo/versiones/{modelo_id}").json()
        if v["estado"] != "entrenando":
            break
        time.sleep(0.5)
    assert v["estado"] == "listo", v.get("mensaje_error")
    assert v["activo"] and v["metricas"]["validacion_cruzada"]["con_planeta"] == 12
    assert cliente.get("/salud").json()["modelo_activo"] == v["version"]

    # la predicción usa la nueva versión
    assert cliente.post("/predicciones", json={"flux": curva(False).tolist()}).json()["modelo_version"] == v["version"]

    # rollback a la versión inicial
    v1 = [x for x in cliente.get("/modelo/versiones").json() if x["version"] == "v1"][0]
    assert cliente.put(f"/modelo/versiones/{v1['id']}/activar").json()["activo"]
    assert cliente.get("/salud").json()["modelo_activo"] == "v1"


def test_reentrenar_archivo_inexistente(cliente):
    assert cliente.post("/modelo/reentrenar", json={"archivo_id": 9999}).status_code == 404
