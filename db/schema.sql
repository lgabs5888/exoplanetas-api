-- Esquema de la base de datos del servicio de IA (SQLite).
-- Se ejecuta automáticamente al iniciar la API. También se puede crear a mano:
--   sqlite3 almacenamiento/exoplanetas.db < db/schema.sql

PRAGMA foreign_keys = ON;

-- Archivos de entrenamiento enviados al servicio (endpoint POST /entrenamiento/archivos)
CREATE TABLE IF NOT EXISTS archivos_entrenamiento (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre_original  TEXT    NOT NULL,
    ruta             TEXT    NOT NULL,
    filas            INTEGER NOT NULL,
    positivos        INTEGER NOT NULL,          -- estrellas con planeta (LABEL = 2)
    tamano_bytes     INTEGER NOT NULL,
    subido_en        TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Versiones del modelo: la inicial (entrenada en el notebook) y cada reentrenamiento
CREATE TABLE IF NOT EXISTS modelos (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    version                   TEXT    NOT NULL UNIQUE,
    algoritmo                 TEXT    NOT NULL,
    ruta                      TEXT,                       -- archivo del modelo (.ubj, formato XGBoost)
    hiperparametros           TEXT    NOT NULL,           -- JSON
    umbral                    REAL,                       -- umbral de decisión
    metricas                  TEXT,                       -- JSON (validación cruzada y prueba)
    estado                    TEXT    NOT NULL CHECK (estado IN ('entrenando', 'listo', 'error')),
    activo                    INTEGER NOT NULL DEFAULT 0 CHECK (activo IN (0, 1)),
    archivo_entrenamiento_id  INTEGER REFERENCES archivos_entrenamiento(id),
    mensaje_error             TEXT,
    creado_en                 TEXT    NOT NULL DEFAULT (datetime('now')),
    finalizado_en             TEXT
);

-- Solo una versión activa a la vez
CREATE UNIQUE INDEX IF NOT EXISTS ux_modelos_un_activo ON modelos(activo) WHERE activo = 1;

-- Historial de predicciones: datos enviados y resultado (endpoint GET /predicciones)
CREATE TABLE IF NOT EXISTS predicciones (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    modelo_id       INTEGER NOT NULL REFERENCES modelos(id),
    identificador   TEXT,                                -- nombre de la estrella o fila del CSV
    origen          TEXT    NOT NULL CHECK (origen IN ('json', 'csv')),
    datos           TEXT    NOT NULL,                    -- JSON con los 3.197 valores de flujo enviados
    probabilidad    REAL    NOT NULL,
    prediccion      INTEGER NOT NULL CHECK (prediccion IN (0, 1)),   -- 1 = con planeta
    etiqueta_real   INTEGER CHECK (etiqueta_real IN (0, 1)),         -- si el CSV traía LABEL
    creado_en       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_predicciones_fecha  ON predicciones(creado_en);
CREATE INDEX IF NOT EXISTS ix_predicciones_modelo ON predicciones(modelo_id);
