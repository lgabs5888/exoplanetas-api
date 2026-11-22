"""Acceso a la base de datos SQLite."""
import json
import sqlite3
from contextlib import contextmanager

from . import config


def iniciar():
    """Crea las carpetas y las tablas (ejecuta db/schema.sql)."""
    for d in (config.ALMACENAMIENTO, config.DIR_ARCHIVOS, config.DIR_MODELOS):
        d.mkdir(parents=True, exist_ok=True)
    with conectar() as con:
        con.executescript(config.SCHEMA.read_text(encoding="utf-8"))


@contextmanager
def conectar():
    con = sqlite3.connect(config.DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def fila_a_dict(fila, json_cols=()):
    if fila is None:
        return None
    d = dict(fila)
    for c in json_cols:
        if d.get(c) is not None:
            d[c] = json.loads(d[c])
    return d
