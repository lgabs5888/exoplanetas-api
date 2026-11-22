"""Genera docs/openapi.json y docs/openapi.yaml (documento Swagger) a partir del código de la API.

Uso:  python scripts/exportar_openapi.py
"""
import json
import pathlib
import sys

import yaml

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from app.main import app  # noqa: E402

esquema = app.openapi()
(RAIZ / "docs" / "openapi.json").write_text(json.dumps(esquema, indent=2, ensure_ascii=False), encoding="utf-8")
(RAIZ / "docs" / "openapi.yaml").write_text(yaml.safe_dump(esquema, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("Generados docs/openapi.json y docs/openapi.yaml")
