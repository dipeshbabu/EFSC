from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {path.suffix}")
