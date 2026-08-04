import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name):
    path = CONFIG_DIR / f"{name.removesuffix('.yaml')}.yaml"
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
