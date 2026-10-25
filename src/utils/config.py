from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """YAML konfigürasyon dosyasını güvenle yükler."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
