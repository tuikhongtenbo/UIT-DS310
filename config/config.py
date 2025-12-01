import yaml
import os
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class AppConfig:
    """
    Configuration loader class to handle YAML config file
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = PROJECT_ROOT / config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
        with open (self.config_path, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                return config
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")
    @property
    def config(self) -> Dict[str, Any]:
        "Get full configuration dict"
        return self._config
    def get(self, key: str, default: Any = None) -> Any:
        "Helper to get a value from config with a default"
        return self._config.get(key, default)

# Global instance
try:
    cfg = AppConfig()
except Exception as e:
    print(f"Warning: Could not load default config. Error: {e}")
    cfg = None

if __name__ == "__main__":
    if cfg:
        print("Config loaded successfully")

        # Access nested values as per my yaml structure
        data_path = cfg.config['pipeline']['data']['legal_corpus_path']
        print(f"Data path: {data_path}")
    else:
        print("Failed to load config")
