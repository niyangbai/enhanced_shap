# src/shap_enhanced/utils/config_loader.py

import yaml
import json

def load_config(path: str):
    """Load configuration from YAML or JSON file.

    :param str path: Path to the configuration file.
    :return dict: Parsed configuration dictionary.
    """
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config file format. Use .yaml or .json")
