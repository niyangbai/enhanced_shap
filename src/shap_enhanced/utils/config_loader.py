"""Load configuration from YAML, JSON, or TOML file."""

import yaml
import json

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # pip install toml

def load_config(path: str):
    """Load configuration from YAML, JSON, or TOML file.

    :param str path: Path to the configuration file.
    :return dict: Parsed configuration dictionary.
    """
    if path.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    elif path.endswith(".toml"):
        with open(path, "rb") as f:
            return tomllib.load(f)
    else:
        raise ValueError("Unsupported config file format. Use .yaml, .json, or .toml")
