"""
Configuration loader utilities for the Credit Risk Detection MLOps project.

Provides functions to load YAML configuration files, resolve relative paths
from the repo root, and merge all configs into a single dictionary.

Exports:
    get_repo_root() -> Path
    load_config(config_path: str) -> dict
    load_all_configs() -> dict
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def get_repo_root() -> Path:
    """Return the repository root directory (parent of src/).

    This file lives at src/utils/config_loader.py, so the repo root
    is two levels up.

    Returns:
        Absolute Path to the repository root.
    """
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and resolve relative paths.

    Relative paths found at any key ending with ``_path``, ``_dir``,
    ``_file``, or ``output`` are resolved against the repository root.

    Args:
        config_path: Absolute or relative path to a YAML config file.
            Relative paths are resolved from the repo root.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    repo_root = get_repo_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo_root / config_file

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh) or {}

    logger.debug("Loaded config from %s", config_file)
    config = _resolve_paths(config, repo_root)
    return config


def _resolve_paths(obj: Any, repo_root: Path) -> Any:
    """Recursively resolve relative path strings within a config object.

    Only string values whose key name contains 'path', 'dir', 'file',
    or 'output' are treated as paths.

    Args:
        obj: The config object (dict, list, or scalar).
        repo_root: Repository root used as the base for relative paths.

    Returns:
        The same structure with path strings resolved.
    """
    if isinstance(obj, dict):
        return {k: _resolve_value(k, v, repo_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_paths(item, repo_root) for item in obj]
    return obj


def _resolve_value(key: str, value: Any, repo_root: Path) -> Any:
    """Resolve a single config value if the key looks like a path field.

    Args:
        key: Config key name.
        value: Config value.
        repo_root: Repository root for resolution.

    Returns:
        Resolved value.
    """
    path_keywords = ("path", "dir", "file", "output", "index", "metadata")
    key_lower = key.lower()
    if isinstance(value, str) and any(kw in key_lower for kw in path_keywords):
        candidate = Path(value)
        if not candidate.is_absolute():
            return str(repo_root / value)
        return value
    return _resolve_paths(value, repo_root)


def load_all_configs() -> dict:
    """Load and merge data, rag, lora, and eval configs.

    Configs are loaded from the ``configs/`` directory relative to the
    repo root.  Each config is stored under a top-level key:
    ``data``, ``rag``, ``lora``, ``eval``.

    Missing config files are skipped with a warning.

    Returns:
        Merged configuration dictionary with keys: data, rag, lora, eval.
    """
    repo_root = get_repo_root()
    config_files = {
        "data": "configs/data_config.yaml",
        "rag": "configs/rag_config.yaml",
        "lora": "configs/lora_config.yaml",
        "eval": "configs/eval_config.yaml",
    }

    merged: dict = {}
    for key, relative_path in config_files.items():
        config_path = repo_root / relative_path
        try:
            merged[key] = load_config(str(config_path))
            logger.info("Loaded %s config from %s", key, config_path)
        except FileNotFoundError:
            logger.warning("Config file not found, skipping: %s", config_path)
            merged[key] = {}
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s config: %s", key, exc)
            merged[key] = {}

    return merged
