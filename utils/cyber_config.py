"""Configuration helpers for the shared FedCCFA cybersecurity benchmark.

The runner deliberately loads the *original* algorithm YAML from ``configs/``
first, then overlays the common cyber config and finally the algorithm-specific
cyber config. This preserves required upstream hyperparameters without copying
or silently changing the authors' defaults.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
import yaml


def deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: str | Path, required: bool = True) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open('r') as f:
        return yaml.safe_load(f) or {}


def load_cyber_config(
    repo_root: str | Path,
    algorithm: str,
    config: str | Path | None = None,
    experiment_config: str | Path | None = None,
):
    repo_root = Path(repo_root)
    algorithm = str(algorithm)
    original = load_yaml(repo_root / 'configs' / f'{algorithm}.yaml', required=True)
    common = load_yaml(repo_root / 'configs' / 'Cyber_Common.yaml', required=True)
    specific_path = Path(config) if config else repo_root / 'configs' / f'{algorithm}_Cyber.yaml'
    if not specific_path.is_absolute():
        specific_path = repo_root / specific_path
    specific = load_yaml(specific_path, required=True)
    args = deep_update(original, common)
    args = deep_update(args, specific)
    if experiment_config:
        experiment_path = Path(experiment_config)
        if not experiment_path.is_absolute():
            experiment_path = repo_root / experiment_path
        args = deep_update(args, load_yaml(experiment_path, required=True))
        args['_experiment_config'] = str(experiment_path)
    args['algorithm'] = algorithm
    args['_original_config'] = str(repo_root / 'configs' / f'{algorithm}.yaml')
    args['_cyber_config'] = str(specific_path)
    return args
