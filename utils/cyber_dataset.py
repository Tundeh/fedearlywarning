"""Dataset primitives for the FedCCFA cybersecurity drift experiments.

This module deliberately exposes ``.data`` and ``.targets`` because the
original FedCCFA implementation reads those attributes directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class CyberTensorDataset(Dataset):
    """Small in-memory tabular dataset compatible with FedCCFA.

    Parameters
    ----------
    data:
        Shape ``[n_samples, n_features]``.
    targets:
        Integer class labels with shape ``[n_samples]``.
    metadata:
        Optional dictionary retained for logging/debugging only.
    """

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> None:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        data = np.asarray(data, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.int64)

        if data.ndim != 2:
            raise ValueError(f"Expected 2-D tabular data, got shape={data.shape}")
        if targets.ndim != 1:
            targets = targets.reshape(-1)
        if len(data) != len(targets):
            raise ValueError("data and targets must contain the same number of samples")

        self.data = data
        self.targets = targets
        self.transform = None  # compatibility with original ClientDataset
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        x = torch.as_tensor(self.data[index], dtype=torch.float32)
        y = torch.as_tensor(self.targets[index], dtype=torch.long)
        return x, y

    @property
    def is_empty(self) -> bool:
        return len(self.targets) == 0

    def subset(self, indices: Sequence[int]) -> "CyberTensorDataset":
        indices = np.asarray(indices, dtype=np.int64)
        return CyberTensorDataset(
            self.data[indices],
            self.targets[indices],
            metadata=dict(self.metadata),
        )

    @classmethod
    def concatenate(cls, datasets: Iterable["CyberTensorDataset"]) -> "CyberTensorDataset":
        datasets = [d for d in datasets if d is not None and len(d) > 0]
        if not datasets:
            raise ValueError("Cannot concatenate an empty dataset list")
        x = np.concatenate([d.data for d in datasets], axis=0)
        y = np.concatenate([d.targets for d in datasets], axis=0)
        return cls(x, y)


@dataclass
class PreparedCyberData:
    """Container produced by ``prepare_cicids2018.py``."""

    x: np.ndarray
    y: np.ndarray
    client_ids: np.ndarray
    times: np.ndarray
    variant_ids: np.ndarray
    class_names: List[str]
    variant_names: List[str]
    feature_names: List[str]
    client_mapping: Dict[str, int]
    metadata: dict

    @property
    def input_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def load_prepared_cyber_data(npz_path: str | Path) -> PreparedCyberData:
    """Load the compact file produced by the preprocessing script."""

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    payload = np.load(npz_path, allow_pickle=False)
    schema_path = npz_path.with_suffix(".json")
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Missing schema file {schema_path}. Run prepare_cicids2018.py first."
        )

    schema = json.loads(schema_path.read_text())
    return PreparedCyberData(
        x=np.asarray(payload["x"], dtype=np.float32),
        y=np.asarray(payload["y"], dtype=np.int64),
        client_ids=np.asarray(payload["client_ids"], dtype=np.int64),
        times=np.asarray(payload["times"], dtype=np.int64),
        variant_ids=np.asarray(payload["variant_ids"], dtype=np.int64) if "variant_ids" in payload.files else np.asarray(payload["y"], dtype=np.int64),
        class_names=list(schema["class_names"]),
        variant_names=list(schema.get("variant_names", schema["class_names"])),
        feature_names=list(schema["feature_names"]),
        client_mapping={str(k): int(v) for k, v in schema.get("client_mapping", {}).items()},
        metadata=schema.get("metadata", {}),
    )
