"""Streaming data manager for emerging cyber-threat experiments.

The manager withholds one attack family from the normal training stream and
injects *real held-out attack flows* only after a client becomes exposed by the
configured propagation schedule. This creates controlled spatial/temporal
threat emergence without label permutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from utils.cyber_dataset import CyberTensorDataset, PreparedCyberData


@dataclass
class ClientRoundState:
    client_id: int
    round_id: int
    exposed: bool
    n_base: int
    n_attack: int
    # Original prepared-data indices used for emerging-threat samples in this
    # round. Keeping indices out of the dataset payload lets the runner measure
    # true unique threat evidence without leaking them into model inputs.
    attack_indices: tuple[int, ...] = ()


class CyberStreamManager:
    def __init__(
        self,
        prepared: PreparedCyberData,
        num_clients: int,
        rounds: int,
        attack_label: int,
        emerging_variant_id: int | None = None,
        window_size: int = 2048,
        attack_mix_fraction: float = 0.20,
        evidence_mode: str = "mix_fraction",
        fixed_total_attack_per_round: int | None = None,
        local_test_fraction: float = 0.15,
        attack_guard_fraction: float = 0.05,
        attack_test_fraction: float = 0.30,
        min_window_samples: int = 32,
        max_global_eval_samples: int | None = 20000,
        max_attack_challenge_samples: int | None = 5000,
        max_target_alert_negative_samples: int | None = 10000,
        target_alert_calibration_fraction: float = 0.50,
        seed: int = 0,
    ) -> None:
        self.data = prepared
        self.num_clients = int(num_clients)
        self.rounds = int(rounds)
        self.attack_label = int(attack_label)
        self.emerging_variant_id = None if emerging_variant_id is None else int(emerging_variant_id)
        self.window_size = int(window_size)
        self.attack_mix_fraction = float(attack_mix_fraction)
        self.evidence_mode = str(evidence_mode).strip().lower()
        self.fixed_total_attack_per_round = (
            None if fixed_total_attack_per_round in (None, 0)
            else int(fixed_total_attack_per_round)
        )
        self.local_test_fraction = float(local_test_fraction)
        self.attack_guard_fraction = float(attack_guard_fraction)
        self.attack_test_fraction = float(attack_test_fraction)
        self.min_window_samples = int(min_window_samples)
        self.max_global_eval_samples = None if max_global_eval_samples in (None, 0) else int(max_global_eval_samples)
        self.max_attack_challenge_samples = None if max_attack_challenge_samples in (None, 0) else int(max_attack_challenge_samples)
        self.max_target_alert_negative_samples = (
            None
            if max_target_alert_negative_samples in (None, 0)
            else int(max_target_alert_negative_samples)
        )
        self.target_alert_calibration_fraction = float(
            target_alert_calibration_fraction
        )
        self.seed = int(seed)

        if not (0.0 <= self.attack_mix_fraction < 1.0):
            raise ValueError("attack_mix_fraction must be in [0,1)")
        if self.evidence_mode not in {"mix_fraction", "fixed_total", "fixed_unique"}:
            raise ValueError(
                "evidence_mode must be 'mix_fraction', 'fixed_total', or 'fixed_unique'"
            )
        if self.evidence_mode in {"fixed_total", "fixed_unique"}:
            if self.fixed_total_attack_per_round is None or self.fixed_total_attack_per_round <= 0:
                raise ValueError(
                    f"{self.evidence_mode} evidence mode requires fixed_total_attack_per_round > 0"
                )
        if not (0.0 < self.attack_test_fraction < 1.0):
            raise ValueError("attack_test_fraction must be in (0,1)")
        if not (0.0 <= self.attack_guard_fraction < 1.0):
            raise ValueError("attack_guard_fraction must be in [0,1)")
        if self.attack_guard_fraction + self.attack_test_fraction >= 1.0:
            raise ValueError(
                "attack_guard_fraction + attack_test_fraction must be < 1"
            )
        if not (0.0 < self.target_alert_calibration_fraction < 1.0):
            raise ValueError(
                "target_alert_calibration_fraction must be in (0,1)"
            )

        self.logical_client_ids = self._remap_client_ids(prepared.client_ids)
        self._build_pools()

        # Deterministic global ordering used by fixed_unique. Each active round
        # consumes the next block of unique attack-training flows, and that same
        # block is repartitioned across exposed sites for every prevalence run
        # with the same seed. This controls both total evidence and evidence
        # identity/diversity while varying only spatial distribution.
        unique_rng = np.random.default_rng(self.seed + 808_771)
        self._fixed_unique_order = self.attack_train_idx.copy()
        unique_rng.shuffle(self._fixed_unique_order)
        self._fixed_unique_cursor = 0
        self._fixed_unique_round_cache: Dict[int, np.ndarray] = {}

    def _remap_client_ids(self, source_ids: np.ndarray) -> np.ndarray:
        source_ids = np.asarray(source_ids, dtype=np.int64)
        unique = sorted(np.unique(source_ids).tolist())
        result = np.empty_like(source_ids)

        if len(unique) >= self.num_clients:
            mapping = {sid: idx % self.num_clients for idx, sid in enumerate(unique)}
            for sid, cid in mapping.items():
                result[source_ids == sid] = cid
            return result

        # Fewer natural source IDs than requested logical clients: subdivide
        # each source chronologically into multiple logical clients.
        next_client = 0
        for source_idx, sid in enumerate(unique):
            idx = np.where(source_ids == sid)[0]
            remaining_sources = len(unique) - source_idx
            remaining_clients = self.num_clients - next_client
            splits = max(1, int(np.ceil(remaining_clients / remaining_sources)))
            chunks = np.array_split(idx, splits)
            for chunk in chunks:
                if len(chunk) == 0:
                    continue
                cid = min(next_client, self.num_clients - 1)
                result[chunk] = cid
                next_client += 1
        # If all desired IDs were not used, that is acceptable; eligible
        # client logic will exclude empty clients.
        return result


    def _cap_indices(self, idx: np.ndarray, max_samples: int | None, seed_offset: int = 0, stratified: bool = True) -> np.ndarray:
        """Deterministically cap a fixed evaluation set.

        Full CSE-CIC-IDS2018 holdouts can contain hundreds of thousands of
        flows. Evaluating every personalized client model on the entire set at
        every FL round is unnecessary and can dominate runtime. The cap keeps
        a fixed, reproducible evaluation sample across all algorithms/seeds.
        """
        idx = np.asarray(idx, dtype=np.int64)
        if max_samples is None or len(idx) <= int(max_samples):
            return idx
        max_samples = int(max_samples)
        rng = np.random.default_rng(self.seed + 100_003 + int(seed_offset))
        if not stratified:
            chosen = rng.choice(idx, size=max_samples, replace=False)
            return np.sort(chosen.astype(np.int64))

        labels = self.data.y[idx]
        classes = np.unique(labels)
        per_class = max(1, max_samples // max(1, len(classes)))
        selected = []
        leftovers = []
        for c in classes:
            cidx = idx[labels == c]
            if len(cidx) <= per_class:
                selected.extend(cidx.tolist())
            else:
                take = rng.choice(cidx, size=per_class, replace=False)
                selected.extend(take.tolist())
                leftovers.extend(np.setdiff1d(cidx, take, assume_unique=False).tolist())
        remaining = max_samples - len(selected)
        if remaining > 0 and leftovers:
            pool = np.asarray(leftovers, dtype=np.int64)
            take = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            selected.extend(take.tolist())
        if len(selected) > max_samples:
            selected = rng.choice(np.asarray(selected, dtype=np.int64), size=max_samples, replace=False).tolist()
        return np.sort(np.asarray(selected, dtype=np.int64))

    # === CICIOT2023_VARIANT_INTERLEAVE_V4 ===
    def _uses_synthetic_variant_interleaving(self) -> bool:
        """Activate only for synthetic CICIoT2023 file-block clients."""
        metadata = getattr(self.data, "metadata", {}) or {}
        dataset = str(metadata.get("dataset", "")).strip().lower()
        client_source = str(metadata.get("client_source", "")).strip().lower()
        return (
            dataset == "ciciot2023"
            and client_source == "synthetic:chronological_file_blocks"
        )

    def _proportional_variant_interleave(
        self,
        idx: np.ndarray,
        client_id: int,
        seed_offset: int = 0,
    ) -> np.ndarray:
        """Spread raw variants across the synthetic stream proportionally."""
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) <= 1:
            return idx.copy()

        vids = self.data.variant_ids[idx]
        unique_vids = np.unique(vids)

        if len(unique_vids) <= 1:
            out = idx.copy()
            rng = np.random.default_rng(
                self.seed + 300_007 + 10_009 * int(client_id) + int(seed_offset)
            )
            rng.shuffle(out)
            return out

        merged_idx = []
        merged_keys = []
        for vid in unique_vids:
            pool = idx[vids == vid].copy()
            rng = np.random.default_rng(
                self.seed + 300_007 + 10_009 * int(client_id)
                + 101 * int(vid) + int(seed_offset)
            )
            rng.shuffle(pool)
            n = len(pool)
            keys = (np.arange(n, dtype=np.float64) + rng.random(n)) / float(n)
            merged_idx.append(pool)
            merged_keys.append(keys)

        all_idx = np.concatenate(merged_idx).astype(np.int64, copy=False)
        all_keys = np.concatenate(merged_keys)
        order = np.argsort(all_keys, kind="stable")
        return all_idx[order]

    def _synthetic_variant_stratified_split(
        self,
        idx: np.ndarray,
        client_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a variant-stratified local holdout and mixed base stream."""
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        test_n = max(1, int(round(len(idx) * self.local_test_fraction)))
        if len(idx) - test_n < self.min_window_samples:
            test_n = max(1, len(idx) // 5)
        test_n = min(test_n, max(0, len(idx) - 1))

        if test_n <= 0:
            return (
                self._proportional_variant_interleave(idx, client_id, seed_offset=11),
                np.empty(0, dtype=np.int64),
            )

        vids = self.data.variant_ids[idx]
        unique_vids, counts = np.unique(vids, return_counts=True)
        ideal = counts.astype(np.float64) * (float(test_n) / float(len(idx)))
        quota = np.floor(ideal).astype(np.int64)
        max_quota = np.maximum(counts - 1, 0)
        quota = np.minimum(quota, max_quota)

        remaining = int(test_n - int(quota.sum()))
        frac = ideal - np.floor(ideal)
        order = np.argsort(-frac, kind="stable")
        while remaining > 0:
            progressed = False
            for j in order:
                if quota[j] < max_quota[j]:
                    quota[j] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break

        train_parts = []
        test_parts = []
        for vid, q in zip(unique_vids, quota):
            pool = idx[vids == vid].copy()
            rng = np.random.default_rng(
                self.seed + 410_011 + 20_011 * int(client_id) + 211 * int(vid)
            )
            rng.shuffle(pool)
            q = int(q)
            if q > 0:
                test_parts.append(pool[:q])
            train_parts.append(pool[q:])

        train_idx = (
            np.concatenate(train_parts).astype(np.int64, copy=False)
            if train_parts else np.empty(0, dtype=np.int64)
        )
        test_idx = (
            np.concatenate(test_parts).astype(np.int64, copy=False)
            if test_parts else np.empty(0, dtype=np.int64)
        )

        train_idx = self._proportional_variant_interleave(
            train_idx, client_id, seed_offset=23
        )

        if len(test_idx):
            rng = np.random.default_rng(
                self.seed + 510_031 + 30_013 * int(client_id)
            )
            rng.shuffle(test_idx)

        return train_idx, test_idx

    def _feature_row_keys(self, idx: np.ndarray) -> np.ndarray:
        """Return exact fixed-width keys for processed model-input rows."""
        idx = np.asarray(idx, dtype=np.int64)
        row_dtype = np.dtype(
            (np.void, self.data.x.dtype.itemsize * self.data.x.shape[1])
        )
        if len(idx) == 0:
            return np.empty(0, dtype=row_dtype)
        features = np.ascontiguousarray(self.data.x[idx])
        if features.ndim != 2:
            raise ValueError("Expected a two-dimensional processed feature matrix")
        return features.view(row_dtype).reshape(-1)

    def _exact_feature_overlap_mask(
        self,
        reference_idx: np.ndarray,
        query_idx: np.ndarray,
    ) -> np.ndarray:
        """Mark query rows whose complete processed feature vector is in reference.

        Rows are compared byte-for-byte after conversion to the prepared matrix's
        contiguous dtype. This is an exact processed-feature audit, not a source-
        or capture-independence guarantee.
        """
        reference_idx = np.asarray(reference_idx, dtype=np.int64)
        query_idx = np.asarray(query_idx, dtype=np.int64)
        if len(query_idx) == 0:
            return np.zeros(0, dtype=bool)
        if len(reference_idx) == 0:
            return np.zeros(len(query_idx), dtype=bool)

        reference_rows = self._feature_row_keys(reference_idx)
        query_rows = self._feature_row_keys(query_idx)
        return np.isin(query_rows, np.unique(reference_rows))

    def _build_purged_emerging_split(
        self,
        attack_idx: np.ndarray,
    ) -> None:
        """Build a chronological split with a guard interval and feature purge."""
        n_attack = len(attack_idx)
        # Normalize decimal configuration arithmetic before applying floor;
        # otherwise 1 - 0.05 - 0.30 may be represented just below 0.65.
        train_fraction = round(
            1.0 - self.attack_guard_fraction - self.attack_test_fraction,
            12,
        )
        challenge_start_fraction = round(1.0 - self.attack_test_fraction, 12)
        # Floor makes the integer boundary convention explicit and guarantees
        # that the training pool never exceeds its configured fraction.
        train_end = int(np.floor(n_attack * train_fraction))
        challenge_start = int(
            np.floor(n_attack * challenge_start_fraction)
        )

        train_end = min(max(train_end, 1), n_attack - 1)
        challenge_start = min(max(challenge_start, train_end), n_attack - 1)
        if self.attack_guard_fraction > 0.0 and challenge_start == train_end:
            if train_end > 1:
                train_end -= 1
            else:
                challenge_start += 1

        self.attack_train_idx = attack_idx[:train_end]
        self.attack_guard_idx = attack_idx[train_end:challenge_start]
        self.attack_candidate_test_idx = attack_idx[challenge_start:]

        candidate_duplicate_mask = self._exact_feature_overlap_mask(
            self.attack_train_idx,
            self.attack_candidate_test_idx,
        )
        self.attack_duplicate_removed_idx = np.asarray(
            self.attack_candidate_test_idx[candidate_duplicate_mask],
            dtype=np.int64,
        )
        self.attack_test_idx = np.asarray(
            self.attack_candidate_test_idx[~candidate_duplicate_mask],
            dtype=np.int64,
        )
        if len(self.attack_test_idx) == 0:
            raise ValueError(
                "Feature purging removed every emerging-threat challenge row; "
                "a leakage-free challenge set cannot be formed from this variant."
            )

        final_duplicate_mask = self._exact_feature_overlap_mask(
            self.attack_train_idx,
            self.attack_test_idx,
        )
        if np.any(final_duplicate_mask):
            raise AssertionError(
                "Emerging-threat feature overlap remains after challenge purging"
            )

        index_overlap = int(
            np.intersect1d(self.attack_train_idx, self.attack_test_idx).size
        )
        if index_overlap:
            raise AssertionError("Emerging-threat train/test index leakage detected")

        duplicate_candidate_idx = self.attack_duplicate_removed_idx
        duplicate_pattern_count = 0
        if len(duplicate_candidate_idx):
            duplicate_rows = np.ascontiguousarray(self.data.x[duplicate_candidate_idx])
            row_dtype = np.dtype(
                (np.void, duplicate_rows.dtype.itemsize * duplicate_rows.shape[1])
            )
            duplicate_pattern_count = int(
                len(np.unique(duplicate_rows.view(row_dtype).reshape(-1)))
            )

        self.emerging_split_audit = {
            "protocol": "purged_chronological_feature_disjoint",
            "n_emerging_total": int(n_attack),
            "train_fraction_configured": float(train_fraction),
            "guard_fraction_configured": float(self.attack_guard_fraction),
            "candidate_challenge_fraction_configured": float(
                self.attack_test_fraction
            ),
            "n_train": int(len(self.attack_train_idx)),
            "n_guard": int(len(self.attack_guard_idx)),
            "n_candidate_challenge": int(len(self.attack_candidate_test_idx)),
            "n_final_challenge": int(len(self.attack_test_idx)),
            "n_candidate_rows_purged_as_train_duplicates": int(
                len(self.attack_duplicate_removed_idx)
            ),
            "n_unique_duplicate_patterns_crossing_before_purge": int(
                duplicate_pattern_count
            ),
            "n_index_overlap_train_final_challenge": index_overlap,
            "n_feature_overlap_rows_after_purge": int(final_duplicate_mask.sum()),
            "n_unique_duplicate_patterns_crossing_after_purge": 0,
            "capture_independence_verified": False,
        }

    def _stratified_two_way_split(
        self,
        idx: np.ndarray,
        first_fraction: float,
        seed_offset: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split indices reproducibly while retaining every feasible class."""
        idx = np.asarray(idx, dtype=np.int64)
        rng = np.random.default_rng(self.seed + 700_001 + int(seed_offset))
        first_parts: list[np.ndarray] = []
        second_parts: list[np.ndarray] = []
        for label in np.unique(self.data.y[idx]):
            pool = idx[self.data.y[idx] == label].copy()
            rng.shuffle(pool)
            if len(pool) == 1:
                # A singleton cannot support independent calibration and test.
                second_parts.append(pool)
                continue
            n_first = int(round(len(pool) * float(first_fraction)))
            n_first = min(max(n_first, 1), len(pool) - 1)
            first_parts.append(pool[:n_first])
            second_parts.append(pool[n_first:])
        first = (
            np.concatenate(first_parts).astype(np.int64, copy=False)
            if first_parts else np.empty(0, dtype=np.int64)
        )
        second = (
            np.concatenate(second_parts).astype(np.int64, copy=False)
            if second_parts else np.empty(0, dtype=np.int64)
        )
        return np.sort(first), np.sort(second)

    def _build_target_alert_negative_sets(self) -> None:
        """Build disjoint calibration and mixed-challenge negative sets.

        The target is the canonical attack family.  Therefore every row used
        here is a target-negative (benign or a different attack family).  Both
        partitions come from the already-held-out local-test tails and are
        purged against all training rows by exact processed feature value.
        """
        heldout_parts = [
            np.asarray(v, dtype=np.int64)
            for v in self.local_test_idx.values()
            if len(v)
        ]
        if not heldout_parts:
            raise ValueError("No local holdout rows for target-alert evaluation")
        source_idx = np.unique(np.concatenate(heldout_parts))
        candidate_idx = source_idx[self.data.y[source_idx] != self.attack_label]
        if len(candidate_idx) < 20:
            raise ValueError(
                "Too few held-out target-negative rows for calibrated target "
                f"evaluation: {len(candidate_idx)}"
            )

        candidate_idx = self._cap_indices(
            candidate_idx,
            self.max_target_alert_negative_samples,
            seed_offset=41,
            stratified=True,
        )
        calibration_idx, test_idx = self._stratified_two_way_split(
            candidate_idx,
            self.target_alert_calibration_fraction,
            seed_offset=43,
        )

        training_parts = [
            np.asarray(v, dtype=np.int64)
            for v in self.base_train_idx.values()
            if len(v)
        ]
        training_parts.append(np.asarray(self.attack_train_idx, dtype=np.int64))
        training_idx = np.unique(np.concatenate(training_parts))

        calibration_train_overlap = self._exact_feature_overlap_mask(
            training_idx, calibration_idx
        )
        calibration_idx = calibration_idx[~calibration_train_overlap]
        test_train_overlap = self._exact_feature_overlap_mask(training_idx, test_idx)
        test_idx = test_idx[~test_train_overlap]

        # Remove contradictory exact patterns shared with target-positive rows.
        calibration_positive_overlap = self._exact_feature_overlap_mask(
            self.attack_test_idx, calibration_idx
        )
        calibration_idx = calibration_idx[~calibration_positive_overlap]
        test_positive_overlap = self._exact_feature_overlap_mask(
            self.attack_test_idx, test_idx
        )
        test_idx = test_idx[~test_positive_overlap]

        # Calibration and test must also have no exact feature pattern in common.
        test_calibration_overlap = self._exact_feature_overlap_mask(
            calibration_idx, test_idx
        )
        test_idx = test_idx[~test_calibration_overlap]
        if len(calibration_idx) < 10 or len(test_idx) < 10:
            raise ValueError(
                "Feature purging left too few target-alert negatives: "
                f"calibration={len(calibration_idx)}, test={len(test_idx)}"
            )

        self.target_alert_calibration_negative_idx = np.asarray(
            calibration_idx, dtype=np.int64
        )
        self.target_alert_test_negative_idx = np.asarray(test_idx, dtype=np.int64)
        self.target_alert_calibration_negative_set = self._dataset_from_indices(
            self.target_alert_calibration_negative_idx,
            metadata={
                "kind": "target_alert_calibration_negatives",
                "target_label": self.attack_label,
            },
        )
        self.target_alert_test_negative_set = self._dataset_from_indices(
            self.target_alert_test_negative_idx,
            metadata={
                "kind": "target_alert_test_negatives",
                "target_label": self.attack_label,
            },
        )

        final_cal_train = self._exact_feature_overlap_mask(
            training_idx, self.target_alert_calibration_negative_idx
        )
        final_test_train = self._exact_feature_overlap_mask(
            training_idx, self.target_alert_test_negative_idx
        )
        final_cal_test = self._exact_feature_overlap_mask(
            self.target_alert_calibration_negative_idx,
            self.target_alert_test_negative_idx,
        )
        final_positive_cal = self._exact_feature_overlap_mask(
            self.attack_test_idx,
            self.target_alert_calibration_negative_idx,
        )
        final_positive_test = self._exact_feature_overlap_mask(
            self.attack_test_idx,
            self.target_alert_test_negative_idx,
        )
        index_train = int(
            np.intersect1d(
                training_idx,
                np.concatenate([
                    self.target_alert_calibration_negative_idx,
                    self.target_alert_test_negative_idx,
                ]),
            ).size
        )
        index_cal_test = int(
            np.intersect1d(
                self.target_alert_calibration_negative_idx,
                self.target_alert_test_negative_idx,
            ).size
        )
        if (
            index_train
            or index_cal_test
            or np.any(final_cal_train)
            or np.any(final_test_train)
            or np.any(final_cal_test)
            or np.any(final_positive_cal)
            or np.any(final_positive_test)
        ):
            raise AssertionError("Target-alert split leakage remains after purging")

        def counts(indices: np.ndarray) -> dict[int, int]:
            labels, values = np.unique(
                self.data.y[indices], return_counts=True
            )
            return {
                int(label): int(value)
                for label, value in zip(labels.tolist(), values.tolist())
            }

        self.target_alert_split_audit = {
            "protocol": "heldout_stratified_feature_disjoint_calibration_test",
            "target_label": int(self.attack_label),
            "n_source_local_holdout": int(len(source_idx)),
            "n_candidate_target_negatives": int(len(candidate_idx)),
            "calibration_fraction": float(
                self.target_alert_calibration_fraction
            ),
            "n_calibration_negatives": int(len(calibration_idx)),
            "n_test_negatives": int(len(test_idx)),
            "n_calibration_rows_purged_as_training_duplicates": int(
                np.sum(calibration_train_overlap)
            ),
            "n_test_rows_purged_as_training_duplicates": int(
                np.sum(test_train_overlap)
            ),
            "n_calibration_rows_purged_as_positive_duplicates": int(
                np.sum(calibration_positive_overlap)
            ),
            "n_test_rows_purged_as_positive_duplicates": int(
                np.sum(test_positive_overlap)
            ),
            "n_test_rows_purged_as_calibration_duplicates": int(
                np.sum(test_calibration_overlap)
            ),
            "n_index_overlap_training_evaluation": index_train,
            "n_index_overlap_calibration_test": index_cal_test,
            "n_feature_overlap_training_calibration": int(
                np.sum(final_cal_train)
            ),
            "n_feature_overlap_training_test": int(np.sum(final_test_train)),
            "n_feature_overlap_calibration_test": int(np.sum(final_cal_test)),
            "n_feature_overlap_positive_calibration": int(
                np.sum(final_positive_cal)
            ),
            "n_feature_overlap_positive_test": int(np.sum(final_positive_test)),
            "calibration_class_counts": counts(calibration_idx),
            "test_class_counts": counts(test_idx),
            "capture_independence_verified": False,
        }

    def _build_pools(self) -> None:
        x, y, times = self.data.x, self.data.y, self.data.times
        if self.emerging_variant_id is None:
            emerging_mask = y == self.attack_label
        else:
            emerging_mask = self.data.variant_ids == self.emerging_variant_id
            labels_in_variant = np.unique(y[emerging_mask])
            if len(labels_in_variant) != 1 or int(labels_in_variant[0]) != self.attack_label:
                raise ValueError(
                    "Configured emerging_variant_id does not map uniquely to attack_label "
                    f"{self.attack_label}; observed canonical labels={labels_in_variant.tolist()}"
                )

        attack_idx = np.where(emerging_mask)[0]
        self.emerging_mask = emerging_mask
        if len(attack_idx) < 20:
            raise ValueError(
                f"Only {len(attack_idx)} examples for emerging attack label "
                f"{self.attack_label}; choose another family or larger dataset."
            )

        attack_idx = attack_idx[np.argsort(times[attack_idx], kind="stable")]
        self._build_purged_emerging_split(attack_idx)

        self.base_train_idx: Dict[int, np.ndarray] = {}
        self.local_test_idx: Dict[int, np.ndarray] = {}

        for cid in range(self.num_clients):
            idx = np.where(
                (self.logical_client_ids == cid) & (~self.emerging_mask)
            )[0]

            if len(idx) == 0:
                self.base_train_idx[cid] = np.empty(0, dtype=np.int64)
                self.local_test_idx[cid] = np.empty(0, dtype=np.int64)
                continue

            if self._uses_synthetic_variant_interleaving():
                train_idx, test_idx = self._synthetic_variant_stratified_split(
                    idx, client_id=cid
                )
                self.base_train_idx[cid] = train_idx
                self.local_test_idx[cid] = test_idx
            else:
                idx = idx[np.argsort(times[idx], kind="stable")]
                test_n = max(1, int(round(len(idx) * self.local_test_fraction)))
                if len(idx) - test_n < self.min_window_samples:
                    test_n = max(1, len(idx) // 5)
                self.base_train_idx[cid] = (
                    idx[:-test_n] if test_n < len(idx) else idx[:0]
                )
                self.local_test_idx[cid] = idx[-test_n:]

            if np.any(self.emerging_mask[self.base_train_idx[cid]]):
                raise AssertionError(
                    f"Emerging threat leaked into base training pool for client {cid}"
                )

        # Split attack training data deterministically among clients. A client
        # only receives its assigned pool after exposure; no attack sample is
        # leaked to unexposed clients through local training.
        rng = np.random.default_rng(self.seed + 17)
        shuffled = self.attack_train_idx.copy()
        rng.shuffle(shuffled)
        chunks = np.array_split(shuffled, self.num_clients)
        self.client_attack_train_idx = {
            cid: np.asarray(chunks[cid], dtype=np.int64)
            for cid in range(self.num_clients)
        }

        # A common fixed challenge set is appropriate for early-warning:
        # models from unexposed clients are tested on exactly the same threat.
        self.attack_challenge_idx = self._cap_indices(
            self.attack_test_idx, self.max_attack_challenge_samples, seed_offset=1, stratified=False
        )
        self.attack_challenge_set = self._dataset_from_indices(
            self.attack_challenge_idx,
            metadata={
                "kind": "attack_challenge",
                "attack_label": self.attack_label,
                "emerging_variant_id": self.emerging_variant_id,
                "split_protocol": self.emerging_split_audit["protocol"],
            },
        )

        # A second challenge gives every distinct processed feature pattern
        # equal weight. Row-weighted challenge recall remains the primary EWS;
        # this pattern-balanced view is a sensitivity analysis.
        final_test_keys = self._feature_row_keys(self.attack_test_idx)
        _, first_positions = np.unique(final_test_keys, return_index=True)
        self.attack_pattern_challenge_idx_full = self.attack_test_idx[
            np.sort(first_positions)
        ]
        self.attack_pattern_challenge_idx = self._cap_indices(
            self.attack_pattern_challenge_idx_full,
            self.max_attack_challenge_samples,
            seed_offset=3,
            stratified=False,
        )
        self.attack_pattern_challenge_set = self._dataset_from_indices(
            self.attack_pattern_challenge_idx,
            metadata={
                "kind": "attack_pattern_balanced_challenge",
                "attack_label": self.attack_label,
                "emerging_variant_id": self.emerging_variant_id,
                "one_row_per_exact_feature_pattern": True,
            },
        )
        self.emerging_split_audit[
            "n_final_challenge_unique_feature_patterns"
        ] = int(len(self.attack_pattern_challenge_idx_full))

        # === DATASET_SPECIFIC_GLOBAL_EVAL_V1 ===
        # CICIoT2023 needs a dedicated stratified non-emerging holdout because
        # its folder-per-attack export has synthetic file-block chronology.
        # Datasets with meaningful chronology retain the original evaluation
        # protocol so earlier CSE-CIC-IDS2018 experiments remain reproducible.
        if self._uses_synthetic_variant_interleaving():
            self.global_eval_idx_full = np.where(~self.emerging_mask)[0].astype(np.int64)
            global_eval_cap = min(
                int(self.max_global_eval_samples or 20000),
                len(self.global_eval_idx_full),
            )
            self.global_eval_idx = self._cap_indices(
                self.global_eval_idx_full,
                global_eval_cap,
                seed_offset=2,
                stratified=True,
            )

            global_eval_set = set(int(i) for i in self.global_eval_idx.tolist())
            for cid in range(self.num_clients):
                base_idx = self.base_train_idx[cid]
                if len(base_idx):
                    keep = np.asarray(
                        [int(i) not in global_eval_set for i in base_idx],
                        dtype=bool,
                    )
                    self.base_train_idx[cid] = base_idx[keep]
                if np.intersect1d(self.base_train_idx[cid], self.global_eval_idx).size:
                    raise AssertionError(
                        f"Global-evaluation leakage detected in base training pool for client {cid}"
                    )

            available_classes = np.unique(self.data.y[self.global_eval_idx_full])
            eval_classes = np.unique(self.data.y[self.global_eval_idx])
            if (
                len(self.global_eval_idx) >= len(available_classes)
                and not np.array_equal(np.sort(available_classes), np.sort(eval_classes))
            ):
                missing = sorted(
                    set(int(x) for x in available_classes.tolist())
                    - set(int(x) for x in eval_classes.tolist())
                )
                raise AssertionError(
                    "Class-stratified global evaluation missed available canonical classes: "
                    f"{missing}"
                )

            class_counts = {
                int(c): int(np.sum(self.data.y[self.global_eval_idx] == c))
                for c in eval_classes
            }
            self.global_eval_set = self._dataset_from_indices(
                self.global_eval_idx,
                metadata={
                    "kind": "global_eval",
                    "stratified": True,
                    "excludes_emerging_variant": True,
                    "class_counts": class_counts,
                },
            )
        else:
            # Original CSE-CIC-IDS2018 protocol: evaluation-only local tails
            # plus held-out emerging-attack test samples. No base-training
            # samples are removed, preserving the established chronological stream.
            global_idx = [idx for idx in self.local_test_idx.values() if len(idx)]
            global_idx.append(self.attack_test_idx)
            self.global_eval_idx_full = np.unique(np.concatenate(global_idx))
            self.global_eval_idx = self._cap_indices(
                self.global_eval_idx_full,
                self.max_global_eval_samples,
                seed_offset=2,
                stratified=True,
            )
            self.global_eval_set = self._dataset_from_indices(
                self.global_eval_idx,
                metadata={"kind": "global_eval", "protocol": "chronological_local_tails_plus_attack_test"},
            )

        # Additive target-alert evaluation.  Existing global and emerging-only
        # challenge sets above are unchanged for backward compatibility.
        self._build_target_alert_negative_sets()

    def _dataset_from_indices(self, idx: np.ndarray, metadata: Optional[dict] = None):
        idx = np.asarray(idx, dtype=np.int64)
        return CyberTensorDataset(self.data.x[idx], self.data.y[idx], metadata=metadata)

    def feature_pattern_tokens(self, idx: Iterable[int]) -> tuple[bytes, ...]:
        """Return collision-free exact feature identities for evidence records."""
        idx_array = np.asarray(list(idx), dtype=np.int64)
        return tuple(key.tobytes() for key in self._feature_row_keys(idx_array))

    @property
    def eligible_clients(self) -> List[int]:
        return [
            cid for cid in range(self.num_clients)
            if len(self.base_train_idx.get(cid, [])) >= self.min_window_samples
        ]

    def local_test_set(self, client_id: int) -> CyberTensorDataset:
        idx = self.local_test_idx[int(client_id)]
        if len(idx) == 0:
            # Keep constructors valid even for an empty natural site.
            return CyberTensorDataset(
                np.empty((0, self.data.input_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                metadata={"client_id": int(client_id), "kind": "local_test"},
            )
        return self._dataset_from_indices(
            idx,
            metadata={"client_id": int(client_id), "kind": "local_test"},
        )

    def _base_window_indices(self, client_id: int, round_id: int) -> np.ndarray:
        pool = self.base_train_idx[int(client_id)]
        if len(pool) == 0:
            return pool
        start = int(round_id) * self.window_size
        if start >= len(pool):
            # A finite benchmark cannot provide an endless real stream. Cycle
            # only the non-emerging background pool, while keeping attack
            # exposure monotonic. The logger records round IDs so this can be
            # disclosed; with full CSE-CIC-IDS2018 and modest windows cycling
            # is usually avoidable.
            start = start % len(pool)
        end = min(start + self.window_size, len(pool))
        idx = pool[start:end]
        if len(idx) < self.min_window_samples and len(pool) >= self.min_window_samples:
            missing = self.min_window_samples - len(idx)
            idx = np.concatenate([idx, pool[:missing]])
        return idx

    def _attack_window_indices(self, client_id: int, round_id: int, n_attack: int) -> np.ndarray:
        if n_attack <= 0:
            return np.empty(0, dtype=np.int64)
        pool = self.client_attack_train_idx[int(client_id)]
        if len(pool) == 0:
            # Some client chunks can be empty for a small debug subset. Fall
            # back to the global attack-training pool without touching test data.
            pool = self.attack_train_idx
        rng = np.random.default_rng(
            self.seed + 97_409 * int(round_id) + 1_009 * int(client_id)
        )
        replace = len(pool) < n_attack
        return rng.choice(pool, size=n_attack, replace=replace).astype(np.int64)

    def round_dataset(
        self,
        client_id: int,
        round_id: int,
        exposed: bool,
        n_attack_override: int | None = None,
        attack_indices_override: np.ndarray | None = None,
    ) -> Tuple[CyberTensorDataset, ClientRoundState]:
        base_idx = self._base_window_indices(client_id, round_id)
        if len(base_idx) == 0:
            empty = CyberTensorDataset(
                np.empty((0, self.data.input_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                metadata={"client_id": int(client_id), "round": int(round_id)},
            )
            return empty, ClientRoundState(client_id, round_id, exposed, 0, 0, ())

        if exposed:
            if attack_indices_override is not None:
                attack_idx = np.asarray(attack_indices_override, dtype=np.int64)
            else:
                if n_attack_override is not None:
                    n_attack = max(0, int(n_attack_override))
                elif self.attack_mix_fraction > 0:
                    # Choose n_attack so attack/(base+attack) approximately equals mix.
                    n_attack = int(round(
                        len(base_idx) * self.attack_mix_fraction /
                        max(1e-12, 1.0 - self.attack_mix_fraction)
                    ))
                else:
                    n_attack = 0
                attack_idx = self._attack_window_indices(client_id, round_id, n_attack)
        else:
            attack_idx = np.empty(0, dtype=np.int64)

        idx = np.concatenate([base_idx, attack_idx])
        # Shuffle within a time window, not across time windows.
        rng = np.random.default_rng(
            self.seed + 31_337 * int(round_id) + 101 * int(client_id)
        )
        rng.shuffle(idx)
        dataset = self._dataset_from_indices(
            idx,
            metadata={
                "client_id": int(client_id),
                "round": int(round_id),
                "exposed": bool(exposed),
                "n_base": int(len(base_idx)),
                "n_attack": int(len(attack_idx)),
            },
        )
        return dataset, ClientRoundState(
            int(client_id), int(round_id), bool(exposed),
            int(len(base_idx)), int(len(attack_idx)),
            tuple(int(i) for i in np.asarray(attack_idx, dtype=np.int64).tolist()),
        )

    def _fixed_evidence_allocation(self, exposed_clients: Set[int]) -> Dict[int, int]:
        """Allocate one fixed per-round threat-evidence budget across exposed sites.

        This makes spatial prevalence and total evidence independently controllable.
        With full participation, changing the number of exposed clients changes only
        where evidence is located, not how much emerging-threat data enters training.
        """
        active = sorted(set(int(i) for i in exposed_clients) & set(self.eligible_clients))
        if not active:
            return {}
        total = int(self.fixed_total_attack_per_round or 0)
        q, r = divmod(total, len(active))
        return {cid: q + (1 if j < r else 0) for j, cid in enumerate(active)}

    def _fixed_unique_round_indices(self, round_id: int, exposed_clients: Set[int]) -> np.ndarray:
        """Return one fixed set of globally unique threat flows for this active round.

        The set depends on seed and active-round order, not on the number of
        exposed clients. Therefore rho=0.10, 0.30, and 0.50 receive the same
        threat identities at the same post-drift round; only their placement
        across clients changes.
        """
        active = sorted(set(int(i) for i in exposed_clients) & set(self.eligible_clients))
        if not active:
            return np.empty(0, dtype=np.int64)
        if int(round_id) in self._fixed_unique_round_cache:
            return self._fixed_unique_round_cache[int(round_id)]

        total = int(self.fixed_total_attack_per_round or 0)
        start = self._fixed_unique_cursor
        end = start + total
        if end > len(self._fixed_unique_order):
            raise RuntimeError(
                "fixed_unique exhausted the held-out attack-training pool: "
                f"need {total} new unique flows at round {round_id}, but only "
                f"{len(self._fixed_unique_order) - start} remain. Reduce --evidence, "
                "reduce post-drift rounds, or use a larger emerging-variant pool."
            )
        idx = np.asarray(self._fixed_unique_order[start:end], dtype=np.int64)
        if len(np.unique(idx)) != len(idx):
            raise AssertionError("fixed_unique produced duplicate threat indices within a round")
        self._fixed_unique_cursor = end
        self._fixed_unique_round_cache[int(round_id)] = idx
        return idx

    def _fixed_unique_assignment(self, exposed_clients: Set[int], round_id: int) -> Dict[int, np.ndarray]:
        active = sorted(set(int(i) for i in exposed_clients) & set(self.eligible_clients))
        if not active:
            return {}
        idx = self._fixed_unique_round_indices(round_id, exposed_clients)
        chunks = np.array_split(idx, len(active))
        return {cid: np.asarray(chunk, dtype=np.int64) for cid, chunk in zip(active, chunks)}

    def apply_round(
        self,
        clients: Sequence,
        round_id: int,
        exposed_clients: Set[int],
    ) -> Dict[int, ClientRoundState]:
        states: Dict[int, ClientRoundState] = {}
        fixed = (
            self._fixed_evidence_allocation(exposed_clients)
            if self.evidence_mode == "fixed_total"
            else {}
        )
        fixed_unique = (
            self._fixed_unique_assignment(exposed_clients, round_id)
            if self.evidence_mode == "fixed_unique"
            else {}
        )
        for client in clients:
            is_exposed = int(client.id) in exposed_clients
            override = fixed.get(int(client.id), 0) if self.evidence_mode == "fixed_total" else None
            unique_override = fixed_unique.get(int(client.id)) if self.evidence_mode == "fixed_unique" else None
            ds, state = self.round_dataset(
                client.id,
                round_id,
                is_exposed,
                n_attack_override=override,
                attack_indices_override=unique_override,
            )
            # Hard leakage invariant: an unexposed client must never receive an
            # emerging-threat sample in its local training window.
            if not is_exposed and state.n_attack != 0:
                raise AssertionError(
                    f"Threat leakage: unexposed client {client.id} received "
                    f"{state.n_attack} emerging samples at round {round_id}"
                )
            client.train_set = ds
            states[client.id] = state
        return states
