from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cyber_dataset import CyberTensorDataset, PreparedCyberData
from utils.cyber_models import CyberMLP
from utils.cyber_schedule import ParticipationScheduler, ThreatPropagationSchedule
from utils.cyber_stream import CyberStreamManager
from utils.cyber_metrics import classification_metrics


def synthetic_prepared():
    rng = np.random.default_rng(0)
    n_clients = 5
    rows_per_client = 500
    xs, ys, cids, times, variants = [], [], [], [], []
    t = 0
    # variant ids: 0 benign, 1 LOIC(old ddos), 2 HOIC(emerging ddos), 3 Hulk(dos)
    for cid in range(n_clients):
        for _ in range(rows_per_client):
            u = rng.random()
            if u < 0.65:
                y, v = 0, 0
            elif u < 0.80:
                y, v = 1, 1
            elif u < 0.90:
                y, v = 1, 2
            else:
                y, v = 2, 3
            x = rng.normal(size=8).astype(np.float32)
            x[0] = float(v)  # debug marker
            xs.append(x); ys.append(y); cids.append(cid); times.append(t); variants.append(v)
            t += 1
    return PreparedCyberData(
        x=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.int64),
        client_ids=np.asarray(cids, dtype=np.int64),
        times=np.asarray(times, dtype=np.int64),
        variant_ids=np.asarray(variants, dtype=np.int64),
        class_names=["benign", "ddos", "dos"],
        variant_names=["benign", "ddos-loic", "ddos-hoic", "dos-hulk"],
        feature_names=[f"f{i}" for i in range(8)],
        client_mapping={},
        metadata={},
    )


def test_stream_variant_withholding():
    data = synthetic_prepared()
    manager = CyberStreamManager(
        data, num_clients=5, rounds=10, attack_label=1,
        emerging_variant_id=2, window_size=64, attack_mix_fraction=0.25,
        min_window_samples=10, seed=2,
    )
    ds0, state0 = manager.round_dataset(0, 0, exposed=False)
    assert state0.n_attack == 0
    # First feature is raw variant marker: HOIC variant 2 must be absent.
    assert not np.any(np.isclose(ds0.data[:, 0], 2.0))
    # Old DDoS variant remains in the background stream.
    assert np.any(np.isclose(ds0.data[:, 0], 1.0))

    ds1, state1 = manager.round_dataset(0, 0, exposed=True)
    assert state1.n_attack > 0
    assert np.any(np.isclose(ds1.data[:, 0], 2.0))


def test_schedule_monotonicity_and_reproducibility():
    milestones = [[0, 0.0], [2, 0.2], [4, 0.6], [6, 1.0]]
    a = ThreatPropagationSchedule(10, milestones, seed=7)
    b = ThreatPropagationSchedule(10, milestones, seed=7)
    assert a.exposed_clients(5) == b.exposed_clients(5)
    assert a.exposed_clients(2).issubset(a.exposed_clients(4))
    assert a.exposed_clients(4).issubset(a.exposed_clients(6))


def test_participation_minimum():
    p = ParticipationScheduler(10, normal_prob=0.0, exposed_prob=0.0, min_selected=3, seed=1)
    chosen = p.select(0, exposed_clients=set(), eligible_clients=range(10))
    assert len(chosen) == 3


def test_model_interface():
    model = CyberMLP(8, 3)
    x = torch.randn(4, 8)
    logits, features = model(x, True)
    assert logits.shape == (4, 3)
    assert features.shape[0] == 4
    keys = list(model.state_dict().keys())
    assert keys[-2:] == ["classifier.weight", "classifier.bias"]


def test_metrics():
    y = np.array([0, 0, 1, 1, 1, 2])
    p = np.array([0, 2, 1, 0, 1, 2])
    m = classification_metrics(y, p, attack_label=1, benign_label=0)
    assert m["attack_tp"] == 2
    assert m["attack_fn"] == 1
    assert m["benign_fp"] == 1


def test_fixed_total_evidence_and_no_leakage():
    data = synthetic_prepared()
    manager = CyberStreamManager(
        data, num_clients=5, rounds=10, attack_label=1,
        emerging_variant_id=2, window_size=64, attack_mix_fraction=0.25,
        evidence_mode='fixed_total', fixed_total_attack_per_round=120,
        min_window_samples=10, seed=3,
    )
    class C:
        def __init__(self, cid): self.id = cid; self.train_set = None
    clients = [C(i) for i in range(5)]
    exposed = {0, 1, 2}
    states = manager.apply_round(clients, 2, exposed)
    assert sum(states[i].n_attack for i in exposed) == 120
    assert all(states[i].n_attack == 0 for i in {3, 4})
    # Equal allocation differs by at most one example.
    vals = [states[i].n_attack for i in sorted(exposed)]
    assert max(vals) - min(vals) <= 1


def test_purged_chronological_feature_disjoint_split():
    data = synthetic_prepared()
    emerging_idx = np.where(data.variant_ids == 2)[0]
    assert len(emerging_idx) >= 20

    # Copy an early training feature vector into two late candidate-challenge
    # rows. Their indices remain distinct, so an index-only audit would miss it.
    data.x[emerging_idx[-1]] = data.x[emerging_idx[0]]
    data.x[emerging_idx[-2]] = data.x[emerging_idx[0]]

    manager = CyberStreamManager(
        data, num_clients=5, rounds=10, attack_label=1,
        emerging_variant_id=2, window_size=64, attack_mix_fraction=0.25,
        attack_guard_fraction=0.05, attack_test_fraction=0.30,
        min_window_samples=10, seed=4,
    )

    audit = manager.emerging_split_audit
    assert audit['protocol'] == 'purged_chronological_feature_disjoint'
    assert audit['n_guard'] > 0
    assert audit['n_candidate_rows_purged_as_train_duplicates'] == 2
    assert audit['n_unique_duplicate_patterns_crossing_before_purge'] == 1
    assert audit['n_index_overlap_train_final_challenge'] == 0
    assert audit['n_feature_overlap_rows_after_purge'] == 0
    assert audit['n_final_challenge_unique_feature_patterns'] == len(
        manager.attack_pattern_challenge_idx_full
    )
    assert len(manager.feature_pattern_tokens(
        manager.attack_pattern_challenge_idx_full
    )) == len(set(manager.feature_pattern_tokens(
        manager.attack_pattern_challenge_idx_full
    )))
    assert set(manager.attack_duplicate_removed_idx.tolist()) == {
        int(emerging_idx[-1]), int(emerging_idx[-2])
    }
    assert emerging_idx[-1] not in set(manager.attack_test_idx.tolist())
    assert emerging_idx[-2] not in set(manager.attack_test_idx.tolist())

    target_audit = manager.target_alert_split_audit
    assert target_audit['n_index_overlap_training_evaluation'] == 0
    assert target_audit['n_index_overlap_calibration_test'] == 0
    assert target_audit['n_feature_overlap_training_calibration'] == 0
    assert target_audit['n_feature_overlap_training_test'] == 0
    assert target_audit['n_feature_overlap_calibration_test'] == 0
    assert target_audit['n_feature_overlap_positive_calibration'] == 0
    assert target_audit['n_feature_overlap_positive_test'] == 0
    assert np.all(
        data.y[manager.target_alert_calibration_negative_idx] != 1
    )
    assert np.all(data.y[manager.target_alert_test_negative_idx] != 1)

if __name__ == "__main__":
    test_stream_variant_withholding()
    test_schedule_monotonicity_and_reproducibility()
    test_participation_minimum()
    test_model_interface()
    test_metrics()
    test_fixed_total_evidence_and_no_leakage()
    test_purged_chronological_feature_disjoint_split()
    print("All cyber component tests passed.")
