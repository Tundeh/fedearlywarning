"""Tests that do not require the external FedCCFA repository classes."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from utils.cyber_config import deep_update
from utils.cyber_models import CyberMLP
from utils.cyber_model_bridge import load_state_everywhere, state_dict_cpu
from utils.cyber_schedule import ThreatPropagationSchedule, ParticipationScheduler


def test_deep_update():
    x=deep_update({'a':1,'b':{'x':1,'y':2}}, {'b':{'x':9},'c':3})
    assert x=={'a':1,'b':{'x':9,'y':2},'c':3}


def test_same_initialization():
    args={'input_dim':5,'num_classes':3,'hidden_dim':8,'feature_dim':4,'dropout':0.,'device':'cpu'}
    torch.manual_seed(123)
    m=CyberMLP(5,3,8,4,0.)
    state=state_dict_cpu(m)
    class E: pass
    a=E(); b=E(); a.model=CyberMLP(5,3,8,4,0.); b.model=CyberMLP(5,3,8,4,0.)
    load_state_everywhere(a,state); load_state_everywhere(b,state)
    for p,q in zip(a.model.parameters(),b.model.parameters()): assert torch.equal(p,q)


def test_schedule_reproducible():
    t1=ThreatPropagationSchedule(20, [[0,0],[4,.1],[8,.5]], seed=7)
    t2=ThreatPropagationSchedule(20, [[0,0],[4,.1],[8,.5]], seed=7)
    p1=ParticipationScheduler(20,.5,.2,3,seed=9)
    p2=ParticipationScheduler(20,.5,.2,3,seed=9)
    for r in range(12):
        assert t1.exposed_clients(r)==t2.exposed_clients(r)
        e=t1.exposed_clients(r)
        assert p1.select(r,e)==p2.select(r,e)


def test_early_warning_tracker():
    from utils.cyber_metrics import EarlyWarningTracker
    t = EarlyWarningTracker(drift_round=10, target=.90)
    t.update(10, .2, 18, .1, 1000, 800, 650)
    assert np.isnan(t.time_to_target)
    t.update(18, .95, 14, .3, 17000, 9000, 7200)
    assert t.time_to_target == 8
    d = t.as_dict()
    assert d['early_warning_threshold_prevalence'] == .3
    assert d['early_warning_threshold_cumulative_presentations'] == 17000
    assert d['early_warning_threshold_cumulative_distinct_records'] == 9000
    assert d['early_warning_threshold_cumulative_distinct_feature_patterns'] == 7200

def test_matched_challenge_groups():
    from utils.cyber_dataset import CyberTensorDataset
    from utils.cyber_metrics import shared_challenge_group_metrics
    class Constant(torch.nn.Module):
        def __init__(self, cls):
            super().__init__(); self.cls = int(cls)
        def forward(self, x):
            out = torch.zeros((x.shape[0], 2), device=x.device)
            out[:, self.cls] = 1.0
            return out
    ds = CyberTensorDataset(
        np.zeros((10, 3), dtype=np.float32),
        np.ones(10, dtype=np.int64),
    )
    models = {0: Constant(1), 1: Constant(0)}
    m = shared_challenge_group_metrics(
        models, ds, 'cpu', attack_label=1, benign_label=0,
        exposed_client_ids={0}, eligible_client_ids={0,1},
    )
    assert m['emerging_recall_exposed'] == 1.0
    assert m['emerging_recall_unexposed'] == 0.0
    assert m['emerging_recall_all'] == 0.5


def test_pattern_balanced_challenge_recall():
    from utils.cyber_dataset import CyberTensorDataset
    from utils.cyber_metrics import shared_challenge_group_metrics

    class FeatureRule(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros((x.shape[0], 2), device=x.device)
            predicted_attack = x[:, 0] < 0.5
            out[:, 0] = (~predicted_attack).float()
            out[:, 1] = predicted_attack.float()
            return out

    # Row-weighted recall is 2/3 because pattern 0 is duplicated. Giving each
    # distinct pattern one representative yields pattern-balanced recall 1/2.
    row_weighted = CyberTensorDataset(
        np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32),
        np.ones(3, dtype=np.int64),
    )
    pattern_balanced = CyberTensorDataset(
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.ones(2, dtype=np.int64),
    )
    metrics = shared_challenge_group_metrics(
        {0: FeatureRule()},
        row_weighted,
        'cpu',
        attack_label=1,
        benign_label=0,
        exposed_client_ids=set(),
        eligible_client_ids={0},
        pattern_balanced_dataset=pattern_balanced,
    )
    assert np.isclose(metrics['emerging_recall_unexposed'], 2.0 / 3.0)
    assert np.isclose(
        metrics['emerging_pattern_balanced_recall_unexposed'], 0.5
    )


def test_calibrated_target_alert_metrics():
    from utils.cyber_dataset import CyberTensorDataset
    from utils.cyber_metrics import calibrated_target_alert_group_metrics

    class ScoreModel(torch.nn.Module):
        def forward(self, x):
            # Softmax target probability is monotonic in x[:, 0].
            return torch.stack([torch.zeros_like(x[:, 0]), x[:, 0]], dim=1)

    def dataset(scores, labels):
        return CyberTensorDataset(
            np.asarray(scores, dtype=np.float32).reshape(-1, 1),
            np.asarray(labels, dtype=np.int64),
        )

    metrics = calibrated_target_alert_group_metrics(
        {0: ScoreModel()},
        dataset([0.1, 0.2, 0.3, 0.4], [0, 2, 0, 2]),
        dataset([0.15, 0.35], [0, 2]),
        dataset([0.8, 0.9], [1, 1]),
        device='cpu',
        target_label=1,
        benign_label=0,
        eligible_client_ids={0},
        pattern_balanced_positive_dataset=dataset([0.25, 0.8], [1, 1]),
        calibration_fpr_limits=(0.0, 0.25),
        primary_calibration_fpr_limit=0.25,
        test_fpr_limit=0.50,
        fpr_confidence=0.95,
    )
    assert metrics['target_alert_calibration_fpr_unexposed'] <= 0.25
    assert metrics['target_alert_recall_unexposed'] == 1.0
    assert metrics['target_alert_pattern_balanced_recall_unexposed'] == 0.5
    assert metrics['target_alert_target_vs_rest_fpr_unexposed'] == 0.5
    assert metrics['target_alert_benign_to_target_fpr_unexposed'] == 0.0
    assert metrics['target_alert_nontarget_attack_to_target_fpr_unexposed'] == 1.0
    assert np.isclose(metrics['target_alert_precision_unexposed'], 2.0 / 3.0)
    assert metrics['target_alert_target_vs_rest_fp_sum_unexposed'] == 1
    assert metrics['target_alert_target_vs_rest_n_sum_unexposed'] == 2
    assert metrics['target_alert_target_vs_rest_fpr_upper_unexposed'] > 0.5
    assert metrics['target_alert_calibration_000bp_recall_unexposed'] <= 1.0
    assert metrics['target_alert_calibration_2500bp_recall_unexposed'] == 1.0


def test_target_alert_tracker_requires_recall_and_fpr():
    from utils.cyber_metrics import TargetAlertTracker

    tracker = TargetAlertTracker(
        drift_round=10,
        recall_target=0.90,
        fpr_limit=0.01,
        required_consecutive_rounds=3,
        prefix='target_alert_empirical',
    )
    tracker.update(10, .95, .02, 14, .30, 3000, 3000, 2900)
    assert np.isnan(tracker.time_to_target)
    tracker.update(11, .95, .01, 14, .30, 6000, 6000, 5800)
    assert np.isnan(tracker.time_to_target)
    tracker.update(12, .95, .01, 14, .30, 9000, 9000, 8700)
    assert np.isnan(tracker.time_to_target)
    tracker.update(13, .89, .00, 14, .30, 12000, 12000, 11600)
    assert tracker.current_streak == 0
    tracker.update(14, .95, .01, 14, .30, 15000, 15000, 14500)
    tracker.update(15, .95, .01, 14, .30, 18000, 18000, 17400)
    tracker.update(16, .95, .01, 14, .30, 21000, 21000, 20300)
    assert tracker.time_to_target == 6
    result = tracker.as_dict()
    assert result['target_alert_empirical_target_round'] == 16
    assert result['target_alert_empirical_current_streak'] == 3


if __name__=='__main__':
    test_deep_update(); test_same_initialization(); test_schedule_reproducible(); test_early_warning_tracker(); test_matched_challenge_groups(); test_pattern_balanced_challenge_recall(); test_calibrated_target_alert_metrics(); test_target_alert_tracker_requires_recall_and_fpr()
    print('shared cyber framework tests: PASS')
