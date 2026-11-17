"""Round-wise cybersecurity metrics for the FedCCFA drift study."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import csv
import json
import math

import numpy as np
import torch
from scipy.stats import beta
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model, dataset, device: str, batch_size: int = 4096):
    if dataset is None or len(dataset) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    model.eval()
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.argmax(logits, dim=1).detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        preds.append(p)
    return np.concatenate(ys), np.concatenate(preds)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_label: int,
    benign_label: int = 0,
) -> Dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": math.nan,
            "macro_f1": math.nan,
            "weighted_f1": math.nan,
            "attack_recall": math.nan,
            "attack_precision": math.nan,
            "attack_f1": math.nan,
            "benign_fpr": math.nan,
            "attack_tp": 0,
            "attack_fn": 0,
            "benign_fp": 0,
            "benign_n": 0,
        }

    attack_mask = y_true == int(attack_label)
    benign_mask = y_true == int(benign_label)
    tp = int(np.sum(attack_mask & (y_pred == attack_label)))
    fn = int(np.sum(attack_mask & (y_pred != attack_label)))
    benign_fp = int(np.sum(benign_mask & (y_pred != benign_label)))
    benign_n = int(np.sum(benign_mask))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true == attack_label,
        y_pred == attack_label,
        average="binary",
        zero_division=0,
    )
    # Balanced accuracy = macro-average recall over classes present in y_true.
    _ba_classes = np.unique(y_true)
    _ba_recalls = [
        float(np.mean(y_pred[y_true == c] == c))
        for c in _ba_classes
        if np.any(y_true == c)
    ]
    balanced_accuracy = float(np.mean(_ba_recalls)) if _ba_recalls else float('nan')
    return {
        'balanced_accuracy': balanced_accuracy,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "attack_recall": float(recall),
        "attack_precision": float(precision),
        "attack_f1": float(f1),
        "benign_fpr": float(benign_fp / benign_n) if benign_n else math.nan,
        "attack_tp": tp,
        "attack_fn": fn,
        "benign_fp": benign_fp,
        "benign_n": benign_n,
    }


def evaluate_model(
    model,
    dataset,
    device: str,
    attack_label: int,
    benign_label: int = 0,
    batch_size: int = 4096,
):
    y, p = predict(model, dataset, device=device, batch_size=batch_size)
    return classification_metrics(y, p, attack_label=attack_label, benign_label=benign_label)


def prequential_metrics(
    clients: Sequence,
    device: str,
    attack_label: int,
    benign_label: int = 0,
) -> Tuple[Dict[str, float | int], Dict[int, Dict[str, float | int]]]:
    """Test every client's current incoming window *before* local training."""
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    per_client: Dict[int, Dict[str, float | int]] = {}

    for client in clients:
        y, p = predict(client.model, client.train_set, device=device)
        per_client[int(client.id)] = classification_metrics(
            y, p, attack_label=attack_label, benign_label=benign_label
        )
        if len(y):
            all_y.append(y)
            all_p.append(p)

    if not all_y:
        return classification_metrics(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            attack_label=attack_label, benign_label=benign_label,
        ), per_client

    global_metrics = classification_metrics(
        np.concatenate(all_y),
        np.concatenate(all_p),
        attack_label=attack_label,
        benign_label=benign_label,
    )
    return global_metrics, per_client


def mean_personalized_global_metrics(
    clients: Sequence,
    dataset,
    device: str,
    attack_label: int,
    benign_label: int = 0,
    only_client_ids: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """Evaluate each personalized client model on the same fixed set, then average."""
    ids = None if only_client_ids is None else set(int(i) for i in only_client_ids)
    rows = []
    for client in clients:
        if ids is not None and int(client.id) not in ids:
            continue
        rows.append(evaluate_model(
            client.model, dataset, device, attack_label, benign_label
        ))
    if not rows:
        return {
            "accuracy": math.nan, "macro_f1": math.nan, "weighted_f1": math.nan,
            "attack_recall": math.nan, "attack_precision": math.nan,
            "attack_f1": math.nan, "benign_fpr": math.nan,
        }
    keys = [
        "accuracy", "macro_f1", "weighted_f1", "attack_recall",
        'balanced_accuracy',
        "attack_precision", "attack_f1", "benign_fpr",
    ]
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        out[k] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else math.nan
    return out


def early_warning_score(
    clients: Sequence,
    attack_challenge_set,
    unexposed_client_ids: Iterable[int],
    device: str,
    attack_label: int,
    benign_label: int = 0,
) -> Dict[str, float]:
    """Attack recall/F1 on clients that have never trained on the emerging threat."""
    ids = set(int(i) for i in unexposed_client_ids)
    rows = []
    for client in clients:
        if int(client.id) not in ids:
            continue
        rows.append(evaluate_model(
            client.model,
            attack_challenge_set,
            device,
            attack_label,
            benign_label,
        ))
    if not rows:
        return {"early_warning_recall": math.nan, "early_warning_f1": math.nan}
    return {
        "early_warning_recall": float(np.mean([r["attack_recall"] for r in rows])),
        "early_warning_f1": float(np.mean([r["attack_f1"] for r in rows])),
    }


def cluster_agreement(
    client_ids: Sequence[int],
    merged_groups: Sequence[Sequence[int]],
    exposed_clients: Set[int],
) -> Dict[str, float]:
    """Compare inferred grouping to binary exposed/unexposed state.

    FedCCFA omits singleton DBSCAN groups in its published implementation. We
    assign each omitted client its own singleton cluster for a complete
    partition before computing ARI/NMI.
    """
    ids = [int(i) for i in client_ids]
    if len(ids) < 2:
        return {"cluster_ari": math.nan, "cluster_nmi": math.nan}

    pred = {}
    cluster_id = 0
    for group in merged_groups:
        for cid in group:
            pred[int(cid)] = cluster_id
        cluster_id += 1
    for cid in ids:
        if cid not in pred:
            pred[cid] = cluster_id
            cluster_id += 1

    y_pred = [pred[cid] for cid in ids]
    y_true = [1 if cid in exposed_clients else 0 for cid in ids]
    if len(set(y_true)) < 2:
        return {"cluster_ari": math.nan, "cluster_nmi": math.nan}
    return {
        "cluster_ari": float(adjusted_rand_score(y_true, y_pred)),
        "cluster_nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }


class CyberMetricLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: List[dict] = []

    def append(self, row: dict) -> None:
        self.rows.append(dict(row))

    def flush(self) -> None:
        if not self.rows:
            return
        keys = []
        seen = set()
        for row in self.rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)


def time_to_recovery(
    rounds: np.ndarray,
    values: np.ndarray,
    drift_round: int,
    target: float = 0.90,
) -> float:
    mask = rounds >= int(drift_round)
    for r, v in zip(rounds[mask], values[mask]):
        if np.isfinite(v) and v >= target:
            return float(r - drift_round)
    return math.nan


def recovery_to_predrift(
    rounds: np.ndarray,
    values: np.ndarray,
    drift_round: int,
    reference_window: int = 5,
    tolerance: float = 0.02,
) -> Dict[str, float]:
    pre_mask = (rounds < drift_round) & (rounds >= drift_round - reference_window)
    post_mask = rounds >= drift_round
    if not np.any(pre_mask) or not np.any(post_mask):
        return {
            "predrift_value": math.nan,
            "max_drop": math.nan,
            "time_to_predrift_recovery": math.nan,
            "cumulative_recovery_loss": math.nan,
        }
    pre = float(np.nanmean(values[pre_mask]))
    post_rounds = rounds[post_mask]
    post_values = values[post_mask]
    finite_post = post_values[np.isfinite(post_values)]
    max_drop = pre - float(np.min(finite_post)) if len(finite_post) else math.nan
    ttr = math.nan
    for r, v in zip(post_rounds, post_values):
        if np.isfinite(v) and v >= pre - tolerance:
            ttr = float(r - drift_round)
            break
    loss = float(np.nansum(np.maximum(0.0, pre - post_values)))
    return {
        "predrift_value": pre,
        "max_drop": max_drop,
        "time_to_predrift_recovery": ttr,
        "cumulative_recovery_loss": loss,
    }


def shared_challenge_group_metrics(
    models_by_client: Dict[int, object],
    dataset,
    device: str,
    attack_label: int,
    benign_label: int = 0,
    exposed_client_ids: Iterable[int] = (),
    eligible_client_ids: Optional[Iterable[int]] = None,
    pattern_balanced_dataset=None,
) -> Dict[str, float | int]:
    """Evaluate every site model once on the same emerging-threat challenge set.

    Returns matched emerging-threat metrics for all, exposed, and unexposed
    clients. This avoids comparing early-warning recall on the HOIC challenge
    against a different global DDoS evaluation distribution.
    """
    exposed = set(int(i) for i in exposed_client_ids)
    eligible = (
        set(int(i) for i in models_by_client)
        if eligible_client_ids is None
        else set(int(i) for i in eligible_client_ids)
    )
    per_client: Dict[int, Dict[str, float | int]] = {}
    pattern_per_client: Dict[int, Dict[str, float | int]] = {}
    for cid, model in models_by_client.items():
        cid = int(cid)
        if cid not in eligible:
            continue
        per_client[cid] = evaluate_model(
            model, dataset, device, attack_label, benign_label
        )
        if pattern_balanced_dataset is not None:
            pattern_per_client[cid] = evaluate_model(
                model,
                pattern_balanced_dataset,
                device,
                attack_label,
                benign_label,
            )

    def aggregate(ids: Iterable[int], prefix: str) -> Dict[str, float | int]:
        ids = [int(i) for i in ids if int(i) in per_client]
        if not ids:
            return {
                f"emerging_recall_{prefix}": math.nan,
                f"emerging_f1_{prefix}": math.nan,
                f"emerging_models_{prefix}": 0,
            }
        recalls = np.asarray([per_client[i]["attack_recall"] for i in ids], dtype=np.float64)
        f1s = np.asarray([per_client[i]["attack_f1"] for i in ids], dtype=np.float64)
        return {
            f"emerging_recall_{prefix}": float(np.nanmean(recalls)),
            f"emerging_f1_{prefix}": float(np.nanmean(f1s)),
            f"emerging_models_{prefix}": int(len(ids)),
        }

    all_ids = sorted(eligible)
    exposed_ids = sorted(eligible & exposed)
    unexposed_ids = sorted(eligible - exposed)
    out: Dict[str, float | int] = {}
    out.update(aggregate(all_ids, "all"))
    out.update(aggregate(exposed_ids, "exposed"))
    out.update(aggregate(unexposed_ids, "unexposed"))
    if pattern_balanced_dataset is not None:
        def aggregate_pattern(ids: Iterable[int], prefix: str) -> None:
            values = [
                pattern_per_client[int(i)]["attack_recall"]
                for i in ids
                if int(i) in pattern_per_client
            ]
            out[f"emerging_pattern_balanced_recall_{prefix}"] = (
                float(np.nanmean(np.asarray(values, dtype=np.float64)))
                if values else math.nan
            )

        aggregate_pattern(all_ids, "all")
        aggregate_pattern(exposed_ids, "exposed")
        aggregate_pattern(unexposed_ids, "unexposed")
    # Backward-compatible names used by existing analysis scripts.
    out["early_warning_recall"] = out["emerging_recall_unexposed"]
    out["early_warning_f1"] = out["emerging_f1_unexposed"]
    if pattern_balanced_dataset is not None:
        out["early_warning_pattern_balanced_recall"] = out[
            "emerging_pattern_balanced_recall_unexposed"
        ]
    return out


@torch.no_grad()
def predict_target_scores(
    model,
    dataset,
    device: str,
    target_label: int,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and softmax scores for one target family."""
    if dataset is None or len(dataset) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
    )
    model.eval()
    ys: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for x, y in loader:
        logits = model(x.to(device))
        probability = torch.softmax(logits, dim=1)[:, int(target_label)]
        ys.append(y.detach().cpu().numpy())
        scores.append(probability.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores).astype(np.float64)


def conservative_fpr_threshold(
    negative_scores: np.ndarray,
    fpr_limit: float,
) -> float:
    """Choose a strict-score threshold with empirical FPR at most the limit.

    Alerts use ``score > threshold``.  Selecting the (allowed+1)-th largest
    calibration score makes the empirical guarantee conservative, including
    in the presence of tied scores.
    """
    scores = np.asarray(negative_scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if not 0.0 <= float(fpr_limit) < 1.0:
        raise ValueError("fpr_limit must be in [0,1)")
    if len(scores) == 0:
        return math.nan
    allowed = int(np.floor(float(fpr_limit) * len(scores)))
    if allowed <= 0:
        return float(np.max(scores))
    ordered = np.sort(scores)
    return float(ordered[-(allowed + 1)])


def clopper_pearson_upper(
    false_positives: int,
    trials: int,
    confidence: float = 0.95,
) -> float:
    """One-sided exact binomial upper confidence bound."""
    k = int(false_positives)
    n = int(trials)
    if n <= 0 or k < 0 or k > n:
        return math.nan
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0,1)")
    if k == n:
        return 1.0
    return float(beta.ppf(float(confidence), k + 1, n - k))


def _calibration_budget_tag(limit: float) -> str:
    """Stable column tag in basis points (0.5% -> 50bp)."""
    return f"{int(round(float(limit) * 10_000)):03d}bp"


def calibrated_target_alert_group_metrics(
    models_by_client: Dict[int, object],
    calibration_negative_dataset,
    test_negative_dataset,
    positive_dataset,
    device: str,
    target_label: int,
    benign_label: int = 0,
    exposed_client_ids: Iterable[int] = (),
    eligible_client_ids: Optional[Iterable[int]] = None,
    pattern_balanced_positive_dataset=None,
    calibration_fpr_limits: Sequence[float] = (0.0025, 0.005, 0.01),
    primary_calibration_fpr_limit: float = 0.005,
    test_fpr_limit: float = 0.01,
    fpr_confidence: float = 0.95,
) -> Dict[str, float | int]:
    """Evaluate a calibrated target-family alert on a fixed mixed challenge.

    Each deployed client model receives its own threshold, estimated only from
    held-out target-negative calibration rows.  The threshold is then frozen
    while recall, precision, AUROC/AUPRC and false-alert rates are measured on
    disjoint positives and target-negative test rows.
    """
    exposed = set(int(i) for i in exposed_client_ids)
    eligible = (
        set(int(i) for i in models_by_client)
        if eligible_client_ids is None
        else set(int(i) for i in eligible_client_ids)
    )
    budgets = tuple(sorted(set(float(x) for x in calibration_fpr_limits)))
    if not budgets or any(not 0.0 <= x < 1.0 for x in budgets):
        raise ValueError("calibration_fpr_limits must be in [0,1)")
    primary = float(primary_calibration_fpr_limit)
    if not any(np.isclose(primary, x, rtol=0.0, atol=1e-12) for x in budgets):
        raise ValueError("primary calibration limit must be in calibration_fpr_limits")
    if not 0.0 <= float(test_fpr_limit) < 1.0:
        raise ValueError("test_fpr_limit must be in [0,1)")

    per_client: Dict[int, Dict[float, Dict[str, float | int]]] = {}
    for raw_cid, model in models_by_client.items():
        cid = int(raw_cid)
        if cid not in eligible:
            continue
        _, calibration_scores = predict_target_scores(
            model,
            calibration_negative_dataset,
            device,
            target_label,
        )
        negative_y, negative_scores = predict_target_scores(
            model,
            test_negative_dataset,
            device,
            target_label,
        )
        _, positive_scores = predict_target_scores(
            model,
            positive_dataset,
            device,
            target_label,
        )
        benign_mask = negative_y == int(benign_label)
        hard_negative_mask = negative_y != int(benign_label)
        mixed_y = np.concatenate([
            np.ones(len(positive_scores), dtype=np.int64),
            np.zeros(len(negative_scores), dtype=np.int64),
        ])
        mixed_scores = np.concatenate([positive_scores, negative_scores])
        pattern_scores = np.empty(0, dtype=np.float64)
        if pattern_balanced_positive_dataset is not None:
            _, pattern_scores = predict_target_scores(
                model,
                pattern_balanced_positive_dataset,
                device,
                target_label,
            )
        auroc = float(roc_auc_score(mixed_y, mixed_scores))
        auprc = float(average_precision_score(mixed_y, mixed_scores))
        per_client[cid] = {}
        for budget in budgets:
            threshold = conservative_fpr_threshold(calibration_scores, budget)
            calibration_alerts = calibration_scores > threshold
            negative_alerts = negative_scores > threshold
            positive_alerts = positive_scores > threshold
            pattern_alerts = pattern_scores > threshold
            calibration_fp = int(np.sum(calibration_alerts))
            test_fp = int(np.sum(negative_alerts))
            benign_fp = int(np.sum(negative_alerts[benign_mask]))
            nontarget_fp = int(np.sum(negative_alerts[hard_negative_mask]))
            tp = int(np.sum(positive_alerts))
            test_n = int(len(negative_scores))
            benign_n = int(np.sum(benign_mask))
            nontarget_n = int(np.sum(hard_negative_mask))
            positive_n = int(len(positive_scores))
            per_client[cid][budget] = {
                "threshold": float(threshold),
                "calibration_fp": calibration_fp,
                "calibration_n": int(len(calibration_scores)),
                "calibration_fpr": float(np.mean(calibration_alerts)),
                "positive_tp": tp,
                "positive_n": positive_n,
                "recall": float(tp / positive_n) if positive_n else math.nan,
                "pattern_balanced_recall": (
                    float(np.mean(pattern_alerts)) if len(pattern_scores) else math.nan
                ),
                "target_vs_rest_fp": test_fp,
                "target_vs_rest_n": test_n,
                "target_vs_rest_fpr": float(test_fp / test_n) if test_n else math.nan,
                "target_vs_rest_fpr_upper": clopper_pearson_upper(
                    test_fp, test_n, fpr_confidence
                ),
                "benign_to_target_fp": benign_fp,
                "benign_to_target_n": benign_n,
                "benign_to_target_fpr": (
                    float(benign_fp / benign_n) if benign_n else math.nan
                ),
                "benign_to_target_fpr_upper": clopper_pearson_upper(
                    benign_fp, benign_n, fpr_confidence
                ),
                "nontarget_attack_to_target_fp": nontarget_fp,
                "nontarget_attack_to_target_n": nontarget_n,
                "nontarget_attack_to_target_fpr": (
                    float(nontarget_fp / nontarget_n)
                    if nontarget_n else math.nan
                ),
                "nontarget_attack_to_target_fpr_upper": clopper_pearson_upper(
                    nontarget_fp, nontarget_n, fpr_confidence
                ),
                "precision": float(tp / (tp + test_fp)) if tp + test_fp else 0.0,
                "auroc": auroc,
                "auprc": auprc,
            }

    def aggregate(
        ids: Iterable[int], suffix: str, budget: float, key_prefix: str
    ) -> Dict[str, float | int]:
        selected = [
            per_client[int(i)][budget]
            for i in ids if int(i) in per_client
        ]
        out: Dict[str, float | int] = {
            f"{key_prefix}models_{suffix}": int(len(selected))
        }
        mean_names = (
            "threshold",
            "calibration_fpr",
            "recall",
            "pattern_balanced_recall",
            "target_vs_rest_fpr",
            "target_vs_rest_fpr_upper",
            "benign_to_target_fpr",
            "benign_to_target_fpr_upper",
            "nontarget_attack_to_target_fpr",
            "nontarget_attack_to_target_fpr_upper",
            "precision",
            "auroc",
            "auprc",
        )
        for name in mean_names:
            values = np.asarray(
                [row[name] for row in selected], dtype=np.float64
            )
            out[f"{key_prefix}{name}_{suffix}"] = (
                float(np.nanmean(values))
                if np.any(np.isfinite(values)) else math.nan
            )
        upper_values = np.asarray(
            [row["target_vs_rest_fpr_upper"] for row in selected],
            dtype=np.float64,
        )
        out[f"{key_prefix}target_vs_rest_fpr_upper_max_{suffix}"] = (
            float(np.nanmax(upper_values))
            if np.any(np.isfinite(upper_values)) else math.nan
        )
        count_names = (
            "calibration_fp", "calibration_n", "positive_tp", "positive_n",
            "target_vs_rest_fp", "target_vs_rest_n",
            "benign_to_target_fp", "benign_to_target_n",
            "nontarget_attack_to_target_fp", "nontarget_attack_to_target_n",
        )
        for name in count_names:
            out[f"{key_prefix}{name}_sum_{suffix}"] = int(
                sum(int(row[name]) for row in selected)
            )
        return out

    all_ids = sorted(eligible)
    exposed_ids = sorted(eligible & exposed)
    unexposed_ids = sorted(eligible - exposed)
    result: Dict[str, float | int] = {
        "target_alert_primary_calibration_fpr_limit": primary,
        "target_alert_test_fpr_limit": float(test_fpr_limit),
        "target_alert_fpr_limit": float(test_fpr_limit),
        "target_alert_fpr_confidence": float(fpr_confidence),
        "target_alert_calibration_fpr_limits": json.dumps(budgets),
        "target_alert_n_calibration_negatives": int(
            len(calibration_negative_dataset)
        ),
        "target_alert_n_test_negatives": int(len(test_negative_dataset)),
        "target_alert_n_test_positives": int(len(positive_dataset)),
    }
    groups = (
        (all_ids, "all"),
        (exposed_ids, "exposed"),
        (unexposed_ids, "unexposed"),
    )
    for budget in budgets:
        budget_prefix = (
            f"target_alert_calibration_{_calibration_budget_tag(budget)}_"
        )
        for ids, suffix in groups:
            result.update(aggregate(ids, suffix, budget, budget_prefix))
        if np.isclose(budget, primary, rtol=0.0, atol=1e-12):
            for ids, suffix in groups:
                result.update(aggregate(ids, suffix, budget, "target_alert_"))
    return result


@dataclass
class EarlyWarningTracker:
    """Track the first round at which unexposed-site recall reaches a target."""

    drift_round: int
    target: float = 0.90
    first_round: Optional[int] = None
    prevalence_at_first: float = math.nan
    cumulative_presentations_at_first: float = math.nan
    cumulative_unique_at_first: float = math.nan
    cumulative_feature_patterns_at_first: float = math.nan

    def update(
        self,
        round_id: int,
        recall: float,
        n_unexposed: int,
        prevalence: float,
        cumulative_presentations: int,
        cumulative_unique: int,
        cumulative_feature_patterns: int | None = None,
    ) -> None:
        if self.first_round is not None:
            return
        if int(round_id) < int(self.drift_round) or int(n_unexposed) <= 0:
            return
        if np.isfinite(recall) and float(recall) >= float(self.target):
            self.first_round = int(round_id)
            self.prevalence_at_first = float(prevalence)
            self.cumulative_presentations_at_first = float(cumulative_presentations)
            self.cumulative_unique_at_first = float(cumulative_unique)
            self.cumulative_feature_patterns_at_first = float(
                cumulative_unique
                if cumulative_feature_patterns is None
                else cumulative_feature_patterns
            )

    @property
    def time_to_target(self) -> float:
        if self.first_round is None:
            return math.nan
        return float(self.first_round - int(self.drift_round))

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "early_warning_target": float(self.target),
            "early_warning_target_round": (
                math.nan if self.first_round is None else int(self.first_round)
            ),
            "time_to_early_warning_target": self.time_to_target,
            "early_warning_threshold_prevalence": self.prevalence_at_first,
            "early_warning_threshold_cumulative_presentations": self.cumulative_presentations_at_first,
            "early_warning_threshold_cumulative_distinct_records": self.cumulative_unique_at_first,
            "early_warning_threshold_cumulative_distinct_feature_patterns": self.cumulative_feature_patterns_at_first,
            # Deprecated compatibility alias: historically "unique" meant a
            # distinct prepared-data record index, not a feature pattern.
            "early_warning_threshold_cumulative_unique": self.cumulative_unique_at_first,
        }


@dataclass
class TargetAlertTracker:
    """Track sustained recall/FPR qualification over consecutive rounds."""

    drift_round: int
    recall_target: float = 0.90
    fpr_limit: float = 0.01
    required_consecutive_rounds: int = 3
    prefix: str = "target_alert_empirical"
    first_round: Optional[int] = None
    streak_start_round: Optional[int] = None
    current_streak: int = 0
    qualified_this_round: bool = False
    prevalence_at_first: float = math.nan
    cumulative_presentations_at_first: float = math.nan
    cumulative_unique_at_first: float = math.nan
    cumulative_feature_patterns_at_first: float = math.nan

    def update(
        self,
        round_id: int,
        recall: float,
        false_positive_rate: float,
        n_unexposed: int,
        prevalence: float,
        cumulative_presentations: int,
        cumulative_unique: int,
        cumulative_feature_patterns: int,
    ) -> None:
        if int(round_id) < int(self.drift_round) or int(n_unexposed) <= 0:
            self.current_streak = 0
            self.streak_start_round = None
            self.qualified_this_round = False
            return
        qualified = (
            np.isfinite(recall)
            and np.isfinite(false_positive_rate)
            and float(recall) >= float(self.recall_target)
            and float(false_positive_rate) <= float(self.fpr_limit)
        )
        self.qualified_this_round = bool(qualified)
        if not qualified:
            self.current_streak = 0
            self.streak_start_round = None
            return
        if self.current_streak == 0:
            self.streak_start_round = int(round_id)
        self.current_streak += 1
        if (
            self.first_round is None
            and self.current_streak >= int(self.required_consecutive_rounds)
        ):
            # Detection occurs when the required streak has been observed.
            self.first_round = int(round_id)
            self.prevalence_at_first = float(prevalence)
            self.cumulative_presentations_at_first = float(cumulative_presentations)
            self.cumulative_unique_at_first = float(cumulative_unique)
            self.cumulative_feature_patterns_at_first = float(
                cumulative_feature_patterns
            )

    @property
    def time_to_target(self) -> float:
        if self.first_round is None:
            return math.nan
        return float(self.first_round - int(self.drift_round))

    def as_dict(self) -> Dict[str, float | int]:
        prefix = str(self.prefix)
        return {
            f"{prefix}_recall_target": float(self.recall_target),
            f"{prefix}_fpr_limit": float(self.fpr_limit),
            f"{prefix}_required_consecutive_rounds": int(
                self.required_consecutive_rounds
            ),
            f"{prefix}_target_round": (
                math.nan if self.first_round is None else int(self.first_round)
            ),
            f"time_to_{prefix}": self.time_to_target,
            f"{prefix}_streak_start_round": (
                math.nan
                if self.streak_start_round is None
                else int(self.streak_start_round)
            ),
            f"{prefix}_current_streak": int(self.current_streak),
            f"{prefix}_qualified_this_round": int(self.qualified_this_round),
            f"{prefix}_threshold_prevalence": self.prevalence_at_first,
            f"{prefix}_threshold_cumulative_presentations": self.cumulative_presentations_at_first,
            f"{prefix}_threshold_cumulative_distinct_records": self.cumulative_unique_at_first,
            f"{prefix}_threshold_cumulative_distinct_feature_patterns": self.cumulative_feature_patterns_at_first,
        }

# === BALANCED_ACCURACY_WRAPPER_V1 ===
_classification_metrics_before_balanced_accuracy = classification_metrics


def classification_metrics(y_true, y_pred, *args, **kwargs):
    metrics = _classification_metrics_before_balanced_accuracy(
        y_true, y_pred, *args, **kwargs
    )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    recalls = []
    for cls in np.unique(y_true_arr):
        mask = y_true_arr == cls
        denom = int(np.sum(mask))
        if denom > 0:
            recalls.append(
                float(np.mean(y_pred_arr[mask] == cls))
            )

    metrics["balanced_accuracy"] = (
        float(np.mean(recalls))
        if recalls
        else float("nan")
    )
    return metrics
