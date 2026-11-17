#!/usr/bin/env python3
"""Shared prequential cybersecurity runner for FedCCFA repository methods.

Supported: FedAvg, FedDrift, FedCCFA (clustered or global-classifier sharing).

Run from repository root:
  export PYTHONPATH=.
  python methods/run_cyber.py --algorithm FedAvg
  python methods/run_cyber.py --algorithm FedCCFA
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cyber_adapters import build_adapter
from utils.cyber_config import load_cyber_config
from utils.cyber_dataset import load_prepared_cyber_data
from utils.cyber_factory import classes_for_algorithm
from utils.cyber_metrics import (
    calibrated_target_alert_group_metrics,
    CyberMetricLogger,
    EarlyWarningTracker,
    TargetAlertTracker,
    cluster_agreement,
    mean_personalized_global_metrics,
    prequential_metrics,
    shared_challenge_group_metrics,
)
from utils.cyber_model_bridge import make_cyber_model, state_dict_cpu
from utils.cyber_schedule import ParticipationScheduler, ThreatPropagationSchedule
from utils.cyber_stream import CyberStreamManager


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_label(prepared, family: str) -> int:
    needle = family.strip().lower()
    for i, name in enumerate(prepared.class_names):
        if name.lower() == needle:
            return int(i)
    raise ValueError(f'Label {family!r} absent; available={prepared.class_names}')


def resolve_variant(prepared, variant):
    if variant is None or str(variant).strip().lower() in {'', 'none', 'null'}:
        return None
    needle = str(variant).strip().lower()
    for i, name in enumerate(prepared.variant_names):
        if name.lower() == needle:
            return int(i)
    partial = [x for x in prepared.variant_names if needle in x.lower()]
    raise ValueError(f'Variant {variant!r} absent; partial matches={partial[:20]}')


def model_nbytes(model) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def fixed_schedule(args, threat, participation, eligible):
    rows = []
    for r in range(int(args['rounds'])):
        exposed = threat.exposed_clients(r)
        rows.append({
            'round': r,
            'prevalence': threat.prevalence(r),
            'exposed_clients': sorted(exposed),
            'selected_clients': participation.select(r, exposed, eligible),
        })
    return rows


def algorithm_site_models(adapter, clients):
    models = adapter.deployment_models()
    if not models:
        return {int(c.id): c.model for c in clients if hasattr(c, 'model')}
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, choices=['FedAvg','FedDrift','FedCCFA'])
    parser.add_argument('--config', default=None, help='Optional algorithm cyber YAML')
    parser.add_argument(
        '--experiment-config', default=None,
        help='Optional experiment overlay applied after the algorithm cyber YAML',
    )
    parser.add_argument('--repo-root', default=str(ROOT))
    cli = parser.parse_args()

    repo_root = Path(cli.repo_root).resolve()
    args = load_cyber_config(
        repo_root, cli.algorithm, cli.config, cli.experiment_config
    )
    seed_everything(int(args.get('seed', 0)))

    prepared = load_prepared_cyber_data(repo_root / args['prepared_data'] if not Path(args['prepared_data']).is_absolute() else args['prepared_data'])
    attack_label = resolve_label(prepared, args['threat']['attack_family'])
    benign_label = resolve_label(prepared, args.get('benign_family', 'benign'))
    variant_id = resolve_variant(prepared, args['threat'].get('emerging_variant'))
    args['input_dim'] = prepared.input_dim
    args['num_classes'] = prepared.num_classes
    # Runtime-resolved labels are exposed to algorithm adapters. This avoids
    # hard-coding CSE-CIC-IDS2018 class indices in the FedCCFA ablation.
    args['attack_label'] = int(attack_label)
    args['benign_label'] = int(benign_label)
    args['dataset'] = args.get('dataset_name_for_logs', 'CSE-CIC-IDS2018')

    stream_cfg = args['stream']
    stream = CyberStreamManager(
        prepared=prepared,
        num_clients=int(args['client_num']),
        rounds=int(args['rounds']),
        attack_label=attack_label,
        emerging_variant_id=variant_id,
        window_size=int(stream_cfg.get('window_size', 2048)),
        attack_mix_fraction=float(args['threat'].get('attack_mix_fraction', 0.20)),
        evidence_mode=str(args['threat'].get('evidence_mode', 'mix_fraction')),
        fixed_total_attack_per_round=args['threat'].get('fixed_total_attack_per_round'),
        local_test_fraction=float(stream_cfg.get('local_test_fraction', 0.15)),
        attack_guard_fraction=float(stream_cfg.get('attack_guard_fraction', 0.05)),
        attack_test_fraction=float(stream_cfg.get('attack_test_fraction', 0.30)),
        min_window_samples=int(stream_cfg.get('min_window_samples', 32)),
        max_global_eval_samples=stream_cfg.get('max_global_eval_samples', 20000),
        max_attack_challenge_samples=stream_cfg.get('max_attack_challenge_samples', 5000),
        max_target_alert_negative_samples=stream_cfg.get(
            'max_target_alert_negative_samples', 10000
        ),
        target_alert_calibration_fraction=stream_cfg.get(
            'target_alert_calibration_fraction', 0.50
        ),
        seed=int(args.get('stream_seed', args.get('seed', 0))),
    )
    if len(stream.eligible_clients) < 2:
        raise RuntimeError(f'Too few eligible clients: {stream.eligible_clients}')

    threat = ThreatPropagationSchedule(
        int(args['client_num']), args['threat']['milestones'],
        seed=int(args.get('schedule_seed', args.get('seed', 0))),
        client_ids=stream.eligible_clients,
    )
    part = args['participation']
    participation = ParticipationScheduler(
        num_clients=int(args['client_num']),
        normal_prob=float(part.get('normal_prob', args.get('sample_ratio', 1.0))),
        exposed_prob=float(part.get('exposed_prob', part.get('normal_prob', 1.0))),
        min_selected=int(part.get('min_selected', 3)),
        seed=int(args.get('participation_seed', args.get('seed', 0))),
    )
    schedule = fixed_schedule(args, threat, participation, stream.eligible_clients)

    ClientClass, ServerClass = classes_for_algorithm(cli.algorithm)
    initial_exposed = threat.exposed_clients(0)
    clients = []
    for cid in range(int(args['client_num'])):
        train_set, _ = stream.round_dataset(cid, 0, cid in initial_exposed)
        client = ClientClass(cid, args, train_set, stream.local_test_set(cid), 0)
        clients.append(client)
    server = ServerClass(args)
    adapter = build_adapter(cli.algorithm, args, clients, server)

    # One deterministic initialization independent of upstream constructors.
    init_seed = int(args.get('initial_model_seed', 424242))
    cpu_state = torch.random.get_rng_state()
    torch.manual_seed(init_seed)
    initial_model = make_cyber_model({**args, 'device': 'cpu'})
    initial_state = state_dict_cpu(initial_model)
    torch.random.set_rng_state(cpu_state)
    adapter.initialize(initial_state)

    result_dir = Path(args.get('result_dir', 'log/cyber'))
    if not result_dir.is_absolute():
        result_dir = repo_root / result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.get('run_name') or f"{cli.algorithm.lower()}_cyber_seed{args.get('seed',0)}"
    # Prevent one shared config run_name from overwriting another algorithm.
    if '{algorithm}' in str(run_name):
        run_name = str(run_name).format(algorithm=cli.algorithm.lower())
    elif not str(run_name).lower().startswith(cli.algorithm.lower()):
        run_name = f'{cli.algorithm.lower()}_{run_name}'

    round_log = CyberMetricLogger(result_dir / f'{run_name}_rounds.csv')
    client_log = CyberMetricLogger(result_dir / f'{run_name}_clients.csv')
    (result_dir / f'{run_name}_schedule.json').write_text(json.dumps(schedule, indent=2))
    model_bytes = model_nbytes(server.model)

    metadata = {
        'algorithm': cli.algorithm,
        'config': args,
        'class_names': prepared.class_names,
        'variant_names': prepared.variant_names,
        'attack_label': attack_label,
        'attack_family': prepared.class_names[attack_label],
        'emerging_variant_id': variant_id,
        'emerging_variant': prepared.variant_names[variant_id] if variant_id is not None else None,
        'benign_label': benign_label,
        'eligible_clients': stream.eligible_clients,
        'initial_model_seed': init_seed,
        'model_parameter_bytes': model_bytes,
        'first_drift_round': threat.first_drift_round,
        'n_global_eval': len(stream.global_eval_set),
        'n_global_eval_full': len(getattr(stream, 'global_eval_idx_full', stream.global_eval_idx)),
        'n_attack_challenge': len(stream.attack_challenge_set),
        'n_attack_challenge_full': len(stream.attack_test_idx),
        'n_attack_pattern_balanced_challenge': len(stream.attack_pattern_challenge_set),
        'n_attack_challenge_unique_feature_patterns_full': len(
            stream.attack_pattern_challenge_idx_full
        ),
        'emerging_split_audit': stream.emerging_split_audit,
        'target_alert_split_audit': stream.target_alert_split_audit,
        'n_target_alert_calibration_negatives': len(
            stream.target_alert_calibration_negative_set
        ),
        'n_target_alert_test_negatives': len(
            stream.target_alert_test_negative_set
        ),
        'evidence_mode': stream.evidence_mode,
        'evidence_identity_note': (
            'fixed_unique controls distinct prepared-data record indices; '
            'distinct processed feature patterns are logged separately'
        ),
        'fixed_total_attack_per_round': stream.fixed_total_attack_per_round,
        'fedccfa_sharing_mode': args.get('fedccfa_sharing_mode', 'clustered') if cli.algorithm == 'FedCCFA' else None,
        'communication_note': 'model-equivalent accounting; method-specific metadata may add bytes',
    }
    (result_dir / f'{run_name}_metadata.json').write_text(json.dumps(metadata, indent=2, default=str))

    print(json.dumps(metadata, indent=2, default=str), flush=True)
    print('[ready] entering federated rounds', flush=True)

    cumulative_incoming_presentations = 0
    cumulative_selected_presentations = 0
    cumulative_selected_distinct_records: set[int] = set()
    cumulative_selected_distinct_feature_patterns: set[bytes] = set()
    ew_target = float(args.get('evaluation', {}).get('early_warning_target', 0.90))
    ew_tracker = EarlyWarningTracker(
        drift_round=int(threat.first_drift_round if threat.first_drift_round is not None else args['rounds']),
        target=ew_target,
    )
    evaluation_cfg = args.get('evaluation', {})
    target_alert_calibration_limits = tuple(
        float(x) for x in evaluation_cfg.get(
            'target_alert_calibration_fpr_limits', [0.0025, 0.005, 0.01]
        )
    )
    target_alert_primary_calibration_limit = float(
        evaluation_cfg.get('target_alert_primary_calibration_fpr_limit', 0.005)
    )
    target_alert_test_fpr_limit = float(
        evaluation_cfg.get('target_alert_test_fpr_limit', 0.01)
    )
    target_alert_confidence = float(
        evaluation_cfg.get('target_alert_fpr_confidence', 0.95)
    )
    target_alert_sustained_rounds = int(
        evaluation_cfg.get('target_alert_sustained_rounds', 3)
    )
    target_alert_empirical_tracker = TargetAlertTracker(
        drift_round=int(
            threat.first_drift_round
            if threat.first_drift_round is not None else args['rounds']
        ),
        recall_target=ew_target,
        fpr_limit=target_alert_test_fpr_limit,
        required_consecutive_rounds=target_alert_sustained_rounds,
        prefix='target_alert_empirical',
    )
    target_alert_certified_tracker = TargetAlertTracker(
        drift_round=int(
            threat.first_drift_round
            if threat.first_drift_round is not None else args['rounds']
        ),
        recall_target=ew_target,
        fpr_limit=target_alert_test_fpr_limit,
        required_consecutive_rounds=target_alert_sustained_rounds,
        prefix='target_alert_certified',
    )

    for round_id in range(int(args['rounds'])):
        t0 = time.time()
        print(f'[{cli.algorithm}] round {round_id:03d}: prepare stream', flush=True)
        exposed = threat.exposed_clients(round_id)
        states = stream.apply_round(clients, round_id, exposed)

        # Prequential test-before-train on the exact incoming stream window.
        phase = time.time()
        print(f'[{cli.algorithm}] round {round_id:03d}: prequential evaluation', flush=True)
        preq, per_client = prequential_metrics(
            clients, device=args['device'], attack_label=attack_label, benign_label=benign_label
        )
        preq_seconds = time.time() - phase
        selected_ids = schedule[round_id]['selected_clients']
        selected = [c for c in clients if int(c.id) in set(selected_ids)]
        if not selected:
            raise RuntimeError(f'No selected clients at round {round_id}')

        for c in clients:
            cm = per_client[int(c.id)]
            state = states[int(c.id)]
            client_log.append({
                'algorithm': cli.algorithm,
                'round': round_id,
                'client_id': int(c.id),
                'exposed': int(c.id in exposed),
                'selected': int(c.id in selected_ids),
                'n_base': state.n_base,
                'n_emerging': state.n_attack,
                **{f'preq_{k}': v for k, v in cm.items()},
            })

        n_incoming_all = int(sum(s.n_attack for s in states.values()))
        selected_state_rows = [states[int(cid)] for cid in selected_ids if int(cid) in states]
        n_selected_emerging = int(sum(s.n_attack for s in selected_state_rows))
        round_selected_distinct_records = set()
        for state in selected_state_rows:
            round_selected_distinct_records.update(
                int(i) for i in state.attack_indices
            )
        cumulative_incoming_presentations += n_incoming_all

        train_start = time.time()
        print(f'[{cli.algorithm}] round {round_id:03d}: train {len(selected)} selected clients', flush=True)
        diagnostics = adapter.train_round(round_id, selected) or {}
        train_seconds = time.time() - train_start
        # Count evidence only after the algorithm successfully consumed the round.
        cumulative_selected_presentations += n_selected_emerging
        round_selected_distinct_feature_patterns = set(
            stream.feature_pattern_tokens(round_selected_distinct_records)
        )
        cumulative_selected_distinct_records.update(
            round_selected_distinct_records
        )
        cumulative_selected_distinct_feature_patterns.update(
            round_selected_distinct_feature_patterns
        )

        phase = time.time()
        print(f'[{cli.algorithm}] round {round_id:03d}: post-round global evaluation (n={len(stream.global_eval_set)})', flush=True)
        post = mean_personalized_global_metrics(
            clients, stream.global_eval_set, args['device'], attack_label, benign_label
        )
        post_seconds = time.time() - phase
        exposed_eligible = set(stream.eligible_clients) & set(exposed)
        unexposed = set(stream.eligible_clients) - set(exposed)
        phase = time.time()
        print(
            f'[{cli.algorithm}] round {round_id:03d}: matched emerging-threat evaluation '
            f'(all={len(stream.eligible_clients)}, exposed={len(exposed_eligible)}, '
            f'unexposed={len(unexposed)}, n={len(stream.attack_challenge_set)})',
            flush=True,
        )
        site_models = algorithm_site_models(adapter, clients)
        emerging = shared_challenge_group_metrics(
            site_models,
            stream.attack_challenge_set,
            args['device'],
            attack_label,
            benign_label,
            exposed_client_ids=exposed_eligible,
            eligible_client_ids=stream.eligible_clients,
            pattern_balanced_dataset=stream.attack_pattern_challenge_set,
        )
        target_alert = calibrated_target_alert_group_metrics(
            site_models,
            stream.target_alert_calibration_negative_set,
            stream.target_alert_test_negative_set,
            stream.attack_challenge_set,
            args['device'],
            target_label=attack_label,
            benign_label=benign_label,
            exposed_client_ids=exposed_eligible,
            eligible_client_ids=stream.eligible_clients,
            pattern_balanced_positive_dataset=(
                stream.attack_pattern_challenge_set
            ),
            calibration_fpr_limits=target_alert_calibration_limits,
            primary_calibration_fpr_limit=(
                target_alert_primary_calibration_limit
            ),
            test_fpr_limit=target_alert_test_fpr_limit,
            fpr_confidence=target_alert_confidence,
        )
        ews_seconds = time.time() - phase

        selected_exposed = set(int(i) for i in selected_ids) & exposed_eligible
        threat_visibility = (
            float(len(selected_exposed) / len(exposed_eligible))
            if exposed_eligible else np.nan
        )
        ew_tracker.update(
            round_id=round_id,
            recall=float(emerging['emerging_recall_unexposed']),
            n_unexposed=len(unexposed),
            prevalence=threat.prevalence(round_id),
            cumulative_presentations=cumulative_selected_presentations,
            cumulative_unique=len(cumulative_selected_distinct_records),
            cumulative_feature_patterns=len(
                cumulative_selected_distinct_feature_patterns
            ),
        )
        target_alert_empirical_tracker.update(
            round_id=round_id,
            recall=float(target_alert['target_alert_recall_unexposed']),
            false_positive_rate=float(
                target_alert['target_alert_target_vs_rest_fpr_unexposed']
            ),
            n_unexposed=len(unexposed),
            prevalence=threat.prevalence(round_id),
            cumulative_presentations=cumulative_selected_presentations,
            cumulative_unique=len(cumulative_selected_distinct_records),
            cumulative_feature_patterns=len(
                cumulative_selected_distinct_feature_patterns
            ),
        )
        target_alert_certified_tracker.update(
            round_id=round_id,
            recall=float(target_alert['target_alert_recall_unexposed']),
            false_positive_rate=float(
                target_alert['target_alert_target_vs_rest_fpr_upper_unexposed']
            ),
            n_unexposed=len(unexposed),
            prevalence=threat.prevalence(round_id),
            cumulative_presentations=cumulative_selected_presentations,
            cumulative_unique=len(cumulative_selected_distinct_records),
            cumulative_feature_patterns=len(
                cumulative_selected_distinct_feature_patterns
            ),
        )
        cluster = {'cluster_ari': np.nan, 'cluster_nmi': np.nan}
        if cli.algorithm == 'FedCCFA':
            merged = diagnostics.get('label_merged_dict', {})
            # Attack-class clustering is the most direct mechanism diagnostic.
            groups = merged.get(attack_label, []) if isinstance(merged, dict) else []
            cluster = cluster_agreement(selected_ids, groups, exposed)

        exposed_recalls = []
        for cid in exposed:
            cm = per_client.get(int(cid), {})
            if int(cm.get('attack_tp', 0)) + int(cm.get('attack_fn', 0)) > 0:
                exposed_recalls.append(float(cm['attack_recall']))

        row = {
            'algorithm': cli.algorithm,
            'round': round_id,
            'prevalence': threat.prevalence(round_id),
            'n_exposed': len(exposed),
            'n_selected': len(selected),
            'n_incoming_emerging': n_incoming_all,
            'n_selected_emerging': n_selected_emerging,
            'n_round_selected_distinct_emerging_records': len(
                round_selected_distinct_records
            ),
            'n_round_selected_distinct_emerging_feature_patterns': len(
                round_selected_distinct_feature_patterns
            ),
            'cumulative_incoming_emerging_presentations': cumulative_incoming_presentations,
            'cumulative_selected_emerging_presentations': cumulative_selected_presentations,
            'cumulative_selected_distinct_emerging_records': len(
                cumulative_selected_distinct_records
            ),
            'cumulative_selected_distinct_emerging_feature_patterns': len(
                cumulative_selected_distinct_feature_patterns
            ),
            # Deprecated aliases retained for existing analysis scripts.
            'n_round_selected_unique_emerging': len(
                round_selected_distinct_records
            ),
            'cumulative_selected_unique_emerging': len(
                cumulative_selected_distinct_records
            ),
            'n_selected_exposed': len(selected_exposed),
            'n_unexposed': len(unexposed),
            'threat_visibility': threat_visibility,
            **{f'preq_{k}': v for k, v in preq.items()},
            **{f'post_{k}': v for k, v in post.items()},
            **emerging,
            **ew_tracker.as_dict(),
            **target_alert,
            **target_alert_empirical_tracker.as_dict(),
            **target_alert_certified_tracker.as_dict(),
            **cluster,
            'worst_exposed_preq_attack_recall': float(np.min(exposed_recalls)) if exposed_recalls else np.nan,
            'p10_exposed_preq_attack_recall': float(np.quantile(exposed_recalls, .10)) if exposed_recalls else np.nan,
            'preq_seconds': preq_seconds,
            'train_seconds': train_seconds,
            'post_eval_seconds': post_seconds,
            'early_warning_seconds': ews_seconds,
            'round_seconds': time.time() - t0,
            'approx_model_exchange_bytes': int(adapter.communication_multiplier() * len(selected) * model_bytes),
            'adapter_path': diagnostics.get(
                'feddrift_path',
                diagnostics.get(
                    'aggregate_method',
                    ('FedCCFA-' + diagnostics.get('fedccfa_sharing_mode', 'clustered'))
                    if cli.algorithm == 'FedCCFA' else ''
                ),
            ),
            'feddrift_n_global_models': diagnostics.get(
                'feddrift_n_global_models',
                np.nan,
            ),
            'feddrift_cluster_assignments': (
                json.dumps(
                    diagnostics.get('feddrift_cluster_assignments', {}),
                    sort_keys=True,
                )
                if cli.algorithm == 'FedDrift'
                else ''
            ),
            'feddrift_models_before_merge': diagnostics.get(
                'feddrift_models_before_merge',
                np.nan,
            ),
            'feddrift_models_after_merge': diagnostics.get(
                'feddrift_models_after_merge',
                np.nan,
            ),
            'feddrift_models_after_creation': diagnostics.get(
                'feddrift_models_after_creation',
                np.nan,
            ),
            'feddrift_new_models_created': diagnostics.get(
                'feddrift_new_models_created',
                np.nan,
            ),
            'feddrift_drifted_client_ids': (
                json.dumps(diagnostics.get('feddrift_drifted_client_ids', []))
                if cli.algorithm == 'FedDrift'
                else ''
            ),
        }
        round_log.append(row)
        print(
            f"{cli.algorithm} r={round_id:03d} exp={len(exposed):02d} sel={len(selected):02d} "
            f"preqR={preq['attack_recall']:.4f} postR={post['attack_recall']:.4f} "
            f"HOICall={emerging['emerging_recall_all']:.4f} "
            f"EWS={emerging['emerging_recall_unexposed']:.4f} "
            f"PB-EWS={emerging['emerging_pattern_balanced_recall_unexposed']:.4f} "
            f"TA-R={target_alert['target_alert_recall_unexposed']:.4f} "
            f"TA-FPR={target_alert['target_alert_target_vs_rest_fpr_unexposed']:.4f} "
            f"TA-UCB={target_alert['target_alert_target_vs_rest_fpr_upper_unexposed']:.4f} "
            f"TA-streak={target_alert_empirical_tracker.current_streak} "
            f"vis={threat_visibility:.2f} ev={n_selected_emerging} "
            f"records={len(cumulative_selected_distinct_records)} "
            f"patterns={len(cumulative_selected_distinct_feature_patterns)} "
            f"FN={preq['attack_fn']} time={row['round_seconds']:.1f}s",
            flush=True,
        )
        if (round_id + 1) % int(args.get('flush_every', 5)) == 0:
            round_log.flush(); client_log.flush()

    round_log.flush(); client_log.flush()
    summary = {
        'algorithm': cli.algorithm,
        'run_name': run_name,
        'first_drift_round': threat.first_drift_round,
        'evidence_mode': stream.evidence_mode,
        'cumulative_incoming_emerging_presentations': cumulative_incoming_presentations,
        'cumulative_selected_emerging_presentations': cumulative_selected_presentations,
        'cumulative_selected_distinct_emerging_records': len(
            cumulative_selected_distinct_records
        ),
        'cumulative_selected_distinct_emerging_feature_patterns': len(
            cumulative_selected_distinct_feature_patterns
        ),
        'cumulative_selected_unique_emerging': len(
            cumulative_selected_distinct_records
        ),
        **ew_tracker.as_dict(),
        **target_alert_empirical_tracker.as_dict(),
        **target_alert_certified_tracker.as_dict(),
        **{
            f'final_{key}': value
            for key, value in target_alert.items()
            if key.endswith('_unexposed')
        },
    }
    (result_dir / f'{run_name}_early_warning_summary.json').write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f'Results written to {result_dir}')


if __name__ == '__main__':
    main()
