"""Algorithm adapters for a common streaming cybersecurity experiment loop.

The common runner owns data, threat exposure, participation, initialization,
and evaluation. Adapters own only the algorithm update. This ensures identical
experimental conditions across methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import inspect
from typing import Dict, Iterable, List, Sequence
import numpy as np
import torch
from torch.nn import Module
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from utils.cyber_model_bridge import (
    load_state_everywhere,
    make_cyber_model,
    state_dict_cpu,
)


class AdapterError(RuntimeError):
    pass


class AlgorithmAdapter(ABC):
    name = 'base'
    personalized = False

    def __init__(self, args: dict, clients: Sequence, server):
        self.args = args
        self.clients = list(clients)
        self.server = server
        self.last_diagnostics: Dict[str, object] = {}

    def initialize(self, initial_state: dict) -> None:
        load_state_everywhere(self.server, initial_state)
        for c in self.clients:
            load_state_everywhere(c, initial_state)
        if hasattr(self.server, 'get_client_data_size'):
            self.server.get_client_data_size(self.clients)
        # Start all deployed client models from the identical server state.
        if hasattr(self.server, 'send_params'):
            try:
                self.server.send_params(self.clients)
            except Exception:
                # Multi-model methods may not use send_params; their model
                # collections were already deterministically initialized.
                pass

    def before_round(self, round_id: int, selected_clients: Sequence) -> None:
        if hasattr(self.server, 'client_data_size'):
            for c in selected_clients:
                try:
                    self.server.client_data_size[c.id] = len(c.train_set)
                except Exception:
                    pass

    @abstractmethod
    def train_round(self, round_id: int, selected_clients: Sequence) -> Dict[str, object]:
        raise NotImplementedError

    def communication_multiplier(self) -> float:
        # Approximate model-equivalent send+receive. Method-specific metadata
        # (e.g. prototypes) is not included unless adapter overrides it.
        return 2.0

    def deployment_models(self):
        """Models used for site-level prequential evaluation."""
        return {int(c.id): c.model for c in self.clients if hasattr(c, 'model')}


class SendTrainAggregateAdapter(AlgorithmAdapter):
    """Exact loop used by FedCCFA repository's simple global methods.

    The public Flash method script follows this sequence:
      server.send_params(selected) -> client.train() ->
      server.aggregate_by_updates(selected) -> server.send_params(selected)
    FedAvg/FedProx in this repository use the same client/server abstraction.
    """
    aggregate_candidates = ('aggregate_by_updates', 'aggregate_by_params', 'aggregate')

    def _aggregate(self, selected_clients):
        for name in self.aggregate_candidates:
            fn = getattr(self.server, name, None)
            if callable(fn):
                fn(selected_clients)
                return name
        raise AdapterError(
            f'{self.name}: server exposes none of {self.aggregate_candidates}. '
            'Run scripts/validate_cyber_compat.py and inspect upstream API.'
        )

    def train_round(self, round_id: int, selected_clients: Sequence):
        self.before_round(round_id, selected_clients)
        if not hasattr(self.server, 'send_params'):
            raise AdapterError(f'{self.name}: missing server.send_params')
        self.server.send_params(selected_clients)
        for client in selected_clients:
            client.train()
        agg = self._aggregate(selected_clients)
        # Keep the selected clients synchronized with the post-aggregation model,
        # matching the repository's Flash runner and allowing next-round inference.
        try:
            self.server.send_params(selected_clients)
        except Exception:
            pass
        self.last_diagnostics = {'aggregate_method': agg}
        return dict(self.last_diagnostics)


class FedAvgAdapter(SendTrainAggregateAdapter):
    name = 'FedAvg'


class FedDriftAdapter(AlgorithmAdapter):
    """Faithful bridge to the repository's clustered/multi-model FedDrift.

    FedDrift is not a send/train/FedAvg-aggregate method.  Its upstream entity
    keeps a list of cluster models in ``server.global_models`` and assigns each
    client a ``cluster_identity`` using ``client.clustering(global_models)``.
    A correct cyber round therefore has to preserve that multi-model state.

    Round sequence used here follows ``methods/FedDrift.py``:
      1) clients select an active cluster or return the new-model signal;
      2) existing clusters are merged when more than one exists;
      3) a new model is created for each drifted client;
      4) assigned models are sent and selected clients train locally;
      5) the server performs cluster-conditional aggregation and sends again;
      6) the current stream window is saved as each client's previous window.

    The adapter intentionally has *no* generic ``aggregate_by_updates``
    fallback.  Falling back to that path silently turns FedDrift into a
    different algorithm and produced the frozen evaluation observed in the
    cybersecurity benchmark.
    """
    name = 'FedDrift'
    personalized = True

    _CLIENT_PARAM_NAMES = {
        'clients', 'selected_clients', 'client_list', 'selected',
        'participating_clients', 'active_clients',
    }
    _ALL_CLIENT_PARAM_NAMES = {
        'all_clients', 'clients_all', 'global_clients',
    }
    _ROUND_PARAM_NAMES = {
        'round_id', 'round', '_round', 'communication_round', 'global_round',
        'epoch', 't',
    }
    _MODEL_PARAM_NAMES = {
        'global_models', 'models', 'cluster_models', 'model_list',
    }

    def initialize(self, initial_state: dict) -> None:
        super().initialize(initial_state)
        if not hasattr(self.server, 'global_models'):
            raise AdapterError(
                'FedDrift requires server.global_models, but the cyber wrapper '
                'does not expose it. Check entities/FedDrift.py and '
                'entities/FedDrift_Cyber.py.'
            )
        if not isinstance(self.server.global_models, (list, tuple)) or len(self.server.global_models) == 0:
            raise AdapterError('FedDrift server.global_models must be a non-empty model collection')
        for client in self.clients:
            current = getattr(client, 'cluster_identity', None)
            if current is None:
                # At initialization FedDrift has exactly one global concept
                # model, so cluster 0 is the only semantically valid choice.
                if len(self.server.global_models) != 1:
                    raise AdapterError(
                        f'FedDrift client {getattr(client, "id", "?")} has '
                        'cluster_identity=None while the server already has '
                        f'{len(self.server.global_models)} global models. Refusing '
                        'to guess a cluster assignment.'
                    )
                client.cluster_identity = 0
        self._sync_clients(self.clients, recluster=False)

    def _call_signature_aware(self, fn, method_name: str, round_id: int, selected_clients: Sequence):
        """Invoke an upstream FedDrift server method without guessing arg order.

        The public repository has changed small method signatures across forks.
        We bind recognized semantic parameter names and fail loudly for an
        unknown *required* argument instead of silently calling a wrong API.
        """
        sig = inspect.signature(fn)
        kwargs = {}
        unresolved = []
        for p in sig.parameters.values():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            lname = p.name.lower()
            if lname in self._CLIENT_PARAM_NAMES:
                kwargs[p.name] = selected_clients
            elif lname in self._ALL_CLIENT_PARAM_NAMES:
                kwargs[p.name] = self.clients
            elif lname in self._ROUND_PARAM_NAMES:
                kwargs[p.name] = round_id
            elif lname in self._MODEL_PARAM_NAMES:
                kwargs[p.name] = self.server.global_models
            elif p.default is inspect.Parameter.empty:
                unresolved.append(p.name)

        if unresolved:
            raise AdapterError(
                f'FedDrift.{method_name}{sig} has unsupported required '
                f'arguments {unresolved}. This hotfix refuses to guess. '
                'Please send this exact signature if encountered.'
            )
        return fn(**kwargs)

    def _cluster_clients(
        self,
        clients: Sequence,
        allow_new_model_signal: bool = False,
    ) -> None:
        global_models = getattr(self.server, 'global_models', None)
        if not isinstance(global_models, (list, tuple)) or len(global_models) == 0:
            raise AdapterError('FedDrift lost its server.global_models collection')
        for client in clients:
            fn = getattr(client, 'clustering', None)
            if not callable(fn):
                raise AdapterError(
                    f'FedDrift client {getattr(client, "id", "?")} has no clustering(global_models) method'
                )
            fn(global_models)

            # In upstream FedDrift, None is an intentional drift signal: the
            # runner creates a new global model for that client. Never coerce
            # it to cluster 0, because doing so suppresses the first split.
            current = getattr(self.server, 'global_models', global_models)
            raw_idx = getattr(client, 'cluster_identity', None)
            if raw_idx is None:
                if allow_new_model_signal:
                    continue
                raise AdapterError(
                    f'FedDrift client {client.id} returned the new-model signal '
                    'outside the pre-training clustering phase'
                )

            idx = int(raw_idx)
            # client.clustering is allowed to mutate/extend global_models. Re-read.
            current = getattr(self.server, 'global_models', current)
            if idx < 0 or idx >= len(current):
                raise AdapterError(
                    f'FedDrift client {client.id} selected cluster {idx}, '
                    f'but server has {len(current)} global models'
                )

    def _send_cluster_models(self, clients: Sequence) -> None:
        send = getattr(self.server, 'send_params', None)
        if not callable(send):
            raise AdapterError('FedDrift server is missing send_params(clients)')
        send(clients)

    def _sync_clients(self, clients: Sequence, recluster: bool = False) -> None:
        if recluster:
            self._cluster_clients(clients)
        self._send_cluster_models(clients)

    def _create_models_for_drifted_clients(
        self,
        selected_clients: Sequence,
    ) -> list[int]:
        """Implement the upstream ``cluster_identity is None`` branch."""
        drifted_ids = []
        for client in selected_clients:
            if getattr(client, 'cluster_identity', None) is not None:
                continue
            self.server.global_models.append(make_cyber_model(self.args))
            client.cluster_identity = len(self.server.global_models) - 1
            drifted_ids.append(int(client.id))
        return drifted_ids

    def train_round(self, round_id: int, selected_clients: Sequence):
        self.before_round(round_id, selected_clients)

        if not callable(getattr(self.server, 'aggregate_with_clustering', None)):
            public = sorted(
                name for name in dir(self.server)
                if not name.startswith('_') and callable(getattr(self.server, name, None))
            )
            raise AdapterError(
                'FedDrift cyber adapter requires server.aggregate_with_clustering. '
                f'Public server methods: {public}'
            )

        # Match methods/FedDrift.py exactly: cluster, merge existing clusters,
        # create one new model per drifted client, send, train, aggregate, send,
        # then persist the current window as prev_train_set.
        self._cluster_clients(
            selected_clients,
            allow_new_model_signal=True,
        )
        models_before_merge = len(self.server.global_models)

        merge = getattr(self.server, 'merge_clusters', None)
        merge_called = False
        if models_before_merge > 1:
            if not callable(merge):
                raise AdapterError(
                    'FedDrift has multiple global models but server.merge_clusters '
                    'is unavailable'
                )
            self._call_signature_aware(
                merge, 'merge_clusters', round_id, selected_clients
            )
            merge_called = True
        models_after_merge = len(self.server.global_models)

        drifted_ids = self._create_models_for_drifted_clients(selected_clients)
        models_after_creation = len(self.server.global_models)

        self._send_cluster_models(selected_clients)
        for client in selected_clients:
            client.train()

        self._call_signature_aware(
            self.server.aggregate_with_clustering,
            'aggregate_with_clustering',
            round_id,
            selected_clients,
        )

        self._send_cluster_models(selected_clients)

        # The upstream runner advances this reference after every round. The
        # streaming manager mutates train_set, so a deep copy is required.
        for client in self.clients:
            client.prev_train_set = deepcopy(client.train_set)

        assignments = {
            int(c.id): int(c.cluster_identity)
            for c in self.clients
            if hasattr(c, 'cluster_identity')
        }
        model_count = len(getattr(self.server, 'global_models', []))
        self.last_diagnostics = {
            'feddrift_path': 'native:clustering'
                             + ('->merge_clusters' if merge_called else '')
                             + '->create_drift_models->send_params->train'
                             + '->aggregate_with_clustering->send_params'
                             + '->update_prev_train_set',
            'feddrift_n_global_models': int(model_count),
            'feddrift_cluster_assignments': assignments,
            'feddrift_models_before_merge': int(models_before_merge),
            'feddrift_models_after_merge': int(models_after_merge),
            'feddrift_new_models_created': int(len(drifted_ids)),
            'feddrift_drifted_client_ids': drifted_ids,
            'feddrift_models_after_creation': int(models_after_creation),
        }
        return dict(self.last_diagnostics)

    def deployment_models(self):
        """Return the actual cluster model deployed at each FedDrift client."""
        global_models = getattr(self.server, 'global_models', None)
        if not isinstance(global_models, (list, tuple)) or len(global_models) == 0:
            raise AdapterError('FedDrift deployment requested without global_models')

        out = {}
        for client in self.clients:
            if not hasattr(client, 'cluster_identity'):
                raise AdapterError(f'FedDrift client {client.id} has no cluster_identity')
            idx = int(client.cluster_identity)
            if idx < 0 or idx >= len(global_models):
                raise AdapterError(
                    f'FedDrift deployment cluster index {idx} is invalid for '
                    f'{len(global_models)} global models'
                )
            out[int(client.id)] = global_models[idx]
        return out


class FedCCFAAdapter(AlgorithmAdapter):
    name = 'FedCCFA'
    personalized = True

    def initialize(self, initial_state: dict) -> None:
        super().initialize(initial_state)
        clf_keys = list(self.server.model.state_dict().keys())[-2:]
        self.server.clf_keys = clf_keys
        for c in self.clients:
            c.clf_keys = clf_keys
            c.p_clf_params = deepcopy(c.get_clf_parameters())
        self.clf_keys = clf_keys

    @staticmethod
    def _client_weight(client) -> float:
        # Sample-size weighting matches the FedAvg-style intervention described
        # in the mechanism ablation. Fall back to one for empty/missing sets.
        try:
            return float(max(len(client.train_set), 1))
        except Exception:
            return 1.0

    def _classifier_parameters(self, client):
        return [
            param for name, param in client.model.named_parameters()
            if name in self.clf_keys
        ]

    def _share_label_globally(self, label: int, selected_clients: Sequence) -> None:
        if not selected_clients:
            return
        ref = self._classifier_parameters(selected_clients[0])
        if not ref:
            raise AdapterError('FedCCFA classifier parameters were not identified')
        totals = [torch.zeros_like(param.data[label]) for param in ref]
        denom = 0.0
        with torch.no_grad():
            for client in selected_clients:
                weight = self._client_weight(client)
                params = self._classifier_parameters(client)
                if len(params) != len(totals):
                    raise AdapterError('Inconsistent classifier parameter structure across clients')
                for acc, param in zip(totals, params):
                    acc.add_(param.data[label], alpha=weight)
                denom += weight
            if denom <= 0:
                return
            for acc in totals:
                acc.div_(denom)
            for client in selected_clients:
                for acc, param in zip(totals, self._classifier_parameters(client)):
                    param.data[label].copy_(acc)

    def _share_classifier_globally(self, selected_clients: Sequence) -> None:
        if not selected_clients:
            return
        ref = self._classifier_parameters(selected_clients[0])
        if not ref:
            raise AdapterError('FedCCFA classifier parameters were not identified')
        totals = [torch.zeros_like(param.data) for param in ref]
        denom = 0.0
        with torch.no_grad():
            for client in selected_clients:
                weight = self._client_weight(client)
                params = self._classifier_parameters(client)
                if len(params) != len(totals):
                    raise AdapterError('Inconsistent classifier parameter structure across clients')
                for acc, param in zip(totals, params):
                    acc.add_(param.data, alpha=weight)
                denom += weight
            if denom <= 0:
                return
            for acc in totals:
                acc.div_(denom)
            for client in selected_clients:
                for acc, param in zip(totals, self._classifier_parameters(client)):
                    param.data.copy_(acc)

    def train_round(self, round_id: int, selected_clients: Sequence):
        self.before_round(round_id, selected_clients)
        server = self.server
        args = self.args
        server.send_params(selected_clients)
        balanced = {}
        for client in selected_clients:
            client.update_label_distribution()
            if int(args.get('balanced_epochs', 0)) > 0:
                client.balance_train()
                balanced[client.id] = deepcopy(client.get_clf_parameters())
            if not bool(args.get('clustered_protos', True)):
                client.global_protos = deepcopy(server.global_protos)
            client.train_with_protos(round_id)
            if int(args.get('balanced_epochs', 0)) == 0:
                balanced[client.id] = deepcopy(client.get_clf_parameters())

        server.aggregate_rep(selected_clients)
        server.aggregate_protos(selected_clients)
        server.send_rep_params(selected_clients)
        if bool(args.get('oracle', False)):
            raise AdapterError('oracle:true is unsupported for real cyber threat schedules')

        label_merged = server.merge_classifiers_sparse(balanced, selected_clients)
        for label, groups in label_merged.items():
            for indices in groups:
                group = [c for c in selected_clients if c.id in indices]
                label_params = server.aggregate_label_params(label, group)
                label_proto = server.aggregate_label_protos(label, group)
                if label_params is None:
                    continue
                for client in group:
                    client_label_params = [
                        param[label]
                        for name, param in client.model.named_parameters()
                        if name in self.clf_keys
                    ]
                    vector_to_parameters(label_params, client_label_params)
                    client.set_label_params(label, client_label_params)
                    if label_proto is not None:
                        client.global_protos[label] = label_proto.detach().clone()
        # Optional mechanism ablations. These are intentionally applied AFTER
        # normal FedCCFA class-wise clustering so representation/prototype
        # learning and the native collaboration logic remain intact. Only the
        # final classifier-sharing topology is intervened on.
        sharing_mode = str(args.get('fedccfa_sharing_mode', 'clustered')).strip().lower()
        if sharing_mode == 'global_ddos_head':
            attack_label = int(args.get('attack_label', -1))
            if not (0 <= attack_label < int(args['num_classes'])):
                raise AdapterError(
                    f'global_ddos_head requires a valid runtime attack_label; got {attack_label}'
                )
            self._share_label_globally(attack_label, selected_clients)
        elif sharing_mode == 'global_classifier':
            self._share_classifier_globally(selected_clients)
        elif sharing_mode != 'clustered':
            raise AdapterError(
                f'Unknown fedccfa_sharing_mode={sharing_mode!r}; choose '
                "clustered, global_ddos_head, or global_classifier"
            )

        # Persist the post-intervention classifier because CyberFedCCFAClient
        # restores p_clf_params at the beginning of its next local round.
        for client in selected_clients:
            client.p_clf_params = deepcopy(client.get_clf_parameters())

        self.last_diagnostics = {
            'label_merged_dict': label_merged,
            'fedccfa_sharing_mode': sharing_mode,
        }
        return dict(self.last_diagnostics)

    def communication_multiplier(self) -> float:
        # Still model-equivalent; classifier/prototype metadata is logged as an
        # acknowledged lower bound in the runner metadata.
        return 2.0


def build_adapter(algorithm: str, args: dict, clients, server):
    table = {
        'FedAvg': FedAvgAdapter,
        'FedDrift': FedDriftAdapter,
        'FedCCFA': FedCCFAAdapter,
    }
    if algorithm not in table:
        raise ValueError(f'Unsupported cyber algorithm {algorithm!r}; choose {sorted(table)}')
    return table[algorithm](args, clients, server)
