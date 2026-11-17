#!/usr/bin/env python3
"""Regression test: FedDrift None signals must create new global models."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cyber_adapters import FedDriftAdapter
from utils.cyber_models import CyberMLP


class Client:
    def __init__(self, cid, drift=False):
        self.id = cid
        self.model = CyberMLP(4, 2, 8, 4, 0.0)
        self.cluster_identity = 0
        self.train_set = [cid, cid + 1]
        self.prev_train_set = [-1]
        self.drift = drift
        self.train_calls = 0

    def clustering(self, global_models):
        self.cluster_identity = None if self.drift else 0

    def set_params(self, params):
        for destination, source in zip(self.model.parameters(), params):
            destination.data.copy_(source.data)

    def train(self):
        self.train_calls += 1
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(0.01 * (self.id + 1))


class Server:
    def __init__(self):
        self.model = CyberMLP(4, 2, 8, 4, 0.0)
        self.global_models = [CyberMLP(4, 2, 8, 4, 0.0)]
        self.client_data_size = {}
        self.aggregate_calls = 0
        self.merge_calls = 0

    def get_client_data_size(self, clients):
        self.client_data_size = {client.id: len(client.train_set) for client in clients}

    def send_params(self, clients):
        for client in clients:
            client.set_params(
                self.global_models[client.cluster_identity].parameters()
            )

    def aggregate_with_clustering(self, clients):
        self.aggregate_calls += 1
        for identity, model in enumerate(self.global_models):
            members = [c for c in clients if c.cluster_identity == identity]
            if not members:
                continue
            with torch.no_grad():
                for parameters in zip(
                    model.parameters(),
                    *[client.model.parameters() for client in members],
                ):
                    destination = parameters[0]
                    sources = parameters[1:]
                    destination.copy_(
                        torch.stack([source.detach() for source in sources]).mean(0)
                    )

    def merge_clusters(self, clients):
        self.merge_calls += 1


args = {
    'device': 'cpu',
    'input_dim': 4,
    'num_classes': 2,
    'hidden_dim': 8,
    'feature_dim': 4,
    'dropout': 0.0,
}
clients = [Client(0, drift=True), Client(1, drift=True), Client(2)]
server = Server()
adapter = FedDriftAdapter(args, clients, server)
initial = {key: value.detach().clone() for key, value in server.model.state_dict().items()}
adapter.initialize(initial)

old_train_sets = [client.train_set for client in clients]
diagnostics = adapter.train_round(10, clients)

assert server.aggregate_calls == 1
assert server.merge_calls == 0
assert diagnostics['feddrift_models_before_merge'] == 1
assert diagnostics['feddrift_models_after_merge'] == 1
assert diagnostics['feddrift_new_models_created'] == 2
assert diagnostics['feddrift_drifted_client_ids'] == [0, 1]
assert diagnostics['feddrift_models_after_creation'] == 3
assert diagnostics['feddrift_n_global_models'] == 3
assert diagnostics['feddrift_cluster_assignments'] == {0: 1, 1: 2, 2: 0}
assert all(client.train_calls == 1 for client in clients)
assert all(client.prev_train_set == client.train_set for client in clients)
assert all(client.prev_train_set is not old for client, old in zip(clients, old_train_sets))
assert server.global_models[0] is adapter.deployment_models()[2]
assert server.global_models[1] is adapter.deployment_models()[0]
assert server.global_models[2] is adapter.deployment_models()[1]

print('FedDrift new-model signal regression test: PASS')
