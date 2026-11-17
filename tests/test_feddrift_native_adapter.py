#!/usr/bin/env python3
"""Minimal API-level regression test for the FedDrift cyber adapter."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cyber_adapters import FedDriftAdapter
from utils.cyber_models import CyberMLP


class Client:
    def __init__(self, cid):
        self.id = cid
        self.model = CyberMLP(4, 2, 8, 4, 0.0)
        self.cluster_identity = 0
        self.train_set = list(range(8))
        self.train_calls = 0
        self.cluster_calls = 0

    def clustering(self, global_models):
        self.cluster_calls += 1
        # Dummy clients remain in cluster 0.
        self.cluster_identity = 0

    def set_params(self, params):
        for dst, src in zip(self.model.parameters(), params):
            dst.data.copy_(src.data)

    def train(self):
        self.train_calls += 1
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(0.01 * (self.id + 1))


class Server:
    def __init__(self):
        self.model = CyberMLP(4, 2, 8, 4, 0.0)
        self.global_models = [CyberMLP(4, 2, 8, 4, 0.0)]
        self.client_data_size = {}
        self.aggregate_calls = 0
        self.merge_calls = 0

    def get_client_data_size(self, clients):
        self.client_data_size = {c.id: len(c.train_set) for c in clients}

    def send_params(self, clients):
        for c in clients:
            c.set_params(self.global_models[c.cluster_identity].parameters())

    def aggregate_with_clustering(self, clients):
        self.aggregate_calls += 1
        with torch.no_grad():
            for dst_params in zip(self.global_models[0].parameters(), *[c.model.parameters() for c in clients]):
                dst = dst_params[0]
                srcs = dst_params[1:]
                dst.copy_(torch.stack([x.detach() for x in srcs]).mean(0))

    def merge_clusters(self):
        self.merge_calls += 1


clients = [Client(i) for i in range(4)]
server = Server()
adapter = FedDriftAdapter({'device': 'cpu'}, clients, server)
initial = {k: v.detach().clone() for k, v in server.model.state_dict().items()}
adapter.initialize(initial)
d = adapter.train_round(0, clients)

assert server.aggregate_calls == 1
assert server.merge_calls == 0  # Native runner merges only when >1 model exists.
assert all(c.train_calls == 1 for c in clients)
assert all(c.cluster_calls == 1 for c in clients)
assert d['feddrift_n_global_models'] == 1
assert d['feddrift_new_models_created'] == 0
assert d['feddrift_drifted_client_ids'] == []
assert d['feddrift_path'].startswith('native:clustering')
models = adapter.deployment_models()
assert set(models) == {0, 1, 2, 3}
assert all(models[i] is server.global_models[0] for i in models)
print('FedDrift native adapter regression test: PASS')
