"""Cybersecurity-compatible FedCCFA client/server subclasses.

Why subclasses?
---------------
The published FedCCFA code assumes image datasets and that every client owns
examples/prototypes for every class. Real intrusion-detection streams violate
that assumption: a client can have zero examples of an emerging attack until
it is exposed. These subclasses preserve the original FedCCFA training logic
while making class/prototype operations sparse-safe and replacing the CNN with
a tabular MLP.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch.nn import CrossEntropyLoss, MSELoss, Parameter
from torch.nn.utils import parameters_to_vector
from torch.optim import SGD
from torch.utils.data import DataLoader

from entities.FedCCFA import FedCCFAClient, FedCCFAServer
from utils.cyber_dataset import CyberTensorDataset
from utils.cyber_models import CyberMLP


def _stub_args(args: dict) -> dict:
    """Arguments that let the unmodified base class build its original CNN.

    The temporary model is immediately replaced by ``CyberMLP``. This avoids
    editing the authors' model factory and keeps their original experiments
    untouched.
    """
    stub = deepcopy(args)
    stub["dataset"] = "CIFAR10"
    stub["model"] = "CNN"
    stub["num_classes"] = 10
    return stub


def _make_cyber_model(args: dict) -> CyberMLP:
    return CyberMLP(
        input_dim=int(args["input_dim"]),
        num_classes=int(args["num_classes"]),
        hidden_dim=int(args.get("hidden_dim", 256)),
        feature_dim=int(args.get("feature_dim", 128)),
        dropout=float(args.get("dropout", 0.2)),
    ).to(args["device"])


class CyberFedCCFAClient(FedCCFAClient):
    def __init__(self, client_id, args, train_set, test_set, global_test_id=0):
        super().__init__(client_id, _stub_args(args), train_set, test_set, global_test_id)
        self.args = deepcopy(args)
        self.model = _make_cyber_model(self.args)
        self.criterion = CrossEntropyLoss().to(self.args["device"])
        if self.args.get("penalize", "CL") == "L2":
            self.proto_criterion = MSELoss().to(self.args["device"])
        else:
            self.proto_criterion = CrossEntropyLoss().to(self.args["device"])

        self.clf_keys: List[str] = []
        self.local_protos: Dict[int, torch.Tensor] = {}
        self.global_protos: Dict[int, torch.Tensor] = {}
        self.p_clf_params = []
        self.label_distribution = np.zeros(self.args["num_classes"], dtype=np.int64)
        self.proto_weight = 0.0

    def update_label_distribution(self):
        """Reset counts each round; absent attack classes must become zero."""
        counts = np.zeros(self.args["num_classes"], dtype=np.int64)
        for label, size in Counter(np.asarray(self.train_set.targets).tolist()).items():
            label = int(label)
            if 0 <= label < len(counts):
                counts[label] = int(size)
        self.label_distribution = counts

        n = int(counts.sum())
        if n == 0:
            self.proto_weight = 0.0
            return
        prob = counts[counts > 0].astype(np.float64) / n
        entropy = float(-np.sum(prob * np.log(prob)))
        gamma = float(self.args.get("gamma", 0.0))
        if gamma != 0.0:
            self.proto_weight = entropy / gamma
        else:
            self.proto_weight = float(self.args.get("lambda", 0.0))

    def class_balance_sample(self):
        """Balance only classes that actually occur in the current stream window."""
        labels = np.asarray(self.train_set.targets, dtype=np.int64)
        if len(labels) == 0:
            return self.train_set

        counter = Counter(labels.tolist())
        min_size = min(min(counter.values()), int(self.args.get("balance_max_per_class", 5)))
        indices: List[int] = []
        rng = np.random.default_rng(
            int(self.args.get("seed", 0)) + 1009 * int(self.id) + len(labels)
        )
        for label in sorted(counter.keys()):
            class_indices = np.where(labels == label)[0]
            indices.extend(
                rng.choice(class_indices, min_size, replace=False).astype(int).tolist()
            )
        rng.shuffle(indices)
        if isinstance(self.train_set, CyberTensorDataset):
            return self.train_set.subset(indices)
        # Fallback for another FedCCFA-compatible dataset implementation.
        return CyberTensorDataset(
            np.asarray(self.train_set.data)[indices],
            labels[indices],
        )

    def _proto_alignment_loss(self, features: torch.Tensor, labels: torch.Tensor):
        if not self.global_protos or self.proto_weight == 0.0:
            return features.new_tensor(0.0)

        available = sorted(int(k) for k in self.global_protos.keys())
        label_to_pos = {label: pos for pos, label in enumerate(available)}
        mask_list = [int(label.item()) in label_to_pos for label in labels]
        if not any(mask_list):
            return features.new_tensor(0.0)

        mask = torch.tensor(mask_list, dtype=torch.bool, device=features.device)
        valid_features = features[mask]
        valid_labels = labels[mask]

        if self.args.get("penalize", "CL") == "L2":
            target_protos = torch.stack([
                self.global_protos[int(label.item())].detach().to(features.device)
                for label in valid_labels
            ])
            return F.mse_loss(valid_features, target_protos)

        proto_matrix = torch.stack([
            self.global_protos[label].detach().to(features.device)
            for label in available
        ])
        logits = F.normalize(valid_features, dim=1) @ F.normalize(proto_matrix, dim=1).T
        target = torch.tensor(
            [label_to_pos[int(label.item())] for label in valid_labels],
            dtype=torch.long,
            device=features.device,
        )
        temperature = float(self.args.get("temperature", 0.1))
        return F.cross_entropy(logits / temperature, target)

    def train_with_protos(self, _round):
        """Original decoupled FedCCFA local training, sparse-prototype safe."""
        if len(self.train_set) == 0:
            self.local_protos = {}
            return

        if self.p_clf_params:
            self.set_clf_params(self.p_clf_params)

        train_loader = DataLoader(
            self.train_set,
            batch_size=min(int(self.args["batch_size"]), len(self.train_set)),
            shuffle=True,
        )

        # ----- classifier phase -----
        self.model.train()
        for name, param in self.model.named_parameters():
            param.requires_grad = name in self.clf_keys
        optimizer = SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.args["clf_lr"]),
            weight_decay=float(self.args.get("weight_decay", 0.0)),
            momentum=float(self.args.get("momentum", 0.0)),
        )
        for _ in range(int(self.args.get("clf_epochs", 1))):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.args["device"])
                labels = labels.to(self.args["device"])
                loss = self.criterion(self.model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ----- representation phase -----
        for name, param in self.model.named_parameters():
            param.requires_grad = name not in self.clf_keys
        optimizer = SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.args["rep_lr"]),
            weight_decay=float(self.args.get("weight_decay", 0.0)),
            momentum=float(self.args.get("momentum", 0.0)),
        )
        proto_start = int(self.args.get("proto_start_round", 20))
        for _ in range(int(self.args.get("rep_epochs", 1))):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.args["device"])
                labels = labels.to(self.args["device"])
                outputs, features = self.model(inputs, True)
                loss_sup = self.criterion(outputs, labels)
                loss_proto = (
                    self._proto_alignment_loss(features, labels)
                    if _round >= proto_start
                    else features.new_tensor(0.0)
                )
                loss = loss_sup + float(self.proto_weight) * loss_proto
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            self.local_protos = self.get_local_protos(self.model)
            # Preserve locally available prototypes even before server/group
            # aggregation; missing classes remain truly absent.
            for label, proto in self.local_protos.items():
                self.global_protos[int(label)] = proto.detach().clone()


class CyberFedCCFAServer(FedCCFAServer):
    def __init__(self, args):
        super().__init__(_stub_args(args))
        self.args = deepcopy(args)
        self.model = _make_cyber_model(self.args)
        self.clf_keys = None
        self.global_protos: Dict[int, torch.Tensor] = {}

    def aggregate_protos(self, clients):
        """Aggregate only prototypes that clients actually possess."""
        sums: Dict[int, torch.Tensor] = {}
        weights: Dict[int, float] = {}
        use_uniform = self.args.get("weights", "uniform") == "uniform"
        for client in clients:
            for label, proto in client.local_protos.items():
                label = int(label)
                weight = 1.0 if use_uniform else float(client.label_distribution[label])
                if weight <= 0:
                    continue
                contribution = proto.detach() * weight
                sums[label] = contribution.clone() if label not in sums else sums[label] + contribution
                weights[label] = weights.get(label, 0.0) + weight
        self.global_protos = {
            label: sums[label] / max(weights[label], 1e-12)
            for label in sums
        }

    @staticmethod
    def _madd_safe(vecs: np.ndarray) -> np.ndarray:
        """FedCCFA MADD with numerical guards; defined for n>=3."""
        vecs = np.asarray(vecs, dtype=np.float64)
        n = len(vecs)
        if n < 3:
            return np.zeros((n, n), dtype=np.float64)

        norms = np.linalg.norm(vecs, axis=1)
        norms[norms < 1e-12] = 1e-12
        sim = (vecs @ vecs.T) / (norms[:, None] * norms[None, :])
        out = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                mask = np.ones(n, dtype=bool)
                mask[[i, j]] = False
                out[i, j] = out[j, i] = np.mean(np.abs(sim[i, mask] - sim[j, mask]))
        return out

    def merge_classifiers_sparse(self, clf_params_dict, clients):
        """Class-wise DBSCAN using only clients that observed each class.

        This is the minimum change required to avoid treating a client with
        zero attack examples as if it had a valid attack-specific classifier.
        """
        client_lookup = {int(c.id): c for c in clients}
        label_merged_dict = {label: [] for label in range(self.args["num_classes"])}

        for label in range(self.args["num_classes"]):
            eligible_ids = [
                int(cid) for cid in clf_params_dict
                if cid in client_lookup
                and int(client_lookup[cid].label_distribution[label]) > 0
            ]
            if len(eligible_ids) < 3:
                # Original MADD is undefined for two clients (division by n-2).
                continue

            params_list = []
            for cid in eligible_ids:
                clf_params = clf_params_dict[cid]
                params = [param[label].detach().cpu().numpy().reshape(-1) for param in clf_params]
                params_list.append(np.concatenate(params))
            dist = self._madd_safe(np.asarray(params_list))
            clustering = DBSCAN(
                eps=float(self.args["eps"]),
                min_samples=1,
                metric="precomputed",
            ).fit(dist)

            groups: List[List[int]] = []
            labels = clustering.labels_
            for cluster_id in sorted(set(labels.tolist())):
                members = [eligible_ids[i] for i in np.where(labels == cluster_id)[0]]
                if len(members) > 1:  # same behavior as published code
                    groups.append(sorted(members))
            label_merged_dict[label] = groups
        return label_merged_dict

    def aggregate_label_params(self, label, clients):
        if not clients:
            return None
        server_label_params = [
            param[label] for name, param in self.model.named_parameters()
            if name in self.clf_keys
        ]
        result = torch.zeros_like(parameters_to_vector(server_label_params))
        total_weight = 0.0
        uniform = self.args.get("weights", "uniform") == "uniform"

        for client in clients:
            if int(client.label_distribution[label]) <= 0:
                continue
            params = [
                param[label] for name, param in client.model.named_parameters()
                if name in self.clf_keys
            ]
            vec = parameters_to_vector(params)
            weight = 1.0 if uniform else float(client.label_distribution[label])
            result += weight * vec
            total_weight += weight
        if total_weight <= 0:
            return None
        return result / total_weight

    def aggregate_label_protos(self, label, clients):
        valid = [c for c in clients if label in c.local_protos]
        if not valid:
            return None
        uniform = self.args.get("weights", "uniform") == "uniform"
        total = None
        total_weight = 0.0
        for client in valid:
            weight = 1.0 if uniform else float(client.label_distribution[label])
            if weight <= 0:
                continue
            term = client.local_protos[label].detach() * weight
            total = term.clone() if total is None else total + term
            total_weight += weight
        if total is None or total_weight <= 0:
            return None
        return total / total_weight
