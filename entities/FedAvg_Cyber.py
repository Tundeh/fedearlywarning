"""CyberMLP-compatible wrappers for the repository's vanilla FedAvg.

The upstream FedCCFA repository does not define ``entities/FedAvg.py``.
Its vanilla FedAvg process uses the generic ``Client`` and ``Server`` from
``entities.base``.  Keep this wrapper thin so the original FedAvg training and
aggregation logic remains unchanged while replacing only the model with the
cybersecurity MLP.
"""
from entities.base import Client, Server
from utils.cyber_model_bridge import install_cyber_model, upstream_stub_args


class CyberFedAvgClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id=0):
        super().__init__(
            client_id,
            upstream_stub_args(args),
            train_set,
            test_set,
            global_test_id,
        )
        install_cyber_model(self, args)


class CyberFedAvgServer(Server):
    def __init__(self, args):
        super().__init__(upstream_stub_args(args))
        install_cyber_model(self, args)
