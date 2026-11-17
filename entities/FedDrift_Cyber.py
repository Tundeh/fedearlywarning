"""CyberMLP-compatible wrappers around the repository's FedDrift entities.

FedDrift can keep multiple model copies. ``install_cyber_model`` replaces
module collections when the upstream class exposes them as ``models``,
``global_models``, ``cluster_models`` or ``model_list``.
"""
from entities.FedDrift import FedDriftClient, FedDriftServer
from utils.cyber_model_bridge import install_cyber_model, upstream_stub_args


class CyberFedDriftClient(FedDriftClient):
    def __init__(self, client_id, args, train_set, test_set, global_test_id=0):
        super().__init__(client_id, upstream_stub_args(args), train_set, test_set, global_test_id)
        install_cyber_model(self, args)


class CyberFedDriftServer(FedDriftServer):
    def __init__(self, args):
        super().__init__(upstream_stub_args(args))
        install_cyber_model(self, args)
