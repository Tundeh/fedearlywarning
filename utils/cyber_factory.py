"""Construct cyber-compatible clients/servers for supported repository methods."""
from __future__ import annotations


def classes_for_algorithm(algorithm: str):
    if algorithm == 'FedAvg':
        from entities.FedAvg_Cyber import CyberFedAvgClient, CyberFedAvgServer
        return CyberFedAvgClient, CyberFedAvgServer
    if algorithm == 'FedDrift':
        from entities.FedDrift_Cyber import CyberFedDriftClient, CyberFedDriftServer
        return CyberFedDriftClient, CyberFedDriftServer
    if algorithm == 'FedCCFA':
        from entities.FedCCFA_Cyber import CyberFedCCFAClient, CyberFedCCFAServer
        return CyberFedCCFAClient, CyberFedCCFAServer
    raise ValueError(f'Unsupported algorithm: {algorithm}')
