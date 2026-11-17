"""Bridge the image-oriented FedCCFA entity classes to tabular CyberMLP.

The upstream repository's entity constructors create their configured image
model. Cyber subclasses call those constructors with the upstream/original
configuration and immediately replace model modules with CyberMLP. Common
module-shaped state is refreshed conservatively.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from utils.cyber_models import CyberMLP


def make_cyber_model(args: dict) -> CyberMLP:
    return CyberMLP(
        input_dim=int(args['input_dim']),
        num_classes=int(args['num_classes']),
        hidden_dim=int(args.get('hidden_dim', 256)),
        feature_dim=int(args.get('feature_dim', 128)),
        dropout=float(args.get('dropout', 0.2)),
    ).to(args['device'])


def upstream_stub_args(args: dict) -> dict:
    """Return arguments safe for the upstream constructor.

    ``args`` is already merged with the original algorithm config by
    ``cyber_config.load_cyber_config``. We only remove cyber-only dataset/model
    dimensions that the original model factory cannot understand.
    """
    out = deepcopy(args)
    out['dataset'] = str(args.get('upstream_stub_dataset', 'CIFAR10'))
    out['model'] = str(args.get('upstream_stub_model', 'CNN'))
    out['num_classes'] = int(args.get('upstream_stub_num_classes', 10))
    return out


def _replace_module_container(value: Any, args: dict):
    if isinstance(value, nn.Module):
        return make_cyber_model(args)
    if isinstance(value, list) and value and all(isinstance(v, nn.Module) for v in value):
        return [make_cyber_model(args) for _ in value]
    if isinstance(value, tuple) and value and all(isinstance(v, nn.Module) for v in value):
        return tuple(make_cyber_model(args) for _ in value)
    if isinstance(value, dict) and value and all(isinstance(v, nn.Module) for v in value.values()):
        return {k: make_cyber_model(args) for k in value}
    return value


def install_cyber_model(entity: Any, args: dict) -> None:
    """Replace the primary model and common cached module copies.

    We intentionally do *not* rewrite arbitrary tensors because some algorithms
    use control variates or update histories with algorithm-specific meaning.
    Such state is initialized by the upstream constructor and refreshed by its
    own first training round. Module copies are safe to replace dimensionally.
    """
    entity.args = deepcopy(args)
    entity.model = make_cyber_model(args)
    if hasattr(entity, 'criterion'):
        entity.criterion = CrossEntropyLoss().to(args['device'])

    for name in [
        'global_model', 'local_model', 'previous_model', 'prev_model',
        'old_model', 'last_model', 'personalized_model', 'p_model'
    ]:
        if hasattr(entity, name) and isinstance(getattr(entity, name), nn.Module):
            setattr(entity, name, make_cyber_model(args))

    # Some clustered/drift methods keep collections of models.
    for name in ['models', 'global_models', 'cluster_models', 'model_list']:
        if hasattr(entity, name):
            value = getattr(entity, name)
            new_value = _replace_module_container(value, args)
            if new_value is not value:
                setattr(entity, name, new_value)


def state_dict_cpu(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_everywhere(entity: Any, state: dict) -> None:
    """Load one deterministic initialization into model modules held by entity."""
    if hasattr(entity, 'model') and isinstance(entity.model, nn.Module):
        entity.model.load_state_dict(state, strict=True)
    for name in [
        'global_model', 'local_model', 'previous_model', 'prev_model',
        'old_model', 'last_model', 'personalized_model', 'p_model'
    ]:
        if hasattr(entity, name) and isinstance(getattr(entity, name), nn.Module):
            getattr(entity, name).load_state_dict(state, strict=True)
    for name in ['models', 'global_models', 'cluster_models', 'model_list']:
        if not hasattr(entity, name):
            continue
        value = getattr(entity, name)
        if isinstance(value, (list, tuple)):
            for m in value:
                if isinstance(m, nn.Module):
                    m.load_state_dict(state, strict=True)
        elif isinstance(value, dict):
            for m in value.values():
                if isinstance(m, nn.Module):
                    m.load_state_dict(state, strict=True)
