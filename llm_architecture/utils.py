"""Checkpoint helpers for saving and loading trained models."""

import torch

from .models import GPTModel


def save_model(model, path):
    """Persist a model's weights to ``path``."""
    torch.save(model.state_dict(), path)


def load_model(config, path, device="cpu"):
    """Rebuild a :class:`GPTModel` from ``config`` and load weights from ``path``."""
    model = GPTModel(config)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
