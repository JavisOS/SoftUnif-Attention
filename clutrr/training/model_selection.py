"""Validation-only model selection helpers shared by CLUTRR runners."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Subset


def stratified_train_validation_split(dataset, strata, validation_fraction=0.1, seed=2027):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(strata) != len(dataset):
        raise ValueError("strata must contain one value per dataset item")

    groups = defaultdict(list)
    for index, stratum in enumerate(strata):
        groups[stratum].append(index)

    rng = random.Random(seed)
    validation_indices = []
    train_indices = []
    for key in sorted(groups, key=str):
        indices = list(groups[key])
        rng.shuffle(indices)
        if len(indices) <= 1:
            train_indices.extend(indices)
            continue
        validation_size = max(1, round(len(indices) * validation_fraction))
        validation_size = min(validation_size, len(indices) - 1)
        validation_indices.extend(indices[:validation_size])
        train_indices.extend(indices[validation_size:])

    if not validation_indices or not train_indices:
        raise ValueError("validation split produced an empty partition")
    return Subset(dataset, sorted(train_indices)), Subset(dataset, sorted(validation_indices))


def clone_model_state(model):
    module = model.module if hasattr(model, "module") else model
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def restore_model_state(model, state):
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(state)


def write_metrics(path, payload):
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
