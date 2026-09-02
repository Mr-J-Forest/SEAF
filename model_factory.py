"""Model factory for SEAF and the retained comparison adapters."""

import torch.nn as nn

from seaf_model import SEAFNet
from dynaseaf_model import DynaSEAFNet


def create_ocean_model(config: dict) -> nn.Module:
    model_type = str(config.get("model_type", "seaf")).lower()
    if model_type == "seaf":
        return SEAFNet(config)
    if model_type == "dynaseaf":
        return DynaSEAFNet(config)

    from recent_baseline_models import create_recent_baseline, is_recent_baseline

    if is_recent_baseline(model_type):
        return create_recent_baseline(config)

    from paper_reimplementation_models import (
        create_paper_reimplementation_model,
        is_paper_reimplementation,
    )

    if is_paper_reimplementation(model_type):
        return create_paper_reimplementation_model(config)
    raise ValueError(f"未知 model_type: {config.get('model_type')!r}")
