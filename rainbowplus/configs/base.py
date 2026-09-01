
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class LLMConfig:
    """Configuration for a Language Model."""

    # Default "vllm" keeps every existing configs/*.yml (which never set
    # this field on this branch) loading a local model exactly as before.
    # Set to "openai" to route through an OpenAI-compatible HTTP API instead
    # -- see apply_api_override() in rainbowplus.py for the .env-driven path.
    type_: str = "vllm"
    api_key: str = None
    base_url: str = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    sampling_params: Dict[str, Any] = field(default_factory=dict)
