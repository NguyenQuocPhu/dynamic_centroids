"""
VLLMModel: A drop-in replacement for HuggingfaceModel that uses vLLM for inference.

This wrapper mirrors the exact interface of HuggingfaceModel (easyjailbreak) so that
GAP.py, IntrospectGenerationAggregateMaxConcatenate.py, and rainbowplus.py can work
without modification — just swap isinstance() checks.
"""

import warnings
import logging
from typing import List, Optional, Dict, Any

from vllm import LLM, SamplingParams
from fastchat.conversation import get_conv_template
from easyjailbreak.models.model_base import ModelBase

logger = logging.getLogger(__name__)


class _DummyTokenizer:
    """Minimal tokenizer shim — vLLM handles tokenisation internally."""
    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id
        self.pad_token = "</s>"
        self.padding_side = "right"


class VLLMModel(ModelBase):
    """
    vLLM-backed model that is API-compatible with easyjailbreak's HuggingfaceModel.

    Key design decisions
    --------------------
    * Uses FastChat `get_conv_template` exactly the same way HuggingfaceModel does,
      so the conversation-formatting code in IntrospectGenerationAggregateMaxConcatenate
      and GAP continues to work unchanged.
    * `generate()` and `batch_generate()` accept the same signatures.
    * `generation_config` dict is supported (max_new_tokens / max_tokens, temperature,
      top_p, do_sample, eos_token_id).
    * Provides all properties that pipeline code probes (`tokenizer`, `model_name`,
      `conversation`, `generation_config`).
    """

    def __init__(
        self,
        model_name: str,
        model_name_or_path: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **vllm_kwargs,
    ):
        """
        Parameters
        ----------
        model_name : str
            FastChat template name (e.g. "llama-2", "qwen-7b-chat", "vicuna_v1.1").
        model_name_or_path : str, optional
            HuggingFace repo id / local path. Falls back to `model_name` when omitted.
        generation_config : dict, optional
            Same keys as HuggingfaceModel (max_new_tokens/max_tokens, temperature, top_p …).
        **vllm_kwargs
            Forwarded directly to `vllm.LLM()` (e.g. max_model_len, gpu_memory_utilization,
            enforce_eager, tensor_parallel_size …).
        """
        self.model_name = model_name
        _path = model_name_or_path or model_name

        super().__init__()

        logger.info(f"[VLLMModel] Loading '{_path}' with vLLM …")
        self._llm = LLM(model=_path, **vllm_kwargs)

        # --- Conversation template (identical to HuggingfaceModel) ---
        try:
            self.conversation = get_conv_template(model_name)
        except KeyError:
            logger.error(
                f"[VLLMModel] Invalid model_name '{model_name}'. "
                "See https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py"
            )
            raise

        if model_name == "llama-2":
            self.conversation.sep2 = self.conversation.sep2.strip()

        if model_name == "zero_shot":
            self.conversation.roles = tuple(
                ["### " + r for r in self.conversation.template.roles]
            )
            self.conversation.sep = "\n"

        # --- Tokenizer shim ---
        # vLLM's own tokenizer lives inside _llm; we expose a minimal shim so that
        # code like `self.target_model.tokenizer.eos_token_id` doesn't crash.
        _eos = self._llm.get_tokenizer().eos_token_id or 2
        self.tokenizer = _DummyTokenizer(eos_token_id=_eos)

        # --- Generation config (same dict as HuggingfaceModel) ---
        self.generation_config: Dict[str, Any] = generation_config if generation_config is not None else {}

    # ------------------------------------------------------------------
    # Interface parity with HuggingfaceModel
    # ------------------------------------------------------------------

    def set_system_message(self, system_message: str) -> None:
        """Sets the system message on the conversation template."""
        self.conversation.system_message = system_message

    def get_name(self) -> str:
        return self.model_name

    def clear_conversation(self) -> None:
        self.conversation.messages = []

    def create_conversation_prompt(self, messages, clear_old_history: bool = True) -> str:
        """Mirrors HuggingfaceModel.create_conversation_prompt exactly."""
        if clear_old_history:
            self.conversation.messages = []
        if isinstance(messages, str):
            messages = [messages]
        for index, message in enumerate(messages):
            self.conversation.append_message(self.conversation.roles[index % 2], message)
        self.conversation.append_message(self.conversation.roles[-1], None)
        return self.conversation.get_prompt()

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def _build_sampling_params(self, **override_kwargs) -> SamplingParams:
        """
        Build a SamplingParams from generation_config + any call-time overrides.

        Mapping (HuggingfaceModel → vLLM):
          max_new_tokens / max_tokens → max_tokens
          temperature                 → temperature
          top_p                       → top_p
          eos_token_id                → stop_token_ids
        """
        cfg = dict(self.generation_config)
        cfg.update(override_kwargs)

        max_tokens = cfg.get("max_new_tokens") or cfg.get("max_tokens") or 512
        temperature = float(cfg.get("temperature", 1.0))
        top_p = float(cfg.get("top_p", 1.0))

        stop_token_ids = None
        if "eos_token_id" in cfg:
            eid = cfg["eos_token_id"]
            stop_token_ids = [eid] if isinstance(eid, int) else list(eid)

        # vLLM requires temperature > 0 when top_p < 1; clamp to avoid assertion errors.
        if temperature == 0.0:
            temperature = 1e-6

        return SamplingParams(
            max_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
        )

    # ------------------------------------------------------------------
    # generate — mirrors HuggingfaceModel.generate
    # ------------------------------------------------------------------

    def generate_raw(self, raw_prompt: str, **kwargs) -> str:
        """
        Pass a pre-formatted prompt string directly to vLLM without any
        conversation-template wrapping. Use this when the caller has already
        built the full prompt (e.g. via conv.get_prompt()).
        """
        sampling_params = self._build_sampling_params(**kwargs)
        outputs = self._llm.generate([raw_prompt], sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def generate(self, messages, clear_old_history: bool = True, **kwargs) -> str:
        """
        Generate a single response.

        Parameters mirror HuggingfaceModel.generate:
          messages           : str | list[str]
          clear_old_history  : bool
          **kwargs           : generation overrides (max_new_tokens, temperature …)

        Returns a plain string (no chat tokens).
        """
        if isinstance(messages, str):
            messages = [messages]

        prompt = self.create_conversation_prompt(messages, clear_old_history=clear_old_history)
        sampling_params = self._build_sampling_params(**kwargs)

        outputs = self._llm.generate([prompt], sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # batch_generate — mirrors HuggingfaceModel.batch_generate
    # ------------------------------------------------------------------

    def batch_generate(self, conversations, **kwargs) -> List[str]:
        """
        Generate responses for a batch of conversations.

        Parameters mirror HuggingfaceModel.batch_generate:
          conversations : list[list[str]] | list[str]
          **kwargs      : generation overrides

        Returns a list of response strings.
        """
        prompt_list = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn(
                    "Pass list[str] per conversation to batch_generate() to avoid "
                    "treating the string as individual characters."
                )
                conversation = [conversation]
            prompt_list.append(self.create_conversation_prompt(conversation))

        sampling_params = self._build_sampling_params(**kwargs)
        outputs = self._llm.generate(prompt_list, sampling_params=sampling_params)
        return [out.outputs[0].text for out in outputs]

    # ------------------------------------------------------------------
    # Properties / shims expected by the pipeline
    # ------------------------------------------------------------------

    @property
    def model(self):
        """Return the underlying vLLM engine (for code that probes self.model)."""
        return self._llm

    # GAP.py probes self.target_model.conversation.messages = []
    # → already handled by self.conversation above.


# ---------------------------------------------------------------------------
# Factory function — mirrors easyjailbreak's from_pretrained signature
# ---------------------------------------------------------------------------

def from_pretrained(
    model_name: str,
    model_name_or_path: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    **vllm_kwargs,
) -> VLLMModel:
    """
    Create a VLLMModel.

    Parameters
    ----------
    model_name : str
        FastChat conversation template name (e.g. "llama-2", "qwen-7b-chat").
    model_name_or_path : str, optional
        HuggingFace hub id or local path. Defaults to model_name.
    generation_config : dict, optional
        Default generation settings (max_new_tokens, temperature, top_p …).
    **vllm_kwargs
        Forwarded to vllm.LLM (max_model_len, gpu_memory_utilization, enforce_eager …).
    """
    return VLLMModel(
        model_name=model_name,
        model_name_or_path=model_name_or_path,
        generation_config=generation_config,
        **vllm_kwargs,
    )
