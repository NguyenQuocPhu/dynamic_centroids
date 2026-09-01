#
# OpenAI-compatible proxy backend for attack_model / target_model / eval_model.
#
# Lets all three GAP/MCTS roles run through a single hosted API (e.g. the
# proxy.onebot.meobeo.ai Qwen endpoint) instead of loading 3 separate local
# vLLM models. Wired up via OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL_ID
# in rainbowplus.py -- see apply_api_override() there.
#
import logging
import time
from typing import Any, Dict, List, Optional, Union

import httpx
import openai
from openai import OpenAI
from fastchat.model import get_conversation_template
from easyjailbreak.models.openai_model import OpenaiModel

from rainbowplus.graph_easyjailbreak.model_utils import count_calls, count_batch_calls

logger = logging.getLogger(__name__)

# Sampling knobs from configs/*.yml that make sense for local vLLM
# (gpu_memory_utilization, max_model_len, ...) or for vLLM's own logprobs-based
# scoring, but that the OpenAI Chat Completions API either rejects outright or
# doesn't need -- BatchEvaluatorGenerativeGetScore parses the judge's score by
# regexing the *text* output ("Rating: [[7]]"), it never reads logprobs, so
# dropping these is safe rather than a loss of behavior.
_UNSUPPORTED_KWARGS = ("eos_token_id", "do_sample", "max_new_tokens", "logprobs")


class OpenAICompatModel(OpenaiModel):
    """
    Subclasses easyjailbreak's OpenaiModel *only* so the `isinstance(model,
    OpenaiModel)` branches already present in GAP.py / IntrospectGeneration /
    BatchEvaluatorGenerativeGetScore route through the right code path.

    We do NOT call super().__init__() -- this sandbox doesn't have
    easyjailbreak installed to verify the parent constructor's exact
    signature/kwargs, so instead of guessing at it, every attribute GAP.py
    actually touches (verified by grep against this repo, not against
    easyjailbreak's docs) is set here directly:
      - self.model_name         (GAP.py get_name(), IntrospectGeneration's
                                   conv_template(self.model.model_name, ...))
      - self.conversation        (GAP.py does `target_model.conversation.messages = []`
                                   unconditionally, before the isinstance branch)
      - generate() / batch_generate() / get_name() / set_system_message()
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        generation_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        retry_backoff: float = 2.0,
    ):
        self.model_name = model_name
        self.generation_config = dict(generation_config) if generation_config else {}
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # Generic blank conversation, same role this plays for VLLMModel
        # (get_conv_template) -- get_conversation_template() falls back to a
        # generic template for a name it doesn't recognize (our model_name is
        # an API model id, not a FastChat template name), it doesn't raise.
        self.conversation = get_conversation_template(model_name)

    def get_name(self) -> str:
        return self.model_name

    def set_system_message(self, system_message: str) -> None:
        self.conversation.system_message = system_message

    def clear_conversation(self) -> None:
        self.conversation.messages = []

    def _to_openai_messages(self, messages: Union[str, List[str]]) -> List[Dict[str, str]]:
        """Alternating user/assistant turns, same convention as
        IntrospectGeneration's `[message[1] for message in conv.messages]`
        (system message first, then turns starting with 'user')."""
        if isinstance(messages, str):
            messages = [messages]
        payload = []
        system_message = getattr(self.conversation, "system_message", None)
        if system_message:
            payload.append({"role": "system", "content": system_message})
        roles = ("user", "assistant")
        for i, content in enumerate(messages):
            payload.append({"role": roles[i % 2], "content": content})
        return payload

    def _clean_sampling_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(self.generation_config)
        params.update(kwargs)
        for key in _UNSUPPORTED_KWARGS:
            params.pop(key, None)

        # Default enable_thinking=false, same default as the proxy's own
        # production config -- verified live: without it, this Qwen3.5
        # "thinking" endpoint can burn the whole max_tokens budget on
        # invisible reasoning and return content=None (finish_reason
        # "length"), silently wasting a real query for a wasted mutation.
        # This has to travel inside extra_body: chat_template_kwargs isn't a
        # real Chat Completions param, so the OpenAI SDK rejects it as a bare
        # top-level kwarg (TypeError) -- it only reaches vLLM's proxy nested
        # under extra_body. generation_config/kwargs can still override by
        # setting extra_body={"chat_template_kwargs": {...}} explicitly.
        extra_body = dict(params.get("extra_body") or {})
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": False,
            **extra_body.get("chat_template_kwargs", {}),
        }
        params["extra_body"] = extra_body
        return params

    @count_calls
    def generate(self, messages: Union[str, List[str]], clear_old_history: bool = True, **kwargs) -> str:
        payload_messages = self._to_openai_messages(messages)
        params = self._clean_sampling_kwargs(kwargs)

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=payload_messages,
                    **params,
                )
                return response.choices[0].message.content or ""
            except (openai.APIConnectionError, openai.RateLimitError, httpx.HTTPError, httpx.RemoteProtocolError) as e:
                last_exception = e
                logger.warning(f"[OpenAICompatModel] API error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                else:
                    raise
        raise last_exception

    @count_batch_calls
    def batch_generate(self, conversations: List[Union[str, List[str]]], **kwargs) -> List[str]:
        return [self.generate(conv, **kwargs) for conv in conversations]
