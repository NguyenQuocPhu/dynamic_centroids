
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from openai import OpenAI
from rainbowplus.llms.base import BaseLLM
from rainbowplus.rate_limit import throttle
from pydantic import BaseModel
import time
import httpx
import openai

from typing import List


class LLMviaOpenAI(BaseLLM):
    def __init__(self, config):
        self.config = config
        self.model_kwargs = config.model_kwargs
        # max_retries=0: the SDK's built-in retry bypasses our global rate
        # gate (throttle()), so a burst of 5xx/429 retries would exceed
        # LLM_MAX_RPS. Retries are handled by our own throttled loop below.
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, max_retries=0, timeout=90.0)

    def get_name(self):
        return self.model_kwargs["model"]

    def _with_thinking_disabled(self, sampling_params: dict) -> dict:
        """Default enable_thinking=false, matching what algo/beam-search and
        algo/mcts already do in OpenAICompatModel._clean_sampling_kwargs().

        Without it this Qwen3.5 "thinking" endpoint spends the whole max_tokens
        budget on invisible reasoning and returns content=None -- with the
        mutator's max_tokens=128 that happened on EVERY call, so every mutation
        came back as "" and the baseline produced an archive of empty prompts
        (reproduced live). The flag has to travel inside extra_body:
        chat_template_kwargs is not a real Chat Completions parameter, so the
        OpenAI SDK rejects it as a bare top-level kwarg. Callers can still
        override by passing their own extra_body["chat_template_kwargs"].
        """
        params = dict(sampling_params or {})
        extra_body = dict(params.get("extra_body") or {})
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": False,
            **extra_body.get("chat_template_kwargs", {}),
        }
        params["extra_body"] = extra_body
        return params

    def generate(self, query: str, sampling_params: dict, max_retries: int = 5, backoff: float = 2.0):
        last_exception = None
        params = self._with_thinking_disabled(sampling_params)
        for attempt in range(max_retries):
            try:
                throttle(self.config.base_url)   # per-endpoint rate cap (rate_limit.py), re-gated each retry
                response = self.client.chat.completions.create(
                    model=self.model_kwargs["model"],
                    messages=[
                        {"role": "user", "content": query},
                    ],
                    **params
                )
                # Some "thinking" models (e.g. Qwen3.5 via this proxy) can spend the
                # entire max_tokens budget on invisible reasoning and return
                # content=None once the budget runs out (reproduced live testing
                # this against the real proxy: finish_reason "length", content
                # None). Every downstream caller expects a str (nltk.word_tokenize,
                # str concatenation, etc.), so a bare None here crashes call sites
                # far from this one -- return "" instead, same as "model said
                # nothing", which every caller already handles.
                return response.choices[0].message.content or ""
            except (openai.APIConnectionError, openai.RateLimitError, openai.APITimeoutError,
                    openai.InternalServerError, httpx.HTTPError, httpx.RemoteProtocolError) as e:
                last_exception = e
                print(f"OpenAI API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff * (2 ** attempt)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)  # Exponential backoff
                else:
                    # Graceful degradation instead of killing a multi-hour run:
                    # after exhausting retries, return "" (same contract as "model
                    # said nothing"). Downstream handles empty -- empty mutation is
                    # filtered / falls back, empty target response scores as refusal,
                    # empty judge output defaults. One lost item beats a dead run.
                    print(f"All {max_retries} attempts failed ({e}); returning '' to keep the run alive.")
                    return ""
        return ""

    def batch_generate(self, queries: List[str], sampling_params: dict):
        responses = [self.generate(query, sampling_params) for query in queries]
        return responses

    def generate_format(
        self, query: str, sampling_params: dict, response_format: BaseModel
    ):
        completion = self.client.beta.chat.completions.parse(
            model=self.model_kwargs["model"],
            messages=[
                {"role": "user", "content": query},
            ],
            **sampling_params,
            response_format=response_format
        )
        message = completion.choices[0].message
        return message.parsed