
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

    def generate(self, query: str, sampling_params: dict, max_retries: int = 5, backoff: float = 2.0):
        last_exception = None
        for attempt in range(max_retries):
            try:
                throttle()   # global rate cap (LLM_MAX_RPS), re-gated each retry
                response = self.client.chat.completions.create(
                    model=self.model_kwargs["model"],
                    messages=[
                        {"role": "user", "content": query},
                    ],
                    **sampling_params
                )
                return response.choices[0].message.content
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
                    # return "" (same contract as "model said nothing").
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