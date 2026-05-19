import warnings
import logging
import os
from typing import List, Optional, Dict, Any

from vllm import LLM, SamplingParams
from fastchat.conversation import get_conv_template
from easyjailbreak.models.model_base import ModelBase
from rainbowplus.graph_easyjailbreak.model_utils import count_calls, count_batch_calls

from fastchat.conversation import (
    Conversation,
    SeparatorStyle,
    register_conv_template,
)

register_conv_template(
    Conversation(
        name="meta-llama-3.1",
        system_message=(
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: {{currentDateTimev2}}"
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="\n",
        stop_str="<|eot_id|>",
    )
)

logger = logging.getLogger(__name__)


class SafeTokenizer:
    # SỬA LỖI: Thêm tham số eos_token_id_override để nhận fallback value
    def __init__(self, real_tokenizer, eos_token_id_override=None):
        self._tok = real_tokenizer
        self.eos_token_id = eos_token_id_override or real_tokenizer.eos_token_id
        self.pad_token_id = getattr(
            real_tokenizer,
            "pad_token_id",
            self.eos_token_id
        )

    def __getattr__(self, name):
        return getattr(self._tok, name)

class VLLMModel(ModelBase):
    def __init__(
        self,
        model_name: str,
        model_name_or_path: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **vllm_kwargs,
    ):
        self.model_name = model_name
        _path = model_name_or_path or model_name

        super().__init__()

        logger.info(f"[VLLMModel] Loading '{_path}' with vLLM …")
        self._llm = LLM(model=_path, **vllm_kwargs)

        # --- Conversation template ---
        try:
            self.conversation = get_conv_template(model_name)
            logger.info(f"[VLLMModel] Successfully loaded conversation template: {self.conversation.name}")
            print(f"✅ Loaded conversation template: {self.conversation.name} for model: {_path}")
        except KeyError:
            logger.error(
                f"[VLLMModel] Invalid model_name '{model_name}'. "
                "See https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py"
            )
            raise

        if model_name == "llama-2":
            self.conversation.sep2 = self.conversation.sep2.strip()

        if model_name == "zero_shot":
            # SỬA LỖI: Bỏ .template, FastChat lưu trực tiếp ở .roles
            self.conversation.roles = tuple(
                ["### " + r for r in self.conversation.roles]
            )
            self.conversation.sep = "\n"

        # --- Tokenizer shim ---
        # SỬA LỖI: Truyền đúng tokenizer object vào tham số đầu tiên
        real_tok = self._llm.get_tokenizer()
        _eos = real_tok.eos_token_id or 2
        self.tokenizer = SafeTokenizer(real_tok, eos_token_id_override=_eos)

        # --- Generation config ---
        self.generation_config: Dict[str, Any] = generation_config if generation_config is not None else {}

    def set_system_message(self, system_message: str) -> None:
        self.conversation.system_message = system_message

    def get_name(self) -> str:
        return self.model_name

    def clear_conversation(self) -> None:
        self.conversation.messages = []
    
    def create_conversation_prompt(self, messages, clear_old_history: bool = True) -> str:
        if clear_old_history:
            self.conversation.messages = []
        if isinstance(messages, str):
            messages = [messages]
        for index, message in enumerate(messages):
            self.conversation.append_message(self.conversation.roles[index % 2], message)
        self.conversation.append_message(self.conversation.roles[-1], None)
        return self.conversation.get_prompt()

    def _build_sampling_params(self, **override_kwargs) -> SamplingParams:
        cfg = dict(self.generation_config)
        cfg.update(override_kwargs)

        max_tokens = cfg.get("max_new_tokens") or cfg.get("max_tokens") or 512
        temperature = float(cfg.get("temperature", 1.0))
        top_p = float(cfg.get("top_p", 1.0))

        stop_token_ids = None
        if "eos_token_id" in cfg:
            eid = cfg["eos_token_id"]
            stop_token_ids = [eid] if isinstance(eid, int) else list(eid)

        if temperature == 0.0:
            temperature = 1e-6

        return SamplingParams(
            max_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
        )

    def generate(self, messages, clear_old_history: bool = True, **kwargs) -> str:
        if isinstance(messages, str):
            messages = [messages]

        prompt = self.create_conversation_prompt(messages, clear_old_history=clear_old_history)
        sampling_params = self._build_sampling_params(**kwargs)

        # TỐI ƯU: Thêm use_tqdm=False để tránh in rác ra màn hình terminal khi generate
        outputs = self._llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    @count_batch_calls
    def batch_generate(self, conversations, **kwargs) -> List[str]:
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
        
        # TỐI ƯU: Thêm use_tqdm=False
        outputs = self._llm.generate(prompt_list, sampling_params=sampling_params, use_tqdm=False)
        return [out.outputs[0].text for out in outputs]

    @property
    def model(self):
        return self._llm
    
def from_pretrained(
    model_name: str,
    model_name_or_path: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    **vllm_kwargs,
) -> VLLMModel:
    return VLLMModel(
        model_name=model_name,
        model_name_or_path=model_name_or_path,
        generation_config=generation_config,
        **vllm_kwargs,
    )
