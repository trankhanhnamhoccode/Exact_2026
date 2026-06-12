from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass
class TransformersQwenSchemaClient:
    """
    Kaggle/GPU implementation of SchemaLLMClient using Hugging Face Transformers.

    Default model:
        Qwen/Qwen2.5-7B-Instruct

    This module is lazy-loaded:
    importing it does not import torch/transformers or load the model.
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 0.9
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    device_map: str = "auto"
    system_message: str = (
        "You are a precise schema extraction engine. "
        "Return only one valid JSON object. Do not solve the physics problem."
    )

    _tokenizer: Any = None
    _model: Any = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Missing transformers. On Kaggle, run: "
                "`pip install -q transformers accelerate sentencepiece`"
            ) from exc

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.load_in_4bit:
            try:
                import torch
                from transformers import BitsAndBytesConfig
            except Exception as exc:
                raise RuntimeError(
                    "4-bit loading requires bitsandbytes. On Kaggle, run: "
                    "`pip install -q bitsandbytes accelerate`, "
                    "or set XAI_QWEN_LOAD_IN_4BIT=0."
                ) from exc

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()

        import torch

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        chat_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer([chat_text], return_tensors="pt")

        if hasattr(inputs, "to"):
            inputs = inputs.to(self._model.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            generated = self._model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        output_ids = generated[0][input_len:]
        text = self._tokenizer.decode(output_ids, skip_special_tokens=True)

        return text.strip()


def qwen_client_from_env() -> TransformersQwenSchemaClient:
    return TransformersQwenSchemaClient(
        model_name=os.environ.get("XAI_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        max_new_tokens=int(os.environ.get("XAI_QWEN_MAX_NEW_TOKENS", "768")),
        temperature=float(os.environ.get("XAI_QWEN_TEMPERATURE", "0.0")),
        top_p=float(os.environ.get("XAI_QWEN_TOP_P", "0.9")),
        load_in_4bit=os.environ.get("XAI_QWEN_LOAD_IN_4BIT", "0") != "0",
        trust_remote_code=os.environ.get("XAI_QWEN_TRUST_REMOTE_CODE", "1") != "0",
        device_map=os.environ.get("XAI_QWEN_DEVICE_MAP", "auto"),
    )
