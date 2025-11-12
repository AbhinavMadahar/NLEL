
from typing import Tuple, Dict, Any, Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Expect TextModel base in same package
try:
    from .base import TextModel
except Exception:
    # Minimal shim if base imports later
    class TextModel:
        def generate(self, prompt: str, **kwargs):
            raise NotImplementedError

class HFLocalTextModel(TextModel):
    """
    Local (CPU/GPU) text-generation backend using Hugging Face Transformers.

    Offline by default (local_files_only=True). Set TRANSFORMERS_OFFLINE=1 for hard offline runs.
    Environment knobs:
      HF_LOCAL_DEVICE: 'cuda' | 'cpu' | 'mps' | 'auto' (default 'auto')
      HF_LOCAL_DTYPE:  'auto' | 'float32' | 'float16' | 'bfloat16' (default 'auto')
      HF_USE_4BIT:     '1' -> enable 4-bit quantization (requires bitsandbytes and NVIDIA GPU)
    """
    def __init__(self,
                 model_name_or_path: str,
                 device: Optional[str] = None,
                 dtype: Optional[str] = None,
                 local_files_only: bool = True,
                 load_in_4bit: Optional[bool] = None,
                 device_map: Optional[str] = None,
                 trust_remote_code: bool = False) -> None:
        self.model_name = model_name_or_path
        self.local_files_only = local_files_only

        dev_pref = (device or os.getenv("HF_LOCAL_DEVICE") or "auto").lower()
        if dev_pref == "auto":
            if torch.cuda.is_available():
                dev_pref = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                dev_pref = "mps"
            else:
                dev_pref = "cpu"
        self.device = dev_pref

        dtype_pref = (dtype or os.getenv("HF_LOCAL_DTYPE") or "auto").lower()
        if dtype_pref == "auto":
            if self.device == "cuda":
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif self.device == "mps":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            mapd = {
                "float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            }
            if dtype_pref not in mapd:
                raise ValueError(f"Unsupported HF_LOCAL_DTYPE: {dtype_pref}")
            torch_dtype = mapd[dtype_pref]
        self.torch_dtype = torch_dtype

        if load_in_4bit is None:
            load_in_4bit = os.getenv("HF_USE_4BIT", "0") == "1"
        self.load_in_4bit = bool(load_in_4bit)
        self.device_map = device_map or ("auto" if self.load_in_4bit else None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            use_fast=True,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = dict(
            local_files_only=self.local_files_only,
            torch_dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except Exception as e:
                raise RuntimeError("HF_USE_4BIT=1 requires 'bitsandbytes' with a compatible NVIDIA GPU.") from e
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = self.device_map or "auto"
        else:
            if self.device_map is not None:
                model_kwargs["device_map"] = self.device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        if not self.load_in_4bit and self.device_map is None:
            self.model.to(self.device)
        self.model.eval()

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def generate(self, prompt: str, **decode_kwargs):
        temperature = float(decode_kwargs.get("temperature", 0.0))
        top_p = float(decode_kwargs.get("top_p", 1.0))
        repetition_penalty = float(decode_kwargs.get("repetition_penalty", 1.0))
        max_new_tokens = int(decode_kwargs.get("max_tokens", 256))

        do_sample = (temperature > 1e-6) or (top_p < 0.999)

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        if not self.load_in_4bit and self.device_map is None:
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(0.01, temperature) if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        new_tokens = outputs[0, input_ids.shape[1]:]
        completion_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        usage = {
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": int(new_tokens.shape[0]),
        }
        return completion_text, {"usage": usage}
