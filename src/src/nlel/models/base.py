from typing import List, Dict, Any, Optional, Tuple
import os, json
from ..tokens import approx_tokens

class TextModel:
    def generate(self, prompt: str, **decode_kwargs) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError
    def batch_generate(self, prompts: List[str], **decode_kwargs) -> List[Tuple[str, Dict[str, Any]]]:
        return [self.generate(p, **decode_kwargs) for p in prompts]

class DummyModel(TextModel):
    def __init__(self, mode: str = "tiny"):
        self.mode = mode
    def generate(self, prompt: str, **decode_kwargs) -> Tuple[str, Dict[str, Any]]:
        import json
        if 'Emit **JSON only**' in prompt or 'JSON object' in prompt:
            s = json.dumps({"temperature":0.2,"top_p":0.9,"max_tokens":64,"repetition_penalty":1.0,"gen_count":2,"branch_quota":2,"beta":0.15,"verify_passes":1,"verify_strictness":0.5,"retrieval_weights":{"general":0.0,"math-lemmas":0.0}})
        elif 'edge labels' in prompt and 'Emit up to' in prompt:
            s = "work backward; seek a counterexample; call retrieval; summarize first"
        elif 'Return only ACCEPT or REJECT' in prompt:
            s = "ACCEPT"
        elif 'Respond as JSON' in prompt and '"mu"' in prompt:
            s = '{"mu": 0.45, "sigma": 0.50}'
        elif 'Final Answer:' in prompt:
            s = "Reasoning...\nFinal Answer: 42"
        else:
            s = "Thought: try a simpler sub-problem."
        usage = {"prompt_tokens": approx_tokens(prompt), "completion_tokens": approx_tokens(s)}
        return s, {"usage": usage}

class OpenAIChatModel(TextModel):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from openai import OpenAI
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url); self.model = model
    def generate(self, prompt: str, **decode_kwargs) -> Tuple[str, Dict[str, Any]]:
        messages=[{"role":"user","content":prompt}]
        params = dict(model=self.model, messages=messages)
        if "temperature" in decode_kwargs: params["temperature"]=decode_kwargs["temperature"]
        if "top_p" in decode_kwargs: params["top_p"]=decode_kwargs["top_p"]
        if "max_tokens" in decode_kwargs: params["max_tokens"]=decode_kwargs["max_tokens"]
        if "repetition_penalty" in decode_kwargs:
            rp = float(decode_kwargs["repetition_penalty"])
            params["frequency_penalty"] = max(-2.0, min(2.0, 1.0 - rp))
        resp = self.client.chat.completions.create(**params)
        msg = resp.choices[0].message.content or ""
        usage = {"prompt_tokens": getattr(resp.usage, "prompt_tokens", 0), "completion_tokens": getattr(resp.usage, "completion_tokens", 0)}
        return msg, {"usage": usage}

def get_model(spec: str):
    """Resolve a model spec into a TextModel.

    Supported:
      - "hf:<model_or_path>" or "local:<...>" — Hugging Face Transformers (offline by default)
      - "dummy:<mode>" — test stub

    No external APIs are used by this resolver.
    """
    if ":" in spec:
        kind, name = spec.split(":", 1)
    else:
        kind, name = "dummy", spec

    if kind == "dummy":
        return DummyModel(mode=name)
    if kind in ("hf", "local", "transformers"):
        from .hf_local import HFLocalTextModel
        return HFLocalTextModel(model_name_or_path=name, local_files_only=True)

    raise ValueError(f"Unsupported model spec for local-only resolver: {spec}")
