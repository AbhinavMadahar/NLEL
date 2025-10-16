from typing import Tuple, Dict, Any, Optional
import os, json
from .base import TextModel
try:
    import boto3  # type: ignore
except Exception:
    boto3 = None

class BedrockTextModel(TextModel):
    """
    Minimal Bedrock runtime adapter.

    Environment:
      - AWS_REGION (or explicit region_name kwarg)
      - BEDROCK_PROVIDER (optional: 'anthropic'|'cohere'|'meta'), else inferred from modelId prefix
    """
    def __init__(self, model_id: str, region_name: Optional[str] = None, provider: Optional[str] = None):
        self.model_id = model_id
        self.region_name = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self.provider = provider or self._infer_provider(model_id)
        if boto3 is None:
            raise RuntimeError("boto3 is required for BedrockTextModel but is not installed.")
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def _infer_provider(self, model_id: str) -> str:
        if model_id.startswith("anthropic."): return "anthropic"
        if model_id.startswith("cohere."):    return "cohere"
        if model_id.startswith("meta."):      return "meta"
        if model_id.startswith("mistral."):   return "mistral"
        if model_id.startswith("ai21."):      return "ai21"
        return os.getenv("BEDROCK_PROVIDER", "anthropic")

    def generate(self, prompt: str, **decode_kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Supports a small cross-provider subset:
          - temperature, top_p, max_tokens (if present)
        NOTE: Bedrock providers each have their own schema; this adapter covers Anthropic Claude 3 and Cohere Command-R.
              For others, adjust the payload mapping below.
        """
        temperature = float(decode_kwargs.get("temperature", 0.2))
        top_p      = float(decode_kwargs.get("top_p", 0.95))
        max_tokens = int(decode_kwargs.get("max_tokens", 256))

        if self.provider == "anthropic":
            # Claude 3 via Bedrock: messages API
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        elif self.provider == "cohere":
            # Command-R via Bedrock
            body = {
                "message": prompt,
                "temperature": temperature,
                "p": top_p,
                "max_tokens": max_tokens,
            }
        else:
            # Generic fallback
            body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}

        resp = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        raw = resp.get("body")
        text = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        if hasattr(raw, "read"):
            raw_text = raw.read().decode("utf-8")
            try:
                obj = json.loads(raw_text)
            except Exception:
                return raw_text, {"usage": usage}
        else:
            obj = raw or {}

        if self.provider == "anthropic":
            # {"content":[{"type":"text","text":"..."}], "usage":{"input_tokens":..,"output_tokens":..}}
            if isinstance(obj.get("content"), list) and obj["content"]:
                text = obj["content"][0].get("text", "")
            u = obj.get("usage", {})
            usage["prompt_tokens"] = int(u.get("input_tokens", 0))
            usage["completion_tokens"] = int(u.get("output_tokens", 0))
        elif self.provider == "cohere":
            # {"text":"..."} or structured "output" with meta.billed_units
            if "text" in obj:
                text = obj.get("text", "")
            elif "output" in obj:
                try:
                    text = obj["output"][0]["content"][0]["text"]
                except Exception:
                    text = ""
            billed = obj.get("meta", {}).get("billed_units", {})
            usage["prompt_tokens"] = int(billed.get("input_tokens", 0))
            usage["completion_tokens"] = int(billed.get("output_tokens", 0))
        else:
            text = obj.get("generation") or obj.get("output") or obj.get("text") or ""

        return text, {"usage": usage}