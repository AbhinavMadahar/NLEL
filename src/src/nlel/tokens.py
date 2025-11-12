def approx_tokens(text: str) -> int:
    if not text: return 0
    return max(1, int(len(text) / 4))

class TokenBank:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
    def add(self, prompt: int = 0, completion: int = 0):
        self.prompt_tokens += int(prompt); self.completion_tokens += int(completion)
    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens
