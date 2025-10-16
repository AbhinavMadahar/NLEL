import os
def load_prompt(name: str) -> str:
    here = os.path.dirname(__file__)
    with open(os.path.join(here, name),"r",encoding="utf-8") as f:
        return f.read()
