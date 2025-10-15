from typing import List, Dict, Any
class Ledger:
    def __init__(self, max_rows: int = 32):
        self.max_rows = max_rows; self.rows: List[Dict[str, Any]] = []
    def add(self, row: Dict[str, Any]):
        self.rows.append(row)
        if len(self.rows) > self.max_rows: self.rows = self.rows[-self.max_rows:]
    def render_block(self) -> str:
        if not self.rows: return "(empty)"
        out = []
        for r in self.rows[-self.max_rows:]:
            out.append(str({"L": r.get("L"), "Π": r.get("Pi"), "µ": r.get("mu"), "σ": r.get("sigma"), "accept": r.get("accept"), "cost": r.get("cost")}))
        return "\n".join(out)
