import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class AuditLog:
    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None

    def record(self, event: str, subject: str, outcome: str) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "subject": subject,
            "outcome": outcome,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
