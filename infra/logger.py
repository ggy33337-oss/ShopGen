import json
from datetime import datetime
from pathlib import Path


def write_log(payload, log_path="logs/llm.jsonl"):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload["created_at"] = datetime.now().isoformat()

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
