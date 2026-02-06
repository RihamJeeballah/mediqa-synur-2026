# src/schema.py
import json
from typing import Dict, Any, List


class SynurSchema:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.schema: List[Dict[str, Any]] = json.load(f)

        for item in self.schema:
            item["id"] = str(item["id"]).strip()

        self.by_id: Dict[str, Dict[str, Any]] = {item["id"]: item for item in self.schema}

    def get(self, obs_id: str):
        return self.by_id.get(str(obs_id).strip())

    def value_type(self, obs_id: str):
        item = self.get(obs_id)
        return item.get("value_type") if item else None

    def value_enum(self, obs_id: str):
        item = self.get(obs_id)
        return item.get("value_enum", []) if item else []
