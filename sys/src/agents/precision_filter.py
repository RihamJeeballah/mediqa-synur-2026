# src/agents/precision_filter.py
import json
from typing import Any, Dict, List

from src.lm_utils import generate_response, extract_json_from_response


class PrecisionFilterAgent:
    def __init__(
        self,
        model: str,
        schema_by_id: Dict[str, Dict[str, Any]],
        max_tokens: int = 450,
        temperature: float = 0.0,
    ):
        self.model = model
        self.schema_by_id = schema_by_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _safe_str(self, x: Any) -> str:
        try:
            return str(x)
        except Exception:
            return ""

    def decide_keep_drop(self, obs: Dict[str, Any], transcript: str) -> str:
        cid = self._safe_str(obs.get("id", "")).strip()
        schema_item = self.schema_by_id.get(cid, {})
        name = schema_item.get("name", obs.get("name", "")) or ""
        vtype = schema_item.get("value_type", obs.get("value_type", "")) or ""
        enum_list = schema_item.get("value_enum", []) or []

        value = obs.get("value", None)
        evidence = obs.get("evidence", "")

        prompt = f"""
You are validating extracted clinical observations for a benchmark evaluation.

TASK:
Decide whether this single extracted observation should be KEPT or DROPPED.

VERY IMPORTANT:
- You must be STRICT.
- Do NOT keep interpretations or abstractions.
- Do NOT keep negatives unless explicitly negated.

OUTPUT (JSON ONLY):
{{ "decision": "KEEP" or "DROP", "reason": "<short reason>" }}

OBSERVATION:
id: {cid}
name: {name}
value_type: {vtype}
value_enum: {json.dumps(enum_list, ensure_ascii=False)}
value: {json.dumps(value, ensure_ascii=False)}
evidence: {json.dumps(evidence, ensure_ascii=False)}

TRANSCRIPT:
{transcript}
""".strip()

        raw = generate_response(
            self.model,
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        parsed = extract_json_from_response(raw)

        if isinstance(parsed, dict):
            decision = self._safe_str(parsed.get("decision", "")).strip().upper()
            if decision in {"KEEP", "DROP"}:
                return decision

        return "DROP"

    def filter_observations(self, observations: List[Dict[str, Any]], transcript: str) -> List[Dict[str, Any]]:
        kept = []
        for o in observations:
            if not isinstance(o, dict):
                continue
            decision = self.decide_keep_drop(o, transcript)
            if decision == "KEEP":
                kept.append(o)
        return kept