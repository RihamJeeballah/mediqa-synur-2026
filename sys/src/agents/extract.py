# src/agents/extract.py
import json
from typing import List, Dict, Any
from src.lm_utils import generate_response, extract_json_from_response


class ExtractorAgent:
    def __init__(self, model: str, schema_by_id: Dict[str, Dict[str, Any]]):
        self.model = model
        self.schema_by_id = schema_by_id

    def _build_schema_block(self, concept_ids: List[str]) -> List[Dict[str, Any]]:
        block = []
        for cid in concept_ids:
            if cid in self.schema_by_id:
                s = self.schema_by_id[cid]
                block.append({
                    "id": cid,
                    "name": s.get("name", ""),
                    "value_type": s.get("value_type", ""),
                    "value_enum": s.get("value_enum", []) or [],
                })
        return block

    def _parse_observations(self, raw: str) -> List[Dict[str, Any]]:
        parsed = extract_json_from_response(raw)
        if isinstance(parsed, dict):
            obs = parsed.get("observations", [])
            return obs if isinstance(obs, list) else []
        if isinstance(parsed, list):
            return parsed
        return []

    def run(self, transcript: str, concept_ids: List[str]) -> List[Dict[str, Any]]:
        if not isinstance(transcript, str) or not transcript.strip() or not concept_ids:
            return []

        schema_block = self._build_schema_block(concept_ids)
        if not schema_block:
            return []

        prompt = f"""
You are a clinical information extraction system.

TASK:
Extract ONLY observations that are explicitly stated in the transcript.

STRICT EVIDENCE RULES (VERY IMPORTANT):
1) The field "evidence" MUST be an EXACT substring copied from the transcript (verbatim).
2) Do NOT write evidence like: "no mention", "not explicitly stated", "not mentioned".
3) Do NOT use hedging in evidence: "could", "likely", "possibly", "suggest", "indicate", "maybe".
4) If you cannot copy a supporting substring from the transcript, SKIP the observation.

NO-INFERENCE RULES:
- Do NOT infer new information.
- Do NOT guess missing values.
- Do NOT convert a number mentioned for one concept into a value for another concept.

NEGATION RULE:
- Only output negative values (e.g., "No", "None", "Absent") if the transcript explicitly negates it
  using words like: "no", "denies", "without", "absent", "none".

OUTPUT FORMAT (JSON ONLY):
{{
  "observations": [
    {{
      "id": "<id>",
      "value": <value>,
      "evidence": "<EXACT copied substring from transcript>"
    }}
  ]
}}

SCHEMA:
{json.dumps(schema_block, indent=2)}

TRANSCRIPT:
{transcript}
""".strip()


        raw = generate_response(self.model, prompt, temperature=0.0, max_tokens=700)
        obs = self._parse_observations(raw)

        clean: List[Dict[str, Any]] = []
        for o in obs:
            if not isinstance(o, dict):
                continue

            cid = str(o.get("id", "")).strip()
            if cid not in self.schema_by_id:
                continue

            if "value" not in o:
                continue

            evidence = o.get("evidence", "")
            if not isinstance(evidence, str) or not evidence.strip():
                continue

            clean.append({
                "id": cid,
                "name": self.schema_by_id[cid].get("name", ""),
                "value": o.get("value"),
                "evidence": evidence.strip(),
            })

        return clean