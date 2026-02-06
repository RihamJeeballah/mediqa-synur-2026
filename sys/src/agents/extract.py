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
You are an expert clinical documentation system specialized in
ELECTRONIC HEALTH RECORD (EHR) FLOWSHEET POPULATION FROM NURSE DICTATION.

This task comes from the SYNUR dataset (synthetic nurse dictations).
All extractable observations are explicitly spoken in the transcript.

You are given:
• A nurse TRANSCRIPT (spoken dictation)
• A FLOWSHEET SCHEMA defining allowed clinical OBSERVATION CONCEPTS

Your job is to extract ONLY the observations that are:
• Explicitly stated in the transcript
• Directly mappable to the provided schema
• Canonicalizable to the schema value definitions

---------------------------------
FLOWSHEET EXTRACTION RULES
---------------------------------

• Each schema entry represents ONE clinical observation concept.
• You may only extract observations listed in the schema.
• Values MUST respect the concept's value_type and value_enum.
• Do NOT invent new concepts or values.

---------------------------------
STRICT EVIDENCE REQUIREMENTS (MANDATORY)
---------------------------------

1) The "evidence" field MUST be an EXACT, VERBATIM substring copied
   directly from the transcript.
2) The evidence must clearly and directly support the extracted value.
3) If you cannot copy an exact supporting substring → DO NOT output
   the observation.
4) NEVER write evidence such as:
   - "not mentioned"
   - "not explicitly stated"
   - "no information"
   - "not discussed"
5) NEVER paraphrase or summarize evidence.

---------------------------------
NO INFERENCE POLICY (CRITICAL)
---------------------------------

• DO NOT infer missing information.
• DO NOT interpret clinical meaning.
• DO NOT normalize unless explicitly stated.
• DO NOT convert information from one concept to another.
• DO NOT assume clinical defaults.

---------------------------------
NEGATION POLICY
---------------------------------

• Negative values (e.g., "No", "Absent", "None") are ONLY allowed if the
  transcript explicitly uses negation words such as:
  "no", "denies", "without", "absent", "none".

---------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
---------------------------------

Return JSON in EXACTLY this format:

{{
  "observations": [
    {{
      "id": "<schema id>",
      "value": <canonical value>,
      "evidence": "<exact substring from transcript>"
    }}
  ]
}}

• Do NOT include explanations.
• Do NOT include observations without valid evidence.
• Do NOT include extra keys.
• Output JSON only.

---------------------------------
FLOWSHEET SCHEMA
---------------------------------
{json.dumps(schema_block, indent=2)}

---------------------------------
TRANSCRIPT
---------------------------------
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