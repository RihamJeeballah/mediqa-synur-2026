# src/agents/detect.py
from typing import List, Dict, Any
from src.lm_utils import generate_response, extract_json_from_response


class DetectorAgent:
    def __init__(self, model: str, schema: List[Dict[str, Any]]):
        self.model = model
        self.schema = schema

    def run(self, transcript: str) -> List[str]:
        schema_list = "\n".join([f"{c['id']}: {c.get('name', '')}" for c in self.schema])

        prompt = f"""
Detect ALL observation concepts that are explicitly mentioned or weakly implied.

Rules:
- Prefer recall over precision
- Do NOT guess values
- Return ONLY ids

OUTPUT:
{{ "concept_ids": ["id1", "id2"] }}

SCHEMA:
{schema_list}

TRANSCRIPT:
{transcript}
"""

        raw = generate_response(self.model, prompt, temperature=0.0, max_tokens=512)
        parsed = extract_json_from_response(raw)

        ids = parsed.get("concept_ids", []) if isinstance(parsed, dict) else []
        return list({str(i).strip() for i in ids if isinstance(i, (str, int))})
