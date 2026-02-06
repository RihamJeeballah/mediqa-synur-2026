# src/agents/validate.py
import re
from typing import Any, Dict, List


class ValidatorAgent:
    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        self.schema = schema

        self.bad_evidence_patterns = [
            r"\bno mention\b",
            r"\bnot mentioned\b",
            r"\bnot explicitly stated\b",
            r"\bnone explicitly stated\b",
            r"\bno specific\b",
            r"\bno evidence\b",
        ]

        self.hedge_patterns = [
            r"\bsuggest\b",
            r"\bindicat(e|es|ed|ing)\b",
            r"\bpossibly\b",
            r"\bcould\b",
            r"\blikely\b",
            r"\bmaybe\b",
        ]

        self.negation_cues = [
            r"\bno\b",
            r"\bdenies\b",
            r"\bwithout\b",
            r"\babsent\b",
            r"\bnone\b",
        ]

        self.id_anchor_required = {
            "71":  [r"\bvomit", r"\bemesis\b"],
            "116": [r"\bwork of breathing\b", r"\bwob\b"],
            "110": [r"\bgcs\b", r"\bglasgow\b"],
            "96":  [r"\bfollow(s)? commands?\b"],
            "0":   [r"\bbroset\b"],
            "167": [r"\bpain\b"],
        }

        self.patient_id_regex = re.compile(r"\b\d{1,3}-year-old\b", re.IGNORECASE)
        self.hard_deny_if_no_anchor = {"0", "110", "96", "116", "167"}

    def _norm(self, x):
        return " ".join(str(x).strip().lower().split())

    def _has_any_pattern(self, text: str, patterns: List[str]) -> bool:
        t = text.lower()
        return any(re.search(p, t) for p in patterns)

    def _evidence_in_transcript(self, evidence: str, transcript: str) -> bool:
        return evidence.strip() in transcript

    def _allow_negative_value(self, value: Any, evidence: str) -> bool:
        if not isinstance(value, str):
            return True
        v = self._norm(value)
        if v in {"no", "none", "absent"}:
            return self._has_any_pattern(evidence, self.negation_cues)
        return True

    def _passes_id_anchor(self, cid: str, evidence: str, transcript: str) -> bool:
        cid = str(cid)
        if cid == "162":
            if not isinstance(evidence, str) or not evidence.strip():
                return False
            if not self.patient_id_regex.search(transcript):
                return False
            if isinstance(evidence, str) and not self.patient_id_regex.search(evidence):
                return False
            return True

        patterns = self.id_anchor_required.get(cid)
        if not patterns:
            return True

        ev = evidence.lower()
        if any(re.search(p, ev) for p in patterns):
            return True

        tr = transcript.lower()
        return any(re.search(p, tr) for p in patterns)

    def run(self, observations: Any, transcript: str) -> List[Dict[str, Any]]:
        valid = []
        if not isinstance(observations, list):
            return valid

        transcript = transcript or ""

        for o in observations:
            if not isinstance(o, dict):
                continue

            cid = o.get("id")
            if cid is None:
                continue
            cid = str(cid).strip()

            s = self.schema.get(cid)
            if not s:
                continue

            name = s.get("name", "")
            vtype = s["value_type"]

            val = o.get("value")
            evidence = o.get("evidence", "")
            if not isinstance(evidence, str) or not evidence.strip():
                continue

            if not self._evidence_in_transcript(evidence, transcript):
                continue

            if self._has_any_pattern(evidence, self.bad_evidence_patterns):
                continue

            if self._has_any_pattern(evidence, self.hedge_patterns):
                continue

            if not self._allow_negative_value(val, evidence):
                continue

            if not self._passes_id_anchor(cid, evidence, transcript):
                continue

            if isinstance(val, str) and self._has_any_pattern(val, [r"not explicitly stated", r"not mentioned", r"no mention"]):
                continue

            if vtype == "STRING":
                if isinstance(val, str) and val.strip():
                    valid.append({
                        "id": cid,
                        "name": name,
                        "value_type": vtype,
                        "value": val.strip(),
                        "evidence": evidence
                    })
                continue

            if vtype == "NUMERIC":
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    num = val
                elif isinstance(val, str):
                    try:
                        num = float(val) if "." in val else int(val)
                    except:
                        continue
                else:
                    continue

                valid.append({
                    "id": cid,
                    "name": name,
                    "value_type": vtype,
                    "value": num,
                    "evidence": evidence
                })
                continue

            if vtype in ["SINGLE_SELECT", "MULTI_SELECT"]:
                enum = s.get("value_enum", [])
                enum_norm = {self._norm(e): e for e in enum}

                if vtype == "MULTI_SELECT":
                    if isinstance(val, str):
                        val = [val]
                    if not isinstance(val, list):
                        continue

                    clean = []
                    for v in val:
                        k = self._norm(v)
                        if k in enum_norm:
                            clean.append(enum_norm[k])

                    if not clean:
                        continue

                    valid.append({
                        "id": cid,
                        "name": name,
                        "value_type": vtype,
                        "value": clean,
                        "evidence": evidence
                    })
                    continue

                if isinstance(val, str):
                    k = self._norm(val)
                    if k in enum_norm:
                        valid.append({
                            "id": cid,
                            "name": name,
                            "value_type": vtype,
                            "value": enum_norm[k],
                            "evidence": evidence
                        })

        return valid