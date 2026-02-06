# src/agents/validate.py
class ValidatorAgent:
    def __init__(self, schema):
        self.schema = schema

    def _norm(self, x):
        return " ".join(str(x).lower().split())

    def run(self, observations):
        valid = []

        for o in observations:
            cid = str(o.get("id")).strip()
            val = o.get("value")

            s = self.schema.get(cid)
            if not s:
                continue

            vtype = s["value_type"]
            name = s["name"]

            # STRING
            if vtype == "STRING" and isinstance(val, str) and val.strip():
                valid.append({"id": cid, "name": name, "value_type": vtype, "value": val.strip()})

            # NUMERIC
            elif vtype == "NUMERIC":
                try:
                    num = float(val)
                    if num.is_integer():
                        num = int(num)
                    valid.append({"id": cid, "name": name, "value_type": vtype, "value": num})
                except:
                    continue

            # SINGLE_SELECT
            elif vtype == "SINGLE_SELECT":
                enum = s.get("value_enum", [])
                m = {self._norm(e): e for e in enum}
                k = self._norm(val)
                if k in m:
                    valid.append({"id": cid, "name": name, "value_type": vtype, "value": m[k]})

            # MULTI_SELECT
            elif vtype == "MULTI_SELECT":
                enum = s.get("value_enum", [])
                m = {self._norm(e): e for e in enum}
                vals = val if isinstance(val, list) else [val]
                clean = [m[self._norm(v)] for v in vals if self._norm(v) in m]
                if clean:
                    valid.append({"id": cid, "name": name, "value_type": vtype, "value": clean})

        return valid
