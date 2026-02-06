# src/run.py
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from src.schema import SynurSchema
from src.agents.extract import ExtractorAgent
from src.agents.validate import ValidatorAgent
from src.agents.precision_filter import PrecisionFilterAgent
from src.agents.schema_retriever import SchemaRetriever


def split_transcript(text: str, max_chars: int = 1400) -> List[str]:
    parts, buf, cur = [], [], 0
    for p in text.split("\n\n"):
        if cur + len(p) > max_chars and buf:
            parts.append("\n\n".join(buf))
            buf, cur = [], 0
        buf.append(p)
        cur += len(p)
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def chunk_schema_ids(ids: List[str], size: int) -> List[List[str]]:
    return [ids[i:i + size] for i in range(0, len(ids), size)]


# ================= SUPPRESSION =================

SUPPRESS_ALWAYS_BY_NAME = {
    "Orientation",
    "Mental status",
    "Memory status",
    "Delirium symptoms",
    "Patient identification",
    "Skin condition",
    "Meal consumption",
    "Voiding function",
    "Pain description",
    "Vaginal discharge",
}

SUPPRESS_NEGATIVE_ONLY_BY_NAME = {
    "Vomiting",
    "Dyspnea",
    "Gas passage",
    "Urinary stone",
}


def has_explicit_negation(evidence: str) -> bool:
    if not isinstance(evidence, str):
        return False
    e = evidence.lower()
    return any(w in e for w in ["no ", "denies", "without", "absent", "none", "not "])


def apply_suppression_table(obs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for o in obs:
        name = o.get("name", "")
        value = o.get("value", None)
        evidence = o.get("evidence", "")

        if name in SUPPRESS_ALWAYS_BY_NAME:
            continue

        if name in SUPPRESS_NEGATIVE_ONLY_BY_NAME:
            if isinstance(value, str) and value.lower() in {"no", "none", "absent"}:
                if not has_explicit_negation(evidence):
                    continue

        cleaned.append(o)
    return cleaned


# ================= CORE =================

def process_record(
    record: Dict[str, Any],
    model: str,
    schema: SynurSchema,
    batch_size: int,
    segment: bool,
    use_suppress_table: bool,
    use_precision_filter: bool,
    use_schema_retrieval: bool,
    top_k_schema: int,
    filter_model: str,
):
    rid = record.get("id")
    text = record.get("transcript") or record.get("text") or ""

    if not isinstance(text, str) or not text.strip():
        return {"id": rid, "observations": []}

    extractor = ExtractorAgent(model, schema.by_id)
    validator = ValidatorAgent(schema.by_id)

    retriever = None
    if use_schema_retrieval:
        retriever = SchemaRetriever(schema.by_id, top_k=top_k_schema)

    text_chunks = split_transcript(text) if segment else [text]

    raw = []

    for chunk in text_chunks:
        if use_schema_retrieval:
            schema_ids = retriever.retrieve(chunk)
            schema_batches = chunk_schema_ids(schema_ids, batch_size)
        else:
            schema_ids = list(schema.by_id.keys())
            schema_batches = chunk_schema_ids(schema_ids, batch_size)

        for sb in schema_batches:
            extracted = extractor.run(chunk, sb)
            if isinstance(extracted, list):
                raw.extend(extracted)

    validated = validator.run(raw, text)

    if use_suppress_table:
        validated = apply_suppression_table(validated)

    if use_precision_filter:
        pf = PrecisionFilterAgent(
            model=filter_model,
            schema_by_id=schema.by_id,
            temperature=0.0,
        )
        validated = pf.filter_observations(validated, text)

    return {"id": rid, "observations": validated}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llama3.3")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--schema_path", required=True)
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--segment", action="store_true")
    ap.add_argument("--suppress_table", action="store_true")
    ap.add_argument("--precision_filter", action="store_true")

    # NEW FLAGS
    ap.add_argument("--schema_retrieval", action="store_true")
    ap.add_argument("--top_k_schema", type=int, default=40)

    ap.add_argument("--filter_model", default=None)

    args = ap.parse_args()

    schema = SynurSchema(args.schema_path)
    inp = Path(args.data_dir) / f"{args.split}.jsonl"
    out = Path(args.out)

    filter_model = args.filter_model or args.model
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            res = process_record(
                record=rec,
                model=args.model,
                schema=schema,
                batch_size=args.batch_size,
                segment=args.segment,
                use_suppress_table=args.suppress_table,
                use_precision_filter=args.precision_filter,
                use_schema_retrieval=args.schema_retrieval,
                top_k_schema=args.top_k_schema,
                filter_model=filter_model,
            )
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"✅ Saved to {out}")


if __name__ == "__main__":
    main()
