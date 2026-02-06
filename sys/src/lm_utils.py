# src/lm_utils.py
import json
import re
import ollama


def generate_response(model, prompt, temperature=0.0, max_tokens=512):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return response["message"]["content"]


def extract_json_from_response(text):
    if not text:
        return {}

    candidates = []

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidates.extend(fenced)

    raw_obj = re.findall(r"\{[\s\S]*\}", text)
    candidates.extend(raw_obj)

    raw_arr = re.findall(r"\[[\s\S]*\]", text)
    candidates.extend(raw_arr)

    for cand in reversed(candidates):
        cand = cand.strip()
        cand = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            return json.loads(cand)
        except:
            continue

    return {}
