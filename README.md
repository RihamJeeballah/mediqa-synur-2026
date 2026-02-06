# MEDIQA-SYNUR 2026 – Nursing Observation Extraction

This repository contains our submission code for the **MEDIQA-SYNUR** shared task (Synthetic Nursing Observation Extraction).

The system performs structured extraction of clinical nursing observations from synthetic nurse dictations (SYNUR dataset), given a predefined flowsheet schema.

---

## Task Description

The goal of MEDIQA-SYNUR is to extract structured clinical observations from nurse dictations and populate an EHR flowsheet schema.  
Each extracted observation must:
- Match a schema-defined concept
- Contain a canonical value
- Be supported by explicit textual evidence copied verbatim from the transcript

No training is performed; the system operates fully at inference time.

---

## Method Overview

Our approach follows a **strict extract–validate–filter pipeline** designed to maximize precision while maintaining recall:

### 1. Observation Extraction
- Prompt-based extraction conditioned on the provided flowsheet schema
- Concept-driven extraction (schema-first)
- No inference or normalization
- Evidence must be an exact substring of the transcript

### 2. Validation Layer
- Enforces schema constraints (value types and enumerations)
- Filters invalid or unsupported evidence
- Handles negation strictly (only when explicitly stated)

### 3. Precision Filtering
- A second-pass LLM-based precision agent
- Decides whether each extracted observation should be kept or dropped
- Reduces false positives caused by semantic overreach or duplication

---

## Model

We use **LLaMA-3.3** via **Ollama** with deterministic decoding
(temperature = 0) for all extraction and precision-filtering steps.
---

## Code Location

All code used for inference is located under the `sys/src/` directory


---

## Best Performing Setup

We evaluated multiple system configurations.  
The **highest  performance** was achieved with the following setup:

- Transcript segmentation: **enabled**
- Suppression table: **enabled**
- Precision filtering agent: **enabled**
- Batch size: **25**

### Inference Command

```bash
python src/run.py \
  --split test \
  --schema_path data/schema.json \
  --model llama3.3 \
  --batch_size 25 \
  --segment\
  --supress_table\
  --precision_filter
