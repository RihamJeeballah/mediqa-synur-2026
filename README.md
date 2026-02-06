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
## Multi-Agent System Architecture

We propose a multi-stage, multi-agent architecture for extracting structured nursing observations from free-text clinical transcripts in the **MEDIQA-SYNUR** task. The system is explicitly designed to prioritize evidence-grounded extraction and to mitigate common failure modes observed during development, including hallucinated observations, implicit inference, and systematic false positives.

The pipeline consists of four complementary components:

1. An LLM-based extraction agent  
2. A deterministic validation agent  
3. A precision-oriented filtering agent  
4. A conservative rule-based suppression mechanism  

Each component targets a specific source of error, enabling progressive refinement from high-recall candidate extraction to high-precision final predictions.

---

### Transcript Segmentation

Clinical transcripts in MEDIQA-SYNUR are often lengthy and information-dense, which may lead to context dilution and missed extractions. To address this, transcripts can be optionally segmented into smaller, non-overlapping chunks based on character length. Each segment is processed independently by downstream agents. This localized processing reduces context overload and improves recall for observations that appear sparsely or late in the transcript.

---

### Schema Segmentation and Batching

The MEDIQA-SYNUR observation schema contains 192 distinct categories, making single-pass extraction impractical. To address this limitation, schema entries are partitioned into smaller batches, with each batch processed independently by the extraction agent. As a result, each prompt conditions the model on only a subset of schema definitions, improving schema adherence and reducing cognitive load.

---

### 1 Extractor Agent (LLM-Based)

The `ExtractorAgent` performs the initial identification of candidate nursing observations using a large language model (**LLaMA-3.3 via Ollama**). To handle the large schema size, schema batching is combined with optional transcript segmentation, resulting in multiple localized extraction passes.

The extraction prompt enforces strict evidence-based constraints. Each extracted observation must include:

1. A valid schema identifier corresponding to an official MEDIQA-SYNUR category  
2. A value consistent with the schema-defined type (numeric, single-select, multi-select, or free text)  
3. An evidence field containing an **exact verbatim substring** copied from the transcript  

Inference, abstraction, and speculation are explicitly prohibited. Negative values (e.g., *“No”*, *“Absent”*) are allowed only when explicit negation cues appear in the transcript. The extractor is intentionally recall-oriented, allowing permissive candidate generation while deferring strict correctness enforcement to downstream agents.

---

## 2 Validation Agent

The `ValidatorAgent` applies deterministic, rule-based validation to the raw outputs produced by the extraction agent. Its primary role is to remove structurally invalid, weakly grounded, or speculative observations before further refinement.

Validation rules include:

- Rejecting observations whose evidence does not appear verbatim in the transcript  
- Filtering evidence containing hedging or speculative language (e.g., *“possibly”*, *“suggests”*, *“likely”*)  
- Enforcing explicit negation rules for negative values  
- Applying schema-specific anchor constraints for error-prone categories (e.g., vomiting, pain, Glasgow Coma Scale), requiring the presence of domain-specific lexical cues  
- Ensuring extracted values conform exactly to the allowed schema enumeration when applicable  

This agent substantially reduces hallucinated and semantically invalid predictions while preserving observations supported by explicit textual evidence.

---

## 3 Precision-Oriented Filtering Agent

Despite strict validation, development analysis revealed residual over-prediction caused by subtle semantic drift or overly permissive extraction. To address this, a `PrecisionFilterAgent` is applied as an additional refinement stage.

The precision filter operates at the level of individual observations. For each validated observation, the agent re-evaluates whether the extraction should be retained or discarded by jointly considering:

- The full transcript context  
- The schema definition and allowed values  
- The extracted evidence span and associated value  

The agent outputs a binary decision (**KEEP** or **DROP**) along with a brief rationale. To ensure reproducibility, the precision filter uses deterministic decoding (`temperature = 0.0`). Its role is strictly eliminative: it does not introduce new observations, but removes residual false positives that survive earlier validation stages.

---

## 4 Conservative Suppression Table

In addition to agent-based filtering, a lightweight deterministic suppression mechanism is applied as a final post-processing step. This suppression table is derived from systematic error analysis on the development set and is applied uniformly across development and test data without any test-time tuning.

Two categories of suppression are defined:

1. **Always-suppressed observations**, corresponding to categories that were consistently over-predicted or semantically redundant  
2. **Negation-sensitive suppression**, in which negative-valued observations are retained only when explicit negation cues are present in the evidence  

This mechanism is intentionally conservative, aiming to further improve precision while minimizing the risk of introducing false negatives.
 
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
