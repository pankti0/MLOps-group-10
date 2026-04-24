# Credit Risk Detection Agent — MLOps Group 10

An AI agent that detects credit risk signals from SEC 10-K filings using three model strategies: **baseline prompting**, **RAG**, and **LoRA fine-tuning**, evaluated against the Altman Z-Score financial benchmark.

Base model: **Mistral-7B-Instruct-v0.2** 

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Architecture](#3-architecture)
4. [Quickstart — Running the Pipeline](#4-quickstart--running-the-pipeline)
5. [Experiment Tracking (W&B)](#5-experiment-tracking-wb)
6. [Results](#6-results)
7. [Extra Mile — LoRA Configuration Comparison](#7-extra-mile--lora-configuration-comparison)
8. [Hallucination Measurement](#8-hallucination-measurement)
9. [Accuracy Threshold Justification](#9-accuracy-threshold-justification)
10. [Repository Structure](#10-repository-structure)
11. [Design Decisions](#11-design-decisions)

---

## 1. Problem Statement

Financial analysts must read hundreds of pages of annual filings to detect early warning signals of credit deterioration. Rating agency downgrades are **lagging indicators** — they follow collapse rather than predict it.

We build a system that:
- Ingests SEC 10-K filings for a set of companies
- Identifies credit risk signals in financial disclosures (debt levels, liquidity, going concern language, covenant violations)
- Generates a structured risk assessment with a 0–100 risk score and cited evidence
- Outperforms the **Altman Z-Score** financial benchmark on a labelled dataset of known credit events

### Ground Truth

Binary default label (1 = high credit risk, 0 = low/medium risk), sourced from publicly available financial outcomes. Companies include AAL, GME, AMC Entertainment (high risk) vs AAPL, MSFT, JNJ (low risk). Labels are in `data/labels/company_labels.csv`.

---

## 2. Dataset

| Component | Details |
|-----------|---------|
| **Source** | SEC 10-K annual filings (PDF) |
| **Companies** | 10 companies across risk categories (3 high, 7 low/medium) |
| **Sections extracted** | Item 1A (Risk Factors), Item 7 (MD&A), Item 8 (Financial Statements) |
| **Fine-tune data** | 40 train / 10 val / 10 test instruction-tuning examples (`data/finetune/`) |
| **Embeddings** | FAISS index with 384-dim sentence-transformer embeddings (`data/embeddings/`) |
| **Labels** | `data/labels/company_labels.csv` — ticker, label (0/1), risk\_category, risk\_score |

The data pipeline (PDF extraction → section parsing → embedding → fine-tune generation) is implemented in `scripts/` and `src/data/`. Its outputs are committed to the repo so inference and evaluation can be run without re-running the pipeline.

---

## 3. Architecture

```
SEC 10-K PDFs
      │
      ▼
 src/data/pdf_extractor.py          Extract text from PDFs
 src/data/section_extractor.py      Parse into Item 1A / 7 / 8 sections
      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                              ▼
 src/embeddings/embedder.py          src/data/label_builder.py
 src/embeddings/faiss_store.py       Ground-truth labels CSV
 FAISS index (data/embeddings/)
      │
      ├─────────────────┬──────────────────┐
      │                 │                  │
      ▼                 ▼                  ▼
 BASELINE            RAG              LoRA FINE-TUNE
 (direct prompt)  (retrieve + prompt)  (SFTTrainer)
 baseline_agent   rag_agent           lora_trainer
      │                 │                  │
      └─────────────────┴──────────────────┘
                        │
                        ▼
              scripts/evaluate_all.py
              Altman Z-Score + AUC/F1/Hallucination metrics
              data/results/metrics_summary.csv
              data/results/roc_curves.png
```

### Three Model Strategies

| Strategy | File | Description |
|----------|------|-------------|
| **Baseline** | `src/models/baseline_agent.py` | Mistral-7B with direct prompting. Extracted 10-K sections are truncated to fit context. Establishes performance floor. |
| **RAG** | `src/models/rag_agent.py` | Retrieves top-K relevant chunks from FAISS for each company, prepends them to the prompt. Reduces hallucination and handles full-length filings. |
| **LoRA** | `src/models/lora_trainer.py` + `lora_agent.py` | QLoRA fine-tuning (4-bit NF4) of Mistral-7B on credit risk instruction-tuning examples using TRL's SFTTrainer. Adapter-only fine-tuning (~0.5% of parameters trained). |

---

## 4. Quickstart — Running the Pipeline

### Prerequisites

- Python 3.9+
- A HuggingFace account 
- Weights & Biases account 
- GPU with ≥16 GB VRAM (A100 recommended; T4 works at batch size 1)

### Setup

```bash
git clone https://github.com/pankti0/MLOps-group-10
cd MLOps-group-10

pip install -r requirements.txt
pip install bitsandbytes>=0.41.0   # ensure CUDA build

cp .env.example .env
# Edit .env and fill in:
#   HF_TOKEN=hf_xxxx
#   WANDB_API_KEY=your_key
```

### Option A — Jupyter Notebooks 

Two notebooks cover the full pipeline end-to-end:

| Notebook | Platform |
|----------|----------|
| `notebooks/colab_full_pipeline.ipynb` | Google Colab (T4 GPU, free tier) |
| `notebooks/cluster_full_pipeline.ipynb` | SUTD GPU cluster / JupyterHub |

Both notebooks run Sections 1–10 in order: setup → data verification → baseline inference → RAG inference → LoRA training → LoRA inference → evaluation → results download.

**For the cluster notebook:** open it in JupyterHub and set `REPO_ROOT` and API keys in the first code cell.

### Option B — CLI Scripts

```bash
# 1. Run baseline inference
python scripts/run_inference_all.py --approach baseline

# 2. Run RAG inference (requires pre-built FAISS index in data/embeddings/)
python scripts/run_inference_all.py --approach rag

# 3. Fine-tune LoRA (r=16 baseline config)
python scripts/train_lora.py --config configs/lora_config.yaml

# 4. Run LoRA inference
python scripts/run_inference_all.py --approach lora \
    --adapter-path data/models/lora_adapter/final_adapter

# 5. Evaluate all approaches + generate ROC curves
python scripts/evaluate_all.py
```

Results are written to `data/results/`.

### Pipeline Testing Without GPU

```bash
# Validate the evaluation pipeline using label-correlated mock scores
python scripts/run_inference_all.py --approach baseline --mock
python scripts/run_inference_all.py --approach rag --mock
python scripts/run_inference_all.py --approach lora --mock
python scripts/evaluate_all.py
```

The `--mock` flag uses ground-truth-correlated Gaussian noise (not random) so the pipeline test produces meaningful ROC curves.

---

## 5. Experiment Tracking (W&B)

All training runs and evaluation metrics are logged to Weights & Biases under project **`credit-risk-detection`**.

| W&B Run | What it logs |
|---------|-------------|
| `lora-sft` (r=16) | Training/eval loss per step, final metrics |
| `lora-r8` | Same, for rank-8 config |
| `lora-r32` | Same, for rank-32 config |
| `evaluation` | AUC-ROC, F1, precision, recall, hallucination rate per approach |

W&B is initialised automatically by `train_lora.py` and `evaluate_all.py` when `WANDB_API_KEY` is set in the environment. The logger gracefully degrades (prints a warning and continues) if W&B is unavailable.

**W&B dashboard:** `https://wandb.ai` → project `credit-risk-detection`

---

## 6. Results

> **Note:** Results in `data/results/metrics_summary.csv` reflect **pipeline validation runs** (mock data with label-correlated noise). Final numbers will be updated after GPU inference completes. The mock data is intentionally designed with the same statistical properties as expected real model output, so the evaluation pipeline itself is fully validated.

### Current metrics (pipeline validation)

| Approach | AUC-ROC | F1 | Precision | Recall | Accuracy |
|----------|--------:|---:|----------:|-------:|---------:|
| **Baseline** | 0.905 | 0.857 | 0.750 | 1.000 | 0.900 |
| **RAG** | 0.857 | 0.750 | 0.600 | 1.000 | 0.800 |
| **LoRA (r=16)** | 0.810 | 0.571 | 0.500 | 0.667 | 0.700 |
| **Altman Z-Score** | 0.810 | 0.444 | 0.333 | 0.667 | 0.500 |

ROC curves: `data/results/roc_curves.png`

### Altman Z-Score as benchmark

The Altman Z-Score is computed from five financial ratios (working capital, retained earnings, EBIT, market cap, total debt) and provides an interpretable threshold-based classifier. Its AUC of ~0.81 on this dataset is the **target to beat**. All LLM approaches must exceed AUC 0.75 to be considered meaningful (see [Section 9](#9-accuracy-threshold-justification)).

---

## 7. Extra Mile — LoRA Configuration Comparison

We train three LoRA adapters with different ranks to study the tradeoff between capacity, compute cost, and credit risk detection performance.

| Config | Rank `r` | Alpha | Target modules | Trainable params |
|--------|----------|-------|---------------|-----------------|
| `lora_config_r8.yaml` | 8 | 16 | q\_proj, v\_proj | ~2M |
| `lora_config.yaml` *(baseline)* | 16 | 32 | q, k, v, o projections | ~4M |
| `lora_config_r32.yaml` | 32 | 64 | q, k, v, o, up\_proj, down\_proj | ~16M |

**Hypothesis:** Higher rank captures more complex financial reasoning patterns but requires more compute and risks overfitting on our small dataset.

**To run all three configs:**
```bash
python scripts/train_lora.py --config configs/lora_config_r8.yaml
python scripts/train_lora.py --config configs/lora_config.yaml
python scripts/train_lora.py --config configs/lora_config_r32.yaml
```

Each run is tracked as a separate W&B run (`lora-r8`, `lora-sft`, `lora-r32`). Adapters are saved to `data/models/lora_adapter_r8/`, `data/models/lora_adapter/`, and `data/models/lora_adapter_r32/` respectively.

The loss curves for all three runs can be overlaid from the W&B dashboard (project → Runs → select all three → Charts → train/loss).

---

## 8. Hallucination Measurement

LLMs frequently fabricate financial figures not present in source documents. We measure this with two metrics:

| Metric | Definition | Target |
|--------|-----------|--------|
| **Fabrication rate** | Fraction of numeric claims in model output that have no fuzzy match (≥85 score) in the retrieved source chunks | ≤ 10% |
| **Citation accuracy** | Fraction of verbatim citations produced by the model that can be matched back to a source passage | ≥ 85% |

**Implementation:** `src/evaluation/hallucination_checker.py`  
- Uses `rapidfuzz` for fuzzy string matching (handles minor phrasing differences)  
- `check_fabrication(output_text, source_chunks)` — checks numeric sentences  
- `check_citation_accuracy(citations, source_chunks)` — verifies verbatim quotes  
- `score_response(output_text, citations, source_chunks)` — returns combined grounding score  

**Integration:** Called during evaluation by `src/evaluation/evaluator.py` for baseline, RAG, and LoRA. Hallucination metrics are logged to W&B alongside AUC/F1.

**Thresholds** are configured in `configs/eval_config.yaml`:
```yaml
hallucination:
  fuzzy_match_threshold: 85
  min_sentence_length: 20
thresholds:
  fabrication_rate_max: 0.10
  citation_accuracy_target: 0.85
```

---

## 9. Accuracy Threshold Justification

**AUC target: 0.75**  
**F1 target: 0.65**

### Why 0.75 AUC?

The Altman Z-Score achieves **AUC ≈ 0.81** on our dataset. This is a well-validated quantitative model that has been used in practice for 50 years. Our LLM approaches must meaningfully exceed random chance (AUC=0.50) to justify the added complexity. We set 0.75 as a **minimum bar** — it represents strong discriminative ability and is achievable by a well-prompted LLM, while still being below the Altman Z-Score to acknowledge that beating a purpose-built financial model is non-trivial on a 10-company dataset.

The primary goal is to **approach or exceed 0.81** (Altman Z-Score AUC) on the full inference run. If LoRA achieves this, it demonstrates that fine-tuning on domain data adds meaningful signal beyond the pre-trained model.

### Why 0.65 F1?

On a balanced binary task, a random classifier achieves F1 ≈ 0.50. We set 0.65 as the minimum target — this ensures the model is correctly classifying at least 65% of true positives while maintaining reasonable precision, which matters in a credit risk context where false negatives (missing a risky company) are costly.

---

## 10. Repository Structure

```
MLOps-group-10/
│
├── configs/                        # All hyperparameters and thresholds
│   ├── lora_config.yaml            # LoRA r=16 (baseline fine-tune config)
│   ├── lora_config_r8.yaml         # LoRA r=8 (Extra Mile: smaller)
│   ├── lora_config_r32.yaml        # LoRA r=32 (Extra Mile: larger)
│   ├── eval_config.yaml            # Evaluation thresholds + hallucination params
│   ├── rag_config.yaml             # RAG retrieval settings
│   └── data_config.yaml            # Data paths and processing settings
│
├── src/
│   ├── data/
│   │   ├── pdf_extractor.py        # Extract text from 10-K PDFs (PyMuPDF)
│   │   ├── section_extractor.py    # Parse Item 1A, 7, 8 from raw text
│   │   ├── preprocessor.py         # Chunk text for embedding
│   │   └── label_builder.py        # Build ground-truth labels CSV
│   │
│   ├── embeddings/
│   │   ├── embedder.py             # sentence-transformers wrapper
│   │   └── faiss_store.py          # FAISS index build/load/query
│   │
│   ├── models/
│   │   ├── base_loader.py          # Load Mistral-7B with 4-bit quantization
│   │   ├── baseline_agent.py       # Direct-prompt inference agent
│   │   ├── rag_agent.py            # RAG inference agent (retrieval + prompt)
│   │   ├── lora_trainer.py         # QLoRA fine-tuning via TRL SFTTrainer
│   │   └── lora_agent.py           # LoRA adapter inference agent
│   │
│   ├── prompts/
│   │   ├── baseline_prompt.py      # Prompt template for baseline
│   │   ├── rag_prompt.py           # Prompt template + output parser for RAG
│   │   └── lora_prompt.py          # Instruction-tuning prompt format
│   │
│   ├── evaluation/
│   │   ├── evaluator.py            # Orchestrates all evaluation
│   │   ├── metrics.py              # AUC, F1, ROC curve plotting
│   │   ├── altman_zscore.py        # Altman Z-Score baseline computation
│   │   └── hallucination_checker.py # Fabrication rate + citation accuracy
│   │
│   └── utils/
│       ├── config_loader.py        # YAML config loader
│       └── wandb_logger.py         # W&B init, metric logging, graceful fallback
│
├── scripts/
│   ├── extract_sections.py         # Step 1: PDF → section JSON files
│   ├── build_labels.py             # Step 2: Build company_labels.csv
│   ├── build_embeddings.py         # Step 3: Build FAISS index
│   ├── generate_finetune_data.py   # Step 4: Build train/val/test JSONL
│   ├── run_inference_all.py        # Step 5: Run baseline / RAG / LoRA inference
│   ├── train_lora.py               # Step 5b: Fine-tune LoRA adapter
│   └── evaluate_all.py             # Step 6: Compute metrics + ROC curves
│
├── notebooks/
│   ├── colab_full_pipeline.ipynb   # Full pipeline on Google Colab
│   └── cluster_full_pipeline.ipynb # Full pipeline on university GPU cluster
│
├── data/
│   ├── labels/company_labels.csv   # Ground-truth credit risk labels
│   ├── processed/                  # Per-company extracted section JSON files
│   ├── embeddings/                 # FAISS index + chunk metadata
│   ├── finetune/                   # train.jsonl, val.jsonl, test.jsonl
│   └── results/                    # Prediction CSVs, metrics_summary.csv, roc_curves.png
│
├── PROJECT.md                      # Full project brief
├── requirements.txt
└── .env.example                    # API key template
```

---

## 11. Design Decisions

### Why Mistral-7B?
Open-source, self-hosted, fits in 16 GB VRAM with 4-bit quantization. No API costs or rate limits, which matters for running repeated inference over 10 companies across 3 approaches.

### Why QLoRA (4-bit NF4)?
Training a full 7B model is infeasible on a single GPU. QLoRA freezes the quantized base model and trains only the low-rank adapter matrices (~0.5% of parameters), making fine-tuning feasible in under 2 hours on an A100.

### Why FAISS over ChromaDB?
FAISS is CPU/GPU-native, has no server dependency, serialises to a single file (`faiss.index`), and is faster for our small-scale retrieval task (10 companies, ~200 chunks). ChromaDB would add operational overhead with no benefit at this scale.

### TRL API compatibility
`lora_trainer.py` uses Python's `inspect` module to detect available parameters in the installed TRL/transformers version at runtime. This makes the training code forward-compatible with TRL 0.8 through 0.12+ without requiring pinned versions.

### load_best_model_at_end = False
PEFT adapter checkpoints cannot be reloaded mid-training by the HuggingFace Trainer (it expects full model weights). Setting this to `True` would silently crash at the end of training. The best adapter is instead saved manually via `trainer.save_model()` after training completes.

### Hallucination via fuzzy matching
Exact string matching fails for paraphrased financial figures ("revenue declined 12%" vs "revenues fell by 12.0%"). `rapidfuzz` partial-ratio matching with threshold 85 handles this while still catching clear fabrications.
