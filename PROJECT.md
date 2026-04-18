# Credit Risk Detection Agent

## Overview

Financial regulators and analysts must manually analyse hundreds of pages of annual filings
to identify early warning signals of credit deterioration. Existing tools either rely on lagging
indicators (e.g., rating agency downgrades) or simple financial ratios that miss qualitative
signals buried in management discussion and risk disclosures.

We build an AI agent that automatically detects **credit risk signals** from financial statement
filings (SEC 10-K reports), comparing documents year-over-year to identify meaningful
changes in financial metrics, management language, and risk disclosures that precede credit
events (defaults, bankruptcies, or significant rating downgrades).

We evaluate three model strategies — baseline prompting, RAG, and LoRA fine-tuning — on
a retrospective dataset of companies with **known credit events**, providing unambiguous
ground truth for quantitative evaluation.

The base model used throughout is **Mistral 7B** (open-source, self-hosted). No commercial
API services (OpenAI, Claude, etc.) are used.

---

## Problem Statement

Detecting emerging credit risk from financial filings is challenging because:

- Annual reports are long and unstructured (often 200–400 pages).
- Important signals are often subtle changes between years rather than absolute values.
- Risk disclosures may be buried in qualitative text (e.g., Item 1A: Risk Factors, MD&A).
- Rating agency downgrades are lagging indicators — they often follow, not precede, collapse.
- Generic language models hallucinate explanations rather than grounding outputs in source documents.

### Why Credit Risk Specifically

Narrowing from "emerging risk" to **credit risk** gives us:
1. **Unambiguous ground truth**: company defaulted or did not (binary label).
2. **Known historical cases**: Enron (2001), WorldCom (2002), Lehman Brothers (2008),
   General Motors (2009), Toys R Us (2017), SVB (2023), WeWork (2023).
3. **Free data**: All historical 10-K filings are available via SEC EDGAR.
4. **A quantitative baseline to beat**: the Altman Z-Score, a well-established financial
   distress model computed from accounting ratios.

### Goal

Build a system that can:
1. Compare 10-K filings year-over-year for the same company.
2. Identify meaningful changes in financial indicators or risk language.
3. Generate a grounded credit risk assessment citing evidence from the filings.
4. Outperform the Altman Z-Score baseline on a retrospective default dataset.
5. Minimise hallucination while maintaining useful analytical depth.

---

## Dataset

### Filings
- Source: **SEC EDGAR** (free, public) — 10-K filings for all listed US companies.
- Format: Parse HTML/text filings, extracting key sections: Item 1A (Risk Factors),
  Item 7 (MD&A), Item 8 (Financial Statements).

### Ground Truth Labels
- **Primary**: Binary default label — did the company file for bankruptcy / default within
  2 years of the filing date? (Source: UCLA-LoPucki Bankruptcy Research Database,
  public SEC bankruptcy filings.)
- **Secondary**: Rating downgrade to speculative grade (BB or below) from investment
  grade — used for companies that did not outright default but showed significant
  credit deterioration.

### Dataset Composition
- Balanced sample: ~50 defaulted companies + ~50 matched healthy controls
  (matched by industry, size, and time period).
- Time range: 2000–2022 (excludes recent years where outcomes are unknown).
- Pairs: For each company, use filings from year T and T-1 (year before the credit event).

### Altman Z-Score Baseline
Computed directly from financial statement data (working capital, retained earnings,
EBIT, market cap, total assets, total liabilities). Provides a quantitative threshold-based
classifier to benchmark our LLM approaches against.

---

## Proposed Architecture

We design a **Credit Risk Detection Agent** with the following pipeline:

1. **Document ingestion**: Parse 10-K PDFs/HTML from SEC EDGAR.
2. **Preprocessing**: Extract key sections (Risk Factors, MD&A, financials).
3. **Year-over-year comparison**: Feed two consecutive years' filings into the model.
4. **Risk assessment generation**: Model outputs a structured credit risk assessment
   with citations, a risk score, and key risk factors identified.
5. **Evaluation**: Compare predicted risk scores against ground truth default labels.

---

## Experimental Plan: Three Model Strategies

### 1. Baseline Prompting

A Mistral 7B model prompted to compare two 10-K filings and generate a credit risk
assessment. No retrieval, no fine-tuning — pure prompt engineering.

- **Input**: Extracted key sections from two consecutive 10-K filings (truncated to fit context).
- **Output**: Structured risk assessment (risk level: low/medium/high, key signals, citations).
- **Purpose**: Establishes performance floor. Compare against Altman Z-Score.

### 2. Retrieval-Augmented Generation (RAG)

A RAG pipeline retrieves the most relevant sections of filings before generation.

**Pipeline:**
1. Parse 10-K filings into structured chunks (by section and paragraph).
2. Embed chunks using `sentence-transformers` (e.g., `all-MiniLM-L6-v2`).
3. Store embeddings in a vector database (e.g., FAISS or ChromaDB).
4. At query time: retrieve top-K relevant chunks for each year's filing.
5. Generate a credit risk assessment grounded in retrieved passages.

**Advantages over baseline:**
- Handles full-length 10-Ks (not just truncated sections).
- Grounding in retrieved text reduces hallucination.
- Explicit citations to source passages.

### 3. LoRA Fine-Tuned Model

Fine-tune Mistral 7B using **LoRA (Low-Rank Adaptation)** on curated financial
credit risk reasoning tasks.

**Training data:**
- Financial question-answer pairs focused on credit risk signals.
- Risk identification tasks derived from 10-K filings of known defaulters.
- Year-over-year comparison tasks with labelled risk outcomes.
- Sources: public financial NLP datasets (FinQA, ECTSum) + synthetically generated
  QA pairs from historical filings using structured prompting.

**Expected improvements:**
- Better domain-specific financial reasoning.
- More accurate identification of credit deterioration signals.
- Improved calibration of risk assessments.

---

## Extra Mile: Training Techniques

After the baseline pipeline is functional, we will go deeper on **Training Techniques**:

- Compare LoRA configurations (rank, alpha, target modules) for optimal performance
  vs. compute cost.
- Explore **RLHF / DPO (Direct Preference Optimisation)** as an alternative to SFT,
  using human-labelled preferences between risk assessments for the same company.
- Investigate whether domain adaptation (continued pre-training on financial text before
  fine-tuning) improves downstream credit risk detection.

This focus area directly addresses our key challenge: reducing hallucination while
maintaining analytical depth in credit risk assessments.

---

## Evaluation Metrics

### Quantitative (Primary)
| Metric | Description | Acceptable Threshold |
|--------|-------------|----------------------|
| AUC-ROC | Discriminative ability on default prediction | > 0.75 (beat Altman Z-Score) |
| F1 Score | Balance of precision and recall on default classification | > 0.65 |
| Precision@High-Risk | When model says "high risk", how often is it correct? | > 0.70 |

### Hallucination / Grounding
| Metric | Description | Acceptable Threshold |
|--------|-------------|----------------------|
| Citation accuracy | % of cited facts verifiable in source filing | > 85% |
| Fabrication rate | % of claims not found in source | < 10% |

### Comparative
- All three LLM approaches compared against each other and against Altman Z-Score.
- RAG grounding measured by citation recall vs. baseline.
- LoRA improvements measured by delta in AUC-ROC and hallucination rate vs. RAG.

### Evaluation will use both:
- **Automatic metrics** (AUC-ROC, F1, citation recall via string matching).
- **Human review** (sample of 20 assessments rated by group members for grounding quality).

---

## Compute & Infrastructure

- **Base model**: Mistral 7B (open-source, Apache 2.0 licence).
- **Fine-tuning**: LoRA via HuggingFace PEFT library.
- **Experiment tracking**: Weights & Biases (W&B) — all training runs, eval metrics,
  and hyperparameter sweeps logged from the start.
- **Hardware**: SUTD Cluster (primary) / cloud GPU (within allocated budget) for fine-tuning.
- **Final inference**: Model checkpoint saved and verified to run on SUTD Cluster.
- **Estimated GPU hours**: ~20–40 hours for LoRA fine-tuning (to be updated after initial runs).

---

## Deliverables

### Week 6 (in progress)
- [x] GitHub repository with source code (work in progress).
- [x] PROJECT.md with problem statement and technical plan.

### Week 13 (final)
- [ ] GitHub repository with `README.md` (repo structure + instructions to run on SUTD Cluster).
- [ ] W&B log files showing training and evaluation evidence.
- [ ] Project report (PDF) detailing experimentation process and results.
- [ ] Group oral presentation.
- [ ] (Optional) Live demo.
