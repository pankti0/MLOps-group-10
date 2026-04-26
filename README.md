# Credit Risk Detection Agent — MLOps Group 10

An AI-powered credit risk detection system that analyses SEC 10-K annual reports to identify financial distress signals and predict company credit risk.

This project compares **four modelling approaches**:

* **Baseline Prompting**
* **Retrieval-Augmented Generation (RAG)**
* **LoRA Fine-Tuning**
* **Hybrid (RAG + LoRA)**

All approaches are benchmarked against the traditional **Altman Z-Score** financial distress model.

Base model used throughout: Mistral AI **Mistral-7B-Instruct-v0.2**

---

# Overview

Financial distress signals are often present in annual reports long before formal defaults, bankruptcies, or major credit downgrades occur. However, these signals are frequently buried across hundreds of pages of financial disclosures, management commentary, and risk statements.

Traditional models such as Altman Z-Score rely primarily on financial ratios and may miss qualitative warning signals hidden in textual disclosures.

This project builds an AI agent that automatically analyses company filings and generates structured credit risk assessments by identifying:

* liquidity concerns
* debt burden signals
* going concern warnings
* covenant breach indicators
* operational deterioration signals
* management risk disclosures

The goal is to determine whether modern language model pipelines can improve financial distress detection compared to classical quantitative approaches.

---

# Problem Statement

Credit risk detection from financial filings is challenging because:

* Annual reports are long and highly unstructured
* Important warning signals may be subtle rather than explicit
* Financial deterioration appears in both quantitative and qualitative sections
* Generic LLM prompting may hallucinate unsupported reasoning
* Credit distress events are relatively rare, creating class imbalance

This project investigates whether grounding and adaptation techniques can improve predictive performance.

---

# Dataset

## Data Source

Source: SEC 10-K annual filings

For each company, three key sections were extracted:

* **Item 1A — Risk Factors**
* **Item 7 — Management Discussion & Analysis (MD&A)**
* **Item 8 — Financial Statements**

These sections contain the strongest operational, strategic, and financial risk signals.

## Dataset Composition

Final dataset:

* **33 companies total**
* **10 low-risk**
* **16 medium-risk**
* **7 high-risk / distressed**

Binary mapping:

* **Label = 1 → distressed / default-positive**
* **Label = 0 → non-distressed**

This creates a naturally imbalanced but realistic credit-risk classification setting.

Ground-truth labels are stored in:

`data/labels/company_labels.csv`

with:

* ticker
* company name
* binary label
* risk category
* numerical risk score (0–100)

---

# Evaluation Protocol

Due to the relatively small dataset size, evaluation was performed using **5-fold stratified cross validation**.

Stratification preserved the distressed / non-distressed class ratio in each fold, ensuring fair evaluation under class imbalance.

Performance was aggregated across folds using:

* ROC-AUC
* F1-score
* Precision
* Recall
* Accuracy

---

# System Architecture

```text
SEC 10-K PDFs
      │
      ▼
PDF Extraction
      │
      ▼
Section Parsing
(Item 1A / Item 7 / Item 8)
      │
      ├──────────────┬──────────────┬──────────────┐
      │              │              │              │
      ▼              ▼              ▼              ▼
 Baseline           RAG            LoRA          Hybrid
 Prompting      Retrieval + LLM   Fine-tuned    RAG + LoRA
      │              │              │              │
      └──────────────┴──────────────┴──────────────┘
                      │
                      ▼
                Evaluation
             + Altman Benchmark
```

---

# Approaches

## 1) Baseline Prompting

Direct prompting using Mistral AI Mistral-7B-Instruct-v0.2.

Features:

* extracted report sections fed directly into prompt
* no retrieval layer
* no fine-tuning
* structured JSON-style output generation

This establishes the base LLM benchmark.

---

## 2) Retrieval-Augmented Generation (RAG)

A retrieval pipeline was introduced to ground model responses in relevant filing sections.

Pipeline:

* report chunking (**512 tokens**)
* overlap (**64 tokens**)
* embeddings generated using BAAI/bge-base-en-v1.5
* vectors indexed in FAISS
* top relevant chunks retrieved at inference time

Benefits:

* stronger grounding
* reduced hallucination
* better citation alignment

---

## 3) LoRA Fine-Tuning

Parameter-efficient fine-tuning using LoRA adapters on domain-specific financial reasoning tasks.

Multiple LoRA ranks were explored:

* rank = 8
* rank = 16
* rank = 32

Final evaluation used the strongest adapter configuration.

Benefits:

* domain adaptation
* improved financial reasoning specialization
* efficient training with limited compute

---

## 4) Hybrid (RAG + LoRA)

Combines:

* retrieval grounding from RAG
* domain-specialized reasoning from LoRA

Goal:

capture benefits of both grounding and specialization in a single pipeline.

---

# Classical Benchmark — Altman Z-Score

Performance is compared against the classical Altman Z-Score distress model.

Z = 1.2X_1 + 1.4X_2 + 3.3X_3 + 0.6X_4 + 1.0X_5

where financial ratios represent:

* working capital / total assets
* retained earnings / total assets
* EBIT / total assets
* market value of equity / total liabilities
* revenue / total assets

This provides a traditional finance benchmark for comparison.

---

# Results

Performance aggregated across all stratified cross-validation folds:

| Approach                |   ROC-AUC |        F1 | Precision |    Recall |  Accuracy |
| ----------------------- | --------: | --------: | --------: | --------: | --------: |
| **Baseline Prompting**  | **0.849** |     0.000 |     0.000 |     0.000 | **0.813** |
| **RAG**                 |     0.692 |     0.429 |     0.273 | **1.000** |     0.500 |
| **LoRA Fine-Tuning**    |     0.701 |     0.000 |     0.000 |     0.000 |     0.733 |
| **Hybrid (RAG + LoRA)** |     0.500 |     0.000 |     0.000 |     0.000 |     0.813 |
| **Altman Z-Score**      |     0.778 | **0.444** | **0.333** |     0.667 |     0.444 |

## Key Findings

* **Baseline Prompting achieved the strongest ROC-AUC (0.849)**, indicating best ranking performance.
* **Altman Z-Score achieved the strongest F1-score (0.444)**, indicating strongest thresholded classification balance.
* **RAG achieved perfect recall (1.000)**, successfully identifying all distressed companies, but suffered from many false positives.
* **LoRA Fine-Tuning did not improve thresholded classification performance** in this setup.
* **Hybrid RAG + LoRA showed no meaningful improvement**, performing close to random ranking.

Overall, **simple prompting with a strong base model proved surprisingly competitive**, while retrieval improved sensitivity and Altman remained a robust classical benchmark.

---

# Repository Structure

```text
configs/
src/
scripts/
notebooks/
data/
README.md
requirements.txt
```

Key folders:

* `src/` → model and pipeline code
* `scripts/` → execution scripts
* `configs/` → YAML configs
* `data/` → labels, embeddings, processed files, results
* `notebooks/` → Colab / experimentation notebooks

---

# Running the Project

Run inference:

```bash
python scripts/run_inference_all.py --approach baseline
python scripts/run_inference_all.py --approach rag
python scripts/run_inference_all.py --approach lora
```

Run evaluation:

```bash
python scripts/evaluate_all.py
```

---

# Conclusion

This project demonstrates that LLM-based analysis of financial filings is feasible for credit-risk detection.

Key takeaways:

* prompting alone can be surprisingly strong
* retrieval improves recall but hurts precision
* classical finance models remain competitive
* fine-tuning requires more task-specific supervision to outperform simpler approaches

This highlights both the promise and current limitations of LLM-based financial risk modelling.
