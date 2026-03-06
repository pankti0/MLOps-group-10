# Emerging Risk Detection Agent

## Overview

Financial risks often emerge gradually through subtly, and regulators must manually analyse hundreds of pages of disclosures to identify early warning signals.

We shall use an AI agent that automatically identifies emerging risks by comparing financial statement filings across different reporting years. The system aims to highlight meaningful changes in financial metrics, management discussion, and risk disclosures that may indicate new or increasing risk exposure.

We evaluate whether LoRA fine-tuning on domain-specific financial reasoning tasks improves the model's ability to extract grounded insights compared to baseline prompting and Retrieval-Augmented Generation (RAG) approaches.

## Problem Statement

Detecting emerging risks from financial filings is challenging because:

- Annual reports are long and unstructured.
- Important signals are often subtle changes between years rather than absolute values.
- Risk disclosures may be buried within qualitative text in as Management Discussions 
- Generic language models often hallucinate explanations rather than grounding outputs in the source documents.

Our goal is to build a system that can:

1. Compare financial statement filings year-over-year.
2. Identify meaningful changes in financial indicators or risk language.
3. Generate grounded explanations citing evidence from the filings.
4. Minimize hallucination while maintaining useful analytical insights.


## Proposed Architecture

We design an **Emerging Risk Detection Agent** with the following capabilities:

- Document ingestion of annual filings (e.g., 10-K reports).
- Year-over-year comparison between financial documents.
- Extraction of financial metrics and qualitative disclosures.
- Risk analysis summarisation with citations.

We plan on experimenting with three model strategies:

### 1. Baseline Prompting

A general-purpose LLM is prompted to compare two financial filings and generate risk insights.

Characteristics:
- No retrieval
- No fine-tuning
- Pure prompt engineering

This serves as the **baseline performance**.

### 2. Retrieval-Augmented Generation (RAG)

A RAG pipeline retrieves relevant sections of filings before generation.

Pipeline:
1. Parse filings into structured chunks
2. Embed text using a sentence embedding model
3. Store embeddings in a vector database
4. Retrieve relevant passages for each query
5. Generate responses grounded in retrieved context

Advantages:
- Reduces hallucination
- Improves grounding in documents

### 3. LoRA Fine-Tuned Model

We fine-tune a base LLM using **LoRA (Low-Rank Adaptation)** on curated financial reasoning tasks.

Training data will include:
- Financial question-answer pairs
- Risk identification tasks
- Document comparison tasks
- Financial statement interpretation examples

Expected improvements:
- Better domain-specific reasoning
- More accurate financial interpretation
- Improved ability to detect meaningful changes across years

## Experimental Plan

We compare three approaches and will test performance based on the following:

- Risk detection accuracy
- Groundedness of explanations
- Hallucination rate
- Quality of financial reasoning

## Evaluation Metrics

The system will be evaluated using both **automatic metrics and human review**.

More Updates to be made to this document.


