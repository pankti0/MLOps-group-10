#!/usr/bin/env python3
"""
run_rag_lora.py

Runs inference using RAG + LoRA (Mistral-7B).

Pipeline:
    Query → FAISS retrieval → build context → LoRA-tuned LLM → prediction

Outputs:
    data/results/rag_lora_predictions.csv
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging

import faiss
import torch
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.utils.config_loader import get_repo_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG + LoRA inference")
    parser.add_argument("--faiss_index", type=str, default="data/embeddings/faiss.index")
    parser.add_argument("--metadata", type=str, default="data/embeddings/chunk_metadata.json")
    parser.add_argument("--labels", type=str, default="data/labels/company_labels.csv")
    parser.add_argument("--lora_path", type=str, default="data/models/lora_adapter/final_adapter")
    parser.add_argument("--output", type=str, default="data/results/rag_lora_predictions.csv")
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


# -----------------------------
# Load retrieval components
# -----------------------------
def load_retriever(index_path, metadata_path):
    logger.info("Loading FAISS index...")
    index = faiss.read_index(index_path)

    logger.info("Loading metadata...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


# -----------------------------
# Load LoRA model
# -----------------------------
def load_lora_model(lora_path):
    logger.info("Loading LoRA model...")

    base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    return model, tokenizer


# -----------------------------
# Build prompt
# -----------------------------
def build_prompt(context, company):
    return f"""
You are a financial risk analyst.

Analyze the following company based on extracted report sections.

Company: {company}

Context:
{context}

Task:
Classify the company as HIGH RISK or LOW RISK.
Also provide a confidence score between 0 and 1.

Answer in this format:
Prediction: <HIGH/LOW>
Confidence: <0-1>
Reason: <brief explanation>
"""


# -----------------------------
# Generate prediction
# -----------------------------
def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -----------------------------
# Extract prediction
# -----------------------------
def parse_output(text):
    text = text.upper()

    if "HIGH" in text:
        pred = 1
    elif "LOW" in text:
        pred = 0
    else:
        pred = -1

    # crude confidence extraction
    import re
    match = re.search(r"CONFIDENCE[: ]+([0-9.]+)", text)
    score = float(match.group(1)) if match else 0.5

    return pred, score


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    repo_root = get_repo_root()

    # Resolve paths
    index_path = os.path.join(repo_root, args.faiss_index)
    metadata_path = os.path.join(repo_root, args.metadata)
    labels_path = os.path.join(repo_root, args.labels)
    output_path = os.path.join(repo_root, args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load components
    index, metadata = load_retriever(index_path, metadata_path)
    model, tokenizer = load_lora_model(args.lora_path)

    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # Load companies
    df = pd.read_csv(labels_path)

    results = []

    logger.info("Running RAG + LoRA inference...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        company = row["company_name"]

        query = f"What is the risk level of {company}?"

        # ---- RETRIEVAL ----
        q_emb = embed_model.encode(query)
        D, I = index.search(q_emb.reshape(1, -1), args.top_k)

        chunks = [metadata[i]["text"] for i in I[0]]
        context = "\n\n".join(chunks)

        # ---- GENERATION ----
        prompt = build_prompt(context, company)
        output = generate(model, tokenizer, prompt)

        pred, score = parse_output(output)

        results.append({
            "ticker": row["ticker"],
            "prediction": pred,
            "score": score
        })

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    logger.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()