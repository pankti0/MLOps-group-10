"""
run_inference_all.py — Run credit-risk inference for all companies.

Usage:
    # Real inference (requires GPU + downloaded model):
    python scripts/run_inference_all.py --approach baseline
    python scripts/run_inference_all.py --approach rag
    python scripts/run_inference_all.py --approach lora_r8  --adapter-path data/models/lora_adapter_r8/final_adapter
    python scripts/run_inference_all.py --approach lora_r16 --adapter-path data/models/lora_adapter/final_adapter
    python scripts/run_inference_all.py --approach lora_r32 --adapter-path data/models/lora_adapter_r32/final_adapter

    # Pipeline smoke-test without GPU:
    python scripts/run_inference_all.py --approach baseline --mock
    python scripts/run_inference_all.py --approach rag --mock

Flags:
    --approach   baseline | rag | lora_r8 | lora_r16 | lora_r32
    --mock       Skip model loading; return random scores (for testing)
    --model-name Override the HuggingFace model identifier
    --adapter-path  Path to LoRA adapter dir (required for lora_* approaches)
    --no-quantize   Disable 4-bit quantisation
"""

import argparse
import json
import logging
import os
import random
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_inference_all")


LABELS_CSV = os.path.join(_REPO_ROOT, "data", "labels", "company_labels.csv")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(_REPO_ROOT, "data", "results")
INDEX_PATH = os.path.join(_REPO_ROOT, "data", "embeddings", "faiss.index")
METADATA_PATH = os.path.join(_REPO_ROOT, "data", "embeddings", "chunk_metadata.json")

_DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

_CSV_COLUMNS = [
    "ticker",
    "company_name",
    "predicted_score",
    "predicted_label",
    "risk_level",
    "key_signals",
    "citations",
    "raw_output",
    "approach",
]

# Mock agent (no GPU required)


class _MockAgent:
    """Returns random predictions without loading a model."""

    def __init__(self, approach: str) -> None:
        self.approach = approach

    def analyze_all(self, companies_df, sections_dir: str):
        import pandas as pd

        rows = []
        for _, row in companies_df.iterrows():
            score = round(random.uniform(10, 90), 1)
            label = 1 if score >= 50 else 0
            level = "low" if score < 35 else ("high" if score >= 65 else "medium")
            rows.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "predicted_score": score,
                    "predicted_label": label,
                    "risk_level": level,
                    "key_signals": json.dumps(["mock_signal_1", "mock_signal_2"]),
                    "citations": json.dumps(["[mock] No real citations."]),
                    "raw_output": f"[MOCK] score={score}",
                    "approach": self.approach,
                }
            )

        df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"{self.approach}_predictions.csv")
        df.to_csv(out_path, index=False)
        logger.info("[mock] Predictions saved to '%s'.", out_path)
        return df


# CLI

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run credit-risk inference for all companies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--approach",
        choices=["baseline", "rag", "lora_r8", "lora_r16", "lora_r32"],
        required=True,
        help="Inference approach to use.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip model loading and return random scores (for pipeline testing).",
    )
    parser.add_argument(
        "--model-name",
        default=_DEFAULT_MODEL,
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--adapter-path",
        default="",
        help="Path to LoRA adapter directory (required for --approach lora).",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantisation (higher VRAM usage).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold number to use for cross-validation (1-5). If not set, uses all companies or legacy test.jsonl.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to use from the fold directory (default: test).",
    )
    return parser.parse_args()

# Model loading

def load_model_for_approach(args: argparse.Namespace):
    """Load the appropriate model based on CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple (model, tokenizer).
    """
    from src.models.base_loader import load_model, load_model_with_adapter

    quantize = not args.no_quantize

    if args.approach in ("lora_r8", "lora_r16", "lora_r32"):
        if not args.adapter_path:
            logger.error("--adapter-path is required for --approach %s.", args.approach)
            sys.exit(1)
        logger.info("Loading base model + LoRA adapter from '%s'.", args.adapter_path)
        return load_model_with_adapter(
            base_model_name=args.model_name,
            adapter_path=args.adapter_path,
            quantize=quantize,
        )
    else:
        logger.info("Loading model '%s' (quantize=%s).", args.model_name, quantize)
        return load_model(
            model_name=args.model_name,
            quantize=quantize,
            device_map="auto",
        )


def print_summary(df) -> None:
    """Print a compact summary table of predictions to stdout."""
    print(f"  PREDICTIONS SUMMARY  (approach: {df['approach'].iloc[0]})")
    header = f"  {'Ticker':<8}  {'Company':<30}  {'Score':>6}  {'Label':>5}  {'Level'}"
    print(header)
    print("  " + "-" * 68)
    for _, row in df.iterrows():
        print(
            f"  {str(row['ticker']):<8}  "
            f"{str(row['company_name'])[:30]:<30}  "
            f"{float(row['predicted_score']):>6.1f}  "
            f"{int(row['predicted_label']):>5}  "
            f"{str(row['risk_level'])}"
        )
    avg_score = df["predicted_score"].mean()
    n_high = (df["risk_level"] == "high").sum()
    n_medium = (df["risk_level"] == "medium").sum()
    n_low = (df["risk_level"] == "low").sum()
    print(f"  Avg score: {avg_score:.1f}  |  High: {n_high}  Medium: {n_medium}  Low: {n_low}")


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger.info("Starting inference: approach=%s, mock=%s.", args.approach, args.mock)

    import pandas as pd


    # --- FOLD/SPLIT LOGIC ---
    fold_dir = None
    split_jsonl = None
    companies_df = None
    split_name = args.split
    if args.fold is not None:
        fold_dir = os.path.join(_REPO_ROOT, "data", "finetune", f"fold_{args.fold}")
        split_jsonl = os.path.join(fold_dir, f"{split_name}.jsonl")
        if not os.path.isfile(split_jsonl):
            logger.error(f"Split file not found for fold {args.fold}: {split_jsonl}")
            sys.exit(1)
        # Load tickers from split
        tickers = []
        with open(split_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        ticker = obj.get("ticker")
                        if ticker:
                            tickers.append(str(ticker))
                    except Exception:
                        continue
        # Load company list
        if not os.path.isfile(LABELS_CSV):
            logger.error("Labels CSV not found: %s", LABELS_CSV)
            sys.exit(1)
        all_companies_df = pd.read_csv(LABELS_CSV)
        companies_df = all_companies_df[all_companies_df["ticker"].astype(str).isin(tickers)].copy()
        logger.info(f"Loaded {len(companies_df)} companies for fold {args.fold} split '{split_name}'.")
    else:
        # Legacy: use all companies or test.jsonl for LoRA
        if not os.path.isfile(LABELS_CSV):
            logger.error("Labels CSV not found: %s", LABELS_CSV)
            sys.exit(1)
        companies_df = pd.read_csv(LABELS_CSV)
        logger.info("Loaded %d companies from '%s'.", len(companies_df), LABELS_CSV)

    # Build agent
    if args.mock:
        logger.warning("--mock flag set: using mock agent (no model loaded).")
        agent = _MockAgent(approach=args.approach)
        predictions_df = agent.analyze_all(companies_df, PROCESSED_DIR)

    elif args.approach == "baseline":
        model, tokenizer = load_model_for_approach(args)
        from src.models.baseline_agent import BaselineAgent

        agent = BaselineAgent(model=model, tokenizer=tokenizer)
        predictions_df = agent.analyze_all(
            companies_df=companies_df,
            sections_dir=PROCESSED_DIR,
        )

    elif args.approach == "rag":
        model, tokenizer = load_model_for_approach(args)

        # RAG requires a pre-built FAISS index
        if not os.path.isfile(INDEX_PATH) or not os.path.isfile(METADATA_PATH):
            logger.error(
                "FAISS index not found. Run scripts/build_embeddings.py first.\n"
                "  Expected: %s\n  Expected: %s",
                INDEX_PATH,
                METADATA_PATH,
            )
            sys.exit(1)

        from src.embeddings.embedder import Embedder
        from src.embeddings.faiss_store import FAISSStore
        from src.models.rag_agent import RAGAgent

        embedder = Embedder()
        store = FAISSStore(dimension=embedder.get_dimension())
        store.load(INDEX_PATH, METADATA_PATH)

        agent = RAGAgent(
            model=model,
            tokenizer=tokenizer,
            faiss_store=store,
            embedder=embedder,
        )
        predictions_df = agent.analyze_all(
            companies_df=companies_df,
            sections_dir=PROCESSED_DIR,
        )

    elif args.approach in ("lora_r8", "lora_r16", "lora_r32"):
        model, tokenizer = load_model_for_approach(args)
        from src.models.lora_agent import LoRAAgent

        agent = LoRAAgent(model=model, tokenizer=tokenizer)
        predictions_df = agent.analyze_all(
            companies_df=companies_df,
            sections_dir=PROCESSED_DIR,
        )
        predictions_df["approach"] = args.approach

    else:
        logger.error("Unknown approach: %s", args.approach)
        sys.exit(1)

    print_summary(predictions_df)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args.approach}_fold{args.fold or 'all'}_{args.split}.csv")
    predictions_df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
