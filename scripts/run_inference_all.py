"""
run_inference_all.py — Run credit-risk inference for all companies.

Usage:
    # Real inference (requires GPU + downloaded model):
    python scripts/run_inference_all.py --approach baseline
    python scripts/run_inference_all.py --approach rag
    python scripts/run_inference_all.py --approach lora --adapter-path data/finetune/lora_adapter

    # Pipeline smoke-test without GPU:
    python scripts/run_inference_all.py --approach baseline --mock
    python scripts/run_inference_all.py --approach rag --mock

Flags:
    --approach   baseline | rag | lora
    --mock       Skip model loading; return random scores (for testing)
    --model-name Override the HuggingFace model identifier
    --adapter-path  Path to LoRA adapter dir (lora approach only)
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

_DEFAULT_MODEL = "microsoft/phi-2"

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
        choices=["baseline", "rag", "lora"],
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

    if args.approach == "lora":
        if not args.adapter_path:
            logger.error("--adapter-path is required for --approach lora.")
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

    # Load company list
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

    elif args.approach in ("rag", "lora"):
        model, tokenizer = load_model_for_approach(args)

        # RAG requires a pre-built FAISS index
        if not os.path.isfile(INDEX_PATH) or not os.path.isfile(METADATA_PATH):
            logger.error(
                "FAISS index not found. Run scripts/03_build_embeddings.py first.\n"
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

        approach_label = args.approach  # "rag" or "lora"
        agent = RAGAgent(
            model=model,
            tokenizer=tokenizer,
            faiss_store=store,
            embedder=embedder,
        )
        # Patch approach label for LoRA
        if approach_label == "lora":
            import src.models.rag_agent as _rag_mod
            _original_csv = _rag_mod._OUTPUT_CSV
            _rag_mod._OUTPUT_CSV = os.path.join(RESULTS_DIR, "lora_predictions.csv")

        predictions_df = agent.analyze_all(
            companies_df=companies_df,
            sections_dir=PROCESSED_DIR,
        )

        # Fix approach column for lora
        if approach_label == "lora":
            predictions_df["approach"] = "lora"
            _rag_mod._OUTPUT_CSV = _original_csv  
    else:
        logger.error("Unknown approach: %s", args.approach)
        sys.exit(1)

    print_summary(predictions_df)
    out_path = os.path.join(RESULTS_DIR, f"{args.approach}_predictions.csv")
    print(f"Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
