#!/usr/bin/env python3
"""
train_lora.py

Orchestrates LoRA fine-tuning of Mistral-7B for credit risk analysis:
    1. Load lora_config.yaml
    2. Prepare model with 4-bit quantization and LoRA adapters
    3. Load training and validation data
    4. Initialize W&B run
    5. Run SFT training
    6. Save final adapter
    7. Log final metrics to W&B

Usage:
    python scripts/train_lora.py [--config configs/lora_config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.models.lora_trainer import load_training_data, prepare_model_for_training, train
from src.utils.config_loader import get_repo_root, load_config
from src.utils.wandb_logger import init_run, log_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Train LoRA adapter for credit risk detection.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_config.yaml",
        help="Path to lora_config.yaml (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold number for cross-validation (optional).",
    )
    return parser.parse_args()


def main() -> None:

    args = parse_args()
    repo_root = get_repo_root()

    #  1. Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(repo_root / config_path)

    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)


    # Determine config name for directory naming
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    # Resolve output directory, add fold and config name if fold is specified
    output_dir = config.get("output_dir", "data/models/lora_adapter")
    if not os.path.isabs(output_dir):
        output_dir = str(repo_root / output_dir)
    if args.fold is not None:
        output_dir = os.path.join(output_dir, f"fold_{args.fold}", config_name)
    else:
        output_dir = os.path.join(output_dir, config_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Model output directory: %s", output_dir)

    #  2. Prepare model 
    logger.info("Preparing model for training...")
    model, tokenizer = prepare_model_for_training(
        base_model_name=config["base_model"],
        lora_config=config["lora"],
        quant_config=config["quantization"],
    )


    #  3. Load training data (support cross-validation folds)
    finetune_dir = repo_root / "data" / "finetune"
    if args.fold is not None:
        fold_dir = finetune_dir / f"fold_{args.fold}"
        train_path = str(fold_dir / "train.jsonl")
        val_path = str(fold_dir / "val.jsonl")
        logger.info(f"Loading training data from fold {args.fold}: {fold_dir}")
    else:
        train_path = str(finetune_dir / "train.jsonl")
        val_path = str(finetune_dir / "val.jsonl")
        logger.info("Loading training data from %s", finetune_dir)
    train_dataset, val_dataset = load_training_data(train_path, val_path)

    #  4. Initialize W&B 
    wandb_cfg = config.get("wandb", {})
    project = wandb_cfg.get("project", "credit-risk-detection")
    run_name = wandb_cfg.get("run_name", "lora-sft")

    logger.info("Initializing W&B run: project=%s, name=%s", project, run_name)
    init_run(project=project, run_name=run_name, config=config)

    #  5. Train 
    logger.info("Starting training...")
    training_config = config.get("training", {})
    training_config["run_name"] = run_name  # pass run name to TrainingArguments

    train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_config=training_config,
        output_dir=output_dir,
    )
    logger.info("Training complete.")

    #  6. Save final adapter 
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)

    logger.info("Saving final adapter to %s", final_adapter_dir)
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        # Reload the trained model from the output directory
        trained_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True),
            output_dir
        )
        trained_model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)
        logger.info("Final adapter saved successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save final adapter: %s", exc)

    #  7. Log final metrics to W&B 
    final_metrics = {
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "num_epochs": training_config.get("num_epochs", 3),
        "learning_rate": training_config.get("learning_rate", 2e-4),
        "lora_r": config["lora"].get("r", 16),
        "lora_alpha": config["lora"].get("lora_alpha", 32),
    }
    log_metrics(final_metrics, approach="lora_training")
    logger.info("Final training metrics logged: %s", final_metrics)

    print("  LORA TRAINING COMPLETE")
    print(f"  Adapter saved to : {final_adapter_dir}")
    print(f"  Train examples   : {len(train_dataset)}")
    print(f"  Val examples     : {len(val_dataset)}")
    print(f"  Epochs           : {training_config.get('num_epochs', 3)}")


if __name__ == "__main__":
    main()
