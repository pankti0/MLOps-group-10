"""
LoRA fine-tuning trainer for Mistral-7B credit risk analysis.

Implements 4-bit quantization + LoRA PEFT training using trl's SFTTrainer.

Exports:
    prepare_model_for_training(base_model_name, lora_config, quant_config)
        -> Tuple[model, tokenizer]
    load_training_data(train_path, val_path) -> Tuple[Dataset, Dataset]
    train(model, tokenizer, train_dataset, val_dataset, training_config,
          output_dir) -> None
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)


def prepare_model_for_training(
    base_model_name: str,
    lora_config: dict,
    quant_config: dict,
) -> Tuple:
    """Load base model with 4-bit quantization and apply LoRA adapters.

    Args:
        base_model_name: HuggingFace model ID, e.g.
            ``"mistralai/Mistral-7B-Instruct-v0.2"``.
        lora_config: Dict with LoRA hyperparameters:
            r, lora_alpha, target_modules, lora_dropout, bias, task_type.
        quant_config: Dict with BitsAndBytes quantization settings:
            load_in_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type,
            bnb_4bit_use_double_quant.

    Returns:
        Tuple of (model, tokenizer) ready for SFT training.

    Raises:
        ImportError: If required libraries are not installed.
        RuntimeError: If model loading fails.
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "Required libraries missing. Install with: "
            "pip install transformers peft bitsandbytes accelerate"
        ) from exc

    logger.info("Loading tokenizer from '%s'...", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Set pad_token = eos_token")

    # Build BitsAndBytes quantization config
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(
        quant_config.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
    )

    logger.info("Loading model '%s' with 4-bit quantization...", base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    logger.info("Applying prepare_model_for_kbit_training...")
    model = prepare_model_for_kbit_training(model)

    # Build LoRA config
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )

    logger.info("Applying LoRA adapters: r=%d, alpha=%d", peft_config.r, peft_config.lora_alpha)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_training_data(train_path: str, val_path: str) -> Tuple:
    """Load JSONL training and validation datasets.

    Each line of the JSONL files must be a JSON object with at least a
    ``"text"`` key containing the full instruction-response string.

    Args:
        train_path: Path to ``train.jsonl``.
        val_path: Path to ``val.jsonl``.

    Returns:
        Tuple of (train_dataset, val_dataset) as HuggingFace Datasets.

    Raises:
        FileNotFoundError: If either file is missing.
    """
    try:
        from datasets import Dataset, load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("Install datasets: pip install datasets") from exc

    for path in (train_path, val_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data file not found: {path}")

    logger.info("Loading training data from %s", train_path)
    train_dataset = load_dataset("json", data_files=train_path, split="train")

    logger.info("Loading validation data from %s", val_path)
    val_dataset = load_dataset("json", data_files=val_path, split="train")

    logger.info(
        "Loaded %d train examples, %d validation examples.",
        len(train_dataset),
        len(val_dataset),
    )
    return train_dataset, val_dataset


def train(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    training_config: dict,
    output_dir: str,
) -> None:
    """Fine-tune the model using trl SFTTrainer.

    Args:
        model: PEFT model with LoRA adapters (from prepare_model_for_training).
        tokenizer: Corresponding tokenizer.
        train_dataset: HuggingFace Dataset for training.
        val_dataset: HuggingFace Dataset for validation.
        training_config: Training hyperparameters dict. Expected keys:
            num_epochs, per_device_train_batch_size,
            gradient_accumulation_steps, learning_rate, warmup_ratio,
            lr_scheduler_type, max_seq_length, fp16, bf16,
            logging_steps, eval_steps, save_steps,
            load_best_model_at_end.
        output_dir: Directory where checkpoints and final adapter are saved.

    Raises:
        ImportError: If trl or transformers are not installed.
    """
    try:
        from transformers import TrainingArguments  # type: ignore
        from trl import SFTTrainer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Required libraries missing. Install with: pip install trl transformers"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 2e-4),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", True),
        logging_steps=training_config.get("logging_steps", 10),
        eval_steps=training_config.get("eval_steps", 50),
        save_steps=training_config.get("save_steps", 100),
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        report_to="wandb",
        run_name=training_config.get("run_name", "lora-sft"),
        dataloader_num_workers=0,
        group_by_length=True,
    )

    logger.info(
        "Starting SFT training: epochs=%d, lr=%s, batch=%d, grad_accum=%d",
        training_args.num_train_epochs,
        training_args.learning_rate,
        training_args.per_device_train_batch_size,
        training_args.gradient_accumulation_steps,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=training_config.get("max_seq_length", 2048),
        args=training_args,
    )

    trainer.train()
    logger.info("Training complete. Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved to %s", output_dir)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from src.utils.config_loader import get_repo_root, load_config
    from src.utils.wandb_logger import init_run

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = get_repo_root()
    config = load_config(str(repo_root / "configs" / "lora_config.yaml"))

    wandb_cfg = config.get("wandb", {})
    init_run(
        project=wandb_cfg.get("project", "credit-risk-detection"),
        run_name=wandb_cfg.get("run_name", "lora-sft"),
        config=config,
    )

    model, tokenizer = prepare_model_for_training(
        base_model_name=config["base_model"],
        lora_config=config["lora"],
        quant_config=config["quantization"],
    )

    train_path = str(repo_root / "data" / "finetune" / "train.jsonl")
    val_path = str(repo_root / "data" / "finetune" / "val.jsonl")
    train_dataset, val_dataset = load_training_data(train_path, val_path)

    output_dir = str(repo_root / config.get("output_dir", "data/models/lora_adapter"))
    train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_config=config["training"],
        output_dir=output_dir,
    )
