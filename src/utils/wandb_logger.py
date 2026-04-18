"""
Weights & Biases integration utilities for the Credit Risk Detection project.

All public functions gracefully handle the case where W&B has not been
initialized — they log a warning and return without raising.

Exports:
    init_run(project, run_name, config) -> wandb.Run
    log_predictions(predictions_df, approach) -> None
    log_metrics(metrics, approach) -> None
    log_roc_curve(fpr, tpr, auc, approach) -> None
    log_training_step(step, loss, lr) -> None
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False
    logger.warning("wandb is not installed. W&B logging will be disabled.")

if TYPE_CHECKING:
    import wandb as _wandb  # noqa: F401


def _check_wandb_initialized(context: str = "") -> bool:
    """Return True if wandb is available and a run is active.

    Args:
        context: Human-readable description of the calling function,
            used in the warning message.

    Returns:
        True if logging should proceed, False otherwise.
    """
    if not _WANDB_AVAILABLE:
        logger.warning("wandb not available%s — skipping.", f" ({context})" if context else "")
        return False
    if wandb.run is None:
        logger.warning(
            "wandb run is not initialized%s — skipping. Call init_run() first.",
            f" ({context})" if context else "",
        )
        return False
    return True


def init_run(
    project: str,
    run_name: str,
    config: dict,
) -> Optional["_wandb.Run"]:
    """Initialize a W&B run.

    Args:
        project: W&B project name.
        run_name: Display name for this run.
        config: Hyperparameter/configuration dictionary to log.

    Returns:
        The wandb.Run object, or None if wandb is unavailable.
    """
    if not _WANDB_AVAILABLE:
        logger.warning("wandb not available — cannot initialize run.")
        return None

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        reinit=True,
    )
    logger.info("W&B run initialized: project=%s, name=%s, id=%s", project, run_name, wandb.run.id)
    return run


def log_predictions(predictions_df: pd.DataFrame, approach: str) -> None:
    """Log model predictions as a W&B Table artifact.

    Args:
        predictions_df: DataFrame with columns matching the predictions schema.
        approach: Approach identifier (e.g. 'baseline', 'rag', 'lora').
    """
    if not _check_wandb_initialized(f"log_predictions/{approach}"):
        return

    try:
        table = wandb.Table(dataframe=predictions_df.astype(str))
        wandb.log({f"predictions/{approach}": table})
        logger.info("Logged predictions table for approach '%s' (%d rows)", approach, len(predictions_df))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to log predictions for '%s': %s", approach, exc)


def log_metrics(metrics: Dict[str, float], approach: str) -> None:
    """Log a flat metrics dictionary to W&B.

    Keys are namespaced as ``metrics/{approach}/{metric_name}``.

    Args:
        metrics: Dictionary of metric name → value.
        approach: Approach identifier.
    """
    if not _check_wandb_initialized(f"log_metrics/{approach}"):
        return

    try:
        namespaced = {f"metrics/{approach}/{k}": v for k, v in metrics.items()}
        wandb.log(namespaced)
        logger.info("Logged %d metrics for approach '%s'", len(metrics), approach)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to log metrics for '%s': %s", approach, exc)


def log_roc_curve(
    fpr: "np.ndarray",
    tpr: "np.ndarray",
    auc: float,
    approach: str,
) -> None:
    """Log an ROC curve to W&B as a custom chart.

    The curve is logged as a W&B Table with columns ``fpr`` and ``tpr``,
    plus the AUC scalar.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.
        auc: Area under the ROC curve.
        approach: Approach identifier.
    """
    if not _check_wandb_initialized(f"log_roc_curve/{approach}"):
        return

    try:
        roc_table = wandb.Table(
            columns=["fpr", "tpr"],
            data=[[float(f), float(t)] for f, t in zip(fpr, tpr)],
        )
        wandb.log(
            {
                f"roc/{approach}/curve": roc_table,
                f"roc/{approach}/auc": auc,
            }
        )
        logger.info("Logged ROC curve for approach '%s' (AUC=%.4f)", approach, auc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to log ROC curve for '%s': %s", approach, exc)


def log_training_step(step: int, loss: float, lr: float) -> None:
    """Log a single training step's loss and learning rate.

    Args:
        step: Global training step number.
        loss: Training loss value.
        lr: Current learning rate.
    """
    if not _check_wandb_initialized("log_training_step"):
        return

    try:
        wandb.log({"train/loss": loss, "train/lr": lr, "train/step": step}, step=step)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to log training step %d: %s", step, exc)
