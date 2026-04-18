"""
Model loading utilities for Mistral 7B (and compatible causal LMs).

Exports:
    load_model(model_name, quantize, device_map) -> (model, tokenizer)
    load_model_with_adapter(base_model_name, adapter_path, quantize) -> (model, tokenizer)
    generate_response(model, tokenizer, prompt, max_new_tokens, temperature) -> str
"""

import logging
import os
from typing import Tuple

import torch

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def _build_bnb_config():
    """Return a BitsAndBytesConfig for 4-bit NF4 quantisation."""
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers is not installed. Run: pip install transformers"
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model(
    model_name: str = _DEFAULT_MODEL,
    quantize: bool = True,
    device_map: str = "auto",
) -> Tuple:
    """Load a causal LM and its tokenizer from HuggingFace.

    When ``quantize=True`` the model is loaded in 4-bit NF4 precision via
    bitsandbytes, which allows Mistral-7B to run on a single consumer GPU.

    Args:
        model_name: HuggingFace model identifier.
        quantize: Whether to apply 4-bit quantisation.
        device_map: Device placement strategy passed to ``from_pretrained``.
            Use ``"auto"`` to let accelerate decide, or ``"cpu"`` for CPU-only.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        RuntimeError: If the model cannot be loaded, with a hint to use
            ``ollama pull mistral`` as a local alternative.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers is not installed. Run: pip install transformers"
        ) from exc

    # Use HF_TOKEN from .env if present (required for gated models like Mistral)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN found — using authenticated HuggingFace download.")
    else:
        logger.warning("HF_TOKEN not set. Set it in .env if model download fails.")

    logger.info("Loading tokenizer for '%s'.", model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, token=hf_token
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer for '{model_name}': {exc}\n"
            "Ensure HF_TOKEN is set in your .env file and you have accepted\n"
            "the model licence at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
        ) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("pad_token set to eos_token ('%s').", tokenizer.eos_token)

    quant_kwargs: dict = {}
    if quantize:
        quant_kwargs["quantization_config"] = _build_bnb_config()
        logger.info("4-bit NF4 quantisation enabled.")
    else:
        logger.info("Quantisation disabled; loading in full precision.")

    logger.info("Loading model '%s' (device_map='%s').", model_name, device_map)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if not quantize else None,
            trust_remote_code=False,
            token=hf_token,
            **quant_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_name}': {exc}\n"
            "Possible causes:\n"
            "  - Model not downloaded (run `huggingface-cli download` or accept licence on HF hub)\n"
            "  - Insufficient GPU VRAM (try quantize=True or device_map='cpu')\n"
            "  - No compatible GPU (bitsandbytes requires CUDA)\n"
            "Alternative: run `ollama pull mistral` and use the Ollama HTTP API."
        ) from exc

    model.eval()
    logger.info("Model loaded and set to eval mode.")
    return model, tokenizer


def load_model_with_adapter(
    base_model_name: str = _DEFAULT_MODEL,
    adapter_path: str = "",
    quantize: bool = True,
) -> Tuple:
    """Load a base causal LM and apply a LoRA adapter.

    Args:
        base_model_name: HuggingFace identifier for the base model.
        adapter_path: Local path to the saved PEFT/LoRA adapter directory.
        quantize: Whether to apply 4-bit quantisation to the base model.

    Returns:
        Tuple of (model_with_adapter, tokenizer).

    Raises:
        RuntimeError: If loading fails.
    """
    try:
        from peft import PeftModel  # type: ignore
    except ImportError as exc:
        raise ImportError("peft is not installed. Run: pip install peft") from exc

    model, tokenizer = load_model(
        model_name=base_model_name,
        quantize=quantize,
        device_map="auto",
    )

    logger.info("Applying LoRA adapter from '%s'.", adapter_path)
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load LoRA adapter from '{adapter_path}': {exc}"
        ) from exc

    model.eval()
    logger.info("LoRA adapter applied successfully.")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Run greedy / low-temperature generation and return the decoded response.

    Only the newly generated tokens (after the input prompt) are returned.

    Args:
        model: A loaded causal LM (e.g. from ``load_model``).
        tokenizer: The corresponding tokenizer.
        prompt: Full prompt string (including any instruction formatting).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature; values close to 0 are near-greedy.

    Returns:
        Decoded generated text (prompt tokens excluded).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Move inputs to the same device as the model's first parameter
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    logger.debug("Generated %d tokens.", len(generated_ids))
    return response
