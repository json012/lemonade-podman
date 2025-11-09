import asyncio
import os
from typing import List, Dict

import torch
from pydantic import validate_call

from app.src.inference import HuggingFaceModelID, DEVICE, DTYPE
from app.src.inference.load import ml_models


@validate_call
def _inference_sync(
    model_id: HuggingFaceModelID,
    text: List[str],
) -> Dict[str, float]:
    """Sycnronous function for running pytorch inference.
    Should not be called directly.
    Use async `inference()` instead.
    """

    model = ml_models[model_id.name]
    with torch.inference_mode():
        enc = model.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        if DEVICE == "cuda":
            enc = {k: v.to(DEVICE, non_blocking=True) for k, v in enc.items()}

        with (
            torch.autocast(device_type="cuda", dtype=DTYPE)
            if DEVICE == "cuda"
            else torch.no_grad()
        ):
            logits = ml_models[model_id.name].model(**enc).logits
            probs = torch.softmax(logits, dim=-1).float().cpu()

        return probs


MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


@validate_call
async def inference(
    model_id: HuggingFaceModelID,
    text: List[str],
) -> Dict[str, float]:
    """Wrapper function to running pytorch inference asynchronously."""
    async with semaphore:
        return await asyncio.to_thread(_inference_sync, model_id, text)
