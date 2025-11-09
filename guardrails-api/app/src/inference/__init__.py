from enum import Enum
from typing import Optional

import torch
from pydantic import BaseModel, model_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
capability = torch.cuda.get_device_capability(0) if DEVICE == "cuda" else (0, 0)
use_bf16 = DEVICE == "cuda" and capability[0] >= 8
DTYPE = (
    torch.bfloat16
    if use_bf16
    else (torch.float16 if DEVICE == "cuda" else torch.float32)
)


class HuggingFaceModelID(str, Enum):
    PROMPT_INJECTION = "protectai/deberta-v3-base-prompt-injection-v2"
    LANGUAGE = "papluca/xlm-roberta-base-language-detection"


class InferenceModel(BaseModel):
    model_id: HuggingFaceModelID
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForSequenceClassification] = None

    @model_validator(mode="after")
    def validate_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id.value)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id.value, dtype=DTYPE
        ).to(DEVICE)

        self.model.eval()

        return self

    class Config:
        arbitrary_types_allowed = True
