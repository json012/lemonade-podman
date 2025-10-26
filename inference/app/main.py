import os
from contextlib import asynccontextmanager
from typing import List, Dict

from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, conlist
from torch import max as torch_max

from app.src.inference import HuggingFaceModelID
from app.src.inference.run import inference
from app.src.inference.load import load_models, ml_models


API_KEY = os.getenv("API_KEY")
bearer_scheme = HTTPBearer(auto_error=True)


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured",
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )

    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()

    yield

    ml_models.clear()


# Request Models
class TextRequest(BaseModel):
    text: conlist(item_type=str, min_length=1, max_length=32)
    threshold: float = 0.5


# Response Models
class InferenceResult(BaseModel):
    flagged: bool
    scores: Dict[str, float]


class InferenceResponse(BaseModel):
    result: List[InferenceResult]


app = FastAPI(
    title="Inference API",
    description="API for inference tasks",
    #     docs_url=None,
    #     redoc_url=None,
    #     openapi_url=None,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inference/prompt_injection", response_model=InferenceResponse)
async def detect_prompt_injection(
    req: TextRequest, api_key: str = Security(verify_api_key)
):
    """Detect prompt injection attacks in text."""
    probs = await inference(HuggingFaceModelID.PROMPT_INJECTION, req.text)

    # Batch text
    # return InferenceResponse(
    #     flagged=[p[1].item() >= req.threshold for p in probs],
    #     scores=[{"injection": p[1].item()} for p in probs],
    # )

    return InferenceResponse(
        result=[
            InferenceResult(
                flagged=p[1].item() >= req.threshold,
                scores={"injection": p[1].item()},
            )
            for p in probs
        ]
    )


@app.post("/inference/non_english", response_model=InferenceResponse)
async def detect_nonenglish(req: TextRequest, api_key: str = Security(verify_api_key)):
    """Detect non-English language text."""
    probs = await inference(HuggingFaceModelID.LANGUAGE, req.text)
    model = ml_models[HuggingFaceModelID.LANGUAGE.name]

    # Batch text
    id2label = model.model.config.id2label

    # Get top predictions for all texts
    top_probs, top_indices = torch_max(probs, dim=1)

    # return InferenceResponse(
    #     flagged=[
    #         id2label[idx.item()] != "en" and prob.item() >= req.threshold
    #         for prob, idx in zip(top_probs, top_indices)
    #     ],
    #     scores=[
    #         {"non_english": prob.item()} for prob, idx in zip(top_probs, top_indices)
    #     ],
    # )

    return InferenceResponse(
        result=[
            InferenceResult(
                flagged=id2label[idx.item()] != "en" and prob.item() >= req.threshold,
                scores={"non_english": prob.item()},
            )
            for prob, idx in zip(top_probs, top_indices)
        ]
    )
