from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.src.inference.load import load_models, ml_models
from app.src.routes import inference, guardrail


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()

    yield

    ml_models.clear()


app = FastAPI(
    title="Guardrails API",
    description="Serves LLM guardrails",
    #     docs_url=None,
    #     redoc_url=None,
    #     openapi_url=None,
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(inference.router)
app.include_router(guardrail.router)
