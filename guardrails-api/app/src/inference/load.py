from app.src.inference import HuggingFaceModelID, InferenceModel
from app.src.logger import logger


ml_models: dict[str, InferenceModel] = {}


async def load_models() -> None:
    for model in HuggingFaceModelID:
        ml_models[model.name] = InferenceModel(model_id=model.value)

        logger.info(f"Loaded {model.value}")
