from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.src.inference import HuggingFaceModelID
from app.src.logger import logger


def install() -> None:
    logger.info("Downloading models...")

    # Download and cache all model artifacts without allocating on GPU
    for m in HuggingFaceModelID:
        logger.info(f"Downloading {m.value}...")

        AutoTokenizer.from_pretrained(m.value)

        AutoModelForSequenceClassification.from_pretrained(m.value)


if __name__ == "__main__":
    install()
