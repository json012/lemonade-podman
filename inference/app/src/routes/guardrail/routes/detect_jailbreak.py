from fastapi import Security
from guardrails import Guard, ValidationOutcome, install

from ..router import router
from ..models import GuardrailsRequest
from app.src.dependencies import verify_api_key

try:
    from guardrails.hub import DetectJailbreak
except ImportError:
    install("hub://guardrails/detect_jailbreak", install_local_models=True)
    from guardrails.hub import DetectJailbreak


@router.post("/detect_jailbreak", response_model=ValidationOutcome)
def detect_jailbreak(req: GuardrailsRequest, api_key: str = Security(verify_api_key)):
    guard = Guard().use(DetectJailbreak, on_fail="noop")

    text_vals = "\n\n".join(
        [item for input in req.inputs if input.name == "text" for item in input.data]
    )

    return guard.parse(llm_output=text_vals, metadata=None)
