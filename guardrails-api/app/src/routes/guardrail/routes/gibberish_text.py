from fastapi import Security
from guardrails import Guard, ValidationOutcome, install

from ..router import router
from ..models import GuardrailsRequest
from app.src.dependencies import verify_api_key

try:
    from guardrails.hub import GibberishText
except ImportError:
    install("hub://guardrails/gibberish_text", install_local_models=True)
    from guardrails.hub import GibberishText


@router.post("/gibberish_text", response_model=ValidationOutcome)
def gibberish_text(req: GuardrailsRequest, api_key: str = Security(verify_api_key)):
    guard = Guard().use(
        GibberishText, threshold=0.5, validation_method="sentence", on_fail="noop"
    )

    text_vals = "\n\n".join(
        [item for input in req.inputs if input.name == "text" for item in input.data]
    )

    result = guard.parse(
        llm_output=text_vals,
        metadata=None,
    )

    return result
