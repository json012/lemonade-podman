from fastapi import Security
from guardrails import Guard, ValidationOutcome, install

from ..router import router
from ..models import GuardrailsRequest
from app.src.dependencies import verify_api_key

try:
    from guardrails.hub import ToxicLanguage
except ImportError:
    install("hub://guardrails/toxic_language", install_local_models=True)
    from guardrails.hub import ToxicLanguage


@router.post("/toxic_language", response_model=ValidationOutcome)
def toxic_language(req: GuardrailsRequest, api_key: str = Security(verify_api_key)):
    guard = Guard().use(
        ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="noop"
    )

    text_vals = "\n\n".join(
        [item for input in req.inputs if input.name == "text" for item in input.data]
    )

    result = guard.parse(
        llm_output=text_vals,
        metadata=None,
    )

    return result
