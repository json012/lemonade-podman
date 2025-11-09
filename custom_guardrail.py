import os
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from litellm import module_level_aclient
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from pydantic import BaseModel


class InferenceResult(BaseModel):
    flagged: bool
    scores: Dict[str, float]


class InferenceResponse(BaseModel):
    result: List[InferenceResult]


class Endpoint(str, Enum):
    INFERENCE_PROMPT_INJECTION = "inference/prompt_injection"
    INFERENCE_NON_ENGLISH = "inference/non_english"

    # HEURISTIC_PROMPT_INJECTION = "heuristic/prompt_injection"

    # GUARDRAILS_AI_PROMPT_INJECTION = "guardrails_ai/detect_jailbreak"
    # GUARDRAILS_AI_GIBBERISH_TEXT = "guardrails_ai/gibberish_text"
    # GUARDRAILS_AI_NSFW_TEXT = "guardrails_ai/nsfw_text"
    # GUARDRAILS_AI_TOXIC_LANGUAGE = "guardrails_ai/toxic_language"


class CustomGuardrailAPI(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str,
        **kwargs,
    ):
        self.api_endpoint = guardrail_name
        self.api_base = os.environ.get("CUSTOM_GUARDRAIL_API", "http://inference:8000")
        self.api_key = os.environ.get("CUSTOM_GUARDRAIL_API_KEY", None)

        if not Endpoint(guardrail_name):
            raise ValueError(
                "Guardrail name must be a valid endpoint: %s",
                guardrail_name,
            )

        if not self.api_base or not self.api_key:
            raise ValueError(
                "Environment variables for CUSTOM_GUARDRAIL_API, "
                "CUSTOM_GUARDRAIL_API_KEY are required"
            )

        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """
        from httpx import URL

        messages = data.get("messages", [])
        messages = [message.get("content", "") for message in messages]

        response = await module_level_aclient.post(
            url=str(URL(self.api_base).join(f"/{self.api_endpoint}")),
            json={
                "text": messages,
                "threshold": 0.5,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        inference_response = InferenceResponse(**response.json())

        is_flagged = any(result.flagged for result in inference_response.result)
        messages_str = "\n\n".join(messages)

        if is_flagged:
            verbose_proxy_logger.error(
                "One or more messages in this conversation are flagged "
                f"for violating the {self.api_endpoint} guardrail. \n\n {messages_str}",
            )

            raise ValueError(
                "One or more messages in this conversation are flagged "
                f"for violating the {self.api_endpoint} guardrail."
            )

        return data
