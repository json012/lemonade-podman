"""
This module contains routes for providing inference endpoints to [Guardrails AI](https://www.guardrailsai.com) guards.
"""

from .router import router
from .routes import (
    detect_jailbreak,
    # detect_pii,
    gibberish_text,
    nsfw_text,
    toxic_language,
)


__all__ = [
    "detect_jailbreak",
    "gibberish_text",
    "nsfw_text",
    "toxic_language",
    "router",
]
