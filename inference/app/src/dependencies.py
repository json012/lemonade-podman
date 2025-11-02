import os

from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


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
