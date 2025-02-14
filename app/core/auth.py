from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth
from typing import Optional

security = HTTPBearer()

async def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify Firebase ID token and return the user ID.
    This uses the Firebase Admin SDK to verify the token from the client.
    """
    try:
        # Remove 'Bearer ' prefix if present
        token = credentials.credentials
        # Verify the token
        decoded_token = auth.verify_id_token(token)
        # Return the user ID from the token
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) 