from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify Firebase ID token and return the user ID.
    This is a simplified version that only returns the user ID.
    """
    if not credentials:
        raise HTTPException(status_code=403, detail="No credentials provided")
    
    try:
        token = credentials.credentials
        # Verify the token and get the user ID
        decoded_token = auth.verify_id_token(token)
        return decoded_token.get('uid')
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication credentials: {str(e)}"
        ) 