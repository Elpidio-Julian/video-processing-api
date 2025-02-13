from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth
from typing import Optional

security = HTTPBearer()

class FirebaseAuth:
    def __init__(self):
        self.auth = auth

    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
        if not credentials:
            raise HTTPException(status_code=403, detail="No credentials provided")
        
        try:
            token = credentials.credentials
            decoded_token = self.auth.verify_id_token(token)
            return {
                "uid": decoded_token["uid"],
                "email": decoded_token.get("email"),
                "email_verified": decoded_token.get("email_verified", False)
            }
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid authentication credentials: {str(e)}")

    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
        user_data = await self.verify_token(credentials)
        return user_data

firebase_auth = FirebaseAuth()

# Dependency to use in routes
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    return await firebase_auth.get_current_user(credentials) 