#!/usr/bin/env python3
import os
import json
import base64

def create_firebase_credentials():
    """Create Firebase credentials file from environment variable."""
    try:
        # Get the base64 encoded credentials from environment variable
        firebase_creds_b64 = os.getenv('FIREBASE_CREDENTIALS')
        if not firebase_creds_b64:
            raise ValueError("FIREBASE_CREDENTIALS environment variable not set")

        # Decode the base64 string
        firebase_creds_json = base64.b64decode(firebase_creds_b64).decode('utf-8')

        # Parse JSON to validate format
        creds_data = json.loads(firebase_creds_json)

        # Required fields for Firebase credentials
        required_fields = [
            "type",
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
            "client_id",
            "auth_uri",
            "token_uri",
            "auth_provider_x509_cert_url",
            "client_x509_cert_url",
            "universe_domain"
        ]

        # Verify all required fields are present
        for field in required_fields:
            if field not in creds_data:
                raise ValueError(f"Missing required field in credentials: {field}")

        # Write the credentials file
        creds_path = "create-tok-firebase-adminsdk-fbsvc-21d11da1ed.json"
        with open(creds_path, 'w') as f:
            json.dump(creds_data, f)

        # Secure the file permissions (only readable by owner)
        os.chmod(creds_path, 0o600)

        print("Firebase credentials file created successfully")
        return True

    except Exception as e:
        print(f"Error creating Firebase credentials: {str(e)}")
        return False

if __name__ == "__main__":
    if not create_firebase_credentials():
        print("Failed to create Firebase credentials file")
        exit(1)
    print("Startup script completed successfully") 