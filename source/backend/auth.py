import os
import time
import requests
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
import logging

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

STACK_NAME = os.getenv("STACK_NAME", "backend")
name = f"{STACK_NAME}-server"
logger = logging.getLogger(name)

# AWS Cognito configuration
COGNITO_REGION = os.getenv("COGNITO_REGION", os.getenv("AWS_REGION", "us-east-1"))
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
COGNITO_APP_CLIENT_ID = os.getenv("COGNITO_APP_CLIENT_ID")

# Construct Cognito issuer URL
COGNITO_ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}" if COGNITO_USER_POOL_ID else None
COGNITO_JWKS_URL = f"{COGNITO_ISSUER}/.well-known/jwks.json" if COGNITO_ISSUER else None

# Global JWKS cache
jwks_cache = {}
jwks_cache_expiry = 0
JWKS_CACHE_DURATION = 3600  # Cache for 1 hour

# JWT validation leeway for clock synchronization issues
# This addresses "The token is not yet valid (iat)" errors caused by
# minor clock differences between token issuer and validator
JWT_LEEWAY_SECONDS = 10

def get_jwks_keys():
    """
    Fetch and cache JWKS keys from AWS Cognito.
    
    Returns:
        dict: JWKS keys from Cognito
    
    Raises:
        Exception: If unable to fetch JWKS keys
    """
    global jwks_cache, jwks_cache_expiry
    
    # Check if we have cached keys that haven't expired
    current_time = time.time()
    if jwks_cache and current_time < jwks_cache_expiry:
        return jwks_cache
    
    if not COGNITO_JWKS_URL:
        raise Exception("Cognito JWKS URL not configured. Set COGNITO_USER_POOL_ID environment variable.")
    
    try:
        logger.info(f"Fetching JWKS from: {COGNITO_JWKS_URL}")
        response = requests.get(COGNITO_JWKS_URL, timeout=10)
        response.raise_for_status()
        
        jwks_data = response.json()
        
        # Cache the keys
        jwks_cache = jwks_data
        jwks_cache_expiry = current_time + JWKS_CACHE_DURATION
        
        logger.info(f"Successfully cached {len(jwks_data.get('keys', []))} JWKS keys")
        return jwks_data
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        raise Exception(f"Unable to fetch JWKS keys: {e}")

def get_signing_key(kid: str):
    """
    Get the signing key for a specific key ID from JWKS.
    
    Args:
        kid: Key ID from JWT header
    
    Returns:
        str: PEM-formatted public key
    
    Raises:
        Exception: If key not found or invalid
    """
    jwks = get_jwks_keys()
    
    # Find the key with matching kid
    for key in jwks.get('keys', []):
        if key.get('kid') == kid:
            # Convert JWK to PEM format
            if key.get('kty') == 'RSA':
                # Extract RSA components
                n = key.get('n')
                e = key.get('e')
                
                if not n or not e:
                    raise Exception(f"Invalid RSA key components for kid: {kid}")
                
                # Decode base64url
                from base64 import urlsafe_b64decode
                
                def base64url_decode(data):
                    # Add padding if needed
                    padding = 4 - len(data) % 4
                    if padding != 4:
                        data += '=' * padding
                    return urlsafe_b64decode(data)
                
                n_bytes = base64url_decode(n)
                e_bytes = base64url_decode(e)
                
                # Convert to integers
                n_int = int.from_bytes(n_bytes, byteorder='big')
                e_int = int.from_bytes(e_bytes, byteorder='big')
                
                # Create RSA public key
                public_numbers = rsa.RSAPublicNumbers(e_int, n_int)
                public_key = public_numbers.public_key()
                
                # Convert to PEM format
                pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                
                return pem.decode('utf-8')
            else:
                raise Exception(f"Unsupported key type: {key.get('kty')}")
    
    raise Exception(f"No key found for kid: {kid}")

def validate_token(token: str, user_id: str = None) -> dict:
    """
    Validate AWS Cognito JWT token with proper signature verification.
    
    Args:
        token: JWT token from AWS Cognito
        user_id: Optional user ID to cross-validate
    
    Returns:
        dict: User information if valid
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token required")
    
    
    try:
        logger.info(f"Validating token: {token}")
        # Decode header without verification to get kid
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get('kid')
        
        if not kid:
            raise HTTPException(status_code=401, detail="Token missing key ID")
        
        # Get the signing key for this kid
        try:
            signing_key = get_signing_key(kid)
        except Exception as e:
            logger.error(f"Failed to get signing key: {e}")
            raise HTTPException(status_code=401, detail="Unable to verify token signature")
        
        # Verify and decode the token
        try:
            decoded_token = jwt.decode(
                token,
                signing_key,
                algorithms=['RS256'],
                audience=COGNITO_APP_CLIENT_ID,
                issuer=COGNITO_ISSUER,
                leeway=JWT_LEEWAY_SECONDS,  # Add leeway for clock synchronization
                options={
                    "verify_signature": True,
                    "verify_aud": True,
                    "verify_iss": True,
                    "verify_exp": True,
                    "verify_iat": True
                }
            )
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except InvalidTokenError as e:
            logger.error(f"Token validation failed: {e}")
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
        
        # Validate token_use
        token_use = decoded_token.get('token_use')
        if token_use not in ['id', 'access']:
            raise HTTPException(status_code=401, detail=f"Invalid token_use: {token_use}")
        
        # Extract user information
        user_info = {
            "user_id": decoded_token.get("sub") or decoded_token.get("username"),
            "email": decoded_token.get("email"),
            "token_use": token_use,
            "exp": decoded_token.get("exp"),
            "iat": decoded_token.get("iat"),
            "iss": decoded_token.get("iss"),
            "aud": decoded_token.get("aud"),
            "cognito_username": decoded_token.get("cognito:username"),
            "token": token
        }
        
        # Ensure we have a user_id
        if not user_info["user_id"]:
            raise HTTPException(status_code=401, detail="Token missing user identifier")
        
        
        # Cross-validate user ID if provided
        # If the provided user_id looks like an email, compare with token email
        # Otherwise compare with token user_id (UUID)
        if user_id:
            if "@" in user_id:
                # Provided user_id is an email, compare with token email
                token_email = user_info.get("email", "").lower()
                provided_email = user_id.lower()
                if token_email != provided_email:
                    logger.warning(f"Email mismatch: token_email={token_email}, provided_email={provided_email}")
                    raise HTTPException(status_code=401, detail="Email mismatch")
            else:
                # Provided user_id is not an email, compare with token user_id
                if user_info["user_id"] != user_id:
                    logger.warning(f"User ID mismatch: token={user_info['user_id']}, provided={user_id}")
                    raise HTTPException(status_code=401, detail="User ID mismatch")
        
        logger.info(f"Cognito token validated successfully for user: {user_info['user_id']}")
        return user_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during token validation: {e}")
        raise HTTPException(status_code=500, detail="Authentication service error")

