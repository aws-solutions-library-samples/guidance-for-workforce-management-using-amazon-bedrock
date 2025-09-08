#!/usr/bin/env python3
"""
Function Calling Test Catalog
----------------------------
This script performs tests for function calling via REST API using AWS Cognito for authentication.
It reads queries from a validation dataset file and sends them to the API, then collects and saves
the responses for analysis in a JSONL file.
"""

import logging
import os
import requests
import json
import time
import uuid
import boto3
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from botocore.exceptions import ClientError

# Default path to validation dataset
DEFAULT_VALIDATION_DATASET_PATH = "data/text_validation_dataset.jsonl"

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def find_dotenv():
    """
    Find the .env file by looking in several possible locations.
    This makes the code more robust when running in different environments like Jupyter notebooks.
    
    Returns:
        Path: Path to the .env file
    """
    # Try different possible locations for the .env file
    possible_paths = [
        # Current directory
        Path('.env'),
        # Parent directory (for when running from a subdirectory)
        Path('..') / '.env',
        Path('..') / 'deployment' /'.env',
        Path('..') / 'source' / 'frontend' / '.env',
        # From the current working directory
        Path(os.getcwd()) / '.env',
        # From the parent of the current working directory
        Path(os.getcwd()).parent / '.env',
    ]
    
    # Try each path
    for path in possible_paths:
        # logger.debug(f"Checking for .env file at: {path.absolute()}")
        if path.exists():
            logger.info(f"Found .env file at: {path.absolute()}")
            return path
    
    # If no .env file is found, log a warning and return the default path
    logger.warning("No .env file found in any of the expected locations")
    return None

def find_frontend_env():
    """
    Find the frontend .env file by looking in several possible locations.
    
    Returns:
        Path: Path to the frontend .env file
    """
    # Try different possible locations for the .env file
    possible_paths = [
        # Frontend directory
        Path('source/frontend/.env'),
        # From the current working directory
        Path(os.getcwd()) / 'source' / 'frontend' / '.env',
        # From the parent of the current working directory
        Path(os.getcwd()).parent / 'source' / 'frontend' / '.env',
    ]
    
    # Try each path
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found frontend .env file at: {path.absolute()}")
            return path
    
    # If no .env file is found, log a warning and return None
    logger.warning("No frontend .env file found in any of the expected locations")
    return None

def update_cognito_client(profile_name=None):
    """
    Update the Cognito App Client to enable required auth flows.
    
    Args:
        profile_name (str, optional): AWS CLI profile name to use for credentials
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Load environment variables
    env_path = find_dotenv()
    if env_path:
        load_dotenv(dotenv_path=env_path)
    
    user_pool_id = os.getenv('COGNITO_USER_POOL_ID') or os.getenv('VITE_USER_POOL_ID')
    client_id = os.getenv('COGNITO_APP_CLIENT_ID') or os.getenv('VITE_USER_POOL_CLIENT_ID')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    try:
        # Create session with the specified profile
        if profile_name:
            logger.info(f"Using AWS credentials from profile: {profile_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region)
        else:
            logger.info("Using default AWS credentials")
            session = boto3.Session(region_name=region)
        
        cognito_client = session.client('cognito-idp')
        
        logger.info("=== Updating Cognito App Client ===")
        
        # Update the client with required auth flows - using USER_PASSWORD_AUTH like the frontend
        response = cognito_client.update_user_pool_client(
            UserPoolId=user_pool_id,
            ClientId=client_id,
            ExplicitAuthFlows=[
                'ALLOW_USER_PASSWORD_AUTH',  # Primary auth flow (same as frontend)
                'ALLOW_REFRESH_TOKEN_AUTH',  # For token refresh
                'ALLOW_USER_SRP_AUTH',       # For SRP authentication
            ]
        )
        
        logger.info("✅ Successfully updated Cognito App Client!")
        logger.info("Enabled auth flows: ALLOW_USER_PASSWORD_AUTH, ALLOW_REFRESH_TOKEN_AUTH, ALLOW_USER_SRP_AUTH")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Failed to update Cognito client ({error_code}): {error_message}")
        return False
    except Exception as e:
        logger.error(f"Failed to update Cognito settings: {e}")
        return False


class CognitoAuthenticator:
    """
    Handles AWS Cognito authentication and JWT token management.
    """
    
    def __init__(self, user_pool_id=None, client_id=None, region=None, aws_profile=None):
        """
        Initialize the Cognito authenticator.
        
        Args:
            user_pool_id (str, optional): Cognito User Pool ID
            client_id (str, optional): Cognito App Client ID  
            region (str, optional): AWS region
        """
        # Load environment variables
        env_path = find_dotenv()
        load_dotenv(dotenv_path=env_path)
        
        self.user_pool_id = user_pool_id or os.getenv('COGNITO_USER_POOL_ID')
        self.client_id = client_id or os.getenv('COGNITO_APP_CLIENT_ID')
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        
        if not self.user_pool_id or not self.client_id:
            raise ValueError("Cognito User Pool ID and Client ID are required. Set COGNITO_USER_POOL_ID and COGNITO_APP_CLIENT_ID environment variables.")
        
        # Initialize Cognito client
        # self.cognito_client = boto3.client('cognito-idp', region_name=self.region)
        
         # Create session with the specified profile
        if aws_profile:
            logger.info(f"Using AWS credentials from profile: {aws_profile}")
            session = boto3.Session(profile_name=aws_profile, region_name=self.region)
        else:
            logger.info("Using default AWS credentials")
            session = boto3.Session(region_name=self.region)
        
        self.cognito_client = session.client('cognito-idp')
        
        # Token storage
        self.access_token = None
        self.id_token = None
        self.refresh_token = None
        self.token_expiry = None
        
        logger.info(f"Initialized Cognito authenticator for region: {self.region}")
        logger.info(f"User Pool ID: {self.user_pool_id}")
        logger.info(f"Client ID: {self.client_id}")
    
    def authenticate_with_password(self, username, password):
        """
        Authenticate with username and password to get JWT tokens.
        Uses the same authentication flow as the frontend (USER_PASSWORD_AUTH).
        
        Args:
            username (str): Username (email)
            password (str): Password
            
        Returns:
            dict: Authentication result with tokens
        """
        try:
            # First try USER_PASSWORD_AUTH (same as frontend)
            response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password
                }
            )
            
            # Extract tokens
            auth_result = response['AuthenticationResult']
            self.access_token = auth_result['AccessToken']
            self.id_token = auth_result['IdToken']
            self.refresh_token = auth_result.get('RefreshToken')
            
            # Calculate token expiry (tokens typically expire in 1 hour)
            expires_in = auth_result.get('ExpiresIn', 3600)  # Default to 1 hour
            self.token_expiry = time.time() + expires_in
            
            logger.info(f"Successfully authenticated user: {username}")
            logger.info(f"Tokens will expire in {expires_in} seconds")
            
            return {
                'access_token': self.access_token,
                'id_token': self.id_token,
                'refresh_token': self.refresh_token,
                'expires_in': expires_in
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            if error_code == 'NotAuthorizedException':
                raise ValueError(f"Authentication failed: {error_message}")
            elif error_code == 'UserNotFoundException':
                raise ValueError(f"User not found: {error_message}")
            elif error_code == 'UserNotConfirmedException':
                raise ValueError(f"User not confirmed: {error_message}")
            elif error_code == 'InvalidParameterException' and 'Auth flow not enabled' in error_message:
                raise ValueError(f"Auth flow not enabled: {error_message}. Please ensure ALLOW_USER_PASSWORD_AUTH is enabled for your Cognito App Client.")
            else:
                raise ValueError(f"Authentication error ({error_code}): {error_message}")
    
    def refresh_tokens(self):
        """
        Refresh the access and ID tokens using the refresh token.
        
        Returns:
            dict: New authentication result with refreshed tokens
        """
        if not self.refresh_token:
            raise ValueError("No refresh token available. Please authenticate again.")
        
        try:
            response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='REFRESH_TOKEN_AUTH',
                AuthParameters={
                    'REFRESH_TOKEN': self.refresh_token
                }
            )
            
            # Extract new tokens
            auth_result = response['AuthenticationResult']
            self.access_token = auth_result['AccessToken']
            self.id_token = auth_result['IdToken']
            # Note: Refresh token might not be returned in refresh response
            
            # Calculate new token expiry
            expires_in = auth_result.get('ExpiresIn', 3600)
            self.token_expiry = time.time() + expires_in
            
            logger.info("Successfully refreshed tokens")
            
            return {
                'access_token': self.access_token,
                'id_token': self.id_token,
                'expires_in': expires_in
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise ValueError(f"Token refresh failed ({error_code}): {error_message}")
    
    def get_valid_token(self, token_type='id'):
        """
        Get a valid token, refreshing if necessary.
        
        Args:
            token_type (str): Type of token to return ('id' or 'access')
            
        Returns:
            str: Valid JWT token
        """
        # Check if tokens need to be refreshed (refresh 5 minutes before expiry)
        if self.token_expiry and time.time() > (self.token_expiry - 300):
            logger.info("Tokens are expiring soon, refreshing...")
            self.refresh_tokens()
        
        if token_type == 'id':
            if not self.id_token:
                raise ValueError("No ID token available. Please authenticate first.")
            return self.id_token
        elif token_type == 'access':
            if not self.access_token:
                raise ValueError("No access token available. Please authenticate first.")
            return self.access_token
        else:
            raise ValueError("token_type must be 'id' or 'access'")


class RetailRestApiClient:
    """
    Client for interacting with the Retail Store Assistant REST API with Cognito authentication.
    This client handles JWT token authentication automatically.
    """
    
    def __init__(self, username=None, password=None, user_id=None, api_url=None, debug=False, 
                 cognito_user_pool_id=None, cognito_client_id=None, region=None, aws_profile=None):
        """
        Initialize the REST API client with Cognito authentication.
        
        Args:
            username (str, optional): Cognito username (email) for authentication
            password (str, optional): Cognito password for authentication
            user_id (str, optional): User ID for the session. If None, will use username
            api_url (str, optional): REST API URL. If None, it will be read from the .env file
            debug (bool, optional): Enable debug logging. Default is False
            cognito_user_pool_id (str, optional): Cognito User Pool ID
            cognito_client_id (str, optional): Cognito App Client ID
            region (str, optional): AWS region
        """
        # Load environment variables from .env file
        env_path = find_dotenv()
        load_dotenv(dotenv_path=env_path)
        
        # Store username for authentication
        self.username = username or os.getenv('EMAIL')
        self.password = password or os.getenv('COGNITO_PASSWORD')
        
        # User ID for API calls (use username if not provided)
        self.user_id = user_id or self.username
        
        # Log user ID but not password
        logger.info(f'Using user ID: {self.user_id}')
        
        self.cognito_user_pool_id = cognito_user_pool_id or os.getenv('COGNITO_USER_POOL_ID')
        self.cognito_client_id = cognito_client_id or os.getenv('COGNITO_APP_CLIENT_ID')
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        
        # Load frontend .env file to get VITE_WEBSOCKET_URL
        frontend_env_path = find_frontend_env()
        rest_url = None
        
        if frontend_env_path:
            # Load the frontend .env file
            with open(frontend_env_path, 'r') as f:
                for line in f:
                    if line.startswith('VITE_RESTAPI_URL='):
                        rest_url = line.strip().split('=', 1)[1]
                        # Remove quotes if present
                        if rest_url.startswith('"') and rest_url.endswith('"'):
                            rest_url = rest_url[1:-1]
                        elif rest_url.startswith("'") and rest_url.endswith("'"):
                            rest_url = rest_url[1:-1]
                        break
            
            if rest_url:
                logger.info(f"Using VITE_RESTAPI_URL from frontend .env: {rest_url}")
            else:
                logger.warning("VITE_RESTAPI_URL not found in frontend .env file")

        # RESTAPI_URL = f'https://backend.{os.getenv("DOMAIN_NAME")}/api'
        self.api_url = rest_url
        self.debug = debug
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
            
        if not self.api_url:
            raise ValueError("API URL not provided and not found in .env file")
        
        # Initialize Cognito authenticator
        self.authenticator = CognitoAuthenticator(
            user_pool_id=self.cognito_user_pool_id,
            client_id=self.cognito_client_id,
            region=self.region,
            aws_profile=aws_profile
        )
        
        # Authenticate if credentials are provided
        if self.username and self.password:
            self.authenticate()
        
        logger.info(f"Initialized REST API client with URL: {self.api_url}")
    
    def authenticate(self):
        """
        Authenticate with Cognito to get JWT tokens.
        """
        if not self.username or not self.password:
            raise ValueError("Username and password are required for authentication")
        
        logger.info(f"Authenticating with Cognito for user: {self.username}")
        
        try:
            result = self.authenticator.authenticate_with_password(self.username, self.password)
            logger.info("Successfully authenticated with Cognito")
            return result
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    def send_query(self, query, session_id=None):
        """
        Send a text query to the REST API with JWT authentication.
        
        Args:
            query (str): The text query to send
            session_id (str, optional): Session ID for continuing a conversation. 
                                       If None, a new conversation will be started.
                                       
        Returns:
            dict: The API response as a dictionary
        """
        if not query:
            raise ValueError("Query cannot be empty")
            
        # Ensure the API URL doesn't end with a slash
        api_url = self.api_url.rstrip('/')
        
        # Construct the endpoint URL
        endpoint = f"{api_url}/chat"
        
        # Get valid JWT token
        try:
            jwt_token = self.authenticator.get_valid_token('id')
        except ValueError as e:
            logger.error(f"Failed to get valid token: {e}")
            raise ValueError("Authentication required. Please call authenticate() first.")
        
        # Prepare headers with JWT token
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {jwt_token}'
        }
            
        # Prepare the request payload
        payload = {
            'userId': self.user_id,
            'query': query,
            'sessionId': session_id or '1234'
        }
        
        logger.debug(f"Sending query to {endpoint}: {query}")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Record start time
            start_time = time.time()
    
            # Send the GET request with increased timeout to handle backend processing delays
            response = requests.get(endpoint, headers=headers, params=payload, timeout=300)
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the JSON response
            response_data = response.json()

            logger.debug(f"Received response with status code: {response.status_code}")
            logger.debug(f"Response data: {json.dumps(response_data, indent=2)}")

            # Record end time and calculate duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Log the duration
            logger.info(f"Request duration: {duration_ms:.2f} ms")
            
            # Add duration to the response data
            response_data['request_duration_ms'] = round(duration_ms, 2)
            
            return response_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
                
                # Handle authentication errors with improved retry logic
                if e.response.status_code == 401:
                    logger.info("Authentication failed, attempting to refresh tokens...")
                    try:
                        self.authenticator.refresh_tokens()
                        logger.info("Tokens refreshed, retrying request...")
                        # Retry the request with new token
                        jwt_token = self.authenticator.get_valid_token('id')
                        headers['Authorization'] = f'Bearer {jwt_token}'
                        
                        # Record retry start time
                        retry_start_time = time.time()
                        response = requests.get(endpoint, headers=headers, params=payload, timeout=300)
                        response.raise_for_status()
                        
                        # Parse retry response
                        retry_response_data = response.json()
                        
                        # Calculate total duration including retry
                        total_duration_ms = (time.time() - start_time) * 1000
                        retry_response_data['request_duration_ms'] = round(total_duration_ms, 2)
                        retry_response_data['retry_attempted'] = True
                        
                        return retry_response_data
                    except Exception as refresh_error:
                        logger.error(f"Token refresh failed: {refresh_error}")
                        raise ValueError("Authentication failed and token refresh unsuccessful. Please authenticate again.")
                
                # Handle rate limiting errors
                elif e.response.status_code == 429:
                    logger.warning("Rate limited by API, waiting before retry...")
                    time.sleep(5)  # Wait 5 seconds before retry
                    raise ValueError("API rate limit exceeded. Please try again later.")
                
                # Handle server errors
                elif e.response.status_code >= 500:
                    logger.error(f"Server error {e.response.status_code}: {e.response.text}")
                    raise ValueError(f"Server error: {e.response.status_code}")
            raise
            
    def reset_chat(self, session_id=None):
        """
        Reset chat history for the session.
        
        Args:
            session_id (str, optional): Session ID for the conversation to reset.
                                       If None, default session ID '1234' will be used.
                                       
        Returns:
            dict: The API response as a dictionary
        """
        # Ensure the API URL doesn't end with a slash
        api_url = self.api_url.rstrip('/')
        
        # Construct the endpoint URL
        endpoint = f"{api_url}/reset_chat"
        
        # Get valid JWT token
        try:
            jwt_token = self.authenticator.get_valid_token('id')
        except ValueError as e:
            logger.error(f"Failed to get valid token: {e}")
            raise ValueError("Authentication required. Please call authenticate() first.")
        
        # Prepare headers with JWT token
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {jwt_token}'
        }
            
        # Prepare the request payload
        payload = {
            'userId': self.user_id,
            'sessionId': session_id or '1234'
        }
        
        logger.debug(f"Resetting chat at {endpoint} for session: {session_id or '1234'}")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Record start time
            start_time = time.time()
    
            # Send the GET request with a timeout of 60 seconds for reset_chat operations
            response = requests.get(endpoint, headers=headers, params=payload, timeout=60)
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the JSON response if available
            try:
                response_data = response.json()
            except:
                # If the response is not JSON, create a simple success response
                response_data = {
                    'status': 'success',
                    'message': 'Chat history reset successfully'
                }
            
            logger.debug(f"Received response with status code: {response.status_code}")
            if isinstance(response_data, dict):
                logger.debug(f"Response data: {json.dumps(response_data, indent=2)}")
            
            # Record end time and calculate duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Log the duration
            logger.info(f"Reset chat request duration: {duration_ms:.2f} ms")
            
            return response_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Reset chat request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
                
                # Handle authentication errors
                if e.response.status_code == 401:
                    logger.info("Authentication failed, attempting to refresh tokens...")
                    try:
                        self.authenticator.refresh_tokens()
                        logger.info("Tokens refreshed, retrying request...")
                        # Retry the request with new token
                        jwt_token = self.authenticator.get_valid_token('id')
                        headers['Authorization'] = f'Bearer {jwt_token}'
                        response = requests.get(endpoint, headers=headers, params=payload, timeout=60)
                        response.raise_for_status()
                        try:
                            return response.json()
                        except:
                            return {
                                'status': 'success',
                                'message': 'Chat history reset successfully'
                            }
                    except Exception as refresh_error:
                        logger.error(f"Token refresh failed: {refresh_error}")
                        raise ValueError("Authentication failed and token refresh unsuccessful. Please authenticate again.")
            raise


def run_single_test_loop(dataset_path=DEFAULT_VALIDATION_DATASET_PATH, run_number=1, model='nova_pro', aws_profile=None):
    """
    Run a single test loop for function calling with all validation dataset queries
    
    Args:
        dataset_path (str): Path to the validation dataset JSONL file
        run_number (int): The current run number (for logging and directory naming)
        model (str): Model name to use for the test
        aws_profile (str, optional): AWS profile name to use for credentials
        
    Returns:
        dict: Results and statistics from this test run
    """
    # Update Cognito client and create API client
    update_cognito_client(profile_name=aws_profile)
    client = RetailRestApiClient(debug=True, aws_profile=aws_profile)
    
    # Define session ID
    session_id = f'test-run-{run_number}-{uuid.uuid4()}'
    
    # Load validation dataset
    logger.info(f"Loading validation dataset from {dataset_path}")
    validation_data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                validation_data.append(json.loads(line))
    
    logger.info(f"Loaded {len(validation_data)} queries from validation dataset")
    
    # List to store results
    results = []
    successful_queries = 0
    failed_queries = 0
    
    
    for i, test_case in enumerate(validation_data):
        query = test_case['query']
        print(f"Running test case {i+1}: {query}")
        expected_function = test_case.get('expected_function', '')
        expected_response = test_case.get('expected_response', '')
        
        logger.info(f"Processing test case {i+1}/{len(validation_data)}: {query}")
        
        try:
            # Process each query in the validation dataset
            
            # Send query to API
            response = client.send_query(query, session_id)
            print(f"Response: {response}")
            
            # Enhanced retry mechanism with exponential backoff for backend processing delays
            max_retries = 3
            retries = 0
            wait_time = 10  # Start with 10 seconds
            
            while retries < max_retries and not response.get('chat_response'):
                logger.info(f"Response doesn't contain chat_response yet. Waiting {wait_time} seconds and retrying...")
                time.sleep(wait_time)
                
                try:
                    # Try fetching the response again with the same query
                    response = client.send_query(query, session_id)
                    print(f"Retry {retries+1} response: {response}")
                except Exception as retry_error:
                    logger.warning(f"Retry {retries+1} failed: {retry_error}")
                    # Continue with the retry loop even if this attempt fails
                    
                retries += 1
                # Exponential backoff with jitter
                wait_time = min(wait_time * 1.5 + (retries * 2), 60)  # Max 60 seconds

            
            # Get category from test case
            category = test_case.get('category', 'Unknown')
            
            # Store result
            result = {
                'query': query,
                'response': response.get('chat_response', 'N/A'),
                'model': model,
                'request_duration_ms': response.get('request_duration_ms', 'N/A'),
                'expected_function': expected_function,
                'expected_response': expected_response,
                'category': category,
                'status': 'success',
                'session_id': session_id
            }
            successful_queries += 1
            
        except Exception as e:
            logger.error(f"Error processing test case {i+1}: {e}")
            # Get category from test case
            category = test_case.get('category', 'Unknown')
            
            # Create result with error information
            result = {
                'query': query,
                'model': model,
                'expected_function': expected_function,
                'expected_response': expected_response,
                'category': category,
                'status': 'error',
                'error_message': str(e),
                'session_id': session_id
            }
            failed_queries += 1
        
        results.append(result)
        
        # Reset chat history before next test case
        try:
            client.reset_chat(session_id)
        except Exception as reset_error:
            logger.error(f"Error resetting chat: {reset_error}")
            
        logger.info(f"Completed test case {i+1}")
    
    
    
    # Group results by category for statistics
    category_stats = {}
    for result in results:
        category = result.get('category', 'Unknown')
        if category not in category_stats:
            category_stats[category] = {
                'total': 0,
                'successful': 0,
                'failed': 0
            }
        
        category_stats[category]['total'] += 1
        if result.get('status') == 'success':
            category_stats[category]['successful'] += 1
        else:
            category_stats[category]['failed'] += 1
    
    # Calculate success rates for each category
    for category in category_stats:
        if category_stats[category]['total'] > 0:
            category_stats[category]['success_rate'] = category_stats[category]['successful'] / category_stats[category]['total']
        else:
            category_stats[category]['success_rate'] = 0
    
    # Create run statistics
    run_stats = {
        'run_number': run_number,
        'model': model,
        'total_queries': len(validation_data),
        'successful_queries': successful_queries,
        'failed_queries': failed_queries,
        'success_rate': successful_queries / len(validation_data) if len(validation_data) > 0 else 0,
        'average_request_duration_ms': sum(result['request_duration_ms'] for result in results if isinstance(result.get('request_duration_ms'), (int, float))) / len(results) if results else 0,
        'category_stats': category_stats
    }
    
    return {
        'results': results,
        'stats': run_stats
    }

def run_test_cases(dataset_path=DEFAULT_VALIDATION_DATASET_PATH, num_runs=1, delay_between_runs=5.0, model='nova_pro', category_filter=None, aws_profile=None):
    """
    Run the test cases with Nova Pro as reasoning model for function calling
    
    Args:
        dataset_path (str): Path to the validation dataset JSONL file
        num_runs (int): Number of times to run the test loop
        delay_between_runs (float): Delay in seconds between test loop runs
        model (str): Model name to use for the test
        category_filter (str, optional): Filter test cases by category
        aws_profile (str, optional): AWS profile name to use for credentials
        
    Returns:
        dict: Results from all test runs and overall statistics
    """
    logger.info(f"Starting function calling test with model {model}")
    logger.info(f"Will run {num_runs} test loops with {delay_between_runs}s delay between runs")
    
    # Create base directory for results
    base_output_dir = f"./responses/model_{model}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Run the specified number of test loops
    all_results = []
    all_stats = []
    
    for run_number in range(1, num_runs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting test run {run_number}/{num_runs}")
        logger.info(f"{'='*60}")
        
        # If category filter is specified, create filtered dataset
        if category_filter:
            logger.info(f"Filtering dataset by category: {category_filter}")
            filtered_dataset_path = f"{dataset_path}.filtered"
            filtered_count = 0
            
            # Load and filter validation dataset
            validation_data = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        test_case = json.loads(line)
                        if test_case.get('category') == category_filter:
                            validation_data.append(test_case)
                            filtered_count += 1
            
            # Write filtered dataset to temporary file
            with open(filtered_dataset_path, 'w') as f:
                for test_case in validation_data:
                    f.write(json.dumps(test_case) + '\n')
                    
            logger.info(f"Created filtered dataset with {filtered_count} test cases matching category '{category_filter}'")
            
            # Use filtered dataset for this run
            run_dataset_path = filtered_dataset_path
        else:
            run_dataset_path = dataset_path
        
        # Run a single test loop
        run_result = run_single_test_loop(
            dataset_path=run_dataset_path,
            run_number=run_number,
            model=model,
            aws_profile=aws_profile
        )
        
        # Remove temporary filtered dataset if it was created
        if category_filter and os.path.exists(f"{dataset_path}.filtered"):
            os.remove(f"{dataset_path}.filtered")
        
        # Extract results and stats
        results = run_result['results']
        run_stats = run_result['stats']
        
        # Add to overall collections
        all_results.append(results)
        all_stats.append(run_stats)
        
        # Create subdirectory for this run
        run_dir = os.path.join(base_output_dir, f"run_{run_number}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save run results to JSONL file
        run_result_file = os.path.join(run_dir, f"{model}_function_calling_responses.jsonl")
        with open(run_result_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
                
        # Save run statistics to JSON file
        run_stats_file = os.path.join(run_dir, f"{model}_function_calling_stats.json")
        with open(run_stats_file, 'w') as f:
            json.dump(run_stats, f, indent=2, default=str)
            
        logger.info(f"Run {run_number} results saved to {run_dir}")
        logger.info(f"Success rate: {run_stats['success_rate']*100:.2f}% ({run_stats['successful_queries']}/{run_stats['total_queries']})")
        
        # Log category-based statistics for this run
        logger.info(f"Category-based statistics for run {run_number}:")
        for category, stats in run_stats['category_stats'].items():
            logger.info(f"  {category}: {stats['success_rate']*100:.2f}% ({stats['successful']}/{stats['total']})")
        
        # Add delay between runs (except after the last one)
        if run_number < num_runs and delay_between_runs > 0:
            logger.info(f"Waiting {delay_between_runs} seconds before next test run...")
            time.sleep(delay_between_runs)
    
    # Generate and save overall summary
    
    # Collect category-based statistics
    categories = {}
    for run_results in all_results:
        for result in run_results:
            category = result.get('category', 'Unknown')
            if category not in categories:
                categories[category] = {
                    'total': 0,
                    'successful': 0,
                    'failed': 0
                }
            
            categories[category]['total'] += 1
            if result.get('status') == 'success':
                categories[category]['successful'] += 1
            else:
                categories[category]['failed'] += 1
    
    # Calculate success rates for each category
    for category in categories:
        if categories[category]['total'] > 0:
            categories[category]['success_rate'] = categories[category]['successful'] / categories[category]['total']
        else:
            categories[category]['success_rate'] = 0
    
    summary_stats = {
        'model': model,
        'num_runs': num_runs,
        'total_queries_per_run': len(all_results[0]) if all_results else 0,
        'total_queries_overall': sum(stat['total_queries'] for stat in all_stats),
        'successful_queries_overall': sum(stat['successful_queries'] for stat in all_stats),
        'failed_queries_overall': sum(stat['failed_queries'] for stat in all_stats),
        'average_success_rate': sum(stat['success_rate'] for stat in all_stats) / len(all_stats) if all_stats else 0,
        'category_stats': categories,
        'individual_run_stats': all_stats
    }
    
    # Save overall summary to JSON file
    summary_file = os.path.join(base_output_dir, f"{model}_function_calling_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    # Also save a merged JSONL file with all results
    merged_result_file = os.path.join(base_output_dir, f"{model}_function_calling_responses.jsonl")
    with open(merged_result_file, 'w') as f:
        for run_results in all_results:
            for result in run_results:
                f.write(json.dumps(result) + '\n')
    
    # Display overall summary
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL SUMMARY FOR {num_runs} TEST RUNS")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {model}")
    logger.info(f"Total queries across all runs: {summary_stats['total_queries_overall']}")
    logger.info(f"Successful queries: {summary_stats['successful_queries_overall']}")
    logger.info(f"Failed queries: {summary_stats['failed_queries_overall']}")
    logger.info(f"Average success rate: {summary_stats['average_success_rate']*100:.2f}%")
    
    # Display category-based summary
    logger.info(f"\n{'='*60}")
    logger.info(f"CATEGORY-BASED SUMMARY")
    logger.info(f"{'='*60}")
    for category, stats in categories.items():
        logger.info(f"Category: {category}")
        logger.info(f"  Total queries: {stats['total']}")
        logger.info(f"  Successful queries: {stats['successful']}")
        logger.info(f"  Failed queries: {stats['failed']}")
        logger.info(f"  Success rate: {stats['success_rate']*100:.2f}%")
    
    logger.info(f"\nSummary saved to {summary_file}")
    logger.info(f"All results merged to {merged_result_file}")
    
    return {
        'all_results': all_results,
        'all_stats': all_stats,
        'summary_stats': summary_stats
    }


def list_categories(dataset_path=DEFAULT_VALIDATION_DATASET_PATH):
    """
    List all unique categories in the validation dataset.
    
    Args:
        dataset_path (str): Path to the validation dataset JSONL file
        
    Returns:
        list: List of unique category names
    """
    categories = set()
    
    try:
        # Load validation dataset
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    test_case = json.loads(line)
                    category = test_case.get('category', 'Unknown')
                    categories.add(category)
        
        return sorted(list(categories))
    except Exception as e:
        logger.error(f"Error reading categories from dataset: {e}")
        return []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run function calling tests with Nova Pro")
    parser.add_argument("--dataset", type=str, default=DEFAULT_VALIDATION_DATASET_PATH,
                      help=f"Path to validation dataset JSONL file (default: {DEFAULT_VALIDATION_DATASET_PATH})")
    parser.add_argument("--num-runs", type=int, default=1,
                      help="Number of test runs to execute (default: 1)")
    parser.add_argument("--delay", type=float, default=60,
                      help="Delay between test runs in seconds (default: 60)")
    parser.add_argument("--model", type=str, default="nova_pro",
                      help="Model name to use for the test (default: nova_pro)")
    parser.add_argument("--category", type=str, default=None,
                      help="Filter test cases by category (e.g., 'Operations', 'Personalization', 'HR')")
    parser.add_argument("--list-categories", action="store_true",
                      help="List all categories in the validation dataset and exit")
    parser.add_argument("--aws-profile", type=str, default=None,
                      help="AWS profile name to use for credentials (default: use default credentials)")
    
    args = parser.parse_args()
    
    # If --list-categories is specified, show categories and exit
    if args.list_categories:
        categories = list_categories(args.dataset)
        print(f"Categories in dataset '{args.dataset}':")
        for category in categories:
            print(f"  - {category}")
        exit(0)
    
    run_test_cases(
        dataset_path=args.dataset,
        num_runs=args.num_runs,
        delay_between_runs=args.delay,
        model=args.model,
        category_filter=args.category,
        aws_profile=args.aws_profile
    )
