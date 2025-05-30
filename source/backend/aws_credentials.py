#!/usr/bin/env python3
"""
AWS Credentials Script for Kubernetes Pods

This script reads the service account token from a Kubernetes pod,
uses it to assume the IAM role associated with the service account,
and sets the AWS credentials as environment variables.

Usage:
    python aws_credentials.py [--output-env-file FILE]

Options:
    --output-env-file FILE    Write credentials to an environment file instead of stdout
"""

import os
import sys
import json
import argparse
import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.credentials import Credentials
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("aws_credentials")

# Default paths for Kubernetes service account token and namespace
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Global credential cache
_credential_cache = {
    'credentials': None,
    'expiry': None,
    'refresh_threshold': 300  # Refresh if less than 5 minutes until expiry
}

class CachedCredentialProvider:
    """A credential provider that caches credentials and only refreshes them when they're close to expiring."""
    
    def __init__(self, refresh_threshold_seconds=300):
        """Initialize the credential provider.
        
        Args:
            refresh_threshold_seconds: The number of seconds before expiry to refresh credentials.
        """
        self.refresh_threshold = refresh_threshold_seconds
        self._credentials = None
        self._expiry = None
    
    def get_credentials(self):
        """Get credentials, refreshing them if necessary."""
        now = datetime.now()
        
        # If we have cached credentials and they're not close to expiring, return them
        if (self._credentials is not None and 
            self._expiry is not None and 
            self._expiry - now > timedelta(seconds=self.refresh_threshold)):
            print(f"Using cached credentials (expires in {(self._expiry - now).total_seconds() / 60:.2f} minutes)", file=sys.stderr)
            return self._credentials
        
        # Otherwise, get fresh credentials
        print("Refreshing cached credentials", file=sys.stderr)
        credentials = self._get_fresh_credentials()
        
        if credentials is not None:
            self._credentials = credentials
            if hasattr(credentials, 'expiry') and credentials.expiry:
                self._expiry = credentials.expiry
                print(f"Cached credentials will expire in {(self._expiry - now).total_seconds() / 60:.2f} minutes", file=sys.stderr)
        
        return credentials
    
    def _get_fresh_credentials(self):
        """Get fresh credentials from the AWS SDK."""
        try:
            # Clear any existing credentials from the environment to force IRSA
            for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']:
                if key in os.environ:
                    del os.environ[key]
            
            # Create a new session without any existing credentials
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if credentials is None:
                print("Failed to get fresh credentials from IRSA", file=sys.stderr)
                return None
                
            print("Successfully obtained fresh credentials from IRSA", file=sys.stderr)
            return credentials
        except Exception as e:
            print(f"Error getting fresh credentials: {e}", file=sys.stderr)
            return None

# Create a global instance of the credential provider
_credential_provider = CachedCredentialProvider()

def get_service_account_info():
    """Get the service account name and namespace from the pod."""
    try:
        
        #TBD
        return "retail-backend-sa", "retail-app"  # Default values for non-Kubernetes
        
    except Exception as e:
        print(f"Error getting service account info: {e}", file=sys.stderr)
        return "retail-backend-sa", "retail-app"  # Default values

def get_aws_credentials(service_account_name, namespace):
    """Get AWS credentials by assuming the role associated with the service account."""
    try:
        # Check if we're running in a Kubernetes environment
        is_kubernetes = os.path.exists(TOKEN_PATH) and os.path.exists(NAMESPACE_PATH)
        
        if is_kubernetes:
            print("Running in Kubernetes environment, using IRSA", file=sys.stderr)
            
            # Use the AWS SDK's built-in IRSA support
            # This will automatically use the service account token and role
            try:
                # Use the cached credential provider to get credentials
                credentials = _credential_provider.get_credentials()
                
                if credentials:
                    print("Successfully obtained credentials from IRSA", file=sys.stderr)
                    print(f"Access key: {credentials.access_key[:5]}...", file=sys.stderr)
                    print(f"Session token present: {'Yes' if credentials.token else 'No'}", file=sys.stderr)
                    
                    # Check if the credentials are about to expire
                    if hasattr(credentials, 'expiry') and credentials.expiry:
                        from datetime import datetime
                        now = datetime.now(credentials.expiry.tzinfo)
                        time_until_expiry = credentials.expiry - now
                        print(f"Credentials will expire in {time_until_expiry.total_seconds() / 60:.2f} minutes", file=sys.stderr)
                    
                    return {
                        'AWS_ACCESS_KEY_ID': credentials.access_key,
                        'AWS_SECRET_ACCESS_KEY': credentials.secret_key,
                        'AWS_SESSION_TOKEN': credentials.token or '',
                        'AWS_REGION': os.environ.get('AWS_REGION', AWS_REGION)
                    }
                else:
                    print("Failed to obtain credentials from IRSA", file=sys.stderr)
                    
                    # Try to get more information about why IRSA failed
                    try:
                        # Check if the token file exists and is readable
                        if os.path.exists(TOKEN_PATH):
                            with open(TOKEN_PATH, 'r') as f:
                                token = f.read().strip()
                                print(f"Token file exists and contains {len(token)} characters", file=sys.stderr)
                        else:
                            print(f"Token file does not exist at {TOKEN_PATH}", file=sys.stderr)
                            
                        # Check if the namespace file exists and is readable
                        if os.path.exists(NAMESPACE_PATH):
                            with open(NAMESPACE_PATH, 'r') as f:
                                namespace = f.read().strip()
                                print(f"Namespace file exists and contains: {namespace}", file=sys.stderr)
                        else:
                            print(f"Namespace file does not exist at {NAMESPACE_PATH}", file=sys.stderr)
                            
                        # Try to get the role ARN from the service account annotation
                        try:
                            import requests
                            with open(TOKEN_PATH, 'r') as f:
                                token = f.read().strip()
                            
                            namespace = open(NAMESPACE_PATH, 'r').read().strip()
                            api_url = f"https://kubernetes.default.svc/api/v1/namespaces/{namespace}/serviceaccounts/{service_account_name}"
                            headers = {"Authorization": f"Bearer {token}"}
                            
                            response = requests.get(api_url, headers=headers, verify=False)
                            if response.status_code == 200:
                                sa_data = response.json()
                                annotations = sa_data.get('metadata', {}).get('annotations', {})
                                role_arn = annotations.get('eks.amazonaws.com/role-arn')
                                if role_arn:
                                    print(f"Service account has role ARN: {role_arn}", file=sys.stderr)
                                else:
                                    print("Service account does not have a role ARN annotation", file=sys.stderr)
                            else:
                                print(f"Failed to get service account info: {response.status_code} - {response.text}", file=sys.stderr)
                        except Exception as e:
                            print(f"Error checking service account annotations: {e}", file=sys.stderr)
                    except Exception as e:
                        print(f"Error diagnosing IRSA failure: {e}", file=sys.stderr)
                    
                    return None
            except Exception as e:
                print(f"Error creating boto3 session: {e}", file=sys.stderr)
                import traceback
                print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
                return None
        else:
            # Not running in Kubernetes, use environment variables or instance profile
            print("Not running in Kubernetes environment, using existing credentials", file=sys.stderr)
            
            # Check if credentials are already set in environment variables
            if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
                return {
                    'AWS_ACCESS_KEY_ID': os.environ.get('AWS_ACCESS_KEY_ID'),
                    'AWS_SECRET_ACCESS_KEY': os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    'AWS_SESSION_TOKEN': os.environ.get('AWS_SESSION_TOKEN', ''),
                    'AWS_REGION': os.environ.get('AWS_REGION', AWS_REGION)
                }
            
            # Try to use instance profile if available
            try:
                session = boto3.Session()
                credentials = session.get_credentials()
                if credentials:
                    return {
                        'AWS_ACCESS_KEY_ID': credentials.access_key,
                        'AWS_SECRET_ACCESS_KEY': credentials.secret_key,
                        'AWS_SESSION_TOKEN': credentials.token or '',
                        'AWS_REGION': os.environ.get('AWS_REGION', AWS_REGION)
                    }
            except Exception as e:
                print(f"Error getting credentials from instance profile: {e}", file=sys.stderr)
            
            # If all else fails, return None
            return None
    except Exception as e:
        print(f"Error getting AWS credentials: {e}", file=sys.stderr)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        return None

def set_environment_variables(credentials):
    """Set AWS credentials as environment variables."""
    if not credentials:
        return False
    
    for key, value in credentials.items():
        os.environ[key] = value
    
    return True

def write_env_file(credentials, filename):
    """Write AWS credentials to an environment file."""
    if not credentials:
        return False
    
    with open(filename, 'w') as f:
        for key, value in credentials.items():
            f.write(f"{key}={value}\n")
    
    return True

def refresh_aws_credentials():
    """Refresh AWS credentials periodically"""
    logger.info("Refreshing AWS credentials")
    try:
        # Get service account info
        service_account_name, namespace = get_service_account_info()
        logger.info(f"Using service account: {service_account_name} in namespace: {namespace}")
        
        # Get AWS credentials
        credentials = get_aws_credentials(service_account_name, namespace)
        
        if not credentials:
            logger.error("Failed to get AWS credentials")
            return False
        
        # Log credential details (without the actual secret values)
        logger.info(f"Got credentials with access key: {credentials.get('AWS_ACCESS_KEY_ID', '')[:5]}...")
        logger.info(f"Session token present: {'Yes' if credentials.get('AWS_SESSION_TOKEN') else 'No'}")
        
        # Set environment variables
        if set_environment_variables(credentials):
            logger.info("AWS credentials refreshed successfully")
            
            # Verify the credentials were set correctly
            access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
            session_token = os.environ.get('AWS_SESSION_TOKEN')
            
            if not access_key or not secret_key:
                logger.error("Environment variables not set correctly after refresh")
                return False
                
            logger.info(f"Environment variables set correctly. Access key: {access_key[:5]}...")
            logger.info(f"Session token present in environment: {'Yes' if session_token else 'No'}")
            
            # Instead of directly importing and calling run_servers.restart_servers(),
            # we'll use a callback mechanism to avoid circular imports
            # The callback will be set by run_servers.py
            if hasattr(refresh_aws_credentials, 'on_credentials_refreshed') and refresh_aws_credentials.on_credentials_refreshed is not None:
                logger.info("Calling credentials refresh callback")
                try:
                    refresh_aws_credentials.on_credentials_refreshed()
                except Exception as e:
                    logger.error(f"Error calling credentials refresh callback: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.info("No credentials refresh callback registered or callback is None")
            
            return True
        else:
            logger.error("Failed to set AWS credentials")
            return False
    except Exception as e:
        logger.error(f"Error refreshing AWS credentials: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
def main():
    parser = argparse.ArgumentParser(description='Set AWS credentials from Kubernetes service account')
    parser.add_argument('--output-env-file', help='Write credentials to an environment file')
    args = parser.parse_args()
    
    # Get service account info
    service_account_name, namespace = get_service_account_info()
    print(f"Using service account: {service_account_name} in namespace: {namespace}", file=sys.stderr)
    
    # Get AWS credentials
    credentials = get_aws_credentials(service_account_name, namespace)
    
    if not credentials:
        print("Failed to get AWS credentials", file=sys.stderr)
        return 1
    
    # Set environment variables or write to file
    if args.output_env_file:
        if write_env_file(credentials, args.output_env_file):
            print(f"Credentials written to {args.output_env_file}", file=sys.stderr)
            return 0
        else:
            print(f"Failed to write credentials to {args.output_env_file}", file=sys.stderr)
            return 1
    else:
        if set_environment_variables(credentials):
            print("AWS credentials set as environment variables", file=sys.stderr)
            # Print the credentials in a format that can be sourced
            for key, value in credentials.items():
                print(f"export {key}={value}")
            return 0
        else:
            print("Failed to set AWS credentials", file=sys.stderr)
            return 1

if __name__ == "__main__":
    sys.exit(main()) 