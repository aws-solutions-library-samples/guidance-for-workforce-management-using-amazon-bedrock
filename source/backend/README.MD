# Retail Backend

This repository contains the backend services for the Retail Store Assistant application.

## Architecture

The application consists of two main backend services:

1. **REST API Server** - Provides RESTful endpoints for the retail store application
2. **WebSocket Server** - Handles real-time communication for speech-to-speech functionality

Both servers are containerized together in a single Docker container and can be deployed to Kubernetes.

## Environment Variables

The following environment variables can be configured:

| Variable | Description | Default |
|----------|-------------|---------|
| COGNITO_USER_POOL_ID | Cognito User Pool Id | us-east-1_XXXXXXXXX |
| COGNITO_APP_CLIENT_ID | Cognito App Client id | 1234567890abcdefghijklmnop |
| AWS_REGION | AWS region | us-east-1 |
| BEDROCK_MODEL_ID | Bedrock model ID | default_model_id |
| BD_GUARDRAIL_IDENTIFIER | Guardrail identifier | default_identifier |
| BD_GUARDRAIL_VERSION | Guardrail version | default_version |
| BD_KB_ID | Knowledge base ID | default_kb_id |
| STACK_NAME | Stack name | (required) |
| STACK_ENVIRONMENT | Stack environment | (required) |
| S3_BUCKET_NAME | website bucket name | (required) |

## Building and Running

### Local Development
0. Setup Python environment: 
   ```
   pyenv virtualenv 3.12.8 retail-store-assistant
   pyenv activate retail-store-assistant
   ```
1. Set environment variables:
   ```
   COGNITO_USER_POOL_ID=us-east-1_XXXXXXXXX
   COGNITO_APP_CLIENT_ID=1234567890abcdefghijklmnop
   AWS_REGION=us-east-1
   BEDROCK_MODEL_ID="us.anthropic.claude-3-5-haiku-20241022-v1:0"
   BD_GUARDRAIL_IDENTIFIER="XXX"
   BD_GUARDRAIL_VERSION="DRAFT"
   BD_KB_ID="XXX"
   STACK_NAME="retail"
   STACK_ENVIRONMENT="dev"
   S3_BUCKET_NAME="XXX"
   AWS_ACCESS_KEY_ID="XXX"
   AWS_SECRET_ACCESS_KEY="XXX"
   AWS_SESSION_TOKEN="XXX"
   ```


2. Install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```

3. Run both servers:
   ```
   python run_servers.py
   ```

### Docker

1. Build the Docker image:
   ```
   docker build --platform=linux/amd64 -t retail-backend:latest .
   docker build -t retail-backend:latest .
   ```

2. Run the container:
   ```
   docker run \
   --name retail-backend \
   -p 8000:8000 \
   --env-file ../.env \
   retail-backend:latest
   ```


## API Endpoints

### Endpoints

- `/api/users` - Get all users
- `/api/chat` - Chat with the retail assistant
- `/api/todos` - Get, create, update, and delete todos
- `/api/store` - Get store information
- `/api/uploadimage` - Upload an image
- `/api/feedback` - Add and get feedback
- `/api/health` - Health check endpoint


### WebSocket Endpoints

- `/ws/{client_id}` - WebSocket endpoint for speech-to-speech functionality

## Health Checks

The health check endpoint is available at:
- `http://<host>:8000/health`



# AWS Credentials Script for Kubernetes Pods

Currently the Bedrock Sonic Python SDK requires AWS environment variables for authentication. That is why we use the aws_credentials.py script as a workaround to automatically obtain AWS credentials using the pod's service account.

The script:
1. Reads the service account token from the pod
2. Uses the token to assume the IAM role associated with the service account
3. Sets the AWS credentials as environment variables or writes them to a file

It is invoked as part of run_servers.py as part of the run_aws_credentials_refresh() method.