# Retail Backend

This repository contains the backend services for the Retail Store Assistant application.

## Architecture

The application is a unified FastAPI server that provides:

1. **REST API Endpoints** - RESTful endpoints for the retail store application with JWT authentication
2. **WebSocket Server** - Real-time bidirectional streaming for speech-to-speech functionality using AWS Bedrock Nova Sonic
3. **Tool Integration** - Comprehensive tool system for retail operations (inventory, scheduling, tasks, recommendations)
4. **AWS Integration** - Bedrock AI models, DynamoDB storage, S3 image handling, and Cognito authentication

The server runs as a single unified service that can be containerized and deployed to Kubernetes.

## Environment Variables

The following environment variables can be configured:

### Authentication & AWS Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| COGNITO_USER_POOL_ID | AWS Cognito User Pool ID for JWT validation | (required) |
| COGNITO_IDENTITY_POOL_ID | AWS Cognito Identity Pool ID for temporary credentials | (required) |
| COGNITO_APP_CLIENT_ID | AWS Cognito App Client ID | (required) |
| AWS_REGION | AWS region for all services | us-east-1 |

### Bedrock Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| BEDROCK_MODEL_ID | Primary Bedrock model ID | us.anthropic.claude-3-5-haiku-20241022-v1:0 |
| BD_GUARDRAIL_IDENTIFIER | Bedrock guardrail identifier | default_identifier |
| BD_GUARDRAIL_VERSION | Bedrock guardrail version | DRAFT |
| BD_KB_ID | Bedrock Knowledge Base ID | default_kb_id |
| BEDROCK_MAX_RETRIES | Maximum retry attempts for Bedrock calls | 3 |
| BEDROCK_BASE_DELAY | Base delay for exponential backoff (seconds) | 1.0 |
| BEDROCK_MAX_DELAY | Maximum delay for exponential backoff (seconds) | 30.0 |
| BEDROCK_MIN_REQUEST_INTERVAL | Minimum interval between requests (seconds) | 0.1 |

### Application Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| STACK_NAME | Stack name prefix for DynamoDB tables | (required) |
| STACK_ENVIRONMENT | Stack environment suffix for DynamoDB tables | (required) |
| S3_BUCKET_NAME | S3 bucket name for image storage | (required) |
| PORT | Server port | 8000 |
| HOST | Server host | 0.0.0.0 |
| DOMAIN_NAME | Domain name for CORS configuration | * |

### System Prompts
| Variable | Description | Default |
|----------|-------------|---------|
| DEFAUL_SYSTEM_PROMPT | Default system prompt template | (comprehensive retail assistant prompt) |

### Observability (Optional)
| Variable | Description | Default |
|----------|-------------|---------|
| AGENT_OBSERVABILITY_ENABLED | Enable OpenTelemetry tracing | false |

## Building and Running

### Local Development

1. Setup Python environment: 
   ```bash
   pyenv virtualenv 3.12.8 retail-store-assistant
   pyenv activate retail-store-assistant
   ```

2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Set environment variables (create a `.env` file):
   ```bash
   # Authentication
   COGNITO_USER_POOL_ID=us-east-1_XXXXXXXXX
   COGNITO_IDENTITY_POOL_ID=us-east-1:XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
   COGNITO_APP_CLIENT_ID=1234567890abcdefghijklmnop
   AWS_REGION=us-east-1
   
   # Bedrock Configuration
   BEDROCK_MODEL_ID="us.anthropic.claude-3-5-haiku-20241022-v1:0"
   BD_GUARDRAIL_IDENTIFIER="XXX"
   BD_GUARDRAIL_VERSION="DRAFT"
   BD_KB_ID="XXX"
   
   # Application
   STACK_NAME="retail"
   STACK_ENVIRONMENT="dev"
   S3_BUCKET_NAME="XXX"
   DOMAIN_NAME="localhost"
   
   # AWS Credentials (if not using IAM roles)
   AWS_ACCESS_KEY_ID="XXX"
   AWS_SECRET_ACCESS_KEY="XXX"
   AWS_SESSION_TOKEN="XXX"
   ```

4. Run the unified server:
   ```bash
   python run_servers.py
   ```

The server will start on `http://localhost:8000` with both REST API and WebSocket endpoints available.

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

All REST API endpoints require JWT authentication via `Authorization: Bearer <token>` header.

### Authentication
- **JWT Token Validation**: Uses AWS Cognito for token verification
- **User Credentials**: Automatically obtains temporary AWS credentials via Cognito Identity Pool
- **CORS**: Configured for cross-origin requests

### REST API Endpoints

#### Core Functionality
- `GET /api/chat?query=<text>&userId=<id>&sessionId=<id>` - Chat with the retail assistant
- `GET /api/reset_chat?userId=<id>&sessionId=<id>` - Reset conversation history and throttling state
- `POST /api/uploadimage` - Upload images with automatic S3 storage and thumbnail generation
- `GET /api/images/{image_id}` - Retrieve uploaded images

#### User & Task Management
- `GET /api/users` - Get all users
- `GET /api/todos?userId=<id>` - Get tasks for a user
- `POST /api/todos` - Create a new task
- `PATCH /api/todos/{userId}/{taskId}` - Update a task
- `DELETE /api/todos/{userId}/{taskId}` - Delete a task

#### Store Operations
- `GET /api/store` - Get comprehensive store data (KPIs, departments, inventory alerts)
- `POST /api/feedback` - Submit feedback
- `GET /api/feedbacks` - Get all feedback

#### System Monitoring
- `GET /api/health` - Health check endpoint
- `GET /api/metrics/throttling` - Get Bedrock API throttling metrics

### WebSocket Endpoints

#### Speech-to-Speech Communication
- `WS /ws/{session_id}?token=<jwt>&userId=<id>` - Bidirectional streaming for real-time voice interaction

**WebSocket Features:**
- **Authentication**: JWT token validation on connection
- **Session Management**: Automatic session lifecycle management
- **Audio Streaming**: Real-time audio input/output processing
- **Tool Integration**: Automatic tool calling during conversations
- **Error Handling**: Graceful error recovery and session cleanup

## Available Tools

The application includes a comprehensive tool system for retail operations:

### Knowledge & Information
- `search_knowledge_database` - FAQ search across standard operating procedures
- `get_products` - Product catalog listing
- `get_product_details` - Specific product information
- `get_customer_details` - Customer information lookup

### Staff Management
- `get_schedule` - User schedules (role-based access)
- `get_timeoff` - Time-off records
- `add_timeoff` - Time-off requests
- `list_tasks` - View assigned/created tasks
- `create_task` - Assign new tasks

### Analytics & Recommendations
- `generate_store_recommendations` - Daily store recommendations for managers
- `create_daily_task_suggestions_for_staff` - Staff task planning
- `customer_recommendation` - Personalized product suggestions
- `get_image_description` - AI-powered image analysis

## Authentication Flow

1. **JWT Token Validation**: Validates AWS Cognito tokens using JWKS
2. **User Credential Generation**: Uses Cognito Identity Pool to generate temporary AWS credentials
3. **Service Access**: Credentials are used for Bedrock, DynamoDB, and S3 operations
4. **Session Management**: User context is maintained throughout WebSocket sessions

## Data Storage

### DynamoDB Tables
- `{STACK_NAME}-SESSION-HISTORY-{STACK_ENVIRONMENT}` - Conversation history
- `{STACK_NAME}-FEEDBACK-{STACK_ENVIRONMENT}` - User feedback
- `{STACK_NAME}-CUSTOMER_TRANSACTIONS-{STACK_ENVIRONMENT}` - Transaction records
- `{STACK_NAME}-CUSTOMER-{STACK_ENVIRONMENT}` - Customer information
- `{STACK_NAME}-TASKLIST-{STACK_ENVIRONMENT}` - Task management
- `{STACK_NAME}-USERROLE-{STACK_ENVIRONMENT}` - User roles and permissions
- `{STACK_NAME}-PRODUCTS-{STACK_ENVIRONMENT}` - Product catalog
- `{STACK_NAME}-TIMEOFF-{STACK_ENVIRONMENT}` - Time-off requests
- `{STACK_NAME}-SCHEDULE-{STACK_ENVIRONMENT}` - Staff schedules
- `{STACK_NAME}-DAILY_TASKS_BY_DAY-{STACK_ENVIRONMENT}` - Daily task templates
- `{STACK_NAME}-IMAGES-{STACK_ENVIRONMENT}` - Image metadata and references

### S3 Storage
- Images are stored with automatic thumbnail generation
- Organized by user and session: `images/{userId}/{sessionId}/{timestamp}_{filename}`

## Performance & Reliability

### Rate Limiting & Throttling
- **Exponential Backoff**: Automatic retry with exponential backoff for Bedrock API calls
- **Circuit Breaker**: Prevents cascade failures during high throttling
- **Metrics Tracking**: Comprehensive throttling and performance metrics
- **Dynamic Rate Limiting**: Adjusts request intervals based on API responses

### Memory Management
- **Resource Monitoring**: Automatic memory and CPU usage tracking
- **Garbage Collection**: Forced cleanup during high memory usage
- **Message History Limits**: Conversation history truncation and summarization
- **Queue Management**: Proper cleanup of audio and response queues

### Error Handling
- **Graceful Degradation**: Continues operation during partial failures
- **Session Recovery**: Automatic session recreation on timeout
- **Validation Errors**: Proper handling of Bedrock validation exceptions
- **Connection Management**: Robust WebSocket connection lifecycle

## Health Checks

The health check endpoint provides system status:
- `GET /health` - Returns server health, AWS configuration status, and timestamp
- Used by Kubernetes liveness and readiness probes
- Includes AWS credential validation (non-blocking)

## Observability

### OpenTelemetry Integration
- **Distributed Tracing**: Full request tracing across services
- **Session Tracking**: Complete session lifecycle monitoring  
- **Tool Usage Tracking**: Individual tool execution spans
- **Token Usage Metrics**: Cost and usage tracking per session
- **Error Tracking**: Comprehensive error and exception tracking

### Logging
- **Structured Logging**: JSON-formatted logs with context
- **Performance Metrics**: Request duration and resource usage
- **Security Events**: Authentication and authorization logging
- **Health Check Filtering**: Reduced noise from health check requests