"""
Web server for retail store assistant with WebSocket and REST API endpoints.

Features:
- WebSocket endpoint with AWS Cognito JWT token authentication
- REST API endpoints for chat and image upload
- AWS Bedrock integration for AI responses
- Automatic AWS credentials refresh
- Tool configuration and S2S session management

Authentication:
- WebSocket connections require valid AWS Cognito JWT tokens passed as query parameters
- Token validation includes signature verification using Cognito's JWKS public keys
- Supports both ID tokens and access tokens from Cognito
- Validates issuer, audience, expiration, and user identity

Environment Variables:
- DEBUG: Enable debug mode for detailed logging (default: false)
- PORT: Server port (default: 8000)
- HOST: Server host (default: 0.0.0.0)
- AWS_REGION: AWS region for Bedrock
- AWS_REFRESH_INTERVAL: AWS credentials refresh interval in seconds
- COGNITO_REGION: AWS region for Cognito (defaults to AWS_REGION)
- COGNITO_USER_POOL_ID: AWS Cognito User Pool ID (required for authentication)
- COGNITO_APP_CLIENT_ID: AWS Cognito App Client ID (required for authentication)
"""

import asyncio
import os
import logging
import threading
import time
import aws_credentials
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from restapi import app as restapi_app
import boto3
import json
import jwt
import requests
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from s2s_events import S2sEvent
from restapi import ToolsList
from s2s_session_manager import S2sSessionManager
from auth import validate_token
from websockets.exceptions import ConnectionClosed
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

STACK_NAME = os.getenv("STACK_NAME", "backend")
name = f"{STACK_NAME}-server"
logger = logging.getLogger(name)

# Debug: Log environment variables on startup
logger.info("="*60)
logger.info("ENVIRONMENT VARIABLES DEBUG")
logger.info("="*60)
logger.info(f"COGNITO_USER_POOL_ID: {os.getenv('COGNITO_USER_POOL_ID')}")
logger.info(f"COGNITO_APP_CLIENT_ID: {os.getenv('COGNITO_APP_CLIENT_ID')}")
logger.info(f"COGNITO_REGION: {os.getenv('COGNITO_REGION')}")
logger.info(f"AWS_REGION: {os.getenv('AWS_REGION')}")
logger.info(f"DISABLE_AUTH: {os.getenv('DISABLE_AUTH')}")
logger.info("="*60)

# Debug mode flag
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Default ports if not specified in environment variables
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"

# Default AWS credentials refresh interval in 4 hours (14400 seconds)
DEFAULT_AWS_REFRESH_INTERVAL = 14400

# Global variables
aws_refresh_thread = None
restart_in_progress = False

def run_aws_credentials_refresh():
    """Run the AWS credentials refresh in a separate thread"""
    # Get the refresh interval from environment variables, default to 15 minutes
    refresh_interval = int(os.getenv("AWS_REFRESH_INTERVAL", DEFAULT_AWS_REFRESH_INTERVAL))
    logger.info(f"Starting AWS credentials refresh with interval of {refresh_interval} seconds")
    
    # Register the callback for when credentials are refreshed
    aws_credentials.refresh_aws_credentials.on_credentials_refreshed = restart_servers
    logger.info("Registered credentials refresh callback")
    
    # Initial refresh - but don't restart servers yet since they haven't been started
    original_callback = aws_credentials.refresh_aws_credentials.on_credentials_refreshed
    aws_credentials.refresh_aws_credentials.on_credentials_refreshed = None
    initial_success = aws_credentials.refresh_aws_credentials()
    aws_credentials.refresh_aws_credentials.on_credentials_refreshed = original_callback
    logger.info(f"Initial AWS credentials refresh {'succeeded' if initial_success else 'failed'}")
    
    def refresh_loop():
        refresh_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while True:
            try:
                time.sleep(DEFAULT_AWS_REFRESH_INTERVAL)
                refresh_count += 1
                logger.info(f"Refreshing AWS credentials at {time.strftime('%Y-%m-%d %H:%M:%S')} (attempt #{refresh_count})")
                
                if not hasattr(aws_credentials.refresh_aws_credentials, 'on_credentials_refreshed') or aws_credentials.refresh_aws_credentials.on_credentials_refreshed is None:
                    logger.warning("Credentials refresh callback is not set, setting it now")
                    aws_credentials.refresh_aws_credentials.on_credentials_refreshed = restart_servers
                
                success = aws_credentials.refresh_aws_credentials()
                
                if success:
                    consecutive_failures = 0
                    logger.info(f"AWS credentials refresh #{refresh_count} succeeded")
                    
                    if refresh_count > 1 and refresh_count % 5 == 0:
                        try:
                            
                            sts = boto3.client('sts')
                            identity = sts.get_caller_identity()
                            logger.info(f"AWS credentials check: Valid (Identity: {identity.get('Arn', 'Unknown')})")
                            
                            if consecutive_failures == 0 and refresh_count > 10:
                                new_interval = min(refresh_interval * 1.5, 3600)
                                if new_interval > refresh_interval:
                                    refresh_interval = new_interval
                                    logger.info(f"Increasing refresh interval to {refresh_interval} seconds")
                        except Exception as e:
                            logger.error(f"AWS credentials check failed: {e}")
                else:
                    consecutive_failures += 1
                    logger.error(f"AWS credentials refresh #{refresh_count} failed ({consecutive_failures} consecutive failures)")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive failures ({consecutive_failures}), attempting to restart refresh thread")
            except Exception as e:
                logger.error(f"Exception in AWS credentials refresh loop: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                consecutive_failures += 1
    
    thread = threading.Thread(target=refresh_loop)
    thread.daemon = True
    thread.name = "aws-credentials-refresh"
    thread.start()
    
    logger.info(f"AWS credentials refresh thread started with ID: {thread.ident}")
    return thread

def restart_servers():
    """Restart the unified server"""
    global restart_in_progress
    
    logger.info("Restarting server due to AWS credentials refresh")
    restart_in_progress = True
    
    # Add a small delay to ensure everything is ready
    time.sleep(2)
    
    restart_in_progress = False
    return True

def extract_websocket_params(websocket: WebSocket) -> dict:
    """
    Extract authentication parameters from WebSocket query string.
    
    Args:
        websocket: WebSocket connection
    
    Returns:
        dict: Extracted parameters
    """
    query_params = dict(websocket.query_params)
    
    token = query_params.get("token")
    user_id = query_params.get("userId")
    
    return {
        "token": token,
        "user_id": user_id,
        "query_params": query_params
    }

app = FastAPI(title=name)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the REST API routes
app.include_router(restapi_app.router)

# Add WebSocket endpoint with session ID
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    user_info = None
    connection_accepted = False
    
    # Extract authentication parameters
    try:
        auth_params = extract_websocket_params(websocket)
        token = auth_params["token"]
        user_id = auth_params["user_id"]
        
        logger.info(f"WebSocket connection attempt from user: {user_id}, session: {client_id}")
        
        # Validate authentication before accepting connection
        try:
            user_info = validate_token(token, user_id)
            logger.info(f"Authentication successful for user: {user_info['user_id']}")
        except HTTPException as e:
            logger.warning(f"Authentication failed for WebSocket connection: {e.detail}")
            try:
                await websocket.close(code=1008, reason=f"Authentication failed: {e.detail}")
            except Exception as close_error:
                logger.error(f"Error closing WebSocket after auth failure: {close_error}")
            return
        except Exception as e:
            logger.error(f"Unexpected authentication error: {e}")
            try:
                await websocket.close(code=1011, reason="Authentication service error")
            except Exception as close_error:
                logger.error(f"Error closing WebSocket after auth error: {close_error}")
            return
        
        # Accept the connection only after successful authentication
        await websocket.accept()
        connection_accepted = True
        
        logger.info(f"WebSocket connection accepted for user: {user_info['user_id']}, session: {client_id}")
        
    except Exception as e:
        logger.error(f"Error extracting WebSocket parameters or accepting connection: {e}")
        if not connection_accepted:
            try:
                await websocket.close(code=1002, reason="Invalid connection parameters")
            except:
                pass  # Connection might already be closed
        return
    
    stream_manager = None
    forward_task = None
    cleanup_initiated = False
    
    try:
        # Create a stream manager for this connection
        stream_manager = S2sSessionManager(model_id='amazon.nova-sonic-v1:0', 
                                          region=os.environ["AWS_REGION"])
        
        # Store user context and session info in stream manager
        stream_manager.user_info = user_info
        stream_manager.client_id = client_id
        
        await stream_manager.initialize_stream()
        
        # Start a task to forward responses from Bedrock to the WebSocket
        forward_task = asyncio.create_task(forward_responses(websocket, stream_manager, client_id))
        
        while connection_accepted:
            try:
                # Receive message from client
                message = await websocket.receive_text()
                
                # Parse the message as JSON
                try:
                    data = json.loads(message)
                    if 'body' in data:
                        data = json.loads(data["body"])
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received from WebSocket (user: {user_info['user_id']}, session: {client_id}): {e}")
                    continue
                
                # Handle S2S protocol messages
                if 'event' in data:
                    event_type = list(data['event'].keys())[0]
                    
                    # Log event type (except for audio which would be too verbose)
                    if event_type != "audioInput":
                        logger.info(f"Received event type: {event_type} from user: {user_info['user_id']}, session: {client_id}")
                    

                    if event_type == 'contentStart' and data['event']['contentStart'].get('type') == 'AUDIO':
                        stream_manager.audio_content_name = data['event']['contentStart']['contentName']

                    elif event_type == 'promptStart':
                        stream_manager.prompt_name = data['event']['promptStart']['promptName']
                        
                        # Define default tool configuration
                        DEFAULT_TOOL_CONFIG = S2sEvent.DEFAULT_TOOL_CONFIG
                        
                        # Create an instance of the ToolsList class
                        tools_instance = ToolsList()
                        
                        # Initialize the tools configuration
                        tools_config = {
                            "tools": [],
                            "toolChoice": { "any": {} }
                        }
                        
                        # Get all methods that have been decorated with bedrock_tool
                        for method_name in dir(tools_instance):
                            if method_name.startswith('_'):
                                continue
                            
                            method = getattr(tools_instance, method_name)
                            if hasattr(method, 'bedrock_schema'):
                                tool_schema = dict(method.bedrock_schema)
                                
                                # Ensure inputSchema.json is serialized as a JSON string
                                if isinstance(tool_schema.get('toolSpec', {}).get('inputSchema', {}).get('json'), (dict, list)):
                                    tool_schema['toolSpec']['inputSchema']['json'] = json.dumps(
                                        tool_schema['toolSpec']['inputSchema']['json']
                                    )
                                
                                tools_config["tools"].append(tool_schema)
                        
                        # If no tools were found, fall back to default
                        if len(tools_config["tools"]) == 0:
                            logger.info("No tools found in ToolsList, using default config")
                            tools_config = DEFAULT_TOOL_CONFIG
                        
                        
                        # Inject our configurations into the original event
                        if 'toolConfiguration' not in data['event']['promptStart']:
                            data['event']['promptStart']['toolConfiguration'] = tools_config

                            logger.info(f"Using tools for user {user_info['user_id']}, session {client_id}: " + str(data['event']['promptStart']['toolConfiguration']))
                        
                        # Set the tool configuration on the stream manager
                        stream_manager.toolConfiguration = tools_config
                        
                    # Handle audio input
                    if event_type == 'audioInput':
                        prompt_name = data['event']['audioInput']['promptName']
                        content_name = data['event']['audioInput']['contentName']
                        audio_base64 = data['event']['audioInput']['content']
                        stream_manager.add_audio_chunk(prompt_name, content_name, audio_base64)
                    # Handle other events
                    else:
                        await stream_manager.send_raw_event(data)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for user: {user_info['user_id']}, session: {client_id}")
                break
            except ConnectionClosed:
                logger.info(f"WebSocket connection closed for user: {user_info['user_id']}, session: {client_id}")
                break
            except Exception as e:
                error_msg = str(e)
                # Handle specific WebSocket state errors more gracefully
                if "WebSocket is not connected" in error_msg or "Need to call \"accept\" first" in error_msg:
                    logger.warning(f"WebSocket connection {client_id} is in invalid state, stopping message processing: {error_msg}")
                    break
                else:
                    logger.error(f"Error processing WebSocket message for user {user_info['user_id']}, session {client_id}: {e}")
                    if DEBUG:
                        import traceback
                        traceback.print_exc()
                    # For other errors, don't break the loop immediately but add a small delay
                    await asyncio.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Unexpected error in websocket handler for user {user_info.get('user_id', 'unknown') if user_info else 'unknown'}, session {client_id}: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
    finally:
        # Clean up resources
        if not cleanup_initiated:
            cleanup_initiated = True
            
            if forward_task and not forward_task.done():
                logger.info(f"Cancelling forward_task for user: {user_info.get('user_id', 'unknown') if user_info else 'unknown'}, session: {client_id}")
                forward_task.cancel()
                try:
                    await forward_task # Directly await the task after cancellation
                    logger.info("forward_task completed after cancellation.")
                except asyncio.CancelledError:
                    logger.info("forward_task was cancelled as expected.")
                except Exception as e:
                    logger.error(f"Exception while awaiting cancelled forward_task: {e}")
            
            if stream_manager:
                try:
                    await stream_manager.close()
                    logger.info(f"Stream manager closed for user: {user_info.get('user_id', 'unknown') if user_info else 'unknown'}, session: {client_id}")
                except Exception as e:
                    logger.error(f"Error closing stream manager: {e}")
            
        logger.info(f"WebSocket handler completed for user: {user_info.get('user_id', 'unknown') if user_info else 'unknown'}, session: {client_id}")


# Add health check endpoint
@app.get("/health")
async def health_check():
    """
    Enhanced health check that responds quickly and provides basic status.
    Used by Kubernetes liveness and readiness probes.
    """
    try:
        status = {
            "status": "healthy",
            "timestamp": time.time()
        }
        
        # Optional: Quick AWS credentials check (but don't block on it)
        try:
            # Just check if we have environment variables set, don't make AWS calls
            if os.environ.get("AWS_REGION") and os.environ.get("AWS_ROLE_ARN"):
                status["aws_config"] = "configured"
            else:
                status["aws_config"] = "missing"
        except Exception:
            status["aws_config"] = "error"
            
        return status
    except Exception as e:
        # Even if there's an error, return a response to prevent probe timeout
        logger.error(f"Health check error: {e}")
        return {"status": "degraded", "error": str(e), "timestamp": time.time()}

async def forward_responses(websocket: WebSocket, stream_manager: S2sSessionManager, client_id: str):
    """Forward responses from Bedrock to the WebSocket."""
    try:
        logger.info(f"Starting forward_responses for session: {client_id}")
        while True:
            try:
                # Get next response from the output queue with a timeout
                try:
                    response = await asyncio.wait_for(stream_manager.output_queue.get(), timeout=0.5)
                    logger.info(f"Got response from output queue for session {client_id}: {response.get('event', {}).keys()}")
                except asyncio.TimeoutError:
                    if not stream_manager.is_active:
                        logger.info(f"Stream no longer active for session {client_id}, stopping forward_responses")
                        break
                    continue
                
                # Send to WebSocket
                try:
                    event = json.dumps(response)
                    logger.info(f"Sending response to WebSocket session {client_id}: {response.get('event', {}).keys()}")
                    await websocket.send_text(event)
                    logger.info(f"Response sent to WebSocket session {client_id} successfully")
                except WebSocketDisconnect:
                    logger.info(f"WebSocket connection {client_id} closed during forward_responses")
                    break
                except Exception as e:
                    logger.error(f"Error sending response to WebSocket session {client_id}: {e}")
                    continue
                    
            except asyncio.CancelledError:
                logger.info(f"Forward responses task cancelled for session {client_id}")
                raise
            except Exception as e:
                logger.error(f"Error in forward_responses loop for session {client_id}: {e}")
                if "retry-after" in str(e).lower():
                    logger.warning(f"Rate limited by Bedrock API in forward_responses for session {client_id}")
                    await asyncio.sleep(1.0)
                    continue
                
                if not stream_manager.is_active:
                    logger.info(f"Stream manager inactive for session {client_id}, stopping forward_responses")
                    break
    except asyncio.CancelledError:
        # Task was cancelled (normal behavior during cleanup)
        logger.info(f"Forward responses task cancelled for session {client_id} - allowing main handler to clean up")
    except Exception as e:
        logger.error(f"Error forwarding responses for session {client_id}: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
    finally:
        logger.info(f"Forward response task completed for session {client_id}")

async def cleanup_session_managers():
    """Clean up inactive session managers - simplified version since each connection manages its own"""
    # This function is kept for compatibility but doesn't need to do much
    # since each WebSocket connection manages its own stream manager
    pass

def main():
    """Main function to start the unified server"""
    global aws_refresh_thread
    
    
    # Start AWS credentials refresh
    aws_refresh_thread = run_aws_credentials_refresh()
    
    # Get port and host from environment variables
    port = int(os.getenv("PORT", DEFAULT_PORT))
    host = os.getenv("HOST", DEFAULT_HOST)
    
    logger.info(f"Starting unified server on {host}:{port}")
    
    # Configure uvicorn logging to suppress health check logs
    import uvicorn.logging
    
    class HealthCheckFilter(logging.Filter):
        def filter(self, record):
            # Filter out health check requests
            return not (hasattr(record, 'getMessage') and 
                       '"GET /health HTTP/1.1" 200' in record.getMessage())
    
    # Apply filter to uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(HealthCheckFilter())
    
    # Start periodic session cleanup task
    async def periodic_cleanup():
        """Periodic cleanup of inactive session managers"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                await cleanup_session_managers()
                logger.info("Periodic session cleanup completed")
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    # Add the cleanup task to the FastAPI startup
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(periodic_cleanup())
        logger.info("Started periodic session cleanup task")
    
    # Start the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main() 