"""
Web server for retail store assistant with WebSocket and REST API endpoints.

Features:
- WebSocket endpoint with AWS Cognito JWT token authentication
- REST API endpoints for chat and image upload
- AWS Bedrock integration for AI responses
- Tool configuration and S2S session management

Authentication:
- WebSocket connections require valid AWS Cognito JWT tokens passed as query parameters
- Token validation includes signature verification using Cognito's JWKS public keys
- Supports ID tokens from Cognito (not access tokens)
- Validates issuer, audience, expiration, and user identity

Environment Variables:
- PORT: Server port (default: 8000)
- HOST: Server host (default: 0.0.0.0)
- AWS_REGION: AWS region for Bedrock
- COGNITO_USER_POOL_ID: AWS Cognito User Pool ID (required for authentication)
- COGNITO_APP_CLIENT_ID: AWS Cognito App Client ID (required for authentication)
"""

import asyncio
import os
import logging
import threading
import time
import socket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# Configure uvicorn logging to suppress health check logs
import uvicorn.logging
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
import os
# Load .env file from the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


print("Custom OpenTelemetry configuration loaded successfully")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

STACK_NAME = os.getenv("STACK_NAME", "backend")
name = f"run_servers"
logger = logging.getLogger(name)
logger.setLevel(logging.INFO)  # Explicitly set level


# Log environment variables on startup
logger.info("="*60)
logger.info(f"COGNITO_USER_POOL_ID: {os.getenv('COGNITO_USER_POOL_ID')}")
logger.info(f"COGNITO_IDENTITY_POOL_ID: {os.getenv('COGNITO_IDENTITY_POOL_ID')}")
logger.info(f"COGNITO_APP_CLIENT_ID: {os.getenv('COGNITO_APP_CLIENT_ID')}")
logger.info(f"AWS_REGION: {os.getenv('AWS_REGION')}")
logger.info("="*60)

# Default ports if not specified in environment variables
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"


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
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    user_info = None
    connection_accepted = False
    
    # Extract authentication parameters
    try:
        auth_params = extract_websocket_params(websocket)
        token = auth_params["token"]
        user_id = auth_params["user_id"]
        
        logger.info(f"WebSocket connection attempt from user: {user_id}, session: {session_id}")
        
        # Validate authentication before accepting connection
        try:
            user_info = validate_token(token, user_id)
            logger.info(f"Authentication successful for user: {user_info}")
        except Exception as auth_error:
            logger.info(f"User info details: {json.dumps({k: v for k, v in user_info.items() if k != 'aws_credentials'})}")
            raise auth_error
        
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
        
        logger.info(f"WebSocket connection accepted for user: {user_info['user_id']}, session: {session_id}")
        
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
        

        logger.info(f"Set user_info on stream_manager for user: {user_info['user_id']}, session: {session_id}")
        stream_manager = S2sSessionManager(model_id='amazon.nova-sonic-v1:0', 
                                            region=os.environ["AWS_REGION"],
                                            session_id=session_id,
                                            user_info=user_info)
        
        logger.info(f"Stream manager user_info contains keys: {list(stream_manager.user_info.keys())}")
        
        await stream_manager.initialize_stream()
        
        # Start a task to forward responses from Bedrock to the WebSocket
        logger.info(f"Starting forward_responses task for session: {session_id}")
        forward_task = asyncio.create_task(forward_responses(websocket, stream_manager, session_id))
        
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
                    logger.error(f"Invalid JSON received from WebSocket (user: {user_info['user_id']}, session: {session_id}): {e}")
                    continue
                
                # Handle S2S protocol messages
                if 'event' in data:
                    event_type = list(data['event'].keys())[0]

                    if event_type == 'sessionStart':
                        # Extract the prompt name from the event
                        prompt_name = data['event']['sessionStart'].get('promptName')
                        logger.info(f"Received sessionStart event with promptName: {prompt_name} for session: {session_id}")
        
                        # Initialize the session
                        logger.info(f"Initializing session with prompt: {prompt_name} for session: {session_id}")
                        await stream_manager.initialize_session_with_prompt(prompt_name)
                        logger.info(f"Session initialized successfully for session: {session_id}")
                        
                        
                    
                    # Log event type (except for audio which would be too verbose)
                    if event_type != "audioInput":
                        logger.info(f"Received event type: {event_type} from user: {user_info['user_id']}, session: {session_id}")
                    

                    if event_type == 'contentStart' and data['event']['contentStart'].get('type') == 'AUDIO':
                        stream_manager.audio_content_name = data['event']['contentStart']['contentName']          
                        
                    # Handle audio input
                    if event_type == 'audioInput':
                        prompt_name = data['event']['audioInput']['promptName']
                        content_name = data['event']['audioInput']['contentName']
                        audio_base64 = data['event']['audioInput']['content']
                        stream_manager.add_audio_chunk(prompt_name, content_name, audio_base64)
                    # Handle other events (except sessionStart which is already handled in initialize_session_with_prompt)
                    elif event_type != 'sessionStart':
                        await stream_manager.send_raw_event(data)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for user: {user_info['user_id']}, session: {session_id}")
                break
            except ConnectionClosed:
                logger.info(f"WebSocket connection closed for user: {user_info['user_id']}, session: {session_id}")
                break
            except Exception as e:
                error_msg = str(e)
                # Handle specific WebSocket state errors more gracefully
                if "WebSocket is not connected" in error_msg or "Need to call \"accept\" first" in error_msg:
                    logger.warning(f"WebSocket connection {session_id} is in invalid state, stopping message processing: {error_msg}")
                    break
                else:
                    logger.error(f"Error processing WebSocket message for user {user_info['user_id']}, session {session_id}: {e}")
                    # For other errors, don't break the loop immediately but add a small delay
                    await asyncio.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Unexpected error in websocket handler for user {user_info.get('user_id', 'unknown') if user_info else 'unknown'}, session {session_id}: {e}")

    finally:
        # Clean up resources
        if not cleanup_initiated:
            cleanup_initiated = True
            
            if forward_task and not forward_task.done():
                logger.info(f"Cancelling forward_task for user: {user_info.get('user_id', 'unknown') if user_info else 'unknown'}, session: {session_id}")
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
                    logger.info(f"Stream manager closed for user: {user_info.get('user_id', 'unknown') if user_info else 'unknown'}, session: {session_id}")
                except Exception as e:
                    logger.error(f"Error closing stream manager: {e}")
            
        logger.info(f"WebSocket handler completed for user: {user_info.get('user_id', 'unknown') if user_info else 'unknown'}, session: {session_id}")


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

async def forward_responses(websocket: WebSocket, stream_manager, session_id: str):
    """Forward responses from Bedrock to the WebSocket."""
    try:
        logger.info(f"Starting forward_responses for session: {session_id}")
        while True:
            try:
                # Get next response from the output queue with a timeout
                try:
                    response = await asyncio.wait_for(stream_manager.output_queue.get(), timeout=0.5)
                    
                    # Log event type for debugging (except audio events which would be too verbose)
                    if "event" in response:
                        event_type = list(response["event"].keys())[0] if "event" in response else "unknown"
                        if event_type not in ["audioOutput", "audioInput"]:
                            logger.debug(f"[Session {session_id}] Got {event_type} event from output queue")
                            
                            # Log tool use events in more detail
                            if event_type == "toolUse":
                                tool_name = response["event"]["toolUse"].get("toolName", "unknown")
                                tool_use_id = response["event"]["toolUse"].get("toolUseId", "unknown")
                                logger.info(f"[Session {session_id}] Tool use detected: {tool_name}, ID: {tool_use_id}")
                            
                            # Log tool result events in more detail
                            elif event_type == "toolResult":
                                tool_use_id = response["event"]["toolResult"].get("toolUseId", "unknown")
                                logger.info(f"[Session {session_id}] Tool result received for ID: {tool_use_id}")
                                
                except asyncio.TimeoutError:
                    if not stream_manager.is_active:
                        logger.info(f"Stream no longer active for session {session_id}, stopping forward_responses")
                        break
                    continue
                
                # Send to WebSocket
                try:
                    event = json.dumps(response)
                    await websocket.send_text(event)
                    
                    # Log important events (not audio)
                    if "event" in response:
                        event_type = list(response["event"].keys())[0] if "event" in response else "unknown"
                        if event_type not in ["audioOutput", "audioInput"]:
                            logger.debug(f"[Session {session_id}] Sent {event_type} event to WebSocket")
                            
                except WebSocketDisconnect:
                    logger.info(f"WebSocket connection {session_id} closed during forward_responses")
                    break
                except Exception as e:
                    logger.error(f"Error sending response to WebSocket session {session_id}: {e}")
                    continue
                    
            except asyncio.CancelledError:
                logger.info(f"Forward responses task cancelled for session {session_id}")
                raise
            except Exception as e:
                logger.error(f"Error in forward_responses loop for session {session_id}: {e}")
                if "retry-after" in str(e).lower():
                    logger.warning(f"Rate limited by Bedrock API in forward_responses for session {session_id}")
                    await asyncio.sleep(1.0)
                    continue
                
                if not stream_manager.is_active:
                    logger.info(f"Stream manager inactive for session {session_id}, stopping forward_responses")
                    break
    except asyncio.CancelledError:
        # Task was cancelled (normal behavior during cleanup)
        logger.info(f"Forward responses task cancelled for session {session_id} - allowing main handler to clean up")
    except Exception as e:
        logger.error(f"Error forwarding responses for session {session_id}: {e}")

    finally:
        logger.info(f"Forward response task completed for session {session_id}")

async def cleanup_session_managers():
    """Clean up inactive session managers - simplified version since each connection manages its own"""
    # This function is kept for compatibility but doesn't need to do much
    # since each WebSocket connection manages its own stream manager
    pass

def main():
    """Main function to start the unified server"""

    # Get port and host from environment variables
    port = int(os.getenv("PORT", DEFAULT_PORT))
    host = os.getenv("HOST", DEFAULT_HOST)
    
    logger.info(f"Starting unified server on {host}:{port}")
    
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
        # Start periodic session cleanup
        asyncio.create_task(periodic_cleanup())
        logger.info("Started periodic session cleanup task")
        
    
    # Start the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
