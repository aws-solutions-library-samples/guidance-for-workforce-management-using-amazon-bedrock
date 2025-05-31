import asyncio
import json
import base64
import warnings
import uuid
from s2s_events import S2sEvent

import time
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart, ServiceError
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver
import logging
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure debugging
DEBUG = True
logger = logging.getLogger(__name__)

def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        logger.debug(message)

# Bedrock rate limit handling constants
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # 1 second base delay


class S2sSessionManager:
    """Manages bidirectional streaming with AWS Bedrock using asyncio"""
    
    def __init__(self, model_id='amazon.nova-sonic-v1:0', region='us-east-1'):
        """Initialize the stream manager."""
        self.model_id = model_id
        self.region = region
        
        # Audio and output queues
        self.audio_input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        self.response_task = None
        self.audio_task = None
        self.stream = None
        self.is_active = False
        self.bedrock_client = None
        
        # Session information
        self.prompt_name = None  # Will be set from frontend
        self.content_name = None  # Will be set from frontend
        self.audio_content_name = None  # Will be set from frontend
        self.toolUseContent = ""
        self.toolUseId = ""
        self.toolName = ""
        
        # Task tracking for proper cleanup
        self.tasks = set()
        
        # Add retry tracking for token refresh
        self.token_refresh_attempts = 0
        self.max_token_refresh_attempts = 3
        self.last_token_refresh_time = 0
        self.token_refresh_cooldown = 60  # Wait 60 seconds between refresh attempts

    def _initialize_client(self):
        """Initialize the Bedrock client."""
        # Clear any existing client first
        self.bedrock_client = None
        
        # Force clear of any cached credentials in the environment resolver
        # by creating a new resolver instance
        credentials_resolver = EnvironmentCredentialsResolver()
        
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=credentials_resolver,
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
        
        # Log environment variables for debugging (without exposing secrets)
        access_key = os.environ.get('AWS_ACCESS_KEY_ID', '')
        logger.info(f"Bedrock client initialized with access key: {access_key[:5]}..." if access_key else "No access key found")
        session_token = os.environ.get('AWS_SESSION_TOKEN', '')
        logger.info(f"Session token present: {'Yes' if session_token else 'No'}")

    async def initialize_stream(self):
        """Initialize the bidirectional stream with Bedrock."""
        try:
            # if not self.bedrock_client:
            #     logger.info("Initializing Bedrock client")
            self._initialize_client()
            
            # If there's already an active stream, close it first
            if self.is_active and self.stream:
                logger.info("Closing existing stream")
                await self.close()
                # Wait a moment for resources to clean up
                await asyncio.sleep(0.5)
                
            logger.info("Initializing new bidirectional stream")
            
            # Initialize the stream with retry logic
            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    # Initialize the stream
                    self.stream = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                        InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
                    )
                    self.is_active = True
                    break
                except ServiceError as e:
                    retry_count += 1
                    if 'retry-after' in str(e).lower() and retry_count < MAX_RETRIES:
                        # Extract retry time if available or use exponential backoff
                        retry_time = RETRY_BASE_DELAY * (2 ** (retry_count - 1))
                        logger.warning(f"Rate limited by Bedrock API, retrying in {retry_time} seconds (attempt {retry_count}/{MAX_RETRIES})")
                        await asyncio.sleep(retry_time)
                    else:
                        if retry_count >= MAX_RETRIES:
                            logger.error(f"Failed to initialize stream after {MAX_RETRIES} retries")
                        raise
            
            # Start listening for responses
            self.response_task = asyncio.create_task(self._process_responses())
            self.tasks.add(self.response_task)
            self.response_task.add_done_callback(self.tasks.discard)

            # Start processing audio input
            self.audio_task = asyncio.create_task(self._process_audio_input())
            self.tasks.add(self.audio_task)
            self.audio_task.add_done_callback(self.tasks.discard)
            
            # Wait a bit to ensure everything is set up
            await asyncio.sleep(0.1)
            
            logger.info("Stream initialized successfully")
            return self
        except Exception as e:
            self.is_active = False
            logger.error(f"Failed to initialize stream: {str(e)}")
            raise
    
    async def send_raw_event(self, event_data, retry_count=0):
        """Send a raw event to the Bedrock stream."""
        if not self.stream or not self.is_active:
            logger.info("Stream not initialized or closed,stream is set to active: {self.is_active}")
            return
            
        # Prevent infinite retries
        max_retries = 2
        if retry_count >= max_retries:
            logger.error(f"Maximum retry attempts ({max_retries}) exceeded for sending event")
            return
            
        try:
            event_json = json.dumps(event_data)
            # Debug logs for better tracking of protocol flow
            if "event" in event_data:
                event_type = list(event_data["event"].keys())[0]
                if event_type != "audioInput":
                    logger.info(f"Sending event type: {event_type}")
                    # log content name
                    if event_type == "contentStart":
                        content_name = event_data["event"]["contentStart"].get("contentName")
                        logger.info(f"Content name: {content_name}")
                    elif event_type == "contentEnd":
                        content_name = event_data["event"]["contentEnd"].get("contentName")
                        logger.info(f"Content name: {content_name}")
                
            
            # Create and send the event
            event = InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
            )
            
            # Check if stream is still active before sending
            if not self.is_active or not self.stream:
                logger.info("Stream is no longer active, cannot send event")
                return
                
            await self.stream.input_stream.send(event)
            
            # Close session if session end event
            if "event" in event_data and "sessionEnd" in event_data["event"]:
                # Wait a moment for the event to be processed before closing
                await asyncio.sleep(0.2)
                logger.info("Closing session with self.close() given we received a sessionEnd event")
                await self.close()
            
        except ServiceError as e:
            if 'retry-after' in str(e).lower():
                logger.warning(f"Rate limited by Bedrock API: {e}")
                # Wait before next attempt
                await asyncio.sleep(1.0)
            elif "ExpiredTokenException" in str(e) or "UnrecognizedClientException" in str(e):
                # Handle expired/invalid token with retry limits
                error_type = "expired token" if "ExpiredTokenException" in str(e) else "invalid token"
                current_time = time.time()
                if (current_time - self.last_token_refresh_time) > self.token_refresh_cooldown:
                    self.token_refresh_attempts = 0  # Reset counter after cooldown
                
                if self.token_refresh_attempts < self.max_token_refresh_attempts:
                    logger.warning(f"{error_type} detected, attempting refresh (attempt {self.token_refresh_attempts + 1}/{self.max_token_refresh_attempts})")
                    self.token_refresh_attempts += 1
                    self.last_token_refresh_time = current_time
                    
                    try:
                        # Import and refresh AWS credentials
                        import aws_credentials
                        aws_credentials.refresh_aws_credentials()
                        
                        if self._handle_expired_token():
                            # For send_raw_event, we can't recreate the stream here as it would interfere with
                            # the current session flow. Instead, just mark the session as needing restart
                            logger.warning(f"Token refresh successful but stream needs recreation - marking session for restart")
                            self.is_active = False  # This will trigger stream recreation in the main handler
                            return  # Don't retry the event - let the session handler deal with it
                        else:
                            logger.error(f"Failed to recover from {error_type} when sending event")
                            self.is_active = False
                    except Exception as refresh_error:
                        logger.error(f"Exception during token refresh in send_raw_event: {refresh_error}")
                        self.is_active = False
                else:
                    logger.error(f"Maximum token refresh attempts ({self.max_token_refresh_attempts}) exceeded for {error_type}, stopping event sending")
                    self.is_active = False  # Stop the session
            else:
                debug_print(f"Error sending event: {str(e)}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            debug_print(f"Error sending event: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        try:
            # logger.info("Processing audio input while is_active is true: {self.is_active}")
            while self.is_active:
                try:
                    # Get audio data from the queue with a timeout to allow clean cancellation
                    try:
                        data = await asyncio.wait_for(self.audio_input_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # No data received within timeout, continue checking is_active
                        continue
                    
                    # Extract data from the queue item
                    prompt_name = data.get('prompt_name')
                    content_name = data.get('content_name')
                    audio_bytes = data.get('audio_bytes')
                    
                    if not audio_bytes or not prompt_name or not content_name:
                        debug_print("Missing required audio data properties")
                        continue

                    # Create the audio input event
                    audio_event = S2sEvent.audio_input(prompt_name, content_name, audio_bytes.decode('utf-8') if isinstance(audio_bytes, bytes) else audio_bytes)
                    # logger.info(f"Sending audio event for prompt_name and content_name: {prompt_name}, {content_name}")
                    # Send the event
                    await self.send_raw_event(audio_event)
                    
                except asyncio.CancelledError:
                    logger.info("Audio processing task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    if DEBUG:
                        import traceback
                        traceback.print_exc()
                    # Don't break the loop on error, continue processing
        except asyncio.CancelledError:
            logger.info("Audio task cancelled during processing")
        except Exception as e:
            logger.error(f"Unexpected error in audio processing task: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
        finally:
            logger.info("Audio processing task completed")
    
    def add_audio_chunk(self, prompt_name, content_name, audio_data):
        """Add an audio chunk to the queue."""
        # The audio_data is already a base64 string from the frontend
        length = len(audio_data)
        logger.info(f"Adding audio chunk to queue: {length} bytes")
        try:
            self.audio_input_queue.put_nowait({
                'prompt_name': prompt_name,
                'content_name': content_name,
                'audio_bytes': audio_data
            })
        except Exception as e:
            logger.error(f"Error adding audio chunk to queue: {e}")
    
    def _handle_expired_token(self):
        """Handle expired token by recreating the stream with fresh credentials."""
        logger.error("Detected expired token, reinitializing Bedrock client and stream")
        try:
            # First, close the existing stream if it exists
            if self.stream:
                try:
                    # Don't await here since this is a sync method - just mark inactive
                    self.is_active = False
                    logger.info("Marked existing stream as inactive")
                except Exception as e:
                    logger.warning(f"Error while marking stream inactive: {e}")
            
            # Clear the existing client to force recreation with fresh credentials
            self.bedrock_client = None
            
            # Force recreation of the client with fresh environment variables
            self._initialize_client()
            
            logger.info("Bedrock client reinitialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reinitialize Bedrock client: {e}")
            return False
    
    async def _recreate_stream_after_token_refresh(self):
        """Recreate the entire stream after token refresh."""
        try:
            logger.info("Recreating stream after token refresh")
            
            # Cancel existing tasks
            if self.response_task and not self.response_task.done():
                self.response_task.cancel()
                try:
                    await self.response_task
                except asyncio.CancelledError:
                    pass
            
            if self.audio_task and not self.audio_task.done():
                self.audio_task.cancel()
                try:
                    await self.audio_task
                except asyncio.CancelledError:
                    pass
            
            # Close existing stream
            if self.stream:
                try:
                    await self.stream.input_stream.close()
                except:
                    pass  # Ignore errors when closing
                self.stream = None
            
            # Clear queues
            while not self.audio_input_queue.empty():
                try:
                    self.audio_input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Reinitialize everything
            await self.initialize_stream()
            logger.info("Stream successfully recreated after token refresh")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recreate stream after token refresh: {e}")
            self.is_active = False
            return False

    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:
            logger.info("Starting response processing task")
            while self.is_active:
                try:
                    # Get response with a timeout for cancellation
                    output = await self.stream.await_output()
                    
                    # Check if stream is still active before processing
                    if not self.is_active or not self.stream:
                        logger.info("Stream is no longer active, stopping response processing")
                        break
                    
                    result = await output[1].receive()
                    
                    if not result.value or not result.value.bytes_:
                        continue
                        
                    response_data = result.value.bytes_.decode('utf-8')
                    logger.info(f"Received response from Bedrock: {response_data[:400]}...")
                    
                    try:
                        json_data = json.loads(response_data)
                        json_data["timestamp"] = int(time.time() * 1000)  # Milliseconds since epoch
                        
                        event_name = None
                        if 'event' in json_data:
                            event_name = list(json_data["event"].keys())[0]
                            logger.info(f"Received event type: {event_name}")

                            if event_name == 'textInput':
                                prompt_name = json_data['event']['textInput'].get("promptName")
                                content_name = json_data['event']['textInput'].get("contentName")
                                system_prompt = json_data['event']['textInput'].get("content")
                                debug_print(f"Received textInput event: {prompt_name}, {content_name}, {system_prompt}")
                                # text_input_event = S2sEvent.text_input(prompt_name, content_name, system_prompt)
                                # await self.send_raw_event(text_input_event)
                            
                            # Handle tool use detection
                            if event_name == 'toolUse':
                                self.toolUseContent = json_data['event']['toolUse']
                                self.toolName = json_data['event']['toolUse']['toolName']
                                self.toolUseId = json_data['event']['toolUse']['toolUseId']
                                debug_print(f"Tool use detected: {self.toolName}, ID: {self.toolUseId}")

                            # Process tool use when content ends
                            elif event_name == 'contentEnd' and json_data['event'][event_name].get('type') == 'TOOL':
                                prompt_name = json_data['event']['contentEnd'].get("promptName")
                                debug_print("Processing tool use and sending result")
                                toolResult = await self.processToolUse(self.toolName, self.toolUseContent)
                                
                                # Send tool start event
                                toolContent = str(uuid.uuid4())
                                tool_start_event = S2sEvent.content_start_tool(prompt_name, toolContent, self.toolUseId)
                                await self.send_raw_event(tool_start_event)
                                
                                # Send tool result event
                                if isinstance(toolResult, dict):
                                    content_json_string = json.dumps(toolResult)
                                else:
                                    content_json_string = toolResult

                                tool_result_event = S2sEvent.text_input_tool(prompt_name, toolContent, content_json_string)
                                await self.send_raw_event(tool_result_event)

                                # Send tool content end event
                                tool_content_end_event = S2sEvent.content_end(prompt_name, toolContent)
                                await self.send_raw_event(tool_content_end_event)
                        
                        # Put the response in the output queue for forwarding to the frontend
                        if event_name != 'usageEvent':
                            await self.output_queue.put(json_data)
                            logger.info(f"Added response to output queue: {json_data.get('event', {}).keys()}")
                        else:
                            # Still log usage events but don't forward them
                            logger.debug(f"Received usage event (not forwarded): {json_data.get('event', {}).keys()}")
                    except json.JSONDecodeError as json_error:
                        logger.error(f"JSON decode error: {json_error}")
                        await self.output_queue.put({"raw_data": response_data})

                except asyncio.CancelledError:
                    debug_print("Response processing task cancelled")
                    break
                except StopAsyncIteration:
                    # Stream has ended
                    debug_print("Stream iteration stopped")
                    break
                except Exception as e:
                    # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        logger.error(f"Validation error: {error_message}")
                        
                        # Handle specific audio content errors gracefully
                        if "No open content found for content name" in error_message and "audio-" in error_message:
                            logger.warning(f"Audio content validation error - continuing processing: {error_message}")
                            # This is a known issue with audio content lifecycle management
                            # Continue processing instead of breaking the session
                            continue
                        
                        # For other validation errors, we may want to break or continue based on severity
                        if "audio" in error_message.lower():
                            logger.warning("Audio-related validation error, continuing session")
                            continue
                    elif "retry-after" in str(e).lower():
                        logger.warning("Rate limited by Bedrock API")
                        # Wait before next attempt
                        await asyncio.sleep(1.0)
                        continue
                    elif "ExpiredTokenException" in str(e) or "UnrecognizedClientException" in str(e):
                        # Handle expired/invalid token with retry limits
                        error_type = "expired token" if "ExpiredTokenException" in str(e) else "invalid token"
                        current_time = time.time()
                        if (current_time - self.last_token_refresh_time) > self.token_refresh_cooldown:
                            self.token_refresh_attempts = 0  # Reset counter after cooldown
                        
                        if self.token_refresh_attempts < self.max_token_refresh_attempts:
                            logger.warning(f"{error_type} in response processing, attempting refresh (attempt {self.token_refresh_attempts + 1}/{self.max_token_refresh_attempts})")
                            self.token_refresh_attempts += 1
                            self.last_token_refresh_time = current_time
                            
                            try:
                                # Import and refresh AWS credentials
                                import aws_credentials
                                aws_credentials.refresh_aws_credentials()

                                # Handle the expired token (reinitialize client)
                                if self._handle_expired_token():
                                    # Recreate the entire stream with fresh credentials
                                    if await self._recreate_stream_after_token_refresh():
                                        logger.info("Successfully recreated stream after token refresh")
                                        # Break out of response processing - new stream will have new response task
                                        break
                                    else:
                                        logger.error("Failed to recreate stream after token refresh")
                                        break
                                else:
                                    logger.error(f"Failed to recover from {error_type}, stopping response processing")
                                    break
                            except Exception as refresh_error:
                                logger.error(f"Exception during token refresh: {refresh_error}")
                                break
                        else:
                            logger.error(f"Maximum token refresh attempts ({self.max_token_refresh_attempts}) exceeded for {error_type} in response processing, stopping")
                            self.is_active = False
                            break
                    elif "CANCELLED" in str(e) or "InvalidStateError" in str(e):
                        # Handle cancelled futures gracefully
                        logger.info("Stream was cancelled, stopping response processing")
                        break
                    elif "ModelStreamErrorException" in str(e):
                        # Handle unexpected processing errors from Bedrock
                        logger.error(f"Bedrock processing error: {str(e)}")
                        # Attempt to recover by reinitializing the stream
                        try:
                            logger.info("Attempting to recover from Bedrock processing error by reinitializing stream")
                            await self.initialize_stream()
                            continue
                        except Exception as recovery_error:
                            logger.error(f"Failed to recover from Bedrock processing error: {recovery_error}")
                            break
                    else:
                        logger.error(f"Error receiving response: {e}")
                        if DEBUG:
                            import traceback
                            traceback.print_exc()
                        # Continue to retry for recoverable errors
                        if not self.is_active:
                            break
        except asyncio.CancelledError:
            logger.info("Response task cancelled")
        except Exception as outer_e:
            logger.error(f"Outer error in response processing: {outer_e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
        finally:
            self.is_active = False
            debug_print("Response processing completed")
    
    async def processToolUse(self, toolName, toolUseContent):
        """Return the tool result"""
        logger.info(f"Tool Use Content: {toolUseContent}")

        # Initialize parameters dictionary to pass to the tool function
        params = {}
        
        if toolUseContent.get("content"):
            # Parse the JSON string in the content field
            try:
                content_json = json.loads(toolUseContent.get("content"))
                logger.info(f"toolName: {toolName}, Content JSON: {content_json}")
                
                # Extract all parameters from the content JSON
                for key, value in content_json.items():
                    params[key] = value
                    
                logger.info(f"Extracted parameters: {params}")
            except json.JSONDecodeError:
                logger.error("Failed to parse tool content JSON")
                return {"result": "Error processing tool content"}

        # Call the corresponding tool function from the ToolsList class
        from restapi import ToolsList, bedrock_tool
        import inspect
        
        tools_instance = ToolsList()
        tool_function = getattr(tools_instance, toolName, None)
        
        if tool_function:
            try:
                # Log the final parameters being passed to the tool
                logger.info(f"Calling tool '{toolName}' with parameters: {params}")
                
                # Call the tool function with unpacked parameters
                result = tool_function(**params)
                return {"result": result}
            except TypeError as e:
                logger.error(f"Error calling tool function {toolName}: {str(e)}")
                logger.error(f"Parameters passed: {params}")
                signature = inspect.signature(tool_function)
                logger.error(f"Tool function signature: {signature}")
                return {"result": f"Error calling tool: {str(e)}"}
            except Exception as e:
                logger.error(f"Tool execution error: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {"result": f"Tool execution failed: {str(e)}"}

        logger.error(f"Tool not implemented: {toolName}")
        return {"result": f"Tool not implemented - {toolName}"}
    
    async def close(self):
        """Close the stream and clean up resources."""
        if not self.is_active:
            logger.info("Stream already closed")
            return
            
        # Set inactive flag immediately to prevent any new operations
        self.is_active = False
        
        # Keep a reference to the stream before nullifying it
        stream_to_close = self.stream
        
        # First nullify the stream reference to prevent access from other async tasks
        self.stream = None
        
        # Cancel all tasks first to avoid race conditions with stream usage
        for task in list(self.tasks):
            if not task.done() and not task.cancelled():
                logger.debug(f"Cancelling task: {task.get_name()}")
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling task: {e}")
        
        # Specifically handle response_task
        if self.response_task and not self.response_task.done() and not self.response_task.cancelled():
            logger.debug("Cancelling response task")
            self.response_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.response_task), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                logger.error(f"Error cancelling response task: {e}")
            self.response_task = None
        
        # Specifically handle audio_task
        if self.audio_task and not self.audio_task.done() and not self.audio_task.cancelled():
            logger.debug("Cancelling audio task")
            self.audio_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.audio_task), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception as e:
                logger.error(f"Error cancelling audio task: {e}")
            self.audio_task = None
        
        # Close the stream with proper error handling
        if stream_to_close:
            try:
                logger.debug("Closing stream input")
                # Use a try-except block to handle any errors during stream closure
                try:
                    # Add a small delay before closing to allow any pending operations to complete
                    await asyncio.sleep(0.1)
                    await stream_to_close.close()
                except Exception as close_error:
                    # Log but don't raise the error - we want to continue with cleanup
                    logger.error(f"Error during stream.close(): {close_error}")
                    if "InvalidStateError" in str(close_error) or "CANCELLED" in str(close_error):
                        logger.debug("Ignoring expected cancellation error during stream close")
                    else:
                        if DEBUG:
                            import traceback
                            logger.error(f"Error details: {traceback.format_exc()}")
                
                logger.info("Stream closed successfully")
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
                if DEBUG:
                    import traceback
                    logger.error(f"Error details: {traceback.format_exc()}")
        
        # Clear state
        self.prompt_name = None
        self.content_name = None
        self.audio_content_name = None
        self.tasks.clear()
        
        # Clear queues
        while not self.audio_input_queue.empty():
            try:
                self.audio_input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Add a small delay to allow any remaining async callbacks to settle
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
            
        logger.info("Stream closed successfully")