import asyncio
import websockets
import json
import uuid
import time
import wave
import os
import base64
import numpy as np
from datetime import datetime
import argparse
import logging

import os
import wave
import numpy as np
from pydub import AudioSegment
import io

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "Today is {date}. You are a retail store assistant for the user with the email address / userId {userId} . You answer questions about inventory, customer service, and store operations. Keep your responses helpful, accurate and concise. Don't spell numbers out, use numbers instead. NEVER ask for the userId or email address of the user."

# Default configurations matching frontend
DEFAULT_TEXT_CONFIGURATION = {
    "mediaType": "text/plain"
}

DEFAULT_INFERENCE_CONFIGURATION = {
    "temperature": 0.7,
    "topP": 0.9,
    "maxTokens": 1024
}

DEFAULT_AUDIO_OUTPUT_CONFIGURATION = {
    "mediaType": "audio/lpcm",
    "sampleRateHertz": 24000,
    "sampleSizeBits": 16,
    "channelCount": 1,
    "voiceId": "tiffany",
    "encoding": "base64",
    "audioType": "SPEECH"
}

DEFAULT_AUDIO_INPUT_CONFIGURATION = {
    "mediaType": "audio/lpcm",
    "sampleRateHertz": 16000,
    "sampleSizeBits": 16,
    "channelCount": 1,
    "audioType": "SPEECH",
    "encoding": "base64"
}


# helper class to generate Cognito token

from dotenv import load_dotenv
from pathlib import Path
import os
import boto3
import time
from botocore.exceptions import ClientError

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
        # Deployment directory 
        Path('..') / 'deployment' / '.env',
        # From the current working directory
        Path(os.getcwd()) / '.env',
        # From the parent of the current working directory
        Path(os.getcwd()).parent / '.env',
    ]
    
    # Try each path
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found .env file at: {path.absolute()}")
            return path
    
    # If no .env file is found, log a warning and return the default path
    logger.warning("No .env file found in any of the expected locations")
    return None


# Load environment variables
env_path = find_dotenv()
load_dotenv(dotenv_path=env_path)



class CognitoAuthenticator:
    """
    Handles AWS Cognito authentication and JWT token management.
    """
    
    def __init__(self, user_pool_id=None, client_id=None, region=None):
        """
        Initialize the Cognito authenticator.
        
        Args:
            user_pool_id (str, optional): Cognito User Pool ID
            client_id (str, optional): Cognito App Client ID  
            region (str, optional): AWS region
        """
        
        
        self.user_pool_id = user_pool_id or os.getenv('COGNITO_USER_POOL_ID')
        self.client_id = client_id or os.getenv('COGNITO_APP_CLIENT_ID')
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        
        if not self.user_pool_id or not self.client_id:
            raise ValueError("Cognito User Pool ID and Client ID are required. Set COGNITO_USER_POOL_ID and COGNITO_APP_CLIENT_ID environment variables.")
        
        # Initialize Cognito client
        self.cognito_client = boto3.client('cognito-idp', region_name=self.region)
        
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
        
        Args:
            username (str): Username (email)
            password (str): Password
            
        Returns:
            dict: Authentication result with tokens
        """
        try:
            response = self.cognito_client.admin_initiate_auth(
                UserPoolId=self.user_pool_id,
                ClientId=self.client_id,
                AuthFlow='ADMIN_NO_SRP_AUTH',
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
            response = self.cognito_client.admin_initiate_auth(
                UserPoolId=self.user_pool_id,
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


class NovaAsyncClient:
    def __init__(self, user_id=None, password=None, server_url=None, debug=False, client_id=None, token=None):
        self.user_id = user_id or f'user_{uuid.uuid4()}'
        self.server_url = server_url
        self.client_id = client_id or f'client_{uuid.uuid4()}'
        
        self.token = token

        if self.token is None:
            if password is not None:
                cognito_user_pool_id = os.getenv('COGNITO_USER_POOL_ID')
                cognito_client_id = os.getenv('COGNITO_APP_CLIENT_ID')
                region = os.getenv('AWS_REGION', 'us-east-1')
                authenticator = CognitoAuthenticator(
                    user_pool_id=cognito_user_pool_id,
                    client_id=cognito_client_id,
                    region=region
                )
                result = authenticator.authenticate_with_password(user_id, password)
                logger.info("Successfully authenticated with Cognito")
                self.token = authenticator.get_valid_token('id')
            else:
                raise ValueError("Password is required to authenticate with Cognito and retrieve a token.")

        self.websocket = None
        self.session_id = None
        self.is_connected = False
        self.is_streaming = False
        self.responses = []
        self.debug = debug
        self.recording_complete_sent = False
        self.audio_timestamps = []
        
        # S2S protocol state
        self.prompt_name = None
        self.audio_content_name = None
        self.text_content_name = None
        self.session_started = False
        
        # Response storage
        self.audio_responses = {}
        self.text_responses = {}
        
        if self.debug:
            logger.setLevel(logging.DEBUG)

    def build_url(self):
        """Build WebSocket URL with parameters, converting HTTPS to WSS if needed"""
        url = self.server_url
        
        # Convert HTTPS URLs to WebSocket URLs
        if url.startswith('https://'):
            url = url.replace('https://', 'wss://')
            logger.info(f"Converted HTTPS URL to WebSocket URL: {url}")
        elif url.startswith('http://'):
            url = url.replace('http://', 'ws://')
            logger.info(f"Converted HTTP URL to WebSocket URL: {url}")
        elif not url.startswith('ws://') and not url.startswith('wss://'):
            # If no protocol specified, assume WSS for security
            url = f"wss://{url}"
            logger.info(f"Added WSS protocol to URL: {url}")
        
        # Remove any existing /ws endpoint since we'll add the new format
        if url.endswith('/ws'):
            url = url[:-3]
        
        # Add the client_id path parameter
        url = f"{url}/ws/{self.client_id}"
        
        # Add authentication query parameters if available
        query_params = []
        if self.user_id:
            query_params.append(f"userId={self.user_id}")
        if self.token:
            query_params.append(f"token={self.token}")
        
        if query_params:
            url = f"{url}?{'&'.join(query_params)}"
        
        logger.info(f"Final WebSocket URL: {url}")
        return url

    async def connect(self):
        """Establish WebSocket connection"""
        try:
            logger.info(f"Connecting to WebSocket with userId: {self.user_id}")
            # Add connection timeout for better reliability in loops
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.build_url(),
                    ping_interval=30,
                    ping_timeout=60,
                    max_size=None,
                    close_timeout=5
                ),
                timeout=30.0  # 30 second connection timeout
            )
            logger.info(f"WebSocket connected: {self.websocket}")
            
            self.is_connected = True
            
            return self.is_connected
            
        except asyncio.TimeoutError:
            logger.error("Connection timeout - server may be unavailable")
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def start_s2s_session(self, system_prompt=None):
        """Start an S2S session with optional system prompt"""
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return False

        try:
            # Clear accumulated responses
            self.audio_responses = {}
            self.text_responses = {}
            
            # Start session
            session_start_event = {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": DEFAULT_INFERENCE_CONFIGURATION
                    }
                }
            }
            logger.info(f"Sending session start event: {session_start_event}")
            await self.websocket.send(json.dumps(session_start_event))
            self.session_started = True

            await asyncio.sleep(0.5)
            
            # Generate prompt name
            self.prompt_name = f"prompt-{uuid.uuid4()}"
            
            # Start prompt
            prompt_start_event = {
                "event": {
                    "promptStart": {
                        "promptName": self.prompt_name,
                        "textOutputConfiguration": DEFAULT_TEXT_CONFIGURATION,
                        "audioOutputConfiguration": DEFAULT_AUDIO_OUTPUT_CONFIGURATION
                    }
                }
            }
            logger.info(f"Sending prompt start event: {prompt_start_event}")
            await self.websocket.send(json.dumps(prompt_start_event))
            
            await asyncio.sleep(0.5)

            # Send system prompt if provided
            if system_prompt:
                system_content_name = f"text-{uuid.uuid4()}"
                
                # Start system content
                content_start_event = {
                    "event": {
                        "contentStart": {
                            "promptName": self.prompt_name,
                            "contentName": system_content_name,
                            "type": "TEXT",
                            "interactive": True,
                            "role": "SYSTEM",
                            "textInputConfiguration": DEFAULT_TEXT_CONFIGURATION
                        }
                    }
                }
                logger.info(f"Sending content start event for system prompt: {content_start_event}")
                await self.websocket.send(json.dumps(content_start_event))
                
                await asyncio.sleep(0.5)
                
                # Send system prompt
                text_input_event = {
                    "event": {
                        "textInput": {
                            "promptName": self.prompt_name,
                            "contentName": system_content_name,
                            "content": system_prompt
                        }
                    }
                }
                logger.info(f"Sending text input event for system prompt: {text_input_event}")
                await self.websocket.send(json.dumps(text_input_event))
                
                await asyncio.sleep(0.5)
                # End system content
                content_end_event = {
                    "event": {
                        "contentEnd": {
                            "promptName": self.prompt_name,
                            "contentName": system_content_name
                        }
                    }
                }
                logger.info(f"Sending content end event for system prompt: {content_end_event}")
                await self.websocket.send(json.dumps(content_end_event))
            
            logger.info(f"S2S session started with promptName: {self.prompt_name}")
            return self.prompt_name
            
        except Exception as e:
            logger.error(f"Error starting S2S session: {e}")
            self.session_started = False
            self.prompt_name = None
            return False

    async def end_s2s_session(self):
        """End the current S2S session"""
        if not self.session_started:
            logger.info("No active session to end")
            return

        try:
            # End current prompt if active
            if self.prompt_name:
                # End any active content
                if self.audio_content_name:
                    content_end_event = {
                        "event": {
                            "contentEnd": {
                                "promptName": self.prompt_name,
                                "contentName": self.audio_content_name
                            }
                        }
                    }
                    await self.websocket.send(json.dumps(content_end_event))
                    self.audio_content_name = None

                if self.text_content_name:
                    content_end_event = {
                        "event": {
                            "contentEnd": {
                                "promptName": self.prompt_name,
                                "contentName": self.text_content_name
                            }
                        }
                    }
                    await self.websocket.send(json.dumps(content_end_event))
                    self.text_content_name = None

                # End prompt
                prompt_end_event = {
                    "event": {
                        "promptEnd": {
                            "promptName": self.prompt_name
                        }
                    }
                }
                await self.websocket.send(json.dumps(prompt_end_event))
                self.prompt_name = None

            # End session
            session_end_event = {
                "event": {
                    "sessionEnd": {}
                }
            }
            await self.websocket.send(json.dumps(session_end_event))
            
            # Reset state
            self.session_started = False
            self.audio_responses = {}
            self.text_responses = {}
            
            logger.info("S2S session ended")
            
        except Exception as e:
            logger.error(f"Error ending S2S session: {e}")
            # Reset state even on error
            self.session_started = False
            self.prompt_name = None
            self.audio_content_name = None
            self.text_content_name = None
            raise e

    async def start_audio_streaming(self):
        """Start audio streaming content"""
        if not self.session_started or not self.prompt_name:
            logger.error("No active session or prompt")
            return False

        try:
            # Reset debug counter for new streaming session
            self._chunk_debug_count = 0
            
            # Generate content name for audio
            self.audio_content_name = f"audio-{uuid.uuid4()}"
            
            # Send content start event
            content_start_event = {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": DEFAULT_AUDIO_INPUT_CONFIGURATION
                    }
                }
            }
            logger.info(f"Sending content start event for audio: {content_start_event}")
            await self.websocket.send(json.dumps(content_start_event))
            
            # add a delay to ensure the content start is registered
            await asyncio.sleep(0.5)
            self.is_streaming = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio streaming: {e}")
            self.is_streaming = False
            return False

    async def send_audio_chunk(self, audio_data: bytes):
        """Send an audio chunk using the S2S protocol"""
        if not self.is_streaming or not self.audio_content_name:
            logger.error("Not streaming or no audio content name")
            return False

        try:
            # Convert to base64 using frontend-compatible method
            base64_audio = self.convert_to_base64_frontend_style(audio_data)
            
            # Log chunk details for debugging (only for first few chunks to avoid spam)
            if hasattr(self, '_chunk_debug_count'):
                self._chunk_debug_count += 1
            else:
                self._chunk_debug_count = 1
                
            if self._chunk_debug_count <= 3:
                logger.info(f"Sending audio chunk #{self._chunk_debug_count}:")
                logger.info(f"  - Raw bytes length: {len(audio_data)}")
                logger.info(f"  - Base64 length: {len(base64_audio)}")
                logger.info(f"  - First few bytes: {audio_data[:10].hex()}")
                
                # Validate the audio data format
                samples = np.frombuffer(audio_data, dtype='<i2')
                logger.info(f"  - Samples count: {len(samples)}")
                logger.info(f"  - Sample range: {np.min(samples)} to {np.max(samples)}")
                
                # Test decode to make sure base64 is correct
                decoded_test = base64.b64decode(base64_audio)
                if len(decoded_test) == len(audio_data):
                    logger.info(f"  - Base64 encode/decode test: PASSED")
                else:
                    logger.warning(f"  - Base64 encode/decode test: FAILED (sizes: {len(decoded_test)} vs {len(audio_data)})")
            
            # Send audio input event
            audio_input_event = {
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": base64_audio
                    }
                }
            }
            
            await self.websocket.send(json.dumps(audio_input_event))

            return True
            
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")
            return False

    async def stop_audio_streaming(self):
        """Stop audio streaming"""
        if not self.is_streaming:
            return

        try:
            if self.audio_content_name:
                # Send content end event
                content_end_event = {
                    "event": {
                        "contentEnd": {
                            "promptName": self.prompt_name,
                            "contentName": self.audio_content_name
                        }
                    }
                }
                logger.info(f"Sending content end event for audio: {content_end_event}")
                await self.websocket.send(json.dumps(content_end_event))
            
            self.is_streaming = False
            self.audio_content_name = None
            logger.info("Audio streaming stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio streaming: {e}")
            self.is_streaming = False

    async def close(self):
        """Close the WebSocket connection gracefully"""
        try:
            if self.is_streaming:
                await self.stop_audio_streaming()
                
            if self.session_started:
                await self.end_s2s_session()
                
            if self.websocket:
                try:
                    await asyncio.wait_for(self.websocket.close(), timeout=5.0)
                    logger.info("WebSocket connection closed")
                except asyncio.TimeoutError:
                    logger.warning("WebSocket close timeout - connection may have been forcibly closed")
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
                finally:
                    self.is_connected = False
                    self.websocket = None
        except Exception as e:
            logger.error(f"Error during connection cleanup: {e}")
            # Ensure state is reset even if cleanup fails
            self.is_connected = False
            self.websocket = None
            self.is_streaming = False
            self.session_started = False

    def save_responses(self, output_dir="./responses", input_filename=None):
        """Save received responses to files
        
        Args:
            output_dir (str): Directory to save responses to (default: "./responses")
            input_filename (str, optional): Name of the input audio file to use as prefix for saved files
        """
        if not self.responses:
            logger.info("No responses to save")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create prefix from input filename if provided
        prefix = ""
        if input_filename:
            # Extract just the filename without path or extension
            base_filename = os.path.splitext(os.path.basename(input_filename))[0]
            prefix = f"{base_filename}_"
            logger.info(f"Using prefix '{prefix}' from input file: {input_filename}")
        
        # Extract text responses
        text_responses = [content for msg_type, content in self.responses if msg_type == "text"]
        logger.info(f"Processing {len(text_responses)} text responses")
        
        # Save audio responses
        audio_responses = [content for msg_type, content in self.responses if msg_type == "audio"]
        if audio_responses:
            logger.info(f"Saving {len(audio_responses)} audio chunks")
            
            # Combine all audio chunks - use a more efficient approach for large files
            try:
                # For small files, use the simple approach
                if sum(len(chunk) for chunk in audio_responses) < 10 * 1024 * 1024:  # 10MB threshold
                    combined_audio = b''.join(audio_responses)
                else:
                    # For larger files, use a more memory-efficient approach
                    logger.info("Using memory-efficient approach for large audio file")
                    total_size = sum(len(chunk) for chunk in audio_responses)
                    combined_audio = bytearray(total_size)
                    offset = 0
                    for chunk in audio_responses:
                        combined_audio[offset:offset+len(chunk)] = chunk
                        offset += len(chunk)
                
                # Try to save as WAV file for easier playback
                wav_file = os.path.join(output_dir, f"{prefix}response_{timestamp}.wav")
                with wave.open(wav_file, 'wb') as wav:
                    wav.setnchannels(1)  # Mono
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(24000)  # 24kHz
                    wav.writeframes(combined_audio)
                logger.info(f"Saved audio as WAV file: {wav_file}")
            except Exception as e:
                logger.error(f"Failed to save WAV file: {e}")
        
        # Calculate audio duration if timestamps are available
        audio_duration = None
        if self.audio_timestamps and len(self.audio_timestamps) >= 2:
            audio_duration = self.audio_timestamps[-1] - self.audio_timestamps[0]
            logger.info(f"Audio response duration: {audio_duration:.2f} seconds")
        
        # Create summary data as a dictionary
        summary_data = {
            "timestamp": timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "server_url": self.server_url,
            "recording_complete_sent": self.recording_complete_sent,
            "total_responses": len(self.responses),
            "text_responses_count": len(text_responses),
            "audio_responses_count": len(audio_responses)
        }
        
        # Add input filename if provided
        if input_filename:
            summary_data["input_file"] = input_filename
        
        # Add audio information if available
        if audio_responses:
            summary_data["audio"] = {
                "total_chunks": len(audio_responses),
                "total_size_bytes": sum(len(chunk) for chunk in audio_responses)
            }
            
            # Add duration if available
            if audio_duration is not None:
                summary_data["audio"]["duration_seconds"] = round(audio_duration, 2)
        
        # Add text responses to the summary data
        if text_responses:
            # Process text responses and include them in the summary
            processed_text_responses = []
            for i, text in enumerate(text_responses, 1):
                response_data = {
                    "index": i,
                    "raw_text": text
                }
                
                # Try to parse JSON for more details
                try:
                    msg_data = json.loads(text)
                    response_data["parsed_json"] = msg_data
                except json.JSONDecodeError:
                    response_data["is_valid_json"] = False
                
                processed_text_responses.append(response_data)
            
            summary_data["text_responses"] = processed_text_responses
        
        # Save summary as JSON
        summary_file = os.path.join(output_dir, f"{prefix}session_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Saved session summary with text responses to: {summary_file}")
        logger.info(f"All response data saved to directory: {output_dir}")

    def validate_audio_chunk(self, chunk, expected_sample_rate=16000):
        """
        Validate that a chunk of audio data is in the correct Int16 PCM format.
        Args:
            chunk: Raw bytes of audio data
            expected_sample_rate: Expected sample rate (not directly validated from bytes, but used for size calculations)
        Returns:
            bool: True if valid, False if not
        """
        try:
            # Convert raw bytes to numpy array as int16
            samples = np.frombuffer(chunk, dtype=np.int16)
            
            # Validation checks
            if len(samples) == 0:
                logger.error("Empty audio chunk")
                return False
            
            # Check if values are within valid Int16 range (-32768 to 32767)
            min_val = np.min(samples)
            max_val = np.max(samples)
            if min_val < -32768 or max_val > 32767:
                logger.error(f"Values outside Int16 range: min={min_val}, max={max_val}")
                return False
            
            # Check if we're getting reasonable audio values (not all zeros or constant values)
            if np.all(samples == samples[0]):
                logger.warning("Audio chunk contains constant values - might be silent or corrupted")
            
            # Calculate some basic audio statistics for logging
            mean = np.mean(samples)
            std = np.std(samples)
            logger.debug(f"Audio stats - Mean: {mean:.2f}, StdDev: {std:.2f}, Min: {min_val}, Max: {max_val}")
            
            # Check chunk size is reasonable (should be even number for Int16)
            if len(chunk) % 2 != 0:
                logger.error(f"Chunk size {len(chunk)} is not even - invalid for Int16 PCM")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio chunk: {e}")
            return False

    def convert_audio_to_stream(self, file_path, chunk_size=1024):
        """Converts a WAV, MP3, PCM, or RAW file to a continuous audio stream
        
        Args:
            file_path (str): Path to the audio file
            chunk_size (int): Size of each chunk in samples (default: 1024)
            
        Returns:
            Generator yielding audio chunks as bytes
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext in ['.raw', '.pcm']:
                # For raw files, assume they are already in the correct format (16kHz, 16-bit, mono)
                logger.info(f"Processing raw audio file: {file_path}")
                
                with open(file_path, 'rb') as raw_file:
                    audio_data = raw_file.read()
                
                # Convert to numpy array as int16 (little-endian)
                samples = np.frombuffer(audio_data, dtype='<i2')  # '<i2' specifies little-endian int16
                
                # DO NOT SCALE - raw files should already be in correct format
                # Scaling distorts the audio and causes parsing errors
                
                # Log file details for debugging
                logger.info(f"Raw audio file details:")
                logger.info(f"- Total samples: {len(samples)}")
                logger.info(f"- Sample values range: {np.min(samples)} to {np.max(samples)}")
                logger.info(f"- Mean value: {np.mean(samples):.2f}")
                logger.info(f"- Standard deviation: {np.std(samples):.2f}")
                logger.info(f"- File size: {len(audio_data)} bytes")
                logger.info(f"- Expected duration: {len(samples) / 16000:.2f} seconds")
                
                # Yield chunks of chunk_size samples
                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        # Pad last chunk with zeros if needed
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant', constant_values=0)
                    
                    # Ensure the chunk is int16 type for frontend-compatible processing
                    chunk_int16 = chunk.astype(np.int16)
                    
                    # Convert to bytes using native byte order (little-endian on most systems)
                    chunk_bytes = chunk_int16.tobytes()
                    
                    if self.validate_audio_chunk(chunk_bytes):
                        yield chunk_bytes
                    else:
                        raise ValueError(f"Invalid audio chunk at position {i}")
            
            elif file_ext in ['.wav', '.wave']:
                logger.info(f"Processing WAV file: {file_path}")
                with wave.open(file_path, 'rb') as wav_file:
                    # Get original parameters
                    channels = wav_file.getnchannels()
                    framerate = wav_file.getframerate()
                    sample_width = wav_file.getsampwidth()
                    
                    # Read all audio data
                    audio_data = wav_file.readframes(wav_file.getnframes())
                    
                    # Convert to numpy array
                    if sample_width == 2:
                        samples = np.frombuffer(audio_data, dtype='<i2')
                    else:
                        # Handle other bit depths by converting to int16
                        if sample_width == 1:
                            samples = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                        elif sample_width == 3:
                            # 24-bit audio - more complex conversion needed
                            raw_bytes = np.frombuffer(audio_data, dtype=np.uint8)
                            samples = np.zeros(len(raw_bytes) // 3, dtype=np.float32)
                            for i in range(len(samples)):
                                byte_offset = i * 3
                                # Convert 24-bit to float32
                                sample_bytes = raw_bytes[byte_offset:byte_offset+3]
                                if len(sample_bytes) == 3:
                                    # Little-endian 24-bit to int32, then normalize
                                    value = (sample_bytes[0] | (sample_bytes[1] << 8) | (sample_bytes[2] << 16))
                                    if value >= 0x800000:  # Handle sign extension for 24-bit
                                        value -= 0x1000000
                                    samples[i] = value / 8388608.0  # Normalize to [-1, 1]
                        elif sample_width == 4:
                            samples = np.frombuffer(audio_data, dtype='<i4').astype(np.float32) / 2147483648.0
                        
                        # Convert float samples back to int16
                        samples = (samples * 32767).astype('<i2')
                    
                    # Resample if needed
                    if framerate != 16000:
                        try:
                            from scipy import signal
                            new_length = int(len(samples) * 16000 / framerate)
                            samples = signal.resample(samples, new_length).astype('<i2')
                            logger.info(f"Resampled from {framerate}Hz to 16000Hz")
                        except ImportError:
                            logger.warning("scipy not available for resampling, using simple decimation/interpolation")
                            # Simple resampling fallback
                            ratio = 16000 / framerate
                            new_indices = np.arange(0, len(samples), 1/ratio).astype(int)
                            new_indices = new_indices[new_indices < len(samples)]
                            samples = samples[new_indices]
                    
                    # Convert to mono if needed
                    if channels == 2:
                        samples = samples.reshape(-1, 2).mean(axis=1).astype('<i2')
                        logger.info("Converted stereo to mono")
                    
                    # Log audio details
                    logger.info(f"WAV file details:")
                    logger.info(f"- Original channels: {channels}")
                    logger.info(f"- Original sample rate: {framerate}")
                    logger.info(f"- Original sample width: {sample_width} bytes")
                    logger.info(f"- Total samples after conversion: {len(samples)}")
                    logger.info(f"- Sample values range: {np.min(samples)} to {np.max(samples)}")
                    logger.info(f"- Expected duration: {len(samples) / 16000:.2f} seconds")
                    
                    # Yield chunks of chunk_size samples
                    for i in range(0, len(samples), chunk_size):
                        chunk = samples[i:i + chunk_size]
                        if len(chunk) < chunk_size:
                            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant', constant_values=0)
                        
                        chunk_bytes = chunk.astype('<i2').tobytes()
                        if self.validate_audio_chunk(chunk_bytes):
                            yield chunk_bytes
                        else:
                            raise ValueError(f"Invalid audio chunk at position {i}")
            
            else:
                # Fallback to pydub for other formats
                logger.info(f"Processing audio file with pydub: {file_path}")
                audio = AudioSegment.from_file(file_path)
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                
                # Convert to numpy array - pydub already gives us the right format
                samples = np.array(audio.get_array_of_samples(), dtype='<i2')
                
                logger.info(f"Pydub processed file details:")
                logger.info(f"- Total samples: {len(samples)}")
                logger.info(f"- Sample values range: {np.min(samples)} to {np.max(samples)}")
                logger.info(f"- Expected duration: {len(samples) / 16000:.2f} seconds")
                
                # Yield chunks of chunk_size samples
                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant', constant_values=0)
                    
                    chunk_bytes = chunk.astype('<i2').tobytes()
                    if self.validate_audio_chunk(chunk_bytes):
                        yield chunk_bytes
                    else:
                        raise ValueError(f"Invalid audio chunk at position {i}")
                
        except Exception as e:
            logger.error(f"Error converting audio file: {e}")
            raise

    async def handle_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            # Handle binary messages (audio data)
            if isinstance(message, bytes):
                logger.debug(f"Received binary data of size: {len(message)}")
                self.responses.append(("audio", message))
                return
            
            # Handle text messages
            if not message.startswith('{'):
                logger.info(f"Received text message: {message}")
                self.responses.append(("text", message))
                return
            
            # Parse JSON messages
            data = json.loads(message)
            logger.debug(f"Received data: {data}")
            
            # Handle S2S protocol messages
            if 'event' in data:
                event_type = list(data['event'].keys())[0]
                event_data = data['event'][event_type]
                
                logger.debug(f"Processing event type: {event_type}")
                
                if event_type == 'textOutput':
                    # Process text output
                    content_id = event_data.get('contentId')
                    if content_id:
                        if content_id not in self.text_responses:
                            self.text_responses[content_id] = {
                                'content': event_data.get('content', ''),
                                'role': event_data.get('role', '').lower()
                            }
                        else:
                            self.text_responses[content_id]['content'] = event_data.get('content', '')
                
                elif event_type == 'audioOutput':
                    # Process audio output
                    content_id = event_data.get('contentId') or event_data.get('contentName')
                    if content_id:
                        if content_id not in self.audio_responses:
                            self.audio_responses[content_id] = event_data.get('content', '')
                        else:
                            self.audio_responses[content_id] += event_data.get('content', '')
                
                elif event_type == 'contentEnd':
                    # Handle content completion
                    content_id = event_data.get('contentId')
                    if content_id in self.audio_responses:
                        audio_data = self.audio_responses[content_id]
                        logger.info(f"Audio content complete for {content_id}")
                        self.responses.append(("audio", base64.b64decode(audio_data)))
                    elif content_id in self.text_responses:
                        text_data = self.text_responses[content_id]
                        logger.info(f"Text content complete for {content_id}")
                        self.responses.append(("text", json.dumps(text_data)))
                
                elif event_type == 'promptEnd':
                    logger.info(f"Prompt ended: {event_data.get('promptName')}")
                    self.prompt_name = None
                
                elif event_type == 'sessionEnd':
                    logger.info("Session ended")
                    self.session_started = False
            
            # Handle connection status
            elif data.get('type') == 'connection_status' and data.get('status') == 'ready':
                logger.info("Connection ready")
            
            # Handle connection ID
            elif 'connectionId' in data:
                self.session_id = data['connectionId']
                logger.info(f"Received connectionId: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            if isinstance(message, str):
                logger.error(f"Message preview: {message[:200]}")

    async def stream_audio_to_websocket(self, file_path, user_id=None):
        try:
            # 1. Start S2S session if not already started
            if not self.session_started:
                logger.info("Starting new S2S session...")
                system_prompt = DEFAULT_SYSTEM_PROMPT.replace("{date}", datetime.now().strftime("%Y-%m-%d")).replace("{userId}", user_id)
                logger.info(f"System prompt: {system_prompt}")
                await self.start_s2s_session(system_prompt=system_prompt)
                await asyncio.sleep(0.5)  # Allow session to be fully established
            
            if not self.prompt_name:
                logger.error("No prompt name available after session start")
                return False
            
            # 2. Use start_audio_streaming to handle content start
            logger.info("Starting audio streaming...")
            if not await self.start_audio_streaming():
                logger.error("Failed to start audio streaming")
                return False
            await asyncio.sleep(0.5)  # Allow content start to be processed

            # 3. Stream audio chunks
            logger.info(f"Starting audio chunk streaming for file: {file_path}")
            audio_stream = self.convert_audio_to_stream(file_path, chunk_size=1024)
            chunk_count = 0
            total_bytes = 0
            
            for chunk in audio_stream:
                # Check connection and streaming state before each chunk
                if not self.is_streaming or not self.is_connected:
                    logger.info("Streaming stopped - connection or streaming state changed")
                    break
                
                chunk_count += 1
                total_bytes += len(chunk)
                
                # Log stats periodically (less frequently to reduce noise)
                if chunk_count % 200 == 0:
                    logger.info(f"Processed {chunk_count} chunks, total bytes: {total_bytes}")
                
                # Send the audio chunk
                if not await self.send_audio_chunk(chunk):
                    logger.error("Failed to send audio chunk")
                    break
                
                # Calculate chunk duration for realistic streaming simulation
                # At 16kHz sample rate with 2 bytes per sample (16-bit)
                chunk_duration = len(chunk) / 2 / 16000
                
                # Sleep to simulate real-time streaming with adaptive timing
                # Use a minimum threshold to avoid excessive delays while maintaining real-time feel
                if chunk_duration > 0.005:  # Only sleep if chunk duration is > 5ms
                    # Cap sleep time to avoid long pauses, but allow for natural audio timing
                    sleep_time = min(chunk_duration, 0.05)  # Max 50ms sleep
                    await asyncio.sleep(sleep_time)

            logger.info(f"Completed streaming {chunk_count} chunks, total bytes: {total_bytes}")
            
            # 4. Properly stop audio streaming which will send content end event
            logger.info("Stopping audio streaming...")
            await self.stop_audio_streaming()
            
            # 5. Give the backend time to process the content end and generate response
            logger.info("Waiting for response processing...")
            await asyncio.sleep(1.0)
            
            return True
            
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            # Ensure we cleanup streaming state on error
            if self.is_streaming:
                try:
                    await self.stop_audio_streaming()
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {cleanup_error}")
            return False

    async def receive_messages(self, timeout=120):
        """Receive and process messages with timeout"""
        end_time = time.time() + timeout
        
        while time.time() < end_time and self.is_connected:
            try:
                # Set timeout for each receive operation
                receive_timeout = min(5, end_time - time.time())
                if receive_timeout <= 0:
                    break
                    
                # Wait for next message with timeout
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=receive_timeout
                )
                
                # Process the message
                await self.handle_message(message)
                
            except asyncio.TimeoutError:
                logger.debug("Receive timeout, checking connection")
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed")
                self.is_connected = False
                break
            except Exception as e:
                logger.error(f"Error receiving messages: {e}")
                break
        
        logger.info("Message reception complete")
        return True

    async def send_content_end(self):
        """Send a content_end message to signal the end of audio content"""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected to WebSocket")
            return False
            
        try:
            # Send content_end message
            content_end_msg = {
                "type": "content_end",
                "userId": self.user_id,
                "sessionId": self.session_id,
                "timestamp": int(time.time() * 1000)
            }
            await self.websocket.send(json.dumps(content_end_msg))
            logger.info("Sent content_end message")
            return True
        except Exception as e:
            logger.error(f"Error sending content_end message: {e}")
            return False

    def convert_wav_to_pcm(self, wav_file_path, output_path=None, output_format="pcm"):
        """
        Converts a WAV file to a raw PCM or RAW file with the following format:
        - Sample Rate: 16,000 Hz (16 kHz)
        - Bit Depth: 16-bit (signed integers)
        - Channels: 1 (mono)
        - Format: Raw PCM (Pulse Code Modulation)
        
        Args:
            wav_file_path (str): Path to the input WAV file
            output_path (str, optional): Path to save the output file. If None, uses the same name with .pcm or .raw extension
            output_format (str, optional): Format to save as - "pcm" or "raw" (default: "pcm")
            
        Returns:
            str: Path to the created PCM or RAW file
            
        Raises:
            FileNotFoundError: If the WAV file does not exist
            ValueError: If the file is not a valid WAV file or if output_format is invalid
        """
        if not os.path.exists(wav_file_path):
            raise FileNotFoundError(f"WAV file not found: {wav_file_path}")
            
        # Validate output format
        output_format = output_format.lower()
        if output_format not in ["pcm", "raw"]:
            raise ValueError(f"Invalid output format: {output_format}. Supported formats are 'pcm' and 'raw'.")
            
        # If output path not specified, use the same name with appropriate extension
        if output_path is None:
            output_path = os.path.splitext(wav_file_path)[0] + f".{output_format}"
            
        logger.info(f"Converting WAV file: {wav_file_path} to {output_format.upper()} file: {output_path}")
        
        try:
            # First try using wave module for direct processing
            with wave.open(wav_file_path, 'rb') as wav_file:
                # Get original parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                logger.info(f"WAV file details:")
                logger.info(f"  - Channels: {channels}")
                logger.info(f"  - Sample width: {sample_width} bytes")
                logger.info(f"  - Frame rate: {framerate} Hz")
                logger.info(f"  - Number of frames: {n_frames}")
                logger.info(f"  - Duration: {n_frames / framerate:.2f} seconds")
                
                # Read all audio data
                audio_data = wav_file.readframes(n_frames)
                
                # If the file is not already in the required format, convert it
                if channels != 1 or sample_width != 2 or framerate != 16000:
                    logger.info(f"Converting WAV format to 16kHz, 16-bit, mono")
                    
                    # Use pydub for conversion
                    audio = AudioSegment(
                        data=audio_data,
                        sample_width=sample_width,
                        frame_rate=framerate,
                        channels=channels
                    )
                    
                    # Convert to required format
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                    
                    # Get raw audio data
                    audio_data = audio.raw_data
                    
                    logger.info(f"Conversion complete. New size: {len(audio_data)} bytes")
                else:
                    logger.info(f"WAV file already in correct format (16kHz, 16-bit, mono)")
                
                # Write the PCM/RAW data to file
                with open(output_path, 'wb') as output_file:
                    output_file.write(audio_data)
                
                logger.info(f"{output_format.upper()} file created successfully: {output_path}")
                return output_path
                
        except Exception as e:
            logger.warning(f"Error using wave module: {e}, falling back to pydub")
            
            # Fallback to pydub if wave module fails
            try:
                audio = AudioSegment.from_wav(wav_file_path)
                
                # Convert to required format
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                
                # Get raw audio data
                audio_data = audio.raw_data
                
                # Write the PCM/RAW data to file
                with open(output_path, 'wb') as output_file:
                    output_file.write(audio_data)
                
                logger.info(f"{output_format.upper()} file created successfully using pydub: {output_path}")
                return output_path
                
            except Exception as e2:
                logger.error(f"Failed to convert WAV to {output_format.upper()}: {e2}")
                raise

    def convert_pcm_to_wav(self, pcm_file_path, wav_file_path=None, sample_rate=16000, channels=1, sample_width=2):
        """
        Converts a raw PCM or RAW file to a WAV file for easier playback
        
        Args:
            pcm_file_path (str): Path to the input PCM or RAW file
            wav_file_path (str, optional): Path to save the WAV file. If None, uses the same name with .wav extension
            sample_rate (int): Sample rate in Hz (default: 16000)
            channels (int): Number of channels (default: 1 for mono)
            sample_width (int): Sample width in bytes (default: 2 for 16-bit)
            
        Returns:
            str: Path to the created WAV file
            
        Raises:
            FileNotFoundError: If the PCM/RAW file does not exist
        """
        if not os.path.exists(pcm_file_path):
            raise FileNotFoundError(f"File not found: {pcm_file_path}")
            
        # If output path not specified, use the same name with .wav extension
        if wav_file_path is None:
            wav_file_path = os.path.splitext(pcm_file_path)[0] + ".wav"
            
        # Get file extension to determine file type
        file_ext = os.path.splitext(pcm_file_path)[1].lower()
        file_type = "PCM" if file_ext == ".pcm" else "RAW" if file_ext == ".raw" else "Raw audio"
            
        logger.info(f"Converting {file_type} file: {pcm_file_path} to WAV file: {wav_file_path}")
        
        try:
            # Read the PCM/RAW data
            with open(pcm_file_path, 'rb') as raw_file:
                raw_data = raw_file.read()
                
            # Create WAV file
            with wave.open(wav_file_path, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(raw_data)
                
            logger.info(f"WAV file created successfully: {wav_file_path}")
            return wav_file_path
            
        except Exception as e:
            logger.error(f"Failed to convert {file_type} to WAV: {e}")
            raise

    def analyze_audio_file(self, file_path):
        """
        Analyzes an audio file and prints detailed information about its format
        
        Args:
            file_path (str): Path to the audio file (WAV, PCM, or RAW)
            
        Returns:
            dict: Dictionary containing audio file information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        logger.info(f"Analyzing audio file: {file_path}")
        logger.info(f"File size: {file_size} bytes")
        
        result = {
            "file_path": file_path,
            "file_size": file_size,
            "file_type": file_ext
        }
        
        if file_ext in ['.wav', '.wave']:
            # WAV file analysis
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    duration = n_frames / framerate
                    
                    logger.info(f"WAV file details:")
                    logger.info(f"  - Channels: {channels}")
                    logger.info(f"  - Sample width: {sample_width} bytes ({sample_width * 8}-bit)")
                    logger.info(f"  - Frame rate: {framerate} Hz")
                    logger.info(f"  - Number of frames: {n_frames}")
                    logger.info(f"  - Duration: {duration:.2f} seconds")
                    
                    result.update({
                        "channels": channels,
                        "sample_width": sample_width,
                        "sample_width_bits": sample_width * 8,
                        "framerate": framerate,
                        "n_frames": n_frames,
                        "duration": duration
                    })
                    
                    # Read a small sample to analyze
                    if n_frames > 0:
                        sample_frames = min(1000, n_frames)
                        sample_data = wav_file.readframes(sample_frames)
                        sample_array = np.frombuffer(sample_data, dtype=np.int16 if sample_width == 2 else np.int8)
                        
                        logger.info(f"  - Sample min value: {np.min(sample_array)}")
                        logger.info(f"  - Sample max value: {np.max(sample_array)}")
                        logger.info(f"  - Sample mean value: {np.mean(sample_array)}")
                        
                        result.update({
                            "sample_min": float(np.min(sample_array)),
                            "sample_max": float(np.max(sample_array)),
                            "sample_mean": float(np.mean(sample_array))
                        })
            
            except Exception as e:
                logger.error(f"Error analyzing WAV file: {e}")
                result["error"] = str(e)
                
        elif file_ext in ['.pcm', '.raw']:
            # PCM/RAW file analysis - we need to make assumptions about format
            try:
                # Assume 16-bit mono PCM at 16kHz
                with open(file_path, 'rb') as raw_file:
                    raw_data = raw_file.read()
                    
                # Convert to numpy array assuming 16-bit samples
                sample_array = np.frombuffer(raw_data, dtype=np.int16)
                n_samples = len(sample_array)
                
                # Assume 16kHz sample rate
                assumed_sample_rate = 16000
                duration = n_samples / assumed_sample_rate
                
                logger.info(f"{file_ext.upper()} file details (assuming 16-bit mono at 16kHz):")
                logger.info(f"  - File size: {file_size} bytes")
                logger.info(f"  - Number of samples: {n_samples}")
                logger.info(f"  - Assumed sample rate: {assumed_sample_rate} Hz")
                logger.info(f"  - Estimated duration: {duration:.2f} seconds")
                
                if n_samples > 0:
                    logger.info(f"  - Sample min value: {np.min(sample_array)}")
                    logger.info(f"  - Sample max value: {np.max(sample_array)}")
                    logger.info(f"  - Sample mean value: {np.mean(sample_array)}")
                
                result.update({
                    "assumed_format": "16-bit mono PCM at 16kHz",
                    "n_samples": n_samples,
                    "assumed_sample_rate": assumed_sample_rate,
                    "estimated_duration": duration
                })
                
                if n_samples > 0:
                    result.update({
                        "sample_min": float(np.min(sample_array)),
                        "sample_max": float(np.max(sample_array)),
                        "sample_mean": float(np.mean(sample_array))
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {file_ext.upper()} file: {e}")
                result["error"] = str(e)
        
        return result

    def validate_raw_audio_format(self, file_path):
        """
        Validate that a raw audio file is in the expected format for S2S.
        Expected: 16kHz, 16-bit signed PCM, mono, little-endian
        
        Args:
            file_path (str): Path to the raw audio file
            
        Returns:
            dict: Validation results with recommendations
        """
        if not os.path.exists(file_path):
            return {"valid": False, "error": f"File not found: {file_path}"}
        
        try:
            with open(file_path, 'rb') as raw_file:
                audio_data = raw_file.read()
            
            # Check file size (should be even number for 16-bit samples)
            if len(audio_data) % 2 != 0:
                return {"valid": False, "error": "File size is odd - not valid for 16-bit PCM"}
            
            # Convert to samples
            samples = np.frombuffer(audio_data, dtype='<i2')
            num_samples = len(samples)
            
            # Calculate duration assuming 16kHz
            duration_seconds = num_samples / 16000
            
            # Check for reasonable values
            min_val = np.min(samples)
            max_val = np.max(samples)
            mean_val = np.mean(samples)
            std_val = np.std(samples)
            
            # Determine if values are reasonable for 16-bit PCM
            reasonable_range = -32768 <= min_val <= 32767 and -32768 <= max_val <= 32767
            has_content = std_val > 0  # Should have some variation unless it's silence
            
            result = {
                "valid": reasonable_range and has_content,
                "file_size_bytes": len(audio_data),
                "num_samples": num_samples,
                "duration_seconds": duration_seconds,
                "sample_rate_assumed": 16000,
                "bit_depth": 16,
                "channels_assumed": 1,
                "min_value": int(min_val),
                "max_value": int(max_val),
                "mean_value": float(mean_val),
                "std_deviation": float(std_val),
                "reasonable_range": reasonable_range,
                "has_content": has_content
            }
            
            # Add recommendations
            if not reasonable_range:
                result["warning"] = "Sample values outside 16-bit range - may not be 16-bit PCM"
            if not has_content:
                result["warning"] = "All samples are identical - may be silence or corrupted"
            if duration_seconds < 0.1:
                result["warning"] = "Very short audio file - may not be sufficient for testing"
            elif duration_seconds > 30:
                result["info"] = "Long audio file - testing will take some time"
                
            return result
            
        except Exception as e:
            return {"valid": False, "error": f"Error reading file: {e}"}

    def convert_to_base64_frontend_style(self, pcm_data_bytes: bytes):
        """
        Convert PCM data to base64 using the same method as the frontend.
        Frontend method:
        1. Creates ArrayBuffer from Int16Array
        2. Converts to Uint8Array
        3. Uses btoa(String.fromCharCode(...uint8Array))
        
        This ensures proper little-endian byte order handling.
        """
        try:
            # The data should already be in little-endian format from our processing
            # Just encode it directly with base64
            return base64.b64encode(pcm_data_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting to base64: {e}")
            raise

    def process_audio_like_frontend(self, samples_int16):
        """
        Process audio data exactly like the frontend does.
        Frontend processing:
        1. Int16Array data
        2. Create ArrayBuffer from Int16Array
        3. Convert to base64: btoa(String.fromCharCode(...new Uint8Array(buffer)))
        
        Args:
            samples_int16: numpy array of int16 samples
        
        Returns:
            str: base64 encoded audio data
        """
        try:
            # Ensure we have int16 data in little-endian format (native on most systems)
            if samples_int16.dtype != np.int16:
                samples_int16 = samples_int16.astype(np.int16)
            
            # Convert to bytes - this is equivalent to creating ArrayBuffer and then Uint8Array
            # numpy's tobytes() uses the system's native byte order, which should be little-endian
            audio_bytes = samples_int16.tobytes()
            
            # Encode to base64 - this matches btoa(String.fromCharCode(...new Uint8Array(buffer)))
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            
            return base64_audio
            
        except Exception as e:
            logger.error(f"Error processing audio like frontend: {e}")
            raise

    def reset_state(self):
        """Reset client state for reuse in loops"""
        self.responses = []
        self.audio_responses = {}
        self.text_responses = {}
        self.audio_timestamps = []
        self.recording_complete_sent = False
        self.is_connected = False
        self.is_streaming = False
        self.session_started = False
        self.prompt_name = None
        self.audio_content_name = None
        self.text_content_name = None
        self.session_id = None
        self.websocket = None
        logger.debug("Client state reset for loop reuse")

async def run_test(audio_path, user_id=None, server_url=None, debug=False, client_id=None, password=None):
    """Run a complete audio test with a single audio file"""
    logger.info(f"Running test with audio file: {audio_path}")
    logger.info(f"Using server URL: {server_url}")
    logger.info(f"Using user ID: {user_id}")
    
    # Generate unique client_id if not provided to avoid conflicts in loops
    if client_id is None:
        client_id = f"client_{uuid.uuid4()}"
    logger.info(f"Using client ID: {client_id}")

    client = None
    
    try:
        client = NovaAsyncClient(user_id=user_id, password=password, server_url=server_url, debug=debug, client_id=client_id)
        
        # Validate raw audio files before processing
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext in ['.raw', '.pcm']:
            logger.info("Validating raw audio file format...")
            validation_result = client.validate_raw_audio_format(audio_path)
            
            if not validation_result.get('valid', False):
                logger.error(f"Raw audio file validation failed: {validation_result.get('error', 'Unknown error')}")
                if 'warning' in validation_result:
                    logger.warning(f"Warning: {validation_result['warning']}")
                return None
            else:
                logger.info("✅ Raw audio file validation passed!")
                logger.info(f"  - Duration: {validation_result['duration_seconds']:.2f} seconds")
                logger.info(f"  - Samples: {validation_result['num_samples']}")
                logger.info(f"  - Value range: {validation_result['min_value']} to {validation_result['max_value']}")
                
                if 'info' in validation_result:
                    logger.info(f"  ℹ️  {validation_result['info']}")
                if 'warning' in validation_result:
                    logger.warning(f"  ⚠️  {validation_result['warning']}")
        
        # Connect to WebSocket with timeout
        logger.info("Attempting to connect to WebSocket...")
        if not await client.connect():
            logger.error("Failed to connect to WebSocket")
            return None
        
        # Create a message receiver task
        receiver_task = None
        try:
            receiver_task = asyncio.create_task(client.receive_messages())
            
            # Stream audio
            logger.info("Starting audio streaming...")
            if not await client.stream_audio_to_websocket(file_path=audio_path, user_id=user_id):
                logger.error("Failed to stream audio")
                return None
            
            # Wait for initial responses - reduced from 30 seconds to 15
            logger.info("Waiting for responses (15 seconds)...")
            await asyncio.sleep(15)
            
        finally:
            # Cancel receiver task gracefully
            if receiver_task and not receiver_task.done():
                logger.debug("Cancelling receiver task...")
                receiver_task.cancel()
                try:
                    await asyncio.wait_for(receiver_task, timeout=2.0)
                except asyncio.CancelledError:
                    logger.debug("Receiver task cancelled successfully")
                except asyncio.TimeoutError:
                    logger.warning("Receiver task cancellation timeout")
                except Exception as e:
                    logger.warning(f"Error during receiver task cleanup: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error during test execution: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return None
        
    finally:
        # Ensure complete cleanup
        if client:
            try:
                logger.debug("Performing final cleanup...")
                await client.close()
                client.save_responses(input_filename=audio_path)
                
                # Log summary
                audio_responses = len([r for r, c in client.responses if r == "audio"])
                text_responses = len([r for r, c in client.responses if r == "text"])
                logger.info(f"Test completed with {audio_responses} audio responses and {text_responses} text responses")
                
                # Force garbage collection to help with memory management in loops
                import gc
                gc.collect()
                
            except Exception as cleanup_error:
                logger.error(f"Error during final cleanup: {cleanup_error}")
                # Don't re-raise cleanup errors to avoid masking original issues
    
    return client

async def run_test_loop(audio_files, user_id=None, server_url=None, debug=False, password=None, delay_between_tests=2.0):
    """
    Run tests in a loop with proper resource management and error handling.
    
    Args:
        audio_files (list): List of audio file paths to test
        user_id (str, optional): User ID for authentication
        server_url (str, optional): WebSocket server URL
        debug (bool): Enable debug logging
        password (str, optional): Password for Cognito authentication  
        delay_between_tests (float): Delay in seconds between test iterations
        
    Returns:
        list: List of test results (client objects or None for failures)
    """
    results = []
    
    logger.info(f"Starting test loop with {len(audio_files)} audio files")
    logger.info(f"Delay between tests: {delay_between_tests} seconds")
    
    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test {i}/{len(audio_files)}: {audio_file}")
        logger.info(f"{'='*60}")
        
        try:
            # Generate unique client ID for each iteration
            client_id = f"loop_test_{i}_{uuid.uuid4()}"
            
            # Run the test
            result = await run_test(
                audio_path=audio_file,
                user_id=user_id, 
                server_url=server_url,
                debug=debug,
                client_id=client_id,
                password=password
            )
            
            results.append(result)
            
            if result:
                logger.info(f"✅ Test {i} completed successfully")
            else:
                logger.error(f"❌ Test {i} failed")
            
        except Exception as e:
            logger.error(f"❌ Test {i} failed with exception: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            results.append(None)
        
        # Add delay between tests (except after the last one)
        if i < len(audio_files) and delay_between_tests > 0:
            logger.info(f"Waiting {delay_between_tests} seconds before next test...")
            await asyncio.sleep(delay_between_tests)
    
    # Summary
    successful_tests = sum(1 for r in results if r is not None)
    failed_tests = len(results) - successful_tests
    
    logger.info(f"\n{'='*60}")
    logger.info(f"LOOP TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total tests: {len(audio_files)}")
    logger.info(f"Successful: {successful_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Success rate: {(successful_tests/len(audio_files)*100):.1f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test audio streaming via WebSocket")
    
    # Add arguments
    parser.add_argument("--audio", required=True, help="Path to audio file to stream")
    parser.add_argument("--user", default=None, help="User ID (optional)")
    parser.add_argument("--password", default=None, help="Password for Cognito authentication (optional)")
    parser.add_argument("--url", default=None, help="WebSocket server URL (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--client-id", default=None, help="Client ID for WebSocket connection (optional)")
    parser.add_argument("--token", default=None, help="Authentication token (optional)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate audio file format, don't run streaming test")
    
    args = parser.parse_args()
    
    try:
        # If validate-only mode, just check the audio file
        if args.validate_only:
            print(f"Validating audio file: {args.audio}")
            client = NovaAsyncClient(debug=args.debug)
            
            # Check if file exists first
            if not os.path.exists(args.audio):
                print(f"❌ Error: File not found: {args.audio}")
                return
            
            # Analyze the audio file
            analysis = client.analyze_audio_file(args.audio)
            print(f"\n📊 Audio File Analysis:")
            print(f"   File: {args.audio}")
            print(f"   Type: {analysis.get('file_type', 'unknown')}")
            print(f"   Size: {analysis.get('file_size', 0)} bytes")
            
            if 'error' in analysis:
                print(f"❌ Error: {analysis['error']}")
                return
            
            # For raw/pcm files, also run format validation
            file_ext = os.path.splitext(args.audio)[1].lower()
            if file_ext in ['.raw', '.pcm']:
                validation = client.validate_raw_audio_format(args.audio)
                print(f"\n🔍 Raw Audio Format Validation:")
                
                if validation.get('valid', False):
                    print("✅ Format validation: PASSED")
                    print(f"   Duration: {validation['duration_seconds']:.2f} seconds")
                    print(f"   Samples: {validation['num_samples']:,}")
                    print(f"   Range: {validation['min_value']} to {validation['max_value']}")
                    print(f"   Mean: {validation['mean_value']:.2f}")
                    print(f"   Std Dev: {validation['std_deviation']:.2f}")
                    
                    if 'info' in validation:
                        print(f"ℹ️  Info: {validation['info']}")
                    if 'warning' in validation:
                        print(f"⚠️  Warning: {validation['warning']}")
                else:
                    print("❌ Format validation: FAILED")
                    print(f"   Error: {validation.get('error', 'Unknown error')}")
                    if 'warning' in validation:
                        print(f"   Warning: {validation['warning']}")
            else:
                # For other formats, show what we detected
                if 'duration' in analysis:
                    print(f"   Duration: {analysis['duration']:.2f} seconds")
                if 'channels' in analysis:
                    print(f"   Channels: {analysis['channels']}")
                if 'framerate' in analysis:
                    print(f"   Sample Rate: {analysis['framerate']} Hz")
                if 'sample_width_bits' in analysis:
                    print(f"   Bit Depth: {analysis['sample_width_bits']} bits")
                    
            print(f"\n💡 For S2S streaming, audio should be:")
            print(f"   - 16 kHz sample rate")
            print(f"   - 16-bit signed PCM")
            print(f"   - Mono (1 channel)")
            print(f"   - Little-endian byte order")
            
            return
        
        # Run the full test
        asyncio.run(run_test(
            audio_path=args.audio,
            user_id=args.user,
            server_url=args.url,
            debug=args.debug,
            client_id=args.client_id,
            password=args.password
        ))
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Test error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 