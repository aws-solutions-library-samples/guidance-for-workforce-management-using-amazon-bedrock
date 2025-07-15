"""
Enhanced LLM as a Judge Evaluation for Speech-to-Speech (S2S) Tests

This script implements an LLM-as-a-Judge evaluation approach for assessing the quality
of speech-to-speech responses from the Nova Sonic API. It evaluates responses based on
criteria defined in the configuration file.

Enhancements:
- AWS Transcribe integration for audio response verification
- Expected response validation from dataset
- Objective metrics for response accuracy

The script uses:
- A config JSON file for LLM judge model, evaluation criteria, and prompt template
- A validation dataset JSONL file for expected function calls, transcriptions, and responses
- AWS Transcribe for audio transcription

Usage:
    python llm_judge_s2s_eval.py --responses_file <path_to_responses_jsonl>
                                 [--config_file <path_to_config_json>]
                                 [--validation_dataset <path_to_validation_jsonl>]
"""

import argparse
import json
import os
import pandas as pd
import boto3
import sys
import time
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, ClassVar, Type
import logging
from pathlib import Path
from dotenv import load_dotenv
import wave
import numpy as np
from pydub import AudioSegment
from difflib import SequenceMatcher
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Default paths for configuration and validation dataset
DEFAULT_CONFIG_PATH = "config/llm_judge_s2s_config.json"
DEFAULT_VALIDATION_DATASET_PATH = "data/s2s_validation_dataset.jsonl"

# Nova Sonic pricing information (USD per 1000 tokens)
NOVA_SONIC_PRICING = {
    "speech_input": 0.0034,  # $0.0034 per 1000 speech input tokens
    "speech_output": 0.0136,  # $0.0136 per 1000 speech output tokens
    "text_input": 0.00006,    # $0.00006 per 1000 text input tokens
    "text_output": 0.00024    # $0.00024 per 1000 text output tokens
}

# Pydantic models for data validation
class JudgeModelConfig(BaseModel):
    """Configuration for the judge model"""
    id: str = Field("anthropic.claude-3-sonnet-20240229-v1:0", description="Model ID to use for evaluation")
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.0, description="Temperature for model generation")

class ConfigModel(BaseModel):
    """Configuration for the evaluation script"""
    judge_model: JudgeModelConfig
    evaluation_criteria: Dict[str, str] = Field(..., description="Evaluation criteria with descriptions")
    prompt_template: str = Field(..., description="Prompt template for the judge model")


class AudioFeatures(BaseModel):
    """Audio features extracted from a WAV file"""
    duration_seconds: float
    channels: int
    sample_width: int
    frame_rate: int
    max_amplitude: float
    rms: float
    error: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def check_error(cls, data):
        """Check if there's an error and return a minimal valid model"""
        if isinstance(data, dict) and "error" in data:
            return {
                "error": data["error"],
                "duration_seconds": 0.0,
                "channels": 0,
                "sample_width": 0,
                "frame_rate": 0,
                "max_amplitude": 0.0,
                "rms": 0.0
            }
        return data

class TranscriptionResult(BaseModel):
    """Transcription result"""
    user_input: Optional[str] = None
    assistant_response: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def check_error(cls, values):
        """Check if there's an error and return a minimal valid model"""
        if isinstance(values, dict) and "error" in values:
            return {"error": values["error"], "transcript": None, "confidence": None}
        return values

class CriterionScore(BaseModel):
    """Score and explanation for a single criterion"""
    score: int = Field(..., ge=0, le=5, description="Score from 0-5")
    explanation: str = Field(..., description="Explanation for the score")

class EvaluationCriteria(BaseModel):
    """Evaluation criteria scores"""
    speech_recognition: Optional[CriterionScore] = None
    response_relevance: Optional[CriterionScore] = None
    response_correctness: Optional[CriterionScore] = None
    function_call_accuracy: Optional[CriterionScore] = None
    speech_synthesis: Optional[CriterionScore] = None

class OverallEvaluation(BaseModel):
    """Overall evaluation score and summary"""
    score: int = Field(..., ge=0, le=5, description="Overall score from 0-5")
    summary: str = Field(..., description="Summary of the evaluation")

class ObjectiveMetrics(BaseModel):
    """Objective metrics for evaluation"""
    response_similarity: Optional[float] = None
    transcription_similarity: Optional[float] = None
    duration_seconds: Optional[float] = None
    cost_usd: Optional[float] = None

class TokenUsage(BaseModel):
    """Token usage information"""
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    input_speech_tokens: Optional[int] = None
    input_text_tokens: Optional[int] = None
    output_speech_tokens: Optional[int] = None
    output_text_tokens: Optional[int] = None

class FunctionCallInfo(BaseModel):
    """Information about a function call"""
    function_name: str = Field(..., description="Name of the function called")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters passed to the function")
    timestamp: Optional[datetime] = Field(None, description="Timestamp when the function was called")
    raw_message: Optional[str] = Field(None, description="Raw log message")

class CloudWatchLogsRetriever:
    """
    Retrieves and parses CloudWatch logs to extract function calls and other relevant information.
    """
    
    def __init__(self, region=None, log_group_name=None):
        """
        Initialize the CloudWatch logs retriever.
        
        Args:
            region (str, optional): AWS region
            log_group_name (str, optional): CloudWatch log group name
        """
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        self.log_group_name = log_group_name or os.getenv('CLOUDWATCH_LOG_GROUP', '/aws/lambda/retail-assistant-function')
        
        # Initialize CloudWatch logs client
        self.logs_client = boto3.client('logs', region_name=self.region)
        
        logger.info(f"Initialized CloudWatch logs retriever for region: {self.region}")
        logger.info(f"Log group name: {self.log_group_name}")
    
    def get_log_streams(self, prefix=None, limit=50):
        """
        Get log streams from the specified log group.
        
        Args:
            prefix (str, optional): Filter log streams by prefix
            limit (int): Maximum number of log streams to return
            
        Returns:
            list: List of log stream names
        """
        try:
            params = {
                'logGroupName': self.log_group_name,
                'descending': True,
                'orderBy': 'LastEventTime',
                'limit': limit
            }
            
            if prefix:
                params['logStreamNamePrefix'] = prefix
                
            response = self.logs_client.describe_log_streams(**params)
            
            log_streams = [stream['logStreamName'] for stream in response.get('logStreams', [])]
            logger.info(f"Found {len(log_streams)} log streams")
            
            return log_streams
            
        except Exception as e:
            logger.error(f"Error getting log streams: {e}")
            return []
    
    def get_log_events(self, log_stream_name, start_time=None, end_time=None, limit=1000):
        """
        Get log events from a specific log stream.
        
        Args:
            log_stream_name (str): Name of the log stream
            start_time (int, optional): Start time in milliseconds since epoch
            end_time (int, optional): End time in milliseconds since epoch
            limit (int): Maximum number of log events to return
            
        Returns:
            list: List of log events
        """
        try:
            params = {
                'logGroupName': self.log_group_name,
                'logStreamName': log_stream_name,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
                
            response = self.logs_client.get_log_events(**params)
            
            events = response.get('events', [])
            logger.info(f"Retrieved {len(events)} log events from {log_stream_name}")
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting log events: {e}")
            return []
    
    def search_logs(self, query, start_time=None, end_time=None, limit=100):
        """
        Search logs using CloudWatch Logs Insights.
        
        Args:
            query (str): CloudWatch Logs Insights query
            start_time (datetime, optional): Start time
            end_time (datetime, optional): End time
            limit (int): Maximum number of results to return
            
        Returns:
            dict: Query results
        """
        try:
            # Convert datetime objects to milliseconds since epoch
            if start_time is None:
                start_time = datetime.now() - timedelta(hours=1)
            if end_time is None:
                end_time = datetime.now()
                
            start_time_ms = int(start_time.timestamp() * 1000)
            end_time_ms = int(end_time.timestamp() * 1000)
            
            # Start query
            start_query_response = self.logs_client.start_query(
                logGroupName=self.log_group_name,
                startTime=start_time_ms,
                endTime=end_time_ms,
                queryString=query,
                limit=limit
            )
            
            query_id = start_query_response['queryId']
            
            # Wait for query to complete
            response = None
            while response is None or response['status'] == 'Running':
                logger.info("Waiting for query to complete...")
                time.sleep(1)
                response = self.logs_client.get_query_results(queryId=query_id)
                
            logger.info(f"Query completed with status: {response['status']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error searching logs: {e}")
            return {'status': 'Failed', 'results': [], 'error': str(e)}
    
    def extract_function_calls(self, start_time=None, end_time=None, user_id=None, session_id=None):
        """
        Extract function calls from CloudWatch logs.
        
        Args:
            start_time (datetime, optional): Start time
            end_time (datetime, optional): End time
            user_id (str, optional): Filter by user ID
            session_id (str, optional): Filter by session ID
            
        Returns:
            list: List of function calls with details
        """
        try:
            # Build query
            query = "fields @timestamp, @message | sort @timestamp desc"
            
            # Add filters
            filters = []
            if user_id:
                filters.append(f"@message like '{user_id}'")
            if session_id:
                filters.append(f"@message like '{session_id}'")
                
            # Add function call filter
            filters.append("@message like 'Invoking function' or @message like 'Function call' or @message like 'Lambda function'")
            
            if filters:
                query += " | filter " + " and ".join(filters)
                
            # Execute query
            response = self.search_logs(query, start_time, end_time)
            
            if response['status'] != 'Complete':
                logger.error(f"Query failed with status: {response['status']}")
                return []
                
            # Parse results to extract function calls
            function_calls = []
            for result in response.get('results', []):
                # Convert result to dict for easier access
                result_dict = {field['field']: field.get('value', '') for field in result}
                
                # Extract timestamp
                timestamp_str = result_dict.get('@timestamp', '')
                timestamp = datetime.fromtimestamp(int(timestamp_str) / 1000) if timestamp_str else None
                
                # Extract message
                message = result_dict.get('@message', '')
                
                # Parse function name and parameters
                function_name = None
                parameters = {}
                
                # Different log formats
                if 'Invoking function' in message:
                    match = re.search(r'Invoking function: (\w+)', message)
                    if match:
                        function_name = match.group(1)
                        
                    # Try to extract parameters as JSON
                    param_match = re.search(r'with parameters: ({.*})', message)
                    if param_match:
                        try:
                            parameters = json.loads(param_match.group(1))
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse parameters as JSON: {param_match.group(1)}")
                
                elif 'Function call' in message:
                    match = re.search(r'Function call: (\w+)', message)
                    if match:
                        function_name = match.group(1)
                        
                    # Try to extract parameters
                    param_match = re.search(r'args: ({.*})', message)
                    if param_match:
                        try:
                            parameters = json.loads(param_match.group(1))
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse parameters as JSON: {param_match.group(1)}")
                
                # Add to function calls if we found a function name
                if function_name:
                    function_calls.append(
                        FunctionCallInfo(
                            function_name=function_name,
                            parameters=parameters,
                            timestamp=timestamp,
                            raw_message=message
                        ).model_dump()
                    )
            
            logger.info(f"Extracted {len(function_calls)} function calls")
            return function_calls
            
        except Exception as e:
            logger.error(f"Error extracting function calls: {e}")
            return []

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
        # Deployment directory (from project root)
        Path('deployment') / '.env',
        # Deployment directory (from tests directory)
        Path('..') / 'deployment' / '.env',
        # From the current working directory
        Path(os.getcwd()) / '.env',
        # From the current working directory's deployment subdirectory
        Path(os.getcwd()) / 'deployment' / '.env',
        # From the parent of the current working directory
        Path(os.getcwd()).parent / '.env',
        # From the parent of the current working directory's deployment subdirectory
        Path(os.getcwd()).parent / 'deployment' / '.env',
    ]
    
    # Try each path
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found .env file at: {path.absolute()}")
            return path
    
    # If no .env file is found, log a warning and return the default path
    logger.warning("No .env file found in any of the expected locations")
    return None

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file and validate with Pydantic.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Dictionary containing validated configuration
    """
    default_config = {
        "judge_model": {
            "id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "max_tokens": 1024,
            "temperature": 0.0
        },
        "evaluation_criteria": {
            "speech_recognition": "How accurately did the system recognize the spoken input?",
            "response_relevance": "How relevant is the response to the recognized input?",
            "response_correctness": "How accurate is the information provided in the response?"
        },
        "prompt_template": "You are an expert evaluator for speech-to-speech AI interactions..."
            }
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Validate with Pydantic
        try:
            config_model = ConfigModel(**config_data)
            logger.info(f"Loaded and validated configuration from {config_path}")
            return config_model.model_dump()
        except Exception as validation_error:
            logger.error(f"Configuration validation error: {validation_error}")
            logger.warning("Using default configuration")
            return default_config
            
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.warning("Using default configuration")
        return default_config

def process_responses_file(responses_file: str) -> List[Dict[str, Any]]:
    """
    Process a JSONL file containing response data.
    
    Args:
        responses_file: Path to the JSONL file containing response data
        
    Returns:
        List of dictionaries with input_file_path, text_responses, and other response data
    """
    responses = []
    
    # Check if file exists
    if not os.path.isfile(responses_file):
        logger.error(f"Responses file not found: {responses_file}")
        return responses
    
    # Get the directory containing the responses file
    responses_dir = os.path.dirname(responses_file)
    
    # Read JSONL file
    try:
        with open(responses_file, 'r') as f:
            line_count = 0
            for line in f:
                line_count += 1
                try:
                    record = json.loads(line.strip())
                    
                    # Extract needed fields
                    input_file_path = record.get('input_file_path', '')
                    text_responses = record.get('text_responses', '')
                    output_file_path = record.get('output_file_path')
                    duration_seconds = record.get('duration_seconds')
                    run = record.get('run')
                    
                    # Extract token usage
                    speech_tokens_input = record.get('speech_tokens_input', 0)
                    text_tokens_input = record.get('text_tokens_input', 0)
                    speech_tokens_output = record.get('speech_tokens_output', 0)
                    text_tokens_output = record.get('text_tokens_output', 0)
                    total_tokens = record.get('total_tokens', 0)
                    
                    # Check if output file path is valid
                    if output_file_path and not os.path.isabs(output_file_path):
                        # If path is relative, make it absolute
                        if os.path.exists(output_file_path):
                            output_file_path = os.path.abspath(output_file_path)
                        elif os.path.exists(os.path.join(responses_dir, os.path.basename(output_file_path))):
                            # Try finding the file by basename in the responses directory
                            output_file_path = os.path.join(responses_dir, os.path.basename(output_file_path))
                        else:
                            # File not found, set to None
                            logger.warning(f"Output file not found: {output_file_path}")
                            output_file_path = None
                    
                    # Create token usage dictionary
                    token_usage = {
                        'total_input_tokens': speech_tokens_input + text_tokens_input,
                        'total_output_tokens': speech_tokens_output + text_tokens_output,
                        'total_tokens': total_tokens,
                        'input_speech_tokens': speech_tokens_input,
                        'input_text_tokens': text_tokens_input,
                        'output_speech_tokens': speech_tokens_output,
                        'output_text_tokens': text_tokens_output
                    }
                    
                    # Create response data dictionary
                    response_data = {
                        'input_file_path': input_file_path,
                        'text_responses': text_responses,
                        'output_file_path': output_file_path,
                        'duration_seconds': duration_seconds,
                        'token_usage': token_usage,
                        'run': run
                    }
                    
                    # Add to responses list
                    responses.append(response_data)
                    
                    logger.info(f"Processed record {line_count}")
                    logger.info(f"  - Input file: {input_file_path}")
                    logger.info(f"  - Output file: {output_file_path}")
                    logger.info(f"  - Text response length: {len(text_responses)} characters")
                    logger.info(f"  - Duration: {duration_seconds} seconds")
                    logger.info(f"  - Token usage: {total_tokens} tokens total")
                    if run is not None:
                        logger.info(f"  - Run: {run}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON in line {line_count}: {e}")
                except Exception as e:
                    logger.error(f"Error processing record in line {line_count}: {e}")
                    
    except Exception as e:
        logger.error(f"Error reading responses file {responses_file}: {e}")
    
    logger.info(f"Processed {len(responses)} records from {responses_file}")
    return responses

def load_validation_dataset(dataset_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load validation dataset from a JSONL file.
    
    Args:
        dataset_path: Path to the validation dataset JSONL file
        
    Returns:
        Dictionary mapping audio file names to transcriptions, responses, and categories
    """
    validation_data = {}
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                audio_file = entry.get('audio_file')
                if audio_file:
                    validation_data[audio_file] = {
                        'expected_user_input': entry.get('expected_transcription', 'unknown'),
                        'expected_response': entry.get('expected_response', ''),
                        'category': entry.get('category', 'Unknown')
                    }
        logger.info(f"Loaded validation dataset from {dataset_path} with {len(validation_data)} entries")
        return validation_data
    except Exception as e:
        logger.error(f"Error loading validation dataset from {dataset_path}: {e}")
        
        raise e

def extract_audio_features(audio_file_path: str) -> Dict[str, Any]:
    """
    Extract basic audio features from a WAV file and validate with Pydantic.
    
    Args:
        audio_file_path: Path to the WAV file
        
    Returns:
        Dictionary of validated audio features
    """
    try:
        # Validate input
        if not audio_file_path or not isinstance(audio_file_path, str):
            logger.error(f"Invalid audio file path: {audio_file_path}")
            return AudioFeatures(
                error="Invalid file path",
                duration_seconds=0.0,
                channels=0,
                sample_width=0,
                frame_rate=0,
                max_amplitude=0.0,
                rms=0.0
            ).model_dump()
            
        # Check if file exists
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return AudioFeatures(
                error="File not found",
                duration_seconds=0.0,
                channels=0,
                sample_width=0,
                frame_rate=0,
                max_amplitude=0.0,
                rms=0.0
            ).model_dump()
        
        # Check if file is readable
        if not os.access(audio_file_path, os.R_OK):
            logger.error(f"Audio file not readable: {audio_file_path}")
            return AudioFeatures(
                error="File not readable",
                duration_seconds=0.0,
                channels=0,
                sample_width=0,
                frame_rate=0,
                max_amplitude=0.0,
                rms=0.0
            ).model_dump()
            
        # Load audio file
        try:
            audio = AudioSegment.from_file(audio_file_path)
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            return AudioFeatures(
                error=f"Failed to load audio: {str(e)}",
                duration_seconds=0.0,
                channels=0,
                sample_width=0,
                frame_rate=0,
                max_amplitude=0.0,
                rms=0.0
            ).model_dump()
        
        # Extract basic features
        features = AudioFeatures(
            duration_seconds=len(audio) / 1000,
            channels=audio.channels,
            sample_width=audio.sample_width,
            frame_rate=audio.frame_rate,
            max_amplitude=float(max(abs(np.array(audio.get_array_of_samples())))),
            rms=float(audio.rms)
        )
        
        return features.model_dump()
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        return AudioFeatures(
            error=str(e),
            duration_seconds=0.0,
            channels=0,
            sample_width=0,
            frame_rate=0,
            max_amplitude=0.0,
            rms=0.0
        ).model_dump()

def parse_text_responses(text_responses) -> Dict[str, Any]:
    """
    Parse text_responses from the test run data.
    
    Args:
        text_responses: The text_responses field from the test run data
        
    Returns:
        Dictionary with parsed user input and assistant responses
    """
    try:
        # Initialize user input and assistant responses
        user_inputs = []
        assistant_responses = []
        
        # Case 1: If text_responses is a list of objects (from JSON parsing)
        if isinstance(text_responses, list):
            for item in text_responses:
                if isinstance(item, dict):
                    if "parsed_json" in item and isinstance(item["parsed_json"], dict):
                        parsed = item["parsed_json"]
                        if parsed.get("role") == "user" and "content" in parsed:
                            user_inputs.append(parsed["content"])
                        elif parsed.get("role") == "assistant" and "content" in parsed:
                            assistant_responses.append(parsed["content"])
                    elif "role" in item and "content" in item:
                        if item["role"] == "user":
                            user_inputs.append(item["content"])
                        elif item["role"] == "assistant":
                            assistant_responses.append(item["content"])
        
        # Case 2: If text_responses is a string
        elif isinstance(text_responses, str):
            # Try to parse as JSON list first
            try:
                json_list = json.loads(text_responses)
                if isinstance(json_list, list):
                    # Recursively call with the parsed list
                    return parse_text_responses(json_list)
                elif isinstance(json_list, dict):
                    # If it's a single object, process it
                    if json_list.get("role") == "user" and "content" in json_list:
                        user_inputs.append(json_list["content"])
                    elif json_list.get("role") == "assistant" and "content" in json_list:
                        assistant_responses.append(json_list["content"])
            except json.JSONDecodeError:
                # Not a JSON list, try to parse as multiple JSON objects one per line
                try:
                    lines = text_responses.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                if isinstance(item, dict):
                                    if item.get("role") == "user" and "content" in item:
                                        user_inputs.append(item["content"])
                                    elif item.get("role") == "assistant" and "content" in item:
                                        assistant_responses.append(item["content"])
                            except json.JSONDecodeError:
                                # If we can't parse as JSON, log and continue
                                logger.debug(f"Could not parse line as JSON: {line}")
                except Exception as line_parse_error:
                    logger.warning(f"Error parsing text response lines: {line_parse_error}")
        
        # If we couldn't parse any structured data but we have text, use it directly
        if not assistant_responses and not user_inputs and isinstance(text_responses, str):
            # Extract any user/assistant content we can find
            try:
                # Look for JSON-like patterns in the string
                user_pattern = r'"role":\s*"user".*?"content":\s*"([^"]+)"'
                assistant_pattern = r'"role":\s*"assistant".*?"content":\s*"([^"]+)"'
                
                user_matches = re.findall(user_pattern, text_responses)
                assistant_matches = re.findall(assistant_pattern, text_responses)
                
                if user_matches:
                    user_inputs = user_matches
                if assistant_matches:
                    assistant_responses = assistant_matches
                
                if not user_inputs and not assistant_responses:
                    # If no structured data could be extracted, use the whole text as assistant response
                    logger.warning("No structured data found in text_responses, using as raw text")
            except Exception as regex_error:
                logger.warning(f"Error in regex extraction: {regex_error}")
        
        # Combine user inputs
        user_input = " ".join(user_inputs) if user_inputs else ""
        
        # Combine assistant responses
        assistant_response = " ".join(assistant_responses) if assistant_responses else ""
        
        if not assistant_response and not user_input:
            # If we still couldn't extract anything, use the original text if it's a string
            if isinstance(text_responses, str):
                assistant_response = text_responses
            else:
                assistant_response = str(text_responses)
            
        logger.info(f"Parsed assistant response: {assistant_response}")
        logger.info(f"Parsed user input: {user_input}")
        
        # Return as TranscriptionResult format for compatibility
        return TranscriptionResult(
            user_input=user_input,
            assistant_response=assistant_response
        ).model_dump()
        
    except Exception as e:
        logger.error(f"Error parsing text responses: {e}")
        return TranscriptionResult(user_input="unknown",
            assistant_response="unknown").model_dump()

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using SequenceMatcher.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    # Normalize texts (lowercase, remove punctuation, extra spaces)
    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    norm_text1 = normalize(text1)
    norm_text2 = normalize(text2)
    
    # Calculate similarity
    matcher = SequenceMatcher(None, norm_text1, norm_text2)
    return matcher.ratio()

class LLMJudge:
    """
    LLM as a Judge for evaluating speech-to-speech responses.
    Uses Amazon Bedrock to evaluate the quality of responses.
    """
    
    def __init__(self, config: Dict[str, Any], validation_dataset: Dict[str, Dict[str, str]]):
        """
        Initialize the LLM Judge.
        
        Args:
            config: Configuration dictionary with judge model, criteria, and prompt template
            validation_dataset: Dictionary mapping audio file names to expected functions and transcriptions
        """
        self.config = config
        self.validation_dataset = validation_dataset
        self.judge_model_id = config.get('judge_model', {}).get('id', "anthropic.claude-3-sonnet-20240229-v1:0")
        self.max_tokens = config.get('judge_model', {}).get('max_tokens', 1024)
        self.temperature = config.get('judge_model', {}).get('temperature', 0.0)
        self.evaluation_criteria = config.get('evaluation_criteria', {})
        self.prompt_template = config.get('prompt_template', "")
        
        # Load environment variables
        env_path = find_dotenv()
        if env_path:
            load_dotenv(dotenv_path=env_path)
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        logger.info(f"Initialized LLM Judge with model: {self.judge_model_id}")
        logger.info(f"Loaded {len(self.evaluation_criteria)} evaluation criteria")
    
    def evaluate_response(self, 
                         input_file_path: str, 
                         text_response: str, 
                         output_file_path: Optional[str] = None,
                         audio_features: Optional[Dict[str, Any]] = None,
                         transcription: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single speech-to-speech interaction.
        
        Args:
            input_file_path: Path to the input audio file
            text_response: The transcribed audio input
            output_file_path: Path to the output audio file (optional)
            audio_features: Pre-extracted audio features (optional)
            transcription: Transcription results (optional)
            
        Returns:
            Dict containing evaluation scores and rationale
        """
        # Extract input file name without extension
        input_file_name = os.path.basename(input_file_path).split('.')[0]
        
        # Determine expected function call, transcription, and response from validation dataset
        # expected_function = "unknown"
        expected_user_input = "unknown"
        expected_response = ""
        
        # Look for exact match first
        if input_file_name in self.validation_dataset:
            # expected_function = self.validation_dataset[input_file_name].get('expected_function', 'unknown')
            expected_user_input = self.validation_dataset[input_file_name].get('expected_transcription', 'unknown')
            expected_response = self.validation_dataset[input_file_name].get('expected_response', '')
        else:
            # Try partial match if exact match not found
            for key, data in self.validation_dataset.items():
                if key in input_file_name:
                    # expected_function = data.get('expected_function', 'unknown')
                    expected_user_input = data.get('expected_transcription', 'unknown')
                    expected_response = data.get('expected_response', '')
                    break
        
        # Extract audio features if not provided and output file exists
        if output_file_path and not audio_features:
            try:
                if os.path.exists(output_file_path):
                    audio_features = extract_audio_features(output_file_path)
                else:
                    logger.warning(f"Output file does not exist: {output_file_path}")
                    # Set to None to avoid further issues
                    output_file_path = None
            except Exception as e:
                logger.error(f"Error extracting audio features: {e}")
                output_file_path = None
        
        # Calculate objective metrics
        objective_metrics = {}
        
        # Calculate similarity between expected and actual response
        if text_response and expected_response:
            response_similarity = calculate_text_similarity(text_response, expected_response)
            objective_metrics["response_similarity"] = response_similarity
            logger.info(f"Response similarity: {response_similarity:.2f}")
        
        # Calculate similarity between transcribed audio and expected response
        if transcription and expected_response:
            assistant_response = transcription.get("assistant_response")
            if assistant_response is not None:
                transcription_similarity = calculate_text_similarity(str(assistant_response), expected_response)
                objective_metrics["transcription_similarity"] = transcription_similarity
                logger.info(f"Transcription similarity: {transcription_similarity:.2f}")
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(
            expected_user_input=expected_user_input,
            user_input=transcription.get("user_input") if transcription else None,
            expected_response=expected_response,
            assistant_response=transcription.get("assistant_response") if transcription else None
        )
        logger.info(f"Created evaluation prompt: {prompt}")
        
        # Call the judge model
        try:
            evaluation_result = self._call_judge_model(prompt)
            parsed_result = self._parse_evaluation_result(evaluation_result)
            
            # Add metadata
            parsed_result["input_file"] = input_file_path
            parsed_result["output_file"] = output_file_path
            
            # Use proper field names and avoid duplicate content
            if transcription and "user_input" in transcription:
                parsed_result["user_input"] = transcription["user_input"]
            else:
                # Parse text_response to extract user input if not already done
                try:
                    parsed_text = parse_text_responses(text_response)
                    if "user_input" in parsed_text:
                        parsed_result["user_input"] = parsed_text["user_input"]
                except:
                    # Fall back to raw text_response only if needed
                    pass
            
            parsed_result["expected_user_input"] = expected_user_input
            parsed_result["expected_response"] = expected_response
            
            # Get category from validation dataset
            audio_name = get_audio_file_name_from_path(input_file_path)
            # Find the category in validation dataset
            for key, data in self.validation_dataset.items():
                if key == audio_name or key in audio_name:
                    if "category" in data:
                        parsed_result["category"] = data["category"]
                    break
            
            if transcription:
                if "user_input" in transcription:
                    parsed_result["user_input"] = transcription["user_input"]
                if "assistant_response" in transcription:
                    parsed_result["transcribed_response"] = transcription["assistant_response"]
            
            if objective_metrics:
                parsed_result["objective_metrics"] = objective_metrics
            
            if audio_features and "error" not in audio_features:
                parsed_result["audio_features"] = audio_features
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {
                "error": str(e),
                "input_file": input_file_path,
                "output_file": output_file_path
            }
    
    def _create_evaluation_prompt(self, 
                                expected_user_input: str = "unknown",
                                user_input: str = "unknonw",
                                expected_response: str= "unknown",
                                assistant_response: str = "unknown") -> str:
        """
        Create a prompt for the judge model to evaluate a speech-to-speech interaction.
        
        Args:

            expected_user_input: The expected user input
            user_input: The user's actual transcribed input 
            expected_response: The expected model response content
            assistant_response: The actual model response content
            audio_features: Extracted audio features from the output audio file (optional)
            objective_metrics: Objective evaluation metrics (optional)
            
        Returns:
            Evaluation prompt string
        """
        criteria_text = "\n".join([f"- {name}: {description}" for name, description in self.evaluation_criteria.items()])        
                
        # Format the prompt template with our extracted values
        prompt = self.prompt_template.format(
            expected_user_input=expected_user_input,
            user_input=user_input,
            expected_response=expected_response,
            transcribed_response=assistant_response,
            criteria_text=criteria_text
        )
        return prompt

        
           
    
    def _call_judge_model(self, prompt: str) -> str:
        """
        Call the judge model with the evaluation prompt.
        
        Args:
            prompt: The evaluation prompt
            
        Returns:
            The model's response
        """
        # Prepare request based on model type
        if "anthropic.claude" in self.judge_model_id:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature
            }
        else:
            # Default format for other models
            request_body = {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        
        # Invoke the model
        response = self.bedrock_client.invoke_model(
            modelId=self.judge_model_id,
            body=json.dumps(request_body)
        )
        
        # Parse response based on model type
        response_body = json.loads(response.get('body').read())
        
        if "anthropic.claude" in self.judge_model_id:
            return response_body.get('content', [{}])[0].get('text', '')
        else:
            return response_body.get('completion', '')
    
    def _parse_evaluation_result(self, result_text: str) -> Dict[str, Any]:
        """
        Parse the evaluation result from the judge model.
        
        Args:
            result_text: The raw text response from the judge model
            
        Returns:
            Parsed evaluation result as a dictionary
        """
        try:
            # Extract JSON from the response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON: {e}")
                    logger.error(f"Extracted JSON: {json_str}")
            else:
                logger.warning("Could not find JSON in response")
            
            # If we couldn't extract valid JSON, create a default response
            logger.warning("Creating default evaluation response")
            return {
                "criteria": {
                    "speech_recognition": {
                        "score": 0,
                        "explanation": "Could not evaluate - model did not return valid JSON"
                    },
                    "response_relevance": {
                        "score": 0,
                        "explanation": "Could not evaluate - model did not return valid JSON"
                    },
                    "response_correctness": {
                        "score": 0,
                        "explanation": "Could not evaluate - model did not return valid JSON"
                    }
                },
                "overall": {
                    "score": 0,
                    "summary": "Could not evaluate - model did not return valid JSON"
                },
                "error": "Failed to parse evaluation result",
                "raw_result": result_text[:500] + ("..." if len(result_text) > 500 else "")
            }
                
        except Exception as e:
            logger.error(f"Error in parse_evaluation_result: {e}")
            logger.error(f"Raw result (first 500 chars): {result_text[:500]}")
            
            # Return a structured error response
            return {
                "criteria": {
                    "speech_recognition": {"score": 0, "explanation": f"Error: {str(e)}"},
                    "response_relevance": {"score": 0, "explanation": f"Error: {str(e)}"},
                    "response_correctness": {"score": 0, "explanation": f"Error: {str(e)}"},
                },
                "overall": {
                    "score": 0,
                    "summary": f"Error parsing evaluation result: {str(e)}"
                },
                "error": f"Exception in parse_evaluation_result: {str(e)}",
                "raw_result": result_text[:500] + ("..." if len(result_text) > 500 else "")
            }

def evaluate_responses(responses_file, config_path=DEFAULT_CONFIG_PATH, validation_dataset_path=DEFAULT_VALIDATION_DATASET_PATH):
    """
    Evaluate all responses in the specified file.
    
    Args:
        responses_file: Path to the JSONL file containing response data
        config_path: Path to the configuration JSON file
        validation_dataset_path: Path to the validation dataset JSONL file
        retrieve_function_calls: Whether to retrieve function calls from CloudWatch logs
        log_group_name: CloudWatch log group name to search for function calls
        
    Returns:
        List of evaluation results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Load validation dataset
    try:
        validation_dataset = load_validation_dataset(validation_dataset_path)
    except Exception as e:
        logger.error(f"Failed to load validation dataset: {e}")
        return []
    
    # Process responses file
    responses = process_responses_file(responses_file)
    if not responses:
        logger.error(f"No responses found in {responses_file}")
        return []
    
    logger.info(f"Found {len(responses)} responses to evaluate")
    
    # Initialize LLM Judge
    judge = LLMJudge(config, validation_dataset)
    
    # Evaluate each response
    results = []
    for i, response in enumerate(responses, 1):
        logger.info(f"Evaluating response {i}/{len(responses)}")
        
        input_file_path = response.get('input_file_path', '')
        text_responses = response.get('text_responses', '')
        output_file_path = response.get('output_file_path')
        summary_file = response.get('summary_file')
        
        # Extract audio features if output file exists
        audio_features = None
        if output_file_path and os.path.exists(output_file_path):
            audio_features = extract_audio_features(output_file_path)
        
        transcription = None
        try:
            # Use parse_text_responses function to extract user input assistant responses
            transcription = parse_text_responses(text_responses)
            logger.info(f"Successfully parsed text responses")
        except Exception as e:
            logger.error(f"Error parsing text responses: {e}")
        
        # Add token usage if available
        token_usage = None
        if 'token_usage' in response:
            token_usage = response['token_usage']
        
        
        # Evaluate response
        result = judge.evaluate_response(
            input_file_path=input_file_path,
            text_response=text_responses,
            output_file_path=output_file_path,
            audio_features=audio_features,
            transcription=transcription
        )
        
        # Add duration_seconds to objective metrics
        if 'duration_seconds' in response and response['duration_seconds'] is not None:
            if "objective_metrics" not in result:
                result["objective_metrics"] = {}
            result["objective_metrics"]["duration_seconds"] = response['duration_seconds']
            logger.info(f"  - Added duration_seconds: {response['duration_seconds']} seconds")
        
        # Add summary file path
        result['summary_file'] = summary_file
        
        # Add token usage if available
        if token_usage:
            result['token_usage'] = token_usage
        
        results.append(result)
    
    # Save overall results to the same file path as the responses file
    with open(responses_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "responses_count": len(responses),
            "evaluations_count": len(results),
            "results": results
        }, f, indent=2)
    
    logger.info(f"Saved evaluation results to {responses_file}")
    
    return results

def generate_evaluation_report(results, output_file=None, validation_dataset_path=DEFAULT_VALIDATION_DATASET_PATH):
    """
    Generate a human-readable evaluation report from the evaluation results.
    
    Args:
        results: List of evaluation results
        output_file: Path to save the report (optional)
        validation_dataset_path: Path to the validation dataset for sorting by category
        
    Returns:
        Report text
    """
    if not results:
        return "No evaluation results to report."
    
    # Load validation dataset for sorting
    validation_data = {}
    try:
        # Load the validation dataset to get categories
        with open(validation_dataset_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                audio_file = entry.get('audio_file')
                category = entry.get('category')
                if audio_file and category:
                    validation_data[audio_file] = category
    except Exception as e:
        logger.error(f"Error loading validation dataset for sorting in report generation: {e}")
    
    # Create a sorting key function
    def get_sort_key(result):
        input_file = result.get("input_file", "")
        audio_name = get_audio_file_name_from_path(input_file)
        
        # Get category, defaulting to "Unknown" if not found
        category = validation_data.get(audio_name, "Unknown")
        
        return (category, audio_name)
    
    # Sort results by category and audio file name to ensure consistent order in report
    sorted_results = sorted(results, key=get_sort_key)
    results = sorted_results
    
    # Calculate average scores
    criteria_scores = {
        "speech_recognition": [],
        "response_relevance": [],
        "response_correctness": [],
    }
    
    overall_scores = []
    
    # Initialize category scores dictionary
    category_scores = {}
    
    for result in results:
        if "criteria" in result and "overall" in result:
            # Extract criteria scores
            for criterion, scores in criteria_scores.items():
                if criterion in result["criteria"] and "score" in result["criteria"][criterion]:
                    scores.append(result["criteria"][criterion]["score"])
            
            # Extract overall score
            if "score" in result["overall"]:
                overall_scores.append(result["overall"]["score"])
    
    # Calculate averages
    avg_criteria = {}
    for criterion, scores in criteria_scores.items():
        if scores:
            avg_criteria[criterion] = sum(scores) / len(scores)
        else:
            avg_criteria[criterion] = 0
    
    avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    
    # Generate report
    report = []
    report.append("# Speech-to-Speech Evaluation Report")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total responses evaluated: {len(results)}")
    
    # Calculate and add total cost to the report
    total_cost = 0.0
    for result in results:
        if "objective_metrics" in result and "cost_usd" in result["objective_metrics"]:
            total_cost += result["objective_metrics"]["cost_usd"]
    report.append(f"Total cost: ${total_cost:.6f}")
    report.append("")
    
    report.append("## Overall Average Scores")
    report.append("")
    report.append("| Criterion | Average Score |")
    report.append("|-----------|---------------|")
    for criterion, avg_score in avg_criteria.items():
        report.append(f"| {criterion.replace('_', ' ').title()} | {avg_score:.2f} |")
    report.append(f"| **Overall** | **{avg_overall:.2f}** |")
    report.append("")
    
    # Group results by category for report and calculate category averages
    results_by_category = {}
    category_scores = {}
    
    for result in results:
        input_file = result.get("input_file", "")
        audio_name = get_audio_file_name_from_path(input_file)
        category = validation_data.get(audio_name, "Unknown")
        
        if category not in results_by_category:
            results_by_category[category] = []
            category_scores[category] = {
                "speech_recognition": [],
                "response_relevance": [],
                "response_correctness": [],
                "overall": []
            }
        
        results_by_category[category].append(result)
        
        # Collect scores for this category
        if "criteria" in result and "overall" in result:
            # Extract criteria scores for this category
            for criterion in ["speech_recognition", "response_relevance", "response_correctness"]:
                if criterion in result.get("criteria", {}) and "score" in result["criteria"][criterion]:
                    category_scores[category][criterion].append(result["criteria"][criterion]["score"])
            
            # Extract overall score for this category
            if "score" in result["overall"]:
                category_scores[category]["overall"].append(result["overall"]["score"])
    
    # Add category-specific average scores
    report.append("## Average Scores by Category")
    report.append("")
    
    for category, scores in category_scores.items():
        report.append(f"### {category}")
        report.append("")
        report.append("| Criterion | Average Score |")
        report.append("|-----------|---------------|")
        
        # Calculate and display average for each criterion in this category
        for criterion, values in scores.items():
            if criterion != "overall" and values:  # Skip overall for now and handle empty lists
                avg_score = sum(values) / len(values) if values else 0
                report.append(f"| {criterion.replace('_', ' ').title()} | {avg_score:.2f} |")
        
        # Add overall average for this category
        overall_values = scores.get("overall", [])
        category_avg_overall = sum(overall_values) / len(overall_values) if overall_values else 0
        report.append(f"| **Overall** | **{category_avg_overall:.2f}** |")
        report.append("")
    
    # Update results_by_category with already collected data
    report.append("## Results by Category")
    report.append("")
    
    report.append("## Results by Category")
    report.append("")
    
    # Generate report by category
    for category in sorted(results_by_category.keys()):
        report.append(f"### {category}")
        report.append("")
        
        category_counter = 1
        for result in results_by_category[category]:
            input_file = result.get("input_file", f"Response {category_counter}")
            audio_name = get_audio_file_name_from_path(input_file)
            report.append(f"#### Test {category_counter}: {audio_name}")
            category_counter += 1
            
            if "expected_user_input" in result:
                report.append(f"**Expected User Input:** {result['expected_user_input']}")
            
            if "user_input" in result:
                report.append(f"**Transcribed User Input:** {result['user_input']}")
            
            if "expected_response" in result:
                report.append(f"**Expected Response:** {result['expected_response']}")
            
            if "transcribed_response" in result:
                report.append(f"**Transcribed Response:** {result['transcribed_response']}")
            
            report.append("")
            report.append("##### Scores")
            report.append("")
            
            if "criteria" in result:
                report.append("| Criterion | Score | Explanation |")
                report.append("|-----------|-------|-------------|")
                for criterion, data in result["criteria"].items():
                    if isinstance(data, dict) and "score" in data and "explanation" in data:
                        report.append(f"| {criterion.replace('_', ' ').title()} | {data['score']} | {data['explanation']} |")
            
            if "overall" in result and "score" in result["overall"] and "summary" in result["overall"]:
                report.append("")
                report.append(f"**Overall Score:** {result['overall']['score']}")
                report.append(f"**Summary:** {result['overall']['summary']}")
            
            if "objective_metrics" in result:
                report.append("")
                report.append("##### Objective Metrics")
                report.append("")
                for metric, value in result["objective_metrics"].items():
                    if metric == "cost_usd":
                        report.append(f"- **{metric.replace('_', ' ').title()}:** ${value:.6f}")
                    else:
                        report.append(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}")
            
            report.append("")
            report.append("---")
            report.append("")
    
    report_text = "\n".join(report)
    
    # Save report if output file is specified
    if output_file:
        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created directory: {output_dir}")
            except Exception as e:
                logger.error(f"Failed to create directory {output_dir}: {e}")
                
        try:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Saved evaluation report to {output_file}")
        except Exception as e:
            logger.error(f"Failed to write evaluation report to {output_file}: {e}")
    
    return report_text

def calculate_cost(token_usage):
    """
    Calculate the cost based on token usage and Nova Sonic pricing.
    
    Args:
        token_usage: Dictionary containing token usage information
        
    Returns:
        Total cost in USD
    """
    if not token_usage:
        return 0.0
    
    speech_input_tokens = token_usage.get('input_speech_tokens', 0)
    text_input_tokens = token_usage.get('input_text_tokens', 0)
    speech_output_tokens = token_usage.get('output_speech_tokens', 0)
    text_output_tokens = token_usage.get('output_text_tokens', 0)
    
    # Calculate cost components (convert from price per 1000 tokens)
    speech_input_cost = (speech_input_tokens / 1000) * NOVA_SONIC_PRICING["speech_input"]
    text_input_cost = (text_input_tokens / 1000) * NOVA_SONIC_PRICING["text_input"]
    speech_output_cost = (speech_output_tokens / 1000) * NOVA_SONIC_PRICING["speech_output"]
    text_output_cost = (text_output_tokens / 1000) * NOVA_SONIC_PRICING["text_output"]
    
    # Calculate total cost
    total_cost = speech_input_cost + text_input_cost + speech_output_cost + text_output_cost
    
    return total_cost

def get_audio_file_name_from_path(file_path):
    """
    Extract audio file name (without extension) from a path.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Audio file name without extension
    """
    if not file_path:
        return ""
    
    # Get basename and remove extension
    basename = os.path.basename(file_path)
    file_name = os.path.splitext(basename)[0]
    
    # Extract just the numeric prefix and base name (like "1_whatismyschedule")
    # This pattern looks for digit(s) followed by underscore and text
    pattern = r'(\d+_[a-zA-Z0-9]+)'
    match = re.search(pattern, file_name)
    if match:
        return match.group(1)
    
    return file_name

def remove_null_fields(obj):
    """
    Recursively remove fields with null values from dictionaries
    
    Args:
        obj: Object to process (dict, list, or other type)
        
    Returns:
        Object with null fields removed
    """
    if isinstance(obj, dict):
        return {k: remove_null_fields(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_null_fields(item) for item in obj]
    else:
        return obj

def main():
    """Main function for the script"""
    parser = argparse.ArgumentParser(description="LLM as a Judge for Speech-to-Speech Evaluation")
    
    parser.add_argument("--responses_file", required=True, help="Path to the JSONL file containing response data")
    parser.add_argument("--config_file", default=DEFAULT_CONFIG_PATH, help=f"Path to configuration JSON file (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--validation_dataset", default=DEFAULT_VALIDATION_DATASET_PATH, help=f"Path to validation dataset JSONL file (default: {DEFAULT_VALIDATION_DATASET_PATH})")
    parser.add_argument("--report_file", default=None, help="Path to save evaluation report (optional, defaults to 'evaluation_report.md' in responses directory)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--retrieve_function_calls", action="store_true", default=False, help="Retrieve function calls from CloudWatch logs")
    parser.add_argument("--log_group_name", default=None, help="CloudWatch log group name to search for function calls")
    parser.add_argument("--json_summary_file", default=None, help="Path to save JSON evaluation summary (optional, defaults to 'evaluation_summary.json' in responses directory)")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Check if responses file exists
    if not os.path.isfile(args.responses_file):
        logger.error(f"Responses file not found: {args.responses_file}")
        return 1
    
    # Check if config file exists
    if not os.path.isfile(args.config_file):
        logger.error(f"Configuration file not found: {args.config_file}")
        return 1
    
    # Check if validation dataset exists
    if not os.path.isfile(args.validation_dataset):
        logger.error(f"Validation dataset not found: {args.validation_dataset}")
        return 1
    
    # Evaluate responses
    logger.info(f"Evaluating responses in {args.responses_file}")
    results = process_responses_file(args.responses_file)
    
    if results:
        # Load configuration
        config = load_config(args.config_file)
        
        # Load validation dataset
        try:
            validation_dataset = load_validation_dataset(args.validation_dataset)
        except Exception as e:
            logger.error(f"Failed to load validation dataset: {e}")
            return 1
        
        # Initialize LLM Judge
        judge = LLMJudge(config, validation_dataset)
        
        # Create directory for results if it doesn't exist
        responses_dir = os.path.dirname(args.responses_file)
        
        # Evaluate each response
        evaluation_results = []
        for i, response in enumerate(results, 1):
            logger.info(f"Evaluating response {i}/{len(results)}")
            
            input_file_path = response.get('input_file_path', '')
            text_responses = response.get('text_responses', '')
            output_file_path = response.get('output_file_path')
            summary_file = response.get('summary_file')
            
            # Extract audio features if output file exists
            audio_features = None
            if output_file_path and os.path.exists(output_file_path):
                audio_features = extract_audio_features(output_file_path)
            
            # Parse text responses to extract user input and assistant responses
            transcription = None
            try:
                transcription = parse_text_responses(text_responses)
            except Exception as e:
                logger.error(f"Error parsing text responses: {e}")
            
            # Add token usage if available
            token_usage = None
            if 'token_usage' in response:
                token_usage = response['token_usage']
            
            # Add latency if available
            duration_seconds = None
            if 'duration_seconds' in response:
                duration_seconds = response['duration_seconds']
                
                # Add to objective metrics
                if duration_seconds is not None:
                    if not hasattr(response, 'objective_metrics'):
                        response['objective_metrics'] = {}
                    response['objective_metrics']['duration_seconds'] = duration_seconds
            
            # Evaluate response
            result = judge.evaluate_response(
                input_file_path=input_file_path,
                text_response=text_responses,
                output_file_path=output_file_path,
                audio_features=audio_features,
                transcription=transcription
            )
            
            # Add summary file path
            result['summary_file'] = summary_file
            
            # Add token usage if available
            if token_usage:
                result['token_usage'] = token_usage
                
                # Calculate cost based on token usage
                cost = calculate_cost(token_usage)
                
                # Add cost to objective metrics
                if "objective_metrics" not in result:
                    result["objective_metrics"] = {}
                result["objective_metrics"]["cost_usd"] = cost
                
                logger.info(f"  - Calculated cost: ${cost:.6f}")
            
            # Add duration_seconds to objective metrics
            if duration_seconds is not None:
                if "objective_metrics" not in result:
                    result["objective_metrics"] = {}
                result["objective_metrics"]["duration_seconds"] = duration_seconds
                logger.info(f"  - Added duration_seconds: {duration_seconds} seconds")
            
            # Remove null summary_file field and any other null fields
            result = remove_null_fields(result)
            
            # Get category from validation dataset if not already set
            if "category" not in result:
                audio_name = get_audio_file_name_from_path(input_file_path)
                for key, data in validation_dataset.items():
                    if key == audio_name or key in audio_name:
                        result["category"] = data.get("category", "Unknown")
                        break
            
            evaluation_results.append(result)
        
        # Load validation dataset for sorting
        validation_data = {}
        try:
            # Load the validation dataset to get categories
            with open(args.validation_dataset, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    audio_file = entry.get('audio_file')
                    category = entry.get('category')
                    if audio_file and category:
                        validation_data[audio_file] = category
        except Exception as e:
            logger.error(f"Error loading validation dataset for sorting: {e}")
        
        # Create a sorting key function
        def get_sort_key(result):
            input_file = result.get("input_file", "")
            audio_name = get_audio_file_name_from_path(input_file)
            
            # Get category, defaulting to "Unknown" if not found
            category = validation_data.get(audio_name, "Unknown")
            
            return (category, audio_name)
        
        # Sort evaluation results by category and audio file name
        evaluation_results.sort(key=get_sort_key)
        logger.info(f"Sorted {len(evaluation_results)} evaluation results by category and audio file name")
        
        # Calculate total cost across all evaluations
        total_cost = 0.0
        for result in evaluation_results:
            if "objective_metrics" in result and "cost_usd" in result["objective_metrics"]:
                total_cost += result["objective_metrics"]["cost_usd"]
        
        logger.info(f"Total cost for all evaluations: ${total_cost:.6f}")
        
        # Save overall results to JSON summary file
        json_summary_file = args.json_summary_file
        if not json_summary_file:
            json_summary_file = os.path.join(responses_dir, "evaluation_summary.json")
        
        # Ensure the directory exists
        summary_dir = os.path.dirname(json_summary_file)
        if summary_dir and not os.path.exists(summary_dir):
            try:
                os.makedirs(summary_dir, exist_ok=True)
                logger.info(f"Created directory: {summary_dir}")
            except Exception as e:
                logger.error(f"Failed to create directory {summary_dir}: {e}")
        
        with open(json_summary_file, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "responses_count": len(results),
                "evaluations_count": len(evaluation_results),
                "total_cost_usd": total_cost,
                "results": evaluation_results,
                "metadata": {
                    "contains_duration_seconds": any("objective_metrics" in r and "duration_seconds" in r["objective_metrics"] 
                                                  for r in evaluation_results)
                }
            }, f, indent=2)
        
        logger.info(f"Saved evaluation summary to {json_summary_file}")
        results = evaluation_results
    
    # Generate report
    report_file = args.report_file
    if not report_file:
        # Default to evaluation_report.md in the responses directory
        report_file = os.path.join(responses_dir, "evaluation_report.md")
    
    logger.info(f"Generating evaluation report: {report_file}")
    generate_evaluation_report(results, report_file, args.validation_dataset)
    
    logger.info("Evaluation complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
