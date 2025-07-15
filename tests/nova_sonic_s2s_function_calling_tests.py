#!/usr/bin/env python3
"""
Speech-to-Speech Function Calling Test Catalog
-------------------------------------------
This script performs tests for function calling via WebSocket API using speech input and output.
It processes audio files from the audio_samples directory and sends them to the API, 
then collects and saves the responses for analysis.
"""

import os
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import logging
import uuid
import argparse

# Import the test harness core functionality
from s2s_test_harness import run_test_loop

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def read_json_to_dataframe(directory_path: str = "./responses/model_sonic") -> pd.DataFrame:
    """
    Read all JSON files in the specified directory into a pandas DataFrame.
    
    Args:
        directory_path: Path to the directory containing JSON files
        
    Returns:
        pandas DataFrame with the following columns:
        - input_file_path: from "input_file" in JSON
        - duration_seconds: from "duration_seconds" in JSON
        - text_responses: concatenated "raw_text" attributes from "text_responses" array
        - output_file_path: identified audio output file based on JSON filename prefix
        - speech_tokens_input: number of speech tokens in the input
        - text_tokens_input: number of text tokens in the input
        - speech_tokens_output: number of speech tokens in the output
        - text_tokens_output: number of text tokens in the output
        - total_tokens: total number of tokens used
    """
    # List to store data from each JSON file
    data_list = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    for json_file in json_files:
        # Construct full path to the JSON file
        json_path = os.path.join(directory_path, json_file)
        
        try:
            # Read JSON file
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Extract required fields
            input_file_path = json_data.get('input_file', '')
            
            # Extract duration_seconds from the audio object correctly
            audio_data = json_data.get('audio', {})
            duration_seconds = audio_data.get('duration_seconds', None)
            
            # Extract and concatenate raw_text from text_responses
            text_responses_array = json_data.get('text_responses', [])
            text_responses = '\n'.join([resp.get('raw_text', '') for resp in text_responses_array])
            
            # Extract token usage information
            token_usage = json_data.get('token_usage', {})
            total_tokens = token_usage.get('totalTokens', 0)
            
            # Get detailed token breakdown
            token_details = token_usage.get('details', {})
            input_tokens = token_details.get('input', {})
            output_tokens = token_details.get('output', {})
            
            speech_tokens_input = input_tokens.get('speechTokens', 0)
            text_tokens_input = input_tokens.get('textTokens', 0)
            speech_tokens_output = output_tokens.get('speechTokens', 0)
            text_tokens_output = output_tokens.get('textTokens', 0)
            
            # Identify corresponding audio output file more robustly
            # First check if there's a direct reference in the JSON
            output_file_path = None
            
            # Try to find the WAV file with a matching prefix
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            if '_session_summary' in base_name:
                file_prefix = base_name.split('_session_summary')[0]
                file_postfix = base_name.split('_session_summary')[1]
                
                # Look for response files with matching prefix
                potential_files = [
                    f for f in os.listdir(directory_path) 
                    if f.startswith(file_prefix) and f.endswith('.wav')
                ]
                
                if potential_files:
                    output_file_path = os.path.join(directory_path, potential_files[0])
            
            # Create a dictionary with the extracted data
            data_dict = {
                'input_file_path': input_file_path,
                'duration_seconds': duration_seconds,
                'text_responses': text_responses,
                'output_file_path': output_file_path,
                'speech_tokens_input': speech_tokens_input,
                'text_tokens_input': text_tokens_input,
                'speech_tokens_output': speech_tokens_output, 
                'text_tokens_output': text_tokens_output,
                'total_tokens': total_tokens
            }
            
            data_list.append(data_dict)
            
        except Exception as e:
            logger.error(f"Error processing {json_file}: {str(e)}")
    
    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    
    return df

async def run_tests(server_url=None, num_runs=1, delay_between_tests=5.0, delay_between_runs=5.0):
    """
    Run the speech-to-speech test suite multiple times.
    
    Args:
        server_url (str): WebSocket server URL (default: from environment variables)
        num_runs (int): Number of times to run the test suite (default: 1)
        delay_between_tests (float): Delay in seconds between test iterations (default: 5.0)
        delay_between_runs (float): Delay in seconds between test runs (default: 5.0)
    """
    # Loading environment variables
    local_env_filename = '../deployment/.env'
    load_dotenv(find_dotenv(local_env_filename), override=True)
    
    # Use provided server URL or default to localhost WebSocket endpoint
    ws_url = server_url or 'http://localhost:8000/ws'
    logger.info(f"Using WebSocket server URL: {ws_url}")
    
    # Create base output directory
    base_output_dir = "./responses/model_sonic"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Find audio files in the audio_samples directory
    audio_files = [f for f in os.listdir("./audio_samples") if f.endswith(".raw")]
    if not audio_files:
        logger.error("No .raw audio files found in ./audio_samples directory")
        return
    
    logger.info(f"Found {len(audio_files)} audio files for testing")
    
    # Run the specified number of test runs
    for run in range(1, num_runs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting test run {run}/{num_runs}")
        logger.info(f"{'='*60}")
        
        # Create run-specific output directory
        run_output_dir = os.path.join(base_output_dir, f"run_{run}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        # Run test loop using the s2s_test_harness
        try:
            results = await run_test_loop(
                audio_files=[f"./audio_samples/{f}" for f in audio_files],
                server_url=ws_url,
                delay_between_tests=delay_between_tests,
                output_dir=run_output_dir
            )
            
            # Process the results from this run
            logger.info(f"Test run {run} completed with {len(results)} results")
            
            # Process the results dataframe for this run
            df = read_json_to_dataframe(run_output_dir)
            
            # Save the dataframe to JSONL in the run directory
            jsonl_path = os.path.join(run_output_dir, "s2s_sonic_function_calling_responses.jsonl")
            with open(jsonl_path, 'w') as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict()) + '\n')
            logger.info(f"Saved results to {jsonl_path}")
            
        except Exception as e:
            logger.error(f"Error during test run {run}: {str(e)}")
        
        # Add delay between runs (except after the last one)
        if run < num_runs:
            logger.info(f"Waiting {delay_between_runs} seconds before next test run...")
            await asyncio.sleep(delay_between_runs)
    
    # Create a merged JSONL with all results
    try:
        merged_jsonl_path = os.path.join(base_output_dir, "s2s_sonic_function_calling_responses.jsonl")
        with open(merged_jsonl_path, 'w') as merged_file:
            for run in range(1, num_runs + 1):
                run_dir = os.path.join(base_output_dir, f"run_{run}")
                jsonl_path = os.path.join(run_dir, "s2s_sonic_function_calling_responses.jsonl")
                
                if os.path.exists(jsonl_path):
                    with open(jsonl_path, 'r') as run_file:
                        for line in run_file:
                            # Parse the JSON record
                            record = json.loads(line.strip())
                            # Add run number to the record
                            record['run'] = run
                            
                            # Ensure token usage fields are present, even if they're zero
                            if 'speech_tokens_input' not in record:
                                record['speech_tokens_input'] = 0
                            if 'text_tokens_input' not in record:
                                record['text_tokens_input'] = 0
                            if 'speech_tokens_output' not in record:
                                record['speech_tokens_output'] = 0
                            if 'text_tokens_output' not in record:
                                record['text_tokens_output'] = 0
                            if 'total_tokens' not in record:
                                record['total_tokens'] = 0
                                
                            # Write to merged file
                            merged_file.write(json.dumps(record) + '\n')
        
        logger.info(f"Saved merged results from all runs to {merged_jsonl_path}")
        
        # Load the merged file to calculate statistics
        all_records = []
        with open(merged_jsonl_path, 'r') as f:
            for line in f:
                all_records.append(json.loads(line.strip()))
            
            # Print summary statistics
            successful_responses = sum(1 for record in all_records if record.get('text_responses'))
            total_tests = len(all_records)
            success_rate = successful_responses / total_tests if total_tests > 0 else 0
            
            # Calculate token usage statistics
            total_input_speech_tokens = sum(record.get('speech_tokens_input', 0) for record in all_records)
            total_input_text_tokens = sum(record.get('text_tokens_input', 0) for record in all_records)
            total_output_speech_tokens = sum(record.get('speech_tokens_output', 0) for record in all_records)
            total_output_text_tokens = sum(record.get('text_tokens_output', 0) for record in all_records)
            total_tokens = sum(record.get('total_tokens', 0) for record in all_records)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"OVERALL SUMMARY FOR {num_runs} TEST RUNS")
            logger.info(f"{'='*60}")
            logger.info(f"Total tests: {total_tests}")
            logger.info(f"Successful responses: {successful_responses}")
            logger.info(f"Success rate: {success_rate*100:.2f}%")
            logger.info(f"\nTOKEN USAGE SUMMARY:")
            logger.info(f"Input speech tokens: {total_input_speech_tokens}")
            logger.info(f"Input text tokens: {total_input_text_tokens}")
            logger.info(f"Output speech tokens: {total_output_speech_tokens}")
            logger.info(f"Output text tokens: {total_output_text_tokens}")
            logger.info(f"Total tokens: {total_tokens}")
            logger.info(f"\nResults saved to: {merged_jsonl_path}")
            
    except Exception as e:
        logger.error(f"Error creating merged results: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speech-to-speech function calling tests")
    parser.add_argument("--server-url", type=str, default=None, 
                        help="WebSocket server URL (default: http://localhost:8000/ws)")
    parser.add_argument("--num-runs", type=int, default=1, 
                        help="Number of test runs to execute (default: 1)")
    parser.add_argument("--delay", type=float, default=5.0, 
                        help="Delay between tests in seconds (default: 5.0)")
    parser.add_argument("--run-delay", type=float, default=5.0, 
                        help="Delay between runs in seconds (default: 5.0)")
    
    args = parser.parse_args()
    
    asyncio.run(run_tests(
        server_url=args.server_url,
        num_runs=args.num_runs,
        delay_between_tests=args.delay,
        delay_between_runs=args.run_delay
    ))
