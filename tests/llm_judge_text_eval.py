"""
LLM as a Judge Evaluation for Function Calling Tests

This script implements an LLM-as-a-Judge evaluation approach for assessing the quality
of function calling responses from the retail assistant API. It evaluates responses based on
criteria defined in the configuration file.

It includes:
- LLM-based evaluation using Amazon Bedrock
- Function call accuracy assessment
- Expected response validation from dataset
- Objective metrics for response accuracy

The script uses:
- A config JSON file for LLM judge model, evaluation criteria, and prompt template
- A validation dataset JSONL file for expected function calls and responses
- Amazon Bedrock for LLM-based evaluation

Usage:
    python llm_judge_function_calling_eval.py --responses_file <path_to_responses_csv>
                                             [--config_file <path_to_config_json>]
                                             [--validation_dataset <path_to_validation_jsonl>]
                                             [--report_file <path_to_report_md>]
"""

import argparse
import json
import os
import pandas as pd
import boto3
import time
import sys
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Default paths for configuration and validation dataset
DEFAULT_CONFIG_PATH = "config/llm_judge_function_calling_config.json"
DEFAULT_VALIDATION_DATASET_PATH = "data/function_calling_validation_dataset.jsonl"

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

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.warning("Using default configuration")
        return {
            "judge_model": {
                "id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "max_tokens": 1024,
                "temperature": 0.0
            },
            "evaluation_criteria": {
                "correctness": "Does the response correctly answer the query with accurate information?",
                "relevance": "Is the response directly relevant to the user's query?",
                "completeness": "Does the response provide all necessary information to fully answer the query?",
                "clarity": "Is the response clear, well-structured, and easy to understand?"
            },
            "prompt_template": "You are an expert evaluator for retail assistant AI responses..."
        }

def load_validation_dataset(dataset_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load validation dataset from a JSONL file.
    
    Args:
        dataset_path: Path to the validation dataset JSONL file
        
    Returns:
        Dictionary mapping queries to expected functions, responses, and categories
    """
    validation_data = {}
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                query = entry.get('query')
                if query:
                    validation_data[query] = {
                        'expected_function': entry.get('expected_function', 'unknown'),
                        'expected_response': entry.get('expected_response', ''),
                        'category': entry.get('category', 'Unknown')
                    }
        logger.info(f"Loaded validation dataset from {dataset_path} with {len(validation_data)} entries")
        return validation_data
    except Exception as e:
        logger.error(f"Error loading validation dataset from {dataset_path}: {e}")
        logger.warning("Using default validation dataset")
        return { }

class LLMJudge:
    """
    LLM as a Judge for evaluating function calling responses.
    Uses Amazon Bedrock to evaluate the quality of responses.
    """
    
    def __init__(self, config: Dict[str, Any], validation_dataset: Dict[str, Dict[str, str]]):
        """
        Initialize the LLM Judge.
        
        Args:
            config: Configuration dictionary with judge model, criteria, and prompt template
            validation_dataset: Dictionary mapping queries to expected functions
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
    
    def evaluate_response(self, query: str, response: str) -> Dict[str, Any]:
        """
        Evaluate a single query-response pair using the LLM judge.
        
        Args:
            query: The user query
            response: The model's response to evaluate
            
        Returns:
            Dict containing evaluation scores and rationale
        """
        # Determine expected function call and category from validation dataset
        validation_info = self.validation_dataset.get(query, {})
        expected_function = validation_info.get('expected_function', 'unknown')
        expected_response = validation_info.get('expected_response', '')
        category = validation_info.get('category', 'Unknown')
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(query, response, expected_response)
        
        # Call the judge model
        try:
            evaluation_result = self._call_judge_model(prompt)
            parsed_result = self._parse_evaluation_result(evaluation_result)
            
            # Add metadata
            parsed_result["query"] = query
            parsed_result["response"] = response
            parsed_result["expected_function"] = expected_function
            parsed_result["expected_response"] = expected_response
            parsed_result["category"] = category
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {
                "error": str(e),
                "query": query,
                "response": response,
                "expected_function": expected_function,
                "expected_response": expected_response,
                "category": category
            }
    
    def _create_evaluation_prompt(self, query: str, response: str, expected_response: str) -> str:
        """
        Create a prompt for the judge model to evaluate a response.
        
        Args:
            query: The user query
            response: The model's response to evaluate
            expected_response: The expected model response
            
        Returns:
            Evaluation prompt string
        """
        criteria_text = "\n".join([f"- {name}: {description}" for name, description in self.evaluation_criteria.items()])
        
        # Use the prompt template from config
        
        prompt = self.prompt_template.format(
            query=query,
            response=response,
            expected_response=expected_response,
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
                return json.loads(json_str)
            else:
                logger.warning("Could not find JSON in response, attempting to parse full text")
                return json.loads(result_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation result: {e}")
            logger.error(f"Raw result: {result_text}")
            
            # Return a structured error response
            return {
                "error": "Failed to parse evaluation result",
                "raw_result": result_text
            }

def process_responses_file(responses_file: str) -> List[Dict[str, Any]]:
    """
    Process a file containing responses (supports both CSV and JSONL formats).
    
    Args:
        responses_file: Path to the responses file
        
    Returns:
        List of dictionaries with query, response, and other metadata
    """
    try:
        file_extension = os.path.splitext(responses_file)[1].lower()
        
        if file_extension == '.jsonl':
            # Load JSONL file
            responses = []
            with open(responses_file, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        responses.append(json.loads(line))
            logger.info(f"Loaded {len(responses)} responses from JSONL file: {responses_file}")
            return responses
        else:
            # Default to CSV
            df = pd.read_csv(responses_file)
            responses = df.to_dict('records')
            logger.info(f"Loaded {len(responses)} responses from CSV file: {responses_file}")
            return responses
    except Exception as e:
        logger.error(f"Error loading responses file: {e}")
        return []

def evaluate_responses_file(responses_file: str, 
                           config_file: str = DEFAULT_CONFIG_PATH,
                           validation_dataset_file: str = DEFAULT_VALIDATION_DATASET_PATH,
                           report_file: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Evaluate all responses in a file (supports both CSV and JSONL formats).
    
    Args:
        responses_file: Path to the file containing responses (CSV or JSONL)
        config_file: Path to the configuration JSON file
        validation_dataset_file: Path to the validation dataset JSONL file
        report_file: Path to save the evaluation report (optional)
        
    Returns:
        Tuple of (list of results, json output path, markdown output path)
    """
    # Get the directory containing the responses file to save results there
    responses_dir = os.path.dirname(responses_file)
    if not responses_dir:
        responses_dir = os.getcwd()
    
    # Load configuration and validation dataset
    config = load_config(config_file)
    validation_dataset = load_validation_dataset(validation_dataset_file)
    
    # Process responses
    responses = process_responses_file(responses_file)
    if not responses:
        logger.error(f"No valid responses found in {responses_file}")
        return [], None, None
    
    # Initialize judge
    judge = LLMJudge(config=config, validation_dataset=validation_dataset)
    
    # Evaluate each response
    results = []
    for i, row in enumerate(responses):
        query = row.get('query', '')
        response = row.get('response', '')
        
        if not query or not response:
            logger.warning(f"Skipping record {i+1}: Missing query or response")
            continue
        
        logger.info(f"Evaluating response {i+1}/{len(responses)}: {query[:50]}...")
        
        # Evaluate response
        evaluation = judge.evaluate_response(query, response)
        
        # Add metadata
        evaluation['model'] = row.get('model', 'unknown')
        evaluation['request_duration_ms'] = row.get('request_duration_ms', None)
        
        results.append(evaluation)
        
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    # Extract model name from file path or responses
    model_name = "unknown"
    if len(responses) > 0:
        model_name = responses[0].get('model', 'unknown')
    if model_name == 'unknown':
        # Try to get from file path
        file_basename = os.path.basename(responses_file)
        if '_' in file_basename:
            model_name = file_basename.split('_')[0]
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON - use the same directory as responses_file
    json_output_file = os.path.join(responses_dir, f"{model_name}_evaluation_{timestamp}.json")
    with open(json_output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "model": model_name,
            "responses_count": len(responses),
            "evaluations_count": len(results),
            "results": results
        }, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {json_output_file}")
    
    # Generate and save markdown report
    if report_file:
        md_output_file = report_file
    else:
        md_output_file = os.path.join(responses_dir, f"{model_name}_evaluation_{timestamp}.md")
    
    report = generate_evaluation_report(results, model_name)
    
    # Ensure the directory exists
    md_output_dir = os.path.dirname(md_output_file)
    if md_output_dir and not os.path.exists(md_output_dir):
        try:
            os.makedirs(md_output_dir, exist_ok=True)
            logger.info(f"Created directory: {md_output_dir}")
        except Exception as e:
            logger.error(f"Failed to create directory {md_output_dir}: {e}")
    
    with open(md_output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation report saved to {md_output_file}")
    
    # Calculate and display summary statistics
    if len(results) > 0:
        try:
            # Extract scores
            overall_scores = [r.get('overall', {}).get('score', 0) for r in results if 'overall' in r]
            criteria_scores = {
                criterion: [r.get('criteria', {}).get(criterion, {}).get('score', 0) 
                           for r in results if 'criteria' in r and criterion in r.get('criteria', {})]
                for criterion in config.get('evaluation_criteria', {}).keys()
            }
            
            # Calculate averages
            if overall_scores:
                avg_overall = sum(overall_scores) / len(overall_scores)
                logger.info(f"Average overall score: {avg_overall:.2f}/5.0")
            
            for criterion, scores in criteria_scores.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    logger.info(f"Average {criterion} score: {avg_score:.2f}/5.0")
        
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
    
    return results, json_output_file, md_output_file

def generate_evaluation_report(results: List[Dict[str, Any]], model_name: str = "unknown") -> str:
    """
    Generate a markdown report from the evaluation results.
    
    Args:
        results: List of evaluation results
        model_name: Name of the model being evaluated
        
    Returns:
        Markdown report as a string
    """
    if not results:
        return "# Function Calling Evaluation Report\n\nNo results to report."
    
    # Group results by category
    results_by_category = {}
    for result in results:
        category = result.get("category", "Unknown")
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(result)
    
    # Calculate average scores
    criteria_scores = {}
    overall_scores = []
    
    for result in results:
        if "criteria" in result and "overall" in result:
            # Extract criteria scores
            criteria = result.get("criteria", {})
            for criterion_name, criterion_data in criteria.items():
                if isinstance(criterion_data, dict) and "score" in criterion_data:
                    if criterion_name not in criteria_scores:
                        criteria_scores[criterion_name] = []
                    criteria_scores[criterion_name].append(criterion_data["score"])
            
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
    report.append(f"# Function Calling Evaluation Report - {model_name}")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total responses evaluated: {len(results)}")
    report.append("")
    
    # Overall averages
    report.append("## Overall Average Scores")
    report.append("")
    report.append("| Criterion | Average Score |")
    report.append("|-----------|---------------|")
    for criterion, avg_score in avg_criteria.items():
        report.append(f"| {criterion.replace('_', ' ').title()} | {avg_score:.2f} |")
    report.append(f"| **Overall** | **{avg_overall:.2f}** |")
    report.append("")
    
    # Category-specific averages
    report.append("## Average Scores by Category")
    report.append("")
    
    for category, cat_results in results_by_category.items():
        # Calculate category averages
        cat_criteria_scores = {}
        cat_overall_scores = []
        
        for result in cat_results:
            if "criteria" in result and "overall" in result:
                # Extract criteria scores for this category
                criteria = result.get("criteria", {})
                for criterion_name, criterion_data in criteria.items():
                    if isinstance(criterion_data, dict) and "score" in criterion_data:
                        if criterion_name not in cat_criteria_scores:
                            cat_criteria_scores[criterion_name] = []
                        cat_criteria_scores[criterion_name].append(criterion_data["score"])
                
                # Extract overall score for this category
                if "score" in result["overall"]:
                    cat_overall_scores.append(result["overall"]["score"])
        
        # Calculate category averages
        cat_avg_criteria = {}
        for criterion, scores in cat_criteria_scores.items():
            if scores:
                cat_avg_criteria[criterion] = sum(scores) / len(scores)
            else:
                cat_avg_criteria[criterion] = 0
        
        cat_avg_overall = sum(cat_overall_scores) / len(cat_overall_scores) if cat_overall_scores else 0
        
        # Add to report
        report.append(f"### {category}")
        report.append("")
        report.append("| Criterion | Average Score |")
        report.append("|-----------|---------------|")
        for criterion, avg_score in cat_avg_criteria.items():
            report.append(f"| {criterion.replace('_', ' ').title()} | {avg_score:.2f} |")
        report.append(f"| **Overall** | **{cat_avg_overall:.2f}** |")
        report.append("")
    
    # Results by category
    report.append("## Results by Category")
    report.append("")
    
    for category, cat_results in results_by_category.items():
        report.append(f"### {category}")
        report.append("")
        
        for i, result in enumerate(cat_results, 1):
            query = result.get("query", f"Query {i}")
            report.append(f"#### Test {i}: {query[:50]}{'...' if len(query) > 50 else ''}")
            report.append("")
            report.append(f"**Query:** {query}")
            report.append("")
            report.append(f"**Response:** {result.get('response', 'N/A')}")
            report.append("")
            report.append(f"**Expected Function:** {result.get('expected_function', 'N/A')}")
            report.append("")
            
            if result.get('expected_response'):
                report.append(f"**Expected Response:** {result.get('expected_response')}")
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
            
            report.append("")
            report.append("---")
            report.append("")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Evaluate function calling responses using LLM as a Judge")
    parser.add_argument("--responses_file", required=True, help="Path to the file containing responses (CSV or JSONL)")
    parser.add_argument("--config_file", default=DEFAULT_CONFIG_PATH, 
                        help=f"Path to the configuration JSON file (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--validation_dataset", default=DEFAULT_VALIDATION_DATASET_PATH, 
                        help=f"Path to the validation dataset JSONL file (default: {DEFAULT_VALIDATION_DATASET_PATH})")
    parser.add_argument("--report_file", default=None,
                        help="Path to save the evaluation report (optional)")
    
    args = parser.parse_args()
    
    # Check if required files exist
    if not os.path.isfile(args.responses_file):
        logger.error(f"Responses file not found: {args.responses_file}")
        return 1
    
    if not os.path.isfile(args.config_file):
        logger.error(f"Configuration file not found: {args.config_file}")
        return 1
    
    if not os.path.isfile(args.validation_dataset):
        logger.error(f"Validation dataset not found: {args.validation_dataset}")
        return 1
    
    # Evaluate responses
    results, json_path, md_path = evaluate_responses_file(
        responses_file=args.responses_file,
        config_file=args.config_file,
        validation_dataset_file=args.validation_dataset,
        report_file=args.report_file
    )
    
    # Display summary
    if results:
        logger.info(f"Evaluation complete. {len(results)} responses evaluated.")
        logger.info(f"Results saved to {json_path}")
        logger.info(f"Report saved to {md_path}")
        
        # Calculate and display average scores
        overall_scores = [r.get('overall', {}).get('score', 0) for r in results if 'overall' in r]
        if overall_scores:
            avg_overall = sum(overall_scores) / len(overall_scores)
            logger.info(f"Average overall score: {avg_overall:.2f}/5.0")
    else:
        logger.error("No results generated.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
