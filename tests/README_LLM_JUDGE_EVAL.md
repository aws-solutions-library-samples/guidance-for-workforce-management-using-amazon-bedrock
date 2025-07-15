# LLM as a Judge Evaluation for Workforce Management Tests
TODO:
- usage data capturing is not complete yet

This directory contains scripts for evaluating the performance of LLM-based systems using the "LLM as a Judge" approach. Two evaluation scripts are provided:

1. `llm_judge_function_calling_eval.py` - For evaluating function calling via REST API
2. `llm_judge_s2s_eval.py` - For evaluating speech-to-speech interactions via WebSocket

## Prerequisites

Before running the evaluation scripts, ensure you have:

1. AWS credentials configured with access to Amazon Bedrock
2. The required Python packages installed:
   ```
   pip install boto3 pandas dotenv pydub numpy requests
   ```
3. Response data from the test scripts in JSONL format

## Function Calling Evaluation

The function calling evaluation script assesses responses from the retail assistant API based on:
- Correctness - Does the response correctly answer the query?
- Function Call Accuracy - Was the appropriate function called?
- Relevance - Is the response relevant to the query?
- Completeness - Does the response provide all necessary information?
- Clarity - Is the response clear and easy to understand?

### Running the Function Calling Tests

1. First, run the function calling test script to generate response data:

```bash
python nova_pro_text_function_calling_tests.py --num-runs 1 --delay 60 --model nova_pro
```

Key parameters:
- `--num-runs`: Number of test runs to execute (default: 1)
- `--delay`: Delay between test runs in seconds (default: 60)
- `--model`: Model name to use for the test (default: nova_pro)
- `--dataset`: Path to validation dataset JSONL file (default: data/function_calling_validation_dataset.jsonl)

Note: The script uses a 120-second timeout for API requests and includes retry mechanisms with exponential backoff to handle cases where responses are delayed.

2. Then run the evaluation script:

```bash
python llm_judge_s2s_eval.py \
  --responses_file ./responses/model_sonic/s2s_sonic_function_calling_responses.jsonl \
  --config_file ./config/llm_judge_s2s_config.json \
  --validation_dataset ./data/s2s_validation_dataset.jsonl \
  --report_file ./evaluation_results/s2s_sonic_evaluation_report.md
```

### Evaluation Parameters

- `--responses_file`: (Required) Path to the JSONL file containing responses
- `--config_file`: Path to configuration JSON file (default: config/llm_judge_s2s_config.json)
- `--validation_dataset`: Path to validation dataset JSONL file (default: data/s2s_validation_dataset.jsonl)
- `--report_file`: Path to save evaluation report (optional)
- `--debug`: Enable debug logging (flag)
- `--retrieve_function_calls`: Retrieve function calls from CloudWatch logs (default: True)
- `--log_group_name`: CloudWatch log group name to search for function calls (optional)

## Speech-to-Speech Evaluation

The speech-to-speech evaluation script assesses responses from the Nova Sonic API based on:
- Speech Recognition Accuracy - How well the system recognized the spoken input
- Response Relevance - How relevant the response is to the recognized input
- Response Correctness - How accurate the information in the response is
- Function Call Accuracy - Whether the appropriate function was called
- Speech Synthesis Quality - The quality of the synthesized speech output

### Running the Speech-to-Speech Tests

1. First, run the speech-to-speech test script to generate response data:

```bash
python nova_sonic_s2s_function_calling_tests.py --num-runs 3 --delay 60.0 --run-delay 10.0 --server-url "http://localhost:8000/ws"
```

Key parameters:
- `--server-url`: WebSocket server URL (default: http://localhost:8000/ws)
- `--num-runs`: Number of test runs to execute (default: 1)
- `--delay`: Delay between tests within each run in seconds (default: 5.0)
- `--run-delay`: Delay between test runs in seconds (default: 5.0)

This script will:
- Connect to the specified WebSocket server
- Process all .raw audio files from the audio_samples directory
- Run the specified number of test iterations, creating run-specific directories
- Save responses in JSON and WAV formats to the responses/model_sonic directory
- Create a JSONL file summarizing results for each run
- Merge results from all runs into a single JSONL file
- Display summary statistics about success rates

2. Then run the evaluation script:

```bash
python llm_judge_s2s_eval.py \
  --responses_file ./responses/model_sonic/s2s_sonic_function_calling_responses.jsonl \
  --config_file ./config/llm_judge_s2s_config.json \
  --validation_dataset ./data/s2s_validation_dataset.jsonl \
  --report_file ./evaluation_results/s2s_sonic_evaluation_report.md
```

### Evaluation Parameters

- `--responses_file`: (Required) Path to the JSONL file containing responses
- `--config_file`: Path to configuration JSON file (default: config/llm_judge_s2s_config.json)
- `--validation_dataset`: Path to validation dataset JSONL file (default: data/s2s_validation_dataset.jsonl)
- `--report_file`: Path to save evaluation report (optional)
- `--debug`: Enable debug logging (flag)
- `--retrieve_function_calls`: Retrieve function calls from CloudWatch logs (default: True)
- `--log_group_name`: CloudWatch log group name to search for function calls (optional)

## Evaluation Results

Both scripts generate two output files:
1. A CSV file with evaluation scores and metadata
2. A JSON file with detailed evaluation results including explanations

The scripts also display summary statistics in the console, including average scores for each criterion and an overall score.

## Customizing the Evaluation

You can customize the evaluation by:
1. Modifying the evaluation criteria in the config file (`config/llm_judge_s2s_config.json`)
2. Changing the judge model by editing the `judge_model.id` in the config file
3. Updating the validation dataset to reflect different expected functions, transcriptions, or responses

## Example Workflow

1. Run the test scripts to generate response data:
   ```
   cd tests
   python nova_pro_text_function_calling_tests.py --num-runs 3 --delay 60
   python nova_sonic_s2s_function_calling_tests.py --num-runs 3 --delay 60 --run-delay 10.0 --server-url https://backend.<yourdomain>
   ```

2. Run the evaluation scripts:
   ```
   python llm_judge_function_calling_eval.py --model_name nova_pro --responses_file ./responses/model_nova_pro/nova_pro_function_calling_responses.jsonl
   python llm_judge_s2s_eval.py --responses_file ./responses/model_sonic/s2s_sonic_function_calling_responses.jsonl
   ```

3. Compare the evaluation results to assess system performance.

## Notes

- The evaluation scripts use Amazon Bedrock to run the judge model, so ensure you have appropriate permissions and quota.
- The scripts include rate limiting (1 second between evaluations) to avoid hitting API rate limits.
- For speech-to-speech evaluation, audio features are extracted from the output files if available.
- The text function calling tests include explicit timeouts (120 seconds for queries, 60 seconds for reset operations) to handle server processing delays.
- Both test scripts include retry mechanisms with exponential backoff to handle cases where responses are delayed or initially empty.
- The speech-to-speech tests rely on the `s2s_test_harness.py` utility for core WebSocket and audio processing functionality.
- When running speech-to-speech tests, ensure that .raw audio files are available in the ./audio_samples directory.
