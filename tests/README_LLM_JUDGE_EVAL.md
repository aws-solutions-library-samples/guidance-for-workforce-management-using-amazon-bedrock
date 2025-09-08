# LLM as a Judge Evaluation for Workforce Management Tests

This directory contains scripts for evaluating the performance of LLM-based systems using the "LLM as a Judge" approach. Two evaluation scripts are provided:

1. `llm_judge_text_eval.py` - For evaluating text based interactions via REST API
2. `llm_judge_s2s_eval.py` - For evaluating speech-to-speech interactions via WebSocket

## Prerequisites

Before running the evaluation scripts, ensure you have:

1. AWS credentials configured with access to Amazon Bedrock and Cognito
2. The required Python packages installed:
   ```
   pip install boto3 pandas dotenv pydub numpy requests websockets
   ```
3. Response data from the test scripts in JSONL format

4. For speech-to-speech evaluation, ensure .raw audio files are available in `./audio_samples/`

## text-based interactions

### Run the text-based interaction tests

1. First, run the function calling test script to generate response data:

```bash
python text_function_calling_tests.py --num-runs 3 --delay 60 --model haiku_35
```

Key parameters:
- `--num-runs`: Number of test runs to execute (default: 1)
- `--delay`: Delay between test runs in seconds (default: 60)
- `--model`: Model name to use for the test (default: nova_pro)
- `--dataset`: Path to validation dataset JSONL file (default: data/function_calling_validation_dataset.jsonl)
- `--category`: Filter test cases by category (e.g., 'Operations', 'Personalization', 'HR')

**Important Notes:**
- The script uses a 300-second timeout for API requests to handle backend processing delays
- Includes enhanced retry mechanisms with exponential backoff for delayed responses
- Automatically handles JWT token refresh for long-running test sessions
- Supports rate limiting detection and appropriate backoff strategies

### Run text based interaction evaluation

2. Then run the evaluation script:

```bash
python llm_judge_text_eval_agent_core_observability.py \
  --responses_file ./responses/model_haiku_35/haiku_35_function_calling_responses.jsonl \
  --config_file ./config/llm_judge_text_config.json \
  --validation_dataset ./data/text_validation_dataset.jsonl \
  --report_file ./evaluation_results/haiku_35_evaluation_report.md
```

### Evaluation Parameters

- `--responses_file`: (Required) Path to the JSONL file containing responses
- `--config_file`: Path to configuration JSON file (default: config/llm_judge_s2s_config.json)
- `--validation_dataset`: Path to validation dataset JSONL file (default: data/s2s_validation_dataset.jsonl)
- `--report_file`: Path to save evaluation report in markdown


## Speech-to-Speech interactions

### Running the Speech-to-Speech Tests

1. First, run the speech-to-speech test script to generate response data:

```bash
python s2s_function_calling_tests.py --model nova_sonic --num-runs 3 --delay 60.0 --run-delay 10.0
```

Key parameters:
- `--model`: Model name to use for the test (default: nova_pro)
- `--num-runs`: Number of test runs to execute (default: 1)
- `--delay`: Delay between tests within each run in seconds (default: 5.0)
- `--run-delay`: Delay between test runs in seconds (default: 5.0)

**Important Notes:**
- Uses backend endpoint from the configuration in the source/frontend directory
- Supports tool use detection and token usage tracking
- Automatically handles session timeouts and reconnections

This script will:
- Authenticate with Cognito and obtain JWT tokens
- Connect to the specified WebSocket server with proper authentication
- Process all .raw audio files from the audio_samples directory
- Run the specified number of test iterations, creating run-specific directories
- Save responses in JSON and WAV formats to the responses/model_sonic directory
- Track token usage and cost information for each session
- Create a JSONL file summarizing results for each run
- Merge results from all runs into a single JSONL file
- Display comprehensive summary statistics including token usage

### Evaluate the Speech-to-Speech Tests

2. Run the evaluation script:

```bash
python llm_judge_s2s_eval_agent_core_observability.py \
  --responses_file ./responses/model_nova_sonic/nova_sonic_function_calling_responses.jsonl \
  --config_file ./config/llm_judge_s2s_config.json \
  --validation_dataset ./data/s2s_validation_dataset.jsonl \
  --report_file ./evaluation_results/nova_sonic_evaluation_report.md
```

### Evaluation Parameters

- `--responses_file`: (Required) Path to the JSONL file containing responses
- `--config_file`: Path to configuration JSON file (default: config/llm_judge_s2s_config.json)
- `--validation_dataset`: Path to validation dataset JSONL file (default: data/s2s_validation_dataset.jsonl)
- `--report_file`: Path to save evaluation report in markdown



## Evaluation Results

The generated report is comprised of two files, a markdown and json file, each containing the detailed evaluation results.

The scripts also display summary statistics in the console, including average scores for each criterion and an overall score.

## Customizing the Evaluation

You can customize the evaluation by:
1. Modifying the evaluation criteria in the config file (`config/llm_judge_s2s_config.json`)
2. Changing the judge model by editing the `judge_model.id` in the config file
3. Updating the validation dataset to reflect different expected functions, transcriptions, or responses

## Example Workflow

1. **Set up environment variables** in your `.env` file as outlined in the overall README.MD

2. **Run the test scripts** to generate response data

3. **Run the evaluation scripts**

4. **Compare the evaluation results** to assess system performance and identify areas for improvement.

## Enhanced Features

### Advanced Retry Logic
- **Exponential backoff**: Intelligent retry timing that adapts to server load
- **Token refresh**: Automatic JWT token renewal during long test sessions
- **Rate limit handling**: Proper detection and handling of API rate limits
- **Connection recovery**: Automatic reconnection for WebSocket failures

### Comprehensive Monitoring
- **Real-time logging**: Detailed progress tracking during test execution
- **Token usage**: Live monitoring of input/output token consumption
- **Cost tracking**: Automatic cost calculation based on current pricing
- **Performance metrics**: Request duration and response time analysis