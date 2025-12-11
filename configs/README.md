# Configuration Files for vLLM Benchmarks

This directory contains configuration files for running vLLM benchmarks. Each JSON file defines a set of test cases with their respective model parameters, environment variables, and output configurations.

## Configuration File Structure

Each configuration file must contain the following fields:

- **`test_cases`**: Array of test case identifiers
- **`env_vars`**: Environment variables for each test case (key-value pairs)
- **`model_params`**: Model name and VLLM parameters for each test case
- **`folder_name`**: Output folder name for each test case's results
- **`VLLM_PORT`**: Port number for the VLLM API server
- **`LOGS_PATH`**: Directory path to store log files
- **`RESULT_DIR`**: Directory path to store benchmark results
- **`in_out_lengths`**: Array of input/output token length configurations

## Sample Configurations

### 1. `sample_config.json`
A basic configuration file that demonstrates the structure with simple test cases. Use this as a starting point for creating your own configs.

### 2. `cuda_graph_sample.json`
Shows how to benchmark different CUDA Graph modes in vLLM:
- **FULL**: Captures the entire forward pass
- **PIECEWISE**: Captures parts of computation
- **NONE**: No CUDA Graph optimization

### 3. `example_config.json`
Original example configuration with multiple test cases and model variants.

## Creating Your Own Configuration

1. Copy one of the sample configurations:
   ```bash
   cp configs/sample_config.json configs/my_config.json
   ```

2. Modify the fields according to your requirements:
   - Set your desired model in `model_params`
   - Adjust environment variables in `env_vars`
   - Define test cases in `test_cases`
   - Configure input/output lengths for your benchmark scenario

3. Run the benchmark:
   ```bash
   python vllm_bench.py configs/my_config.json
   ```

## Notes

- All config files (except samples) are ignored by git to prevent exposing sensitive information
- Make sure to use appropriate model names from Hugging Face or your local model paths
- Adjust `VLLM_PORT` if running multiple benchmarks simultaneously
- Results will be saved in the specified `RESULT_DIR` with subdirectories for each test case