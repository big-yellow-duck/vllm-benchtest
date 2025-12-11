# vLLM Benchmarking Tool

Automated benchmarking script for vLLM model serving performance across multiple configurations and token lengths.

## Requirements

```bash
# Install vLLM (choose your platform)
pip install vllm

# For AMD GPUs
pip install vllm[rocm]

# For NVIDIA GPUs
pip install vllm[cuda]

# Additional dependencies
pip install requests
```

## Quick Start

1. **Copy a sample config:**
   ```bash
   cp configs/sample_config.json configs/my_bench.json
   ```

2. **Edit the config** with your model and test parameters

3. **Run benchmarks:**
   ```bash
   python vllm_bench.py --config configs/my_bench.json
   ```

## Configuration

Config files define test cases with:
- `test_cases`: List of test identifiers
- `env_vars`: Environment variables per test
- `model_params`: Model name and vLLM args per test
- `folder_name`: Output directory name per test
- `VLLM_PORT`: API server port
- `in_out_lengths`: Array of `{ "in": tokens, "out": tokens }`

**Example:**
```json
{
  "test_cases": ["baseline", "optimized"],
  "env_vars": {
    "baseline": "CUDA_VISIBLE_DEVICES=0",
    "optimized": "CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=FLASHINFER"
  },
  "model_params": {
    "baseline": "facebook/opt-125m --max-model-len 2048",
    "optimized": "facebook/opt-125m --max-model-len 2048 --kv-cache-dtype fp8"
  },
  "folder_name": {
    "baseline": "baseline_results",
    "optimized": "optimized_results"
  },
  "VLLM_PORT": 11169,
  "in_out_lengths": [
    {"in": 100, "out": 100},
    {"in": 512, "out": 512},
    {"in": 1024, "out": 1024}
  ]
}
```

## Usage

```bash
# Basic run
python vllm_bench.py --config <config_file>

# Stop on any server failure
python vllm_bench.py --config <config_file> --stop-on-server-failure
```

## Output Structure

```
logs/
└── serverlog_<testcase>.log

results/
└── <folder_name>/
    ├── ISL-<in>-OSL-<out>.json
    └── ...

benchmark_<timestamp>.log  # Master log file
```

## Available Sample Configs

- `sample_config.json` - Basic template
- `cuda_graph_sample.json` - CUDA Graph mode comparison
- `example_config.json` - Original example

## Cleanup

Kill stray vLLM processes:
```bash
pkill -f "^vllm serve"
```

## Notes

- Clears `~/.cache/vllm` before each test
- Each test runs `vllm bench serve` for all input/output length pairs
- Results saved as JSON with throughput and latency metrics