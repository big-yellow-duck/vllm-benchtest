# vLLM Benchmarking Tool

Automated benchmarking script for vLLM model serving performance across multiple configurations and token lengths.

## Install dependencies first

```bash
# Install vLLM and dependencies
# Alternatively, use the vLLM docker container
pip install vllm

# install dependencies
pip install -r requirements.txt

```
## Quick Start

1. **Copy a sample config:**
   ```bash
   cp configs/example_config.yaml configs/my_bench.yaml
   ```

2. **Edit the config** with your model and test parameters

3. **Run benchmarks:**
   ```bash
   python vllm_bench.py --config configs/my_bench.yaml
   ```

## Configuration

Configuration files use **YAML format** and define test cases with:
- `test_cases`: List of test identifiers
- `env_vars`: Environment variables per test (string or dict format)
- `model_params`: Model name and vLLM args per test
- `VLLM_PORT`: API server port (default: 11169)
- `output_dir`: Output directory for logs and results
- `in_out_lengths`: Array of input/output token length pairs
- `max_concurrency`: Optional max concurrency shared across test cases (default: 64)
- `num_prompts`: Optional prompt count shared across test cases (default: 640)
- `WAIT_RETRIES`: Optional server readiness check retries (default: 100)
- `WAIT_TIME`: Optional wait time between checks in seconds (default: 10)
- `plot_config`: Optional post-run plotting config with `baseline` and `overview_title`

**Example:**
```yaml
test_cases:
  - baseline
  - optimized

env_vars:
  baseline: "CUDA_VISIBLE_DEVICES=0"
  optimized: "CUDA_VISIBLE_DEVICES=0 VLLM_ROCM_USE_AITER=0"

model_params:
  baseline: "facebook/opt-125m --max-model-len 2048"
  optimized: "facebook/opt-125m --max-model-len 2048 --kv-cache-dtype fp8"

VLLM_PORT: 11169
output_dir: "benchmark_results"

in_out_lengths:
  - in: 100
    out: 100
  - in: 512
    out: 512
  - in: 1024
    out: 1024

max_concurrency: 64

num_prompts: 640

plot_config:
  baseline: baseline
  overview_title: "Benchmark Comparison Overview"
```

## Usage

```bash
# Basic run
python vllm_bench.py --config <config_file>

# Stop on any server failure
python vllm_bench.py --config <config_file> --stop-on-server-failure

# Show live server logs during startup
python vllm_bench.py --config <config_file> --live-logs
```

If `plot_config` is present, `vllm_bench.py` automatically runs `plot_benchmark_results.py` after benchmarking completes with:
- `--results-dir <output_dir_timestamp>/results`
- `--output-dir <output_dir_timestamp>/plots`
- optional `--baseline` from `plot_config.baseline`
- optional `--overview-title` from `plot_config.overview_title`

## Output Structure

```
logs/
└── serverlog_<testcase>.log

results/
└── <folder_name>/
    ├── ISL-<in>-OSL-<out>.json
    └── ...

plots/
├── comparison_overview.png
├── mean_ttft_ms.png
└── ...

benchmark_<timestamp>.log  # Master log file
```

## Available Sample Configs

- `example_config.yaml` - Basic template with all configuration options
- `cuda_graph_sample.yaml` - CUDA Graph mode comparison
- `sample_config.yaml` - Minimal configuration example

## Cleanup

Kill stray vLLM processes:
```bash
pkill -f "^vllm serve"
```

## Notes

- Clears `~/.cache/vllm` before each test
- Each test runs `vllm bench serve` for all input/output length pairs
- Results saved as JSON with throughput and latency metrics
- `max_concurrency` and `num_prompts` accept a single integer for all test cases; the older per-test mapping format is still accepted for compatibility
