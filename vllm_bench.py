#!/usr/bin/env python3
"""
vLLM Benchmark Script - Python Implementation

This script benchmarks vLLM performance with different configurations and parameters.
It replaces the original bash script with a more robust Python implementation using
subprocess for better process management and error handling.
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import requests


def tail_log_file(file_path: str, lines: int = 10) -> str:
    """Tail a log file and return the last N lines"""
    try:
        with open(file_path, "r") as f:
            # Read the last N lines
            file_lines = f.readlines()
            return "".join(file_lines[-lines:]) if file_lines else ""
    except Exception:
        return ""


class VLLMBenchmark:
    """Main class for orchestrating vLLM benchmarks"""

    def __init__(
        self, config_file: str, stop_on_failure: bool = False, live_logs: bool = False
    ):
        """
        Initialize the benchmark runner

        Args:
            config_file: Path to configuration file
            stop_on_failure: Whether to stop on server failure
            live_logs: Whether to show live log tailing during server startup
        """
        self.config_file = config_file
        self.stop_on_failure = stop_on_failure
        self.live_logs = live_logs
        self.current_server_log_file = None  # Track current server log for live tailing

        # Default configuration (can be overridden by config file)
        self.vllm_port = 11169
        self.logs_path = "default_logs"
        self.result_dir = "default_results"
        # Temporary console logger before config is loaded
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)
        # Default wait settings (can be overridden by config file)
        self.wait_retries = 100
        self.wait_time = 10

        # Input/output lengths for random benchmarks
        self.in_len = [2000, 1000, 5000, 10000, 3200]
        self.out_len = [150, 1000, 1000, 1000, 800]
        self.in_out_lengths = []

        # Configuration arrays (will be populated by config file)
        self.test_cases = []
        self.env_vars = {}
        self.model_params = {}
        self.folder_name = {}

        # Process tracking
        self.server_process = None
        self.benchmark_process = None

        # Results tracking
        self.test_results = {}

        # Load configuration
        self.load_config()
        # Ensure log directory exists before full logging setup
        Path(self.logs_path).mkdir(parents=True, exist_ok=True)
        # Setup logging
        self.setup_logging()

        # Register signal handlers for cleanup
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle SIGINT signal for graceful shutdown"""
        self.logger.info("Caught Ctrl+C. Killing background processes...")
        if self.server_process:
            self.server_process.terminate()
        if self.benchmark_process:
            self.benchmark_process.terminate()
        sys.exit(1)

    def setup_logging(self):
        """Configure logging for the benchmark"""
        # Ensure logs directory exists
        log_dir = Path(self.logs_path)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create benchmark log file with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.benchmark_log = log_dir / f"benchmark_{timestamp}.log"

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(self.benchmark_log)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(file_handler)

        self.logger = logger

    def load_config(self):
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_file):
            self.logger.error(f"Configuration file {self.config_file} not found!")
            sys.exit(1)

        self.logger.info(f"Loading configuration from {self.config_file}")

        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)

            # Extract configuration values
            self.test_cases = config.get("test_cases", [])
            self.env_vars = config.get("env_vars", {})
            self.model_params = config.get("model_params", {})
            self.folder_name = config.get("folder_name", {})
            self.vllm_port = config.get("VLLM_PORT", self.vllm_port)
            self.logs_path = config.get("LOGS_PATH", self.logs_path)
            self.result_dir = config.get("RESULT_DIR", self.result_dir)

            # Extract benchmark parameters
            self.max_concurrency = config.get("max_concurrency", {})
            self.num_prompts = config.get("num_prompts", {})

            # Enforce the new in_out_lengths format
            if "in_out_lengths" not in config:
                self.logger.error(
                    "Configuration must include 'in_out_lengths' array with objects containing 'in' and 'out' properties"
                )
                self.logger.error(
                    'Example: "in_out_lengths": [{"in": 1000, "out": 1000}, {"in": 2000, "out": 2000}]'
                )
                sys.exit(1)

            self.in_out_lengths = config["in_out_lengths"]

            # Load optional wait settings with defaults
            if "WAIT_RETRIES" in config:
                self.wait_retries = config["WAIT_RETRIES"]
            else:
                self.logger.info(
                    "WAIT_RETRIES not provided in config; using default of 100"
                )
            if "WAIT_TIME" in config:
                self.wait_time = config["WAIT_TIME"]
            else:
                self.logger.info(
                    "WAIT_TIME not provided in config; using default of 10"
                )

            # Validate in_out_lengths format
            if not isinstance(self.in_out_lengths, list) or not self.in_out_lengths:
                self.logger.error("'in_out_lengths' must be a non-empty array")
                sys.exit(1)

            for i, length_config in enumerate(self.in_out_lengths):
                if (
                    not isinstance(length_config, dict)
                    or "in" not in length_config
                    or "out" not in length_config
                ):
                    self.logger.error(
                        f"Invalid format at in_out_lengths[{i}]. Each item must be an object with 'in' and 'out' properties"
                    )
                    sys.exit(1)

                if not isinstance(length_config["in"], int) or not isinstance(
                    length_config["out"], int
                ):
                    self.logger.error(
                        f"Invalid values at in_out_lengths[{i}]. 'in' and 'out' must be integers"
                    )
                    sys.exit(1)

            self.logger.info(
                f"Loaded {len(self.test_cases)} test cases from configuration"
            )

        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON configuration: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            sys.exit(1)

    def get_log_filename(self, name: str) -> str:
        """Convert test case name to valid filename"""
        # Replace spaces and slashes with underscores
        return re.sub(r"[\/ ]", "_", name)

    def is_server_running(self) -> bool:
        """Check if server process is still running"""
        if self.server_process is None:
            return False

        # Check if process is still running
        if self.server_process.poll() is not None:
            self.logger.info(
                f"Server process {self.server_process.pid} is no longer running."
            )
            return False

        return True

    def check_server_health(self) -> Tuple[bool, str]:
        """Check server health and return status with message"""

        try:
            response = requests.get(
                f"http://localhost:{self.vllm_port}/v1/models", timeout=5
            )
            if response.status_code == 200:
                return True, "Server is responding"
            else:
                return False, f"Server returned status code {response.status_code}"
        except requests.RequestException as e:
            return False, f"Server is not responding to HTTP requests: {e}"

    def wait_for_server_ready(self) -> bool:
        """Wait for server to be ready"""
        self.logger.info("Waiting for server to be ready...")

        for i in range(self.wait_retries):
            # check if server is alive
            if not self.is_server_running():
                return False
            is_healthy, message = self.check_server_health()

            if is_healthy:
                self.logger.info("Server is ready!")
                return True

            # Show live log tail if enabled
            if self.live_logs and self.current_server_log_file:
                log_tail = tail_log_file(self.current_server_log_file, 5)
                if log_tail:
                    self.logger.info(f"Server log tail:\n{log_tail}")

            self.logger.info(
                f"Waiting for server to be ready... (attempt {i + 1}/{self.wait_retries}): {message}"
            )
            time.sleep(self.wait_time)

        self.logger.error("Server did not become ready in time.")

        # Show final server log content if available
        if self.live_logs and self.current_server_log_file:
            try:
                with open(self.current_server_log_file, "r") as f:
                    full_log = f.read()
                    if full_log:
                        self.logger.error(f"Full server log content:\n{full_log}")
            except Exception as e:
                self.logger.error(f"Could not read full server log: {e}")

        return False

    def start_server(self, test_case: str) -> bool:
        """Start vLLM server for the given test case"""
        self.logger.info(f"Starting vLLM server for test case: {test_case}")

        # Prepare environment variables
        env = os.environ.copy()

        # Add test case specific environment variables
        if test_case in self.env_vars:
            test_env = self.env_vars[test_case]
            if isinstance(test_env, dict):
                env.update(test_env)
            elif isinstance(test_env, str):
                # Parse string environment variables
                for env_var in test_env.split():
                    if "=" in env_var:
                        key, value = env_var.split("=", 1)
                        env[key] = value

        # Clear vLLM cache
        cache_dir = Path.home() / ".cache" / "vllm"
        if cache_dir.exists():
            subprocess.run(["rm", "-rf", str(cache_dir)], check=True)

        # Prepare server command
        if test_case not in self.model_params:
            self.logger.error(f"No model parameters found for test case: {test_case}")
            return False

        model_params = self.model_params[test_case]
        
        # Sanitize JSON strings in model parameters to remove escaping issues
        # Split the parameters and sanitize any JSON content
        params_list = model_params.split()
        sanitized_params = []
        
        for param in params_list:
            # Check if this parameter contains JSON content (starts with single quote and has JSON structure)
            if param.startswith("'") and param.endswith("'") and ("{" in param and "}" in param):
                # Remove the surrounding single quotes from JSON strings
                param = param.strip("'")
            sanitized_params.append(param)
        
        cmd = (
            ["vllm", "serve"]
            + sanitized_params
            + [
                "--trust-remote-code",
                "--port",
                str(self.vllm_port),
            ]
        )

        # Prepare server log file with timestamp for uniqueness
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        server_log_file = (
            Path(self.logs_path)
            / f"serverlog_{self.get_log_filename(test_case)}_{timestamp}.log"
        )

        self.logger.info(f"Server command: {' '.join(cmd)}")
        self.logger.info(f"Server log file: {server_log_file}")

        try:
            # Start server process and write command to log file first
            # Use unbuffered mode for real-time logging
            with open(server_log_file, "w", buffering=1) as log_file:  # line buffering
                log_file.write(f"=== Server Command ===\n{' '.join(cmd)}\n")
                log_file.write(f"=== Environment Variables ===\n")
                for key, value in env.items():
                    log_file.write(f"{key}={value}\n")
                log_file.write(f"=== Server Output ===\n")
                log_file.flush()  # Ensure header is written before starting process

                # sanitized_cmd = [c.replace("\\", "") for c in cmd]
                print(cmd)
                self.server_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    bufsize=1,  # line buffering
                )

            self.logger.info(f"Server started with PID: {self.server_process.pid}")
            self.current_server_log_file = str(
                server_log_file
            )  # Track for live tailing

            # Give the server a moment to start and check if it immediately fails
            time.sleep(2)
            if not self.is_server_running():
                self.logger.error(
                    f"Server process died immediately for test case: {test_case}"
                )
                # Read the server log to see what went wrong
                try:
                    with open(server_log_file, "r") as f:
                        server_log_content = f.read()
                        if server_log_content:
                            self.logger.error(
                                f"Server log content:\n{server_log_content}"
                            )
                except Exception as log_error:
                    self.logger.error(f"Could not read server log: {log_error}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            # Try to capture any error output
            try:
                with open(server_log_file, "a") as log_file:
                    log_file.write(f"\nError starting server: {e}\n")
            except Exception as log_error:
                self.logger.error(f"Could not write error to server log: {log_error}")
            return False

    def stop_server(self) -> bool:
        """Stop the vLLM server"""
        if not self.is_server_running():
            self.logger.info("Server process already died.")
            return True

        if self.server_process is None:
            self.logger.info("Server process is None.")
            return True

        self.logger.info(f"Stopping vLLM server (PID: {self.server_process.pid})...")

        try:
            # Try graceful termination first
            self.server_process.terminate()

            # Wait for graceful termination
            wait_count = 0
            while self.is_server_running() and wait_count < 60:
                time.sleep(1)
                wait_count += 1

            # If server is still running, force kill
            if self.is_server_running():
                self.logger.info("Server did not terminate gracefully, forcing kill...")
                self.server_process.kill()
            else:
                self.logger.info("Server terminated gracefully.")

            return True

        except Exception as e:
            self.logger.error(f"Error stopping server: {e}")
            return False

    def run_benchmark(self, test_case: str, in_len: int, out_len: int) -> bool:
        """Run a single benchmark with given parameters"""
        self.logger.info(
            f"Running random benchmark: in_len={in_len}, out_len={out_len}"
        )

        # Extract model name from model_params
        model_name = self.model_params[test_case].split()[0]

        # Prepare result directory
        res_dir = Path(self.result_dir) / self.folder_name[test_case]
        res_dir.mkdir(parents=True, exist_ok=True)

        # Get benchmark parameters for this test case or use defaults
        max_concurrency = self.max_concurrency.get(test_case, 64)
        num_prompts = self.num_prompts.get(test_case, 640)

        # Prepare benchmark command
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--model",
            model_name,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(in_len),
            "--save-result",
            "--result-filename",
            f"ISL-{in_len}-OSL-{out_len}.json",
            "--result-dir",
            str(res_dir),
            "--port",
            str(self.vllm_port),
            "--max-concurrency",
            str(max_concurrency),
            "--num-prompts",
            str(num_prompts),
            "--ignore-eos",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--random-output-len",
            str(out_len),
        ]

        # Prepare log file
        log_file = Path(self.logs_path) / f"{self.get_log_filename(test_case)}.log"

        try:
            # Start benchmark process with line buffering for real-time logging
            with open(log_file, "a", buffering=1) as f:  # line buffering
                f.write(
                    f"\n=== Benchmark started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
                f.write(f"Command: {' '.join(cmd)}\n")
                f.flush()  # Ensure header is written

                self.benchmark_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    bufsize=1,  # line buffering
                )

            self.logger.info(
                f"Benchmark started with PID: {self.benchmark_process.pid}"
            )

            # Monitor both server and benchmark
            while self.benchmark_process.poll() is None:
                # Check if server is still running
                if not self.is_server_running():
                    self.logger.error("Server died during benchmark")
                    self.benchmark_process.terminate()
                    return False

                time.sleep(5)

            # Wait for benchmark to complete
            self.benchmark_process.wait()

            return self.benchmark_process.returncode == 0

        except Exception as e:
            self.logger.error(f"Error running benchmark: {e}")
            return False

    def run_test_case(self, test_case: str) -> bool:
        """Run all benchmarks for a single test case"""
        self.logger.info(f"Running test case: {test_case}")

        # Skip test cases marked as FAILED
        if "FAILED" in test_case:
            self.logger.info(f"Skipping failed test case: {test_case}")
            self.test_results[test_case] = "SKIPPED"
            return False

        # Initialize test case result
        self.test_results[test_case] = "UNKNOWN"

        # Start server
        if not self.start_server(test_case):
            self.logger.error(f"Failed to start server for test case: {test_case}")
            self.test_results[test_case] = "FAILED"

            if self.stop_on_failure:
                self.logger.error("STOP ON FAILURE mode enabled. Aborting script.")
                self.generate_summary()
                sys.exit(1)

            return False

        # Wait for server to be ready
        time.sleep(3)  # Give server time to start
        if not self.wait_for_server_ready():
            self.logger.error(f"Server failed to start for test case: {test_case}")
            self.test_results[test_case] = "FAILED"

            # Stop server if it's running
            if self.is_server_running():
                self.stop_server()

            if self.stop_on_failure:
                self.logger.error("STOP ON FAILURE mode enabled. Aborting script.")
                self.generate_summary()
                sys.exit(1)

            return False

        # Run benchmarks with different input/output lengths
        benchmark_success = True

        # Run benchmarks using the in_out_lengths format
        for length_config in self.in_out_lengths:
            in_len = length_config["in"]
            out_len = length_config["out"]

            # Check if server is still running before each benchmark
            if not self.is_server_running():
                self.logger.error(
                    f"Server died before benchmark for test case: {test_case}"
                )
                benchmark_success = False
                break

            # Run benchmark
            if not self.run_benchmark(test_case, in_len, out_len):
                self.logger.error(f"Benchmark failed for test case: {test_case}")
                benchmark_success = False
                break

            # Sleep between benchmarks
            time.sleep(10)

        # Determine test case result
        if benchmark_success and self.is_server_running():
            self.logger.info(f"Test case completed successfully: {test_case}")
            self.test_results[test_case] = "SUCCESS"
        else:
            self.logger.error(f"Test case failed: {test_case}")
            self.test_results[test_case] = "FAILED"

        # Append updated summary to benchmark log
        self.append_summary_to_log(test_case)

        # Stop server
        self.stop_server()

        # Sleep between test cases
        time.sleep(30)

        return benchmark_success

    def append_summary_to_log(self, test_case: str):
        """Append updated summary to benchmark log"""
        if not self.benchmark_log:
            return

        try:
            with open(self.benchmark_log, "a") as f:
                f.write("\n")
                f.write("=" * 42 + "\n")
                f.write(
                    "PER-RUN BENCHMARK SUMMARY (after test case: " + test_case + ")\n"
                )
                f.write("Runtime: " + time.strftime("%Y%m%d_%H%M%S") + "\n")
                f.write("Timestamp: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("-" * 42 + "\n")

                # Generate summary
                summary = self.generate_summary(return_string=True)
                if summary:
                    f.write(summary)

                f.write("=" * 42 + "\n")
                f.write("\n")

            self.logger.info(f"Appended updated summary to {self.benchmark_log}")

        except Exception as e:
            self.logger.error(f"Error appending summary to log: {e}")

    def generate_summary(self, return_string: bool = False) -> Optional[str]:
        """Generate summary report"""
        summary_lines = []
        summary_lines.append("")
        summary_lines.append("=" * 42)
        summary_lines.append("BENCHMARK SUMMARY REPORT")
        summary_lines.append("=" * 42)
        summary_lines.append(f"Total test cases: {len(self.test_cases)}")

        success_count = sum(
            1 for result in self.test_results.values() if result == "SUCCESS"
        )
        failure_count = sum(
            1 for result in self.test_results.values() if result == "FAILED"
        )
        skipped_count = sum(
            1 for result in self.test_results.values() if result == "SKIPPED"
        )

        summary_lines.append(f"Successful test cases: {success_count}")
        summary_lines.append(f"Failed test cases: {failure_count}")
        summary_lines.append(f"Skipped test cases: {skipped_count}")
        summary_lines.append("")

        if failure_count > 0:
            summary_lines.append("FAILED TEST CASES:")
            for test_case, result in self.test_results.items():
                if result == "FAILED":
                    summary_lines.append(f"  - {test_case}")
            summary_lines.append("")

        if skipped_count > 0:
            summary_lines.append("SKIPPED TEST CASES:")
            for test_case, result in self.test_results.items():
                if result == "SKIPPED":
                    summary_lines.append(f"  - {test_case}")
            summary_lines.append("")

        if success_count > 0:
            summary_lines.append("SUCCESSFUL TEST CASES:")
            for test_case, result in self.test_results.items():
                if result == "SUCCESS":
                    summary_lines.append(f"  - {test_case}")

        summary_lines.append("=" * 42)

        summary_text = "\n".join(summary_lines)

        if return_string:
            return summary_text
        else:
            print(summary_text)
            return None

    def run_all_tests(self):
        """Run all test cases"""
        self.logger.info(f"Starting benchmark with {len(self.test_cases)} test cases")
        self.logger.info(
            f"Server failure behavior: {'STOP ON FAILURE' if self.stop_on_failure else 'CONTINUE ON FAILURE'}"
        )

        # Create logs directory
        Path(self.logs_path).mkdir(parents=True, exist_ok=True)

        # Run each test case
        for test_case in self.test_cases:
            self.run_test_case(test_case)

        # Generate final summary
        self.generate_summary()

    @staticmethod
    def print_help():
        """Print help message"""
        print("Usage:")
        print(
            "python vllm_bench.py --config /path/to/config.py [--stop-on-server-failure]"
        )
        print("Options:")
        print(
            "  --config /path/to/config.py     Path to configuration script (required)"
        )
        print(
            "  --stop-on-server-failure        Stop the entire script if server dies (default: continue)"
        )
        print(
            "  --live-logs                     Show live server log tailing during server startup (default: disabled)"
        )
        print("  --help                          Show this help message")
        print("")
        print("The configuration script should define the following variables:")
        print("  - test_cases[]             : Array of test case definitions")
        print("  - env_vars[test_case]      : Dictionary of environment variables")
        print("  - model_params[test_case]      : Dictionary of model parameters")
        print("  - folder_name[test_case]       : Dictionary of folder names")
        print("  - VLLM_PORT                    : VLLM server port for benchmark")
        print("  - LOGS_PATH                    : Path to store run logs")
        print("  - RESULT_DIR                   : Path to store benchmark results")
        print(
            "  - in_out_lengths[]            : Array of {'in': x, 'out': y} objects (required)"
        )
        print("")
        print("Example: python vllm_bench.py --config ./example_config.json")
        print("")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="vLLM Benchmark Script")
    parser.add_argument("--config", required=True, help="Path to configuration script")
    parser.add_argument(
        "--stop-on-server-failure",
        action="store_true",
        help="Stop the entire script if server dies (default: continue)",
    )
    parser.add_argument(
        "--live-logs",
        action="store_true",
        help="Show live server log tailing during server startup (default: disabled)",
    )

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = VLLMBenchmark(args.config, args.stop_on_server_failure, args.live_logs)

    # Run all tests
    benchmark.run_all_tests()


if __name__ == "__main__":
    main()
