"""
DGM (Darwin Gödel Machine) Integration Adapter

Provides safe, sandboxed execution of DGM within ShinkaEvolve framework.
Maintains isolation while enabling integration with Context-Aware Thompson Sampling.
"""

import os
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DGMConfig:
    """Configuration for DGM execution."""
    benchmark_type: str = "swe_bench"  # "swe_bench" or "polyglot"
    seed: int = 42
    budget_steps: int = 200
    model: str = "gpt-4o"  # "gpt-4o", "claude-sonnet", "mock"
    
    # Safety settings
    enable_network: bool = False
    unsafe_allow: bool = False
    timeout_seconds: int = 1800  # 30 minutes
    
    # Resource limits
    max_memory_gb: float = 4.0
    max_cpus: float = 2.0
    
    # Tool allowlist
    allowed_tools: List[str] = None
    
    def __post_init__(self):
        if self.allowed_tools is None:
            # Strict default allowlist
            self.allowed_tools = [
                "python3",
                "python",
                "pytest",
                "flake8",
                "mypy",
                "black",
                "isort",
                "cat",
                "head",
                "tail",
                "grep",
                "find",
                "ls",
                "pwd"
            ]
            
            # Add bash only if explicitly unsafe_allow
            if self.unsafe_allow:
                self.allowed_tools.extend([
                    "bash",
                    "sh",
                    "str_replace_editor",
                    "vim",
                    "nano"
                ])


class DGMSafetyError(Exception):
    """Raised when DGM execution violates safety constraints."""
    pass


class DGMRunner:
    """Safe DGM execution runner with sandboxing and monitoring."""
    
    def __init__(self, config: DGMConfig):
        self.config = config
        self.dgm_root = Path("/app/third_party/dgm")
        self.output_dir = Path("/app/reports/dgm")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure DGM submodule exists
        if not self.dgm_root.exists():
            raise RuntimeError("DGM submodule not found. Run: git submodule update --init")
    
    def _check_docker_available(self) -> bool:
        """Check if Docker is available and DGM sandbox can be started."""
        try:
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _start_dgm_sandbox(self) -> str:
        """Start DGM Docker sandbox and return container ID."""
        if not self._check_docker_available():
            raise DGMSafetyError("Docker not available for DGM sandbox")
        
        logger.info("Starting DGM sandbox container...")
        
        try:
            # Build and start container
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.dgm.yml",
                "up", "-d", "dgm-sandbox"
            ], cwd="/app", capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise DGMSafetyError(f"Failed to start DGM sandbox: {result.stderr}")
            
            # Get container ID
            result = subprocess.run([
                "docker", "ps", "-q", "--filter", "name=shinka-dgm-sandbox"
            ], capture_output=True, text=True)
            
            if not result.stdout.strip():
                raise DGMSafetyError("DGM sandbox container not found after startup")
            
            container_id = result.stdout.strip()
            logger.info(f"DGM sandbox started: {container_id}")
            
            # Wait for container to be ready
            time.sleep(5)
            
            return container_id
            
        except subprocess.TimeoutExpired:
            raise DGMSafetyError("Timeout starting DGM sandbox")
    
    def _stop_dgm_sandbox(self):
        """Stop and cleanup DGM sandbox."""
        try:
            subprocess.run([
                "docker-compose", "-f", "docker-compose.dgm.yml",
                "down", "-v"
            ], cwd="/app", capture_output=True, timeout=30)
            logger.info("DGM sandbox stopped and cleaned up")
        except Exception as e:
            logger.warning(f"Error stopping DGM sandbox: {e}")
    
    def _execute_in_sandbox(self, command: List[str], timeout: int = None) -> subprocess.CompletedProcess:
        """Execute command in DGM sandbox with safety monitoring."""
        if timeout is None:
            timeout = self.config.timeout_seconds
        
        # Safety check: validate command against allowlist
        if not self.config.unsafe_allow:
            self._validate_command_safety(command)
        
        try:
            result = subprocess.run([
                "docker", "exec", "shinka-dgm-sandbox"
            ] + command, capture_output=True, text=True, timeout=timeout)
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s: {' '.join(command)}")
            # Kill container on timeout
            subprocess.run(["docker", "kill", "shinka-dgm-sandbox"], capture_output=True)
            raise DGMSafetyError(f"Command execution timed out: {' '.join(command)}")
    
    def _validate_command_safety(self, command: List[str]):
        """Validate command against safety constraints."""
        if not command:
            raise DGMSafetyError("Empty command not allowed")
        
        cmd_name = command[0]
        
        # Check against tool allowlist
        if cmd_name not in self.config.allowed_tools:
            raise DGMSafetyError(f"Tool '{cmd_name}' not in allowlist: {self.config.allowed_tools}")
        
        # Check for dangerous patterns (expanded list)
        dangerous_patterns = [
            "rm -rf", "sudo", "chmod +x", "curl", "wget", "git clone",
            "pip install", "npm install", "exec", "eval", "> /dev/",
            "dd if=", "mkfs", "fdisk", "mount", "umount", "systemctl",
            "&& rm", "; rm", "$(", "`", "nc -l", "netcat", "telnet",
            "ssh", "scp", "rsync", "ftp", "wget", "curl -X POST",
            "> /etc/", "> ~/.ssh", "chmod 777", "chmod -R",
            "kill -9", "killall", "pkill", "nohup", "/proc/",
            "docker", "podman", "kubectl", "helm"
        ]
        
        cmd_str = " ".join(command).lower()
        for pattern in dangerous_patterns:
            if pattern in cmd_str:
                raise DGMSafetyError(f"Dangerous pattern detected: '{pattern}' in command")
        
        # Additional argument validation
        for arg in command[1:]:
            # Check for suspicious file paths
            suspicious_paths = [
                "/etc/passwd", "/etc/shadow", "/root/", "/home/",
                "~/.ssh/", "~/.aws/", "/var/log/", "/sys/",
                "/proc/", "/dev/", "/boot/"
            ]
            
            for path in suspicious_paths:
                if path in arg.lower():
                    raise DGMSafetyError(f"Suspicious file path detected: '{path}' in argument '{arg}'")
    
    def _create_dgm_config(self, run_id: str) -> Dict[str, Any]:
        """Create DGM-specific configuration."""
        dgm_config = {
            "run_id": run_id,
            "benchmark": self.config.benchmark_type,
            "seed": self.config.seed,
            "steps": self.config.budget_steps,
            "model": self.config.model,
            "output_dir": f"/dgm/output_dgm/{run_id}",
            "safety": {
                "tools_allowlist": self.config.allowed_tools,
                "network_enabled": self.config.enable_network,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
        
        return dgm_config
    
    def run_dgm_eval(self, config_override: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run DGM evaluation and return metrics compatible with ShinkaEvolve.
        
        Returns:
            Dict with keys: final_best_fitness, auc, time_to_first_improve, 
                           logs, artifacts, dgm_metadata
        """
        run_id = f"dgm_eval_{int(time.time())}"
        container_id = None
        
        logger.info(f"Starting DGM evaluation run: {run_id}")
        
        try:
            # Start sandbox
            container_id = self._start_dgm_sandbox()
            
            # Create run configuration
            dgm_config = self._create_dgm_config(run_id)
            if config_override:
                dgm_config.update(config_override)
            
            # Write config to sandbox
            config_json = json.dumps(dgm_config, indent=2)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(config_json)
                config_file = f.name
            
            # Copy config to container
            subprocess.run([
                "docker", "cp", config_file, f"{container_id}:/dgm/run_config.json"
            ], check=True)
            os.unlink(config_file)
            
            # Execute DGM based on benchmark type
            if self.config.benchmark_type == "swe_bench":
                result = self._run_swe_bench_evaluation(run_id)
            elif self.config.benchmark_type == "polyglot":
                result = self._run_polyglot_evaluation(run_id)
            else:
                raise ValueError(f"Unknown benchmark type: {self.config.benchmark_type}")
            
            logger.info(f"DGM evaluation completed: {run_id}")
            return result
            
        except Exception as e:
            logger.error(f"DGM evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "final_best_fitness": 0.0,
                "auc": 0.0,
                "time_to_first_improve": float('inf'),
                "logs": [],
                "artifacts": [],
                "dgm_metadata": {"run_id": run_id, "error": str(e)}
            }
        finally:
            # Always cleanup
            if container_id:
                self._stop_dgm_sandbox()
    
    def _run_swe_bench_evaluation(self, run_id: str) -> Dict[str, Any]:
        """Run DGM SWE-bench evaluation."""
        logger.info("Running DGM SWE-bench evaluation...")
        
        # Create minimal SWE-bench test config
        swe_config = {
            "instances": ["django__django-10914"],  # Small test instance
            "model": self.config.model,
            "max_iterations": self.config.budget_steps,
            "seed": self.config.seed
        }
        
        # Execute DGM_outer.py with SWE-bench mode
        cmd = [
            "python", "DGM_outer.py",
            "--mode", "swe_bench",
            "--config", "run_config.json",
            "--output", f"output_dgm/{run_id}"
        ]
        
        start_time = time.time()
        
        try:
            result = self._execute_in_sandbox(cmd, timeout=self.config.timeout_seconds)
            execution_time = time.time() - start_time
            
            # Parse results from DGM output
            metrics = self._parse_dgm_output(run_id, result, execution_time)
            metrics["benchmark_type"] = "swe_bench"
            
            return metrics
            
        except DGMSafetyError:
            raise
        except Exception as e:
            logger.error(f"SWE-bench execution error: {e}")
            raise DGMSafetyError(f"SWE-bench execution failed: {e}")
    
    def _run_polyglot_evaluation(self, run_id: str) -> Dict[str, Any]:
        """Run DGM Polyglot evaluation."""
        logger.info("Running DGM Polyglot evaluation...")
        
        # Execute DGM with Polyglot mode
        cmd = [
            "python", "DGM_outer.py", 
            "--mode", "polyglot",
            "--config", "run_config.json",
            "--output", f"output_dgm/{run_id}"
        ]
        
        start_time = time.time()
        
        try:
            result = self._execute_in_sandbox(cmd, timeout=self.config.timeout_seconds)
            execution_time = time.time() - start_time
            
            # Parse results
            metrics = self._parse_dgm_output(run_id, result, execution_time)
            metrics["benchmark_type"] = "polyglot"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Polyglot execution error: {e}")
            raise DGMSafetyError(f"Polyglot execution failed: {e}")
    
    def _parse_dgm_output(self, run_id: str, result: subprocess.CompletedProcess, 
                         execution_time: float) -> Dict[str, Any]:
        """Parse DGM output and convert to ShinkaEvolve metrics format."""
        
        # Default metrics
        metrics = {
            "success": result.returncode == 0,
            "final_best_fitness": 0.0,
            "auc": 0.0,
            "time_to_first_improve": execution_time,
            "llm_queries_total": 0,
            "llm_queries_while_stuck": 0,
            "execution_time_seconds": execution_time,
            "logs": [],
            "artifacts": [],
            "dgm_metadata": {}
        }
        
        try:
            # Look for DGM output files
            output_dir = self.output_dir / run_id
            
            if output_dir.exists():
                # Parse DGM logs and results
                log_files = list(output_dir.glob("*.log"))
                json_files = list(output_dir.glob("*.json"))
                
                metrics["logs"] = [str(f) for f in log_files]
                metrics["artifacts"] = [str(f) for f in (log_files + json_files)]
                
                # Try to parse metrics from DGM output
                for json_file in json_files:
                    try:
                        with open(json_file) as f:
                            dgm_data = json.load(f)
                        
                        # Map DGM metrics to our format
                        if "fitness" in dgm_data:
                            metrics["final_best_fitness"] = dgm_data["fitness"]
                        if "iterations" in dgm_data:
                            metrics["llm_queries_total"] = dgm_data["iterations"]
                        
                        metrics["dgm_metadata"].update(dgm_data)
                        
                    except Exception as e:
                        logger.debug(f"Could not parse {json_file}: {e}")
            
            # If execution failed but no exception, try to extract error info
            if not metrics["success"]:
                metrics["dgm_metadata"]["stderr"] = result.stderr
                metrics["dgm_metadata"]["stdout"] = result.stdout
            
            # Estimate AUC (simple approximation)
            if metrics["final_best_fitness"] > 0:
                # Assume linear improvement for AUC calculation
                metrics["auc"] = metrics["final_best_fitness"] * execution_time / 2
            
        except Exception as e:
            logger.error(f"Error parsing DGM output: {e}")
            metrics["dgm_metadata"]["parse_error"] = str(e)
        
        return metrics
    
    def run_dgm_mutation(self, agent_state: Dict[str, Any]) -> Optional[str]:
        """
        Run DGM mutation to generate code patch candidate.
        
        Args:
            agent_state: Current agent state with code and context
            
        Returns:
            Optional code patch as string, or None if mutation failed
        """
        run_id = f"dgm_mutation_{int(time.time())}"
        
        logger.info(f"Starting DGM mutation run: {run_id}")
        
        try:
            container_id = self._start_dgm_sandbox()
            
            # Create mutation config
            mutation_config = {
                "mode": "mutation",
                "target_code": agent_state.get("code", ""),
                "context": agent_state.get("context", {}),
                "seed": self.config.seed
            }
            
            # Write mutation config
            config_json = json.dumps(mutation_config, indent=2)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(config_json)
                config_file = f.name
            
            subprocess.run([
                "docker", "cp", config_file, f"{container_id}:/dgm/mutation_config.json"
            ], check=True)
            os.unlink(config_file)
            
            # Execute mutation
            cmd = [
                "python", "self_improve_step.py",
                "--config", "mutation_config.json",
                "--output", f"output_dgm/{run_id}"
            ]
            
            result = self._execute_in_sandbox(cmd, timeout=300)  # 5 min for mutations
            
            if result.returncode == 0:
                # Extract generated patch
                output_dir = self.output_dir / run_id
                patch_files = list(output_dir.glob("*.patch"))
                
                if patch_files:
                    with open(patch_files[0]) as f:
                        return f.read()
                
                # Fallback: extract from stdout
                if "diff --git" in result.stdout:
                    return result.stdout
            
            logger.warning(f"DGM mutation failed or produced no output: {result.stderr}")
            return None
            
        except Exception as e:
            logger.error(f"DGM mutation error: {e}")
            return None
        finally:
            if 'container_id' in locals():
                self._stop_dgm_sandbox()


# Convenience functions
def run_dgm_swe_bench_quick(seed: int = 42, budget_steps: int = 200) -> Dict[str, Any]:
    """Run quick DGM SWE-bench evaluation."""
    config = DGMConfig(
        benchmark_type="swe_bench",
        seed=seed,
        budget_steps=budget_steps,
        model="mock"  # Use mock for quick runs
    )
    runner = DGMRunner(config)
    return runner.run_dgm_eval()


def run_dgm_polyglot_quick(seed: int = 42, budget_steps: int = 200) -> Dict[str, Any]:
    """Run quick DGM Polyglot evaluation."""
    config = DGMConfig(
        benchmark_type="polyglot", 
        seed=seed,
        budget_steps=budget_steps,
        model="mock"
    )
    runner = DGMRunner(config)
    return runner.run_dgm_eval()