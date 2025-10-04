#!/usr/bin/env python3
"""
Context-Aware Thompson Sampling Benchmark Harness

Validates ContextAwareThompsonSamplingBandit vs baseline Thompson on 
reproducible benchmarks with comprehensive metrics and automated reporting.
"""

import argparse
import csv
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import sys
import os
from datetime import datetime
import logging

# Add shinka to path
sys.path.insert(0, '/app')

from shinka.llm.dynamic_sampling import ThompsonSamplingBandit, ContextAwareThompsonSamplingBandit


class EpsilonGreedyBandit:
    """Epsilon-Greedy bandit for baseline comparison."""
    
    def __init__(self, arm_names: List[str], seed: int = 42, epsilon: float = 0.1, decay_rate: float = 0.995):
        self.arm_names = arm_names
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.initial_epsilon = epsilon
        
        self.rng = np.random.RandomState(seed)
        self.n_arms = len(arm_names)
        
        # Track statistics
        self.counts = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)
        self.submitted = []
        
    def posterior(self) -> np.ndarray:
        """Return selection probabilities (for compatibility)."""
        # Epsilon-greedy selection probabilities
        if np.sum(self.counts) == 0:
            return np.ones(self.n_arms) / self.n_arms
        
        avg_rewards = np.divide(self.rewards, self.counts, 
                              out=np.zeros_like(self.rewards), where=self.counts!=0)
        
        # Find best arm
        best_arm = np.argmax(avg_rewards)
        probs = np.ones(self.n_arms) * (self.epsilon / self.n_arms)
        probs[best_arm] += (1 - self.epsilon)
        
        return probs
    
    def sample(self) -> str:
        """Sample an arm name using epsilon-greedy strategy."""
        probs = self.posterior()
        selected_idx = self.rng.choice(self.n_arms, p=probs)
        return self.arm_names[selected_idx]
    
    def sample_n(self, n: int) -> List[str]:
        """Sample multiple arms."""
        return [self.sample() for _ in range(n)]
    
    def update_submitted(self, arm_name: str):
        """Track submitted queries."""
        self.submitted.append(arm_name)
    
    def update(self, arm_name: str, reward: float, baseline: float = 0.5):
        """Update arm statistics."""
        arm_idx = self.arm_names.index(arm_name)
        self.counts[arm_idx] += 1
        self.rewards[arm_idx] += reward
        
        # Decay epsilon
        self.epsilon *= self.decay_rate
        self.epsilon = max(0.01, self.epsilon)  # Minimum epsilon

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    benchmark: str
    algorithm: str
    seed: int
    budget_steps: int
    model: str = "mock"
    output_dir: str = "reports/context_bandit/raw"
    ablation: str = "none"
    hyperparams: str = "2.0,1.0,0.99"


class MockLLMScorer:
    """Mock LLM scorer for consistent, reproducible benchmarks."""
    
    def __init__(self, problem_type: str, seed: int = 42):
        self.problem_type = problem_type
        self.rng = np.random.RandomState(seed)
        self.query_count = 0
        
        # Initialize problem-specific data
        if problem_type == "tsp":
            self._init_tsp_berlin52()
        elif problem_type == "toy":
            self._init_hard_toy()
        elif problem_type == "synthetic":
            self._init_hard_synthetic()
    
    def _init_tsp_berlin52(self):
        """Initialize Berlin52 TSP coordinates (TSPLIB format)."""
        # Berlin52 coordinates from TSPLIB (first 50 cities for consistency)
        self.berlin52_coords = np.array([
            [565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
            [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
            [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
            [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
            [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
            [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
            [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
            [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
            [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
            [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0]
        ])
        # Known optimal tour length for Berlin52 (first 50 cities approximation)
        self.optimal_tour_length = 7544.0
        
    def _init_hard_toy(self):
        """Initialize hard multi-modal toy function with plateaux."""
        self.toy_peaks = [
            {'center': np.array([2.0, 2.0]), 'height': 1.0, 'width': 0.8},
            {'center': np.array([-1.5, 1.0]), 'height': 0.85, 'width': 0.6},
            {'center': np.array([1.0, -2.0]), 'height': 0.7, 'width': 0.7},
            {'center': np.array([-2.0, -1.5]), 'height': 0.6, 'width': 0.5},
            {'center': np.array([0.0, 0.0]), 'height': 0.4, 'width': 1.2}  # Wide plateau
        ]
        
    def _init_hard_synthetic(self):
        """Initialize hard synthetic function with constraints and penalties."""
        self.previous_solutions = []
        self.penalty_window = 10  # Track last 10 solutions for repetition penalty
        
    def score_solution(self, solution: np.ndarray, context_info: Dict = None) -> float:
        """Score a solution based on problem type."""
        self.query_count += 1
        
        if self.problem_type == "toy":
            return self._toy_function(solution)
        elif self.problem_type == "tsp":
            return self._tsp_function(solution)
        elif self.problem_type == "synthetic":
            return self._synthetic_function(solution, context_info)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")
    
    def _toy_function(self, solution: np.ndarray) -> float:
        """Hard multi-modal toy function with plateaux and controlled noise."""
        x, y = solution[0], solution[1]
        
        # Multi-modal landscape with plateaux
        fitness = 0.0
        for peak in self.toy_peaks:
            distance = np.linalg.norm(np.array([x, y]) - peak['center'])
            # Create plateaux effect with tanh
            contribution = peak['height'] * np.exp(-distance**2 / peak['width']**2)
            # Add plateau effect
            plateau_factor = 0.5 * (1 + np.tanh(2 - distance))
            fitness += contribution * plateau_factor
        
        # Controlled noise that increases when "stuck" 
        base_noise = 0.08
        noise_amplitude = base_noise * (1 + 0.5 * self.rng.random())
        noise = self.rng.normal(0, noise_amplitude)
        
        # Deceptive valleys between peaks
        deception = -0.2 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.3)
        
        return np.clip(fitness + noise + deception, 0, 1)
    
    def _tsp_function(self, solution: np.ndarray) -> float:
        """Real TSP function using Berlin52 TSPLIB coordinates."""
        n_cities = len(self.berlin52_coords)
        
        # Convert solution to permutation
        # Use solution values to create a tour order
        if len(solution) < n_cities:
            # Extend solution by cycling
            extended = np.tile(solution, (n_cities // len(solution) + 1))[:n_cities]
        else:
            extended = solution[:n_cities]
        
        # Create permutation from continuous values
        sorted_indices = np.argsort(extended)
        tour = sorted_indices
        
        # Calculate tour length using Berlin52 coordinates
        tour_length = 0.0
        for i in range(n_cities):
            curr_city = tour[i]
            next_city = tour[(i + 1) % n_cities]
            
            curr_pos = self.berlin52_coords[curr_city]
            next_pos = self.berlin52_coords[next_city]
            
            distance = np.sqrt(np.sum((curr_pos - next_pos)**2))
            tour_length += distance
        
        # Convert to fitness (normalized by optimal)
        # Better tours get higher fitness
        fitness = max(0, 1.0 - (tour_length - self.optimal_tour_length) / self.optimal_tour_length)
        
        # Add small controlled noise
        noise = self.rng.normal(0, 0.01)
        
        return np.clip(fitness + noise, 0, 1)
    
    def _synthetic_function(self, solution: np.ndarray, context_info: Dict) -> float:
        """Hard synthetic function with constraints and repetition penalties."""
        # Base fitness from multi-objective Rosenbrock + Rastrigin
        fitness = 0
        n_dims = len(solution)
        
        # Rosenbrock component (smooth but narrow valley)
        rosenbrock = 0
        for i in range(n_dims - 1):
            rosenbrock += 100 * (solution[i+1] - solution[i]**2)**2 + (1 - solution[i])**2
        
        # Rastrigin component (many local optima)
        rastrigin = 10 * n_dims + sum(x**2 - 10 * np.cos(2 * np.pi * x) for x in solution)
        
        # Combine objectives
        combined_penalty = rosenbrock / 1000 + rastrigin / 100
        base_fitness = 1.0 / (1.0 + combined_penalty)
        
        # Repetition penalty (anti-cycling constraint)
        repetition_penalty = 0.0
        current_hash = hash(tuple(np.round(solution, 2)))
        
        # Store solution history
        self.previous_solutions.append(current_hash)
        if len(self.previous_solutions) > self.penalty_window:
            self.previous_solutions.pop(0)
        
        # Count recent repetitions
        recent_count = self.previous_solutions.count(current_hash)
        if recent_count > 1:
            repetition_penalty = 0.2 * (recent_count - 1)  # Escalating penalty
        
        # Context-dependent modulation (more aggressive)
        context_modifier = 1.0
        if context_info:
            gen_progress = context_info.get('gen_progress', 0.5)
            no_improve = context_info.get('no_improve_steps', 0)
            
            # Early phase: exploration bonus but with noise
            if gen_progress < 0.2:
                context_modifier *= (0.7 + self.rng.uniform(0, 0.6))
            
            # Mid phase: stable evaluation
            elif gen_progress < 0.6:
                context_modifier *= (0.95 + self.rng.uniform(0, 0.1))
            
            # Late phase: precision matters
            else:
                context_modifier *= (0.98 + self.rng.uniform(0, 0.04))
            
            # Stuck phase: make it hard but allow breakthroughs
            if no_improve > 15:
                if self.rng.random() < 0.05:  # 5% chance of breakthrough
                    context_modifier *= 1.5
                else:
                    context_modifier *= 0.6
        
        # Controlled noise
        noise_level = 0.05 + 0.03 * self.rng.random()
        noise = self.rng.normal(0, noise_level)
        
        final_fitness = base_fitness * context_modifier - repetition_penalty
        
        return np.clip(final_fitness + noise, 0, 1)


class EvolutionSimulator:
    """Simulates evolutionary process for benchmarking."""
    
    def __init__(self, config: BenchmarkConfig, scorer: MockLLMScorer):
        self.config = config
        self.scorer = scorer
        self.rng = np.random.RandomState(config.seed)
        
        # Evolution state
        self.current_step = 0
        self.best_fitness_history = []
        self.fitness_history = []
        self.population_diversity_history = []
        self.current_best_fitness = 0.0
        self.no_improve_steps = 0
        
        # Metrics tracking
        self.llm_queries_total = 0
        self.llm_queries_while_stuck = 0
        self.stuck_windows = []
        self.time_to_first_improve = None
        self.context_history = []
        
        # Parse hyperparameters
        hyperparam_parts = config.hyperparams.split(',')
        prior_alpha = float(hyperparam_parts[0]) if len(hyperparam_parts) > 0 else 2.0
        prior_beta = float(hyperparam_parts[1]) if len(hyperparam_parts) > 1 else 1.0
        auto_decay = float(hyperparam_parts[2]) if len(hyperparam_parts) > 2 else 0.99
        
        # Initialize bandit based on algorithm type
        models = ["fast_model", "accurate_model", "balanced_model"]
        
        if config.algorithm == "baseline":
            # Pure Thompson Sampling (no decay)
            self.bandit = ThompsonSamplingBandit(
                arm_names=models,
                seed=config.seed,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
                auto_decay=1.0,  # No decay
                reward_mapping="adaptive"
            )
        elif config.algorithm == "decay":
            # Thompson Sampling with decay (no contexts)
            self.bandit = ThompsonSamplingBandit(
                arm_names=models,
                seed=config.seed,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
                auto_decay=auto_decay,
                reward_mapping="adaptive"
            )
        elif config.algorithm == "context":
            # Context-Aware Thompson Sampling with ablations
            contexts = ["early", "mid", "late", "stuck"]
            features = ["gen_progress", "no_improve", "fitness_slope", "pop_diversity"]
            
            # Apply ablations
            if config.ablation == "no_gen_progress":
                features.remove("gen_progress")
            elif config.ablation == "no_no_improve":
                features.remove("no_improve")
            elif config.ablation == "no_fitness_slope":
                features.remove("fitness_slope")
            elif config.ablation == "no_pop_diversity":
                features.remove("pop_diversity")
            elif config.ablation == "3contexts":
                contexts = ["early", "mid_late", "stuck"]  # Merge mid/late
            
            self.bandit = ContextAwareThompsonSamplingBandit(
                arm_names=models,
                seed=config.seed,
                contexts=contexts,
                features=features,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
                auto_decay=auto_decay,
                reward_mapping="adaptive"
            )
        elif config.algorithm == "ucb":
            # UCB1 baseline
            try:
                from shinka.llm.dynamic_sampling import AsymmetricUCB
                self.bandit = AsymmetricUCB(
                    arm_names=models,
                    seed=config.seed,
                    c_param=1.4  # Standard UCB1 exploration parameter
                )
            except ImportError:
                logger.warning("AsymmetricUCB not available, using Thompson baseline")
                self.bandit = ThompsonSamplingBandit(
                    arm_names=models,
                    seed=config.seed,
                    prior_alpha=prior_alpha,
                    prior_beta=prior_beta,
                    auto_decay=auto_decay,
                    reward_mapping="adaptive"
                )
        elif config.algorithm == "epsilon":
            # Epsilon-greedy baseline
            self.bandit = EpsilonGreedyBandit(
                arm_names=models,
                seed=config.seed,
                epsilon=0.1,
                decay_rate=0.995
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
    
    def simulate_llm_query(self, model_name: str) -> Tuple[np.ndarray, float]:
        """Simulate LLM generating a solution candidate."""
        self.llm_queries_total += 1
        
        # Context-dependent solution generation
        if model_name == "fast_model":
            # Quick exploration, more random
            solution = self.rng.uniform(-3, 3, 5)
            if self.current_step < 10:  # Early boost
                solution += self.rng.uniform(-1, 1, 5)
        elif model_name == "accurate_model":
            # Precise, slower, better for stuck situations
            if self.no_improve_steps > 10:  # Stuck boost
                solution = self.rng.uniform(-2, 2, 5)
            else:
                solution = self.rng.uniform(-2.5, 2.5, 5)
        else:  # balanced_model
            solution = self.rng.uniform(-2.5, 2.5, 5)
        
        # Add model-specific noise
        noise_levels = {"fast_model": 0.3, "accurate_model": 0.1, "balanced_model": 0.2}
        noise = self.rng.normal(0, noise_levels[model_name], len(solution))
        solution += noise
        
        # Score the solution
        context_info = {
            'gen_progress': self.current_step / self.config.budget_steps,
            'no_improve_steps': self.no_improve_steps,
            'current_best': self.current_best_fitness
        }
        
        fitness = self.scorer.score_solution(solution, context_info)
        return solution, fitness
    
    def detect_stuck_phase(self, window_size: int = 25) -> bool:
        """Detect if evolution is in stuck phase."""
        if len(self.best_fitness_history) < window_size:
            return False
        
        recent_best = self.best_fitness_history[-window_size:]
        
        # Stuck if no improvement in recent window
        return max(recent_best) <= min(recent_best) + 1e-6
    
    def calculate_population_diversity(self) -> float:
        """Calculate population diversity (mock implementation)."""
        # Simple diversity based on recent fitness variance
        if len(self.fitness_history) < 5:
            return 0.8  # High diversity initially
        
        recent_fitness = self.fitness_history[-10:]
        diversity = np.std(recent_fitness) / (np.mean(recent_fitness) + 1e-6)
        return np.clip(diversity, 0, 1)
    
    def run_step(self) -> Dict[str, Any]:
        """Run one evolution step."""
        step_start_time = time.time()
        
        # Update context for context-aware bandit
        current_context = None
        if hasattr(self.bandit, 'update_context'):
            current_context = self.bandit.update_context(
                generation=self.current_step,
                total_generations=self.config.budget_steps,
                no_improve_steps=self.no_improve_steps,
                best_fitness_history=self.best_fitness_history,
                population_diversity=self.calculate_population_diversity()
            )
        
        # Select model using bandit
        probs = self.bandit.posterior()
        selected_model_idx = np.argmax(probs)
        selected_model = ["fast_model", "accurate_model", "balanced_model"][selected_model_idx]
        
        # Generate solution candidate
        solution, fitness = self.simulate_llm_query(selected_model)
        
        # Update fitness tracking
        self.fitness_history.append(fitness)
        
        # Update best fitness
        improvement = False
        if fitness > self.current_best_fitness:
            self.current_best_fitness = fitness
            self.no_improve_steps = 0
            improvement = True
            
            # Track time to first improvement
            if self.time_to_first_improve is None and fitness > 0.1:
                self.time_to_first_improve = self.current_step
        else:
            self.no_improve_steps += 1
        
        self.best_fitness_history.append(self.current_best_fitness)
        
        # Track stuck phase queries
        if self.detect_stuck_phase():
            self.llm_queries_while_stuck += 1
        
        # Update population diversity
        pop_diversity = self.calculate_population_diversity()
        self.population_diversity_history.append(pop_diversity)
        
        # Calculate fitness slope
        fitness_slope = 0.0
        if len(self.best_fitness_history) >= 3:
            recent = self.best_fitness_history[-min(10, len(self.best_fitness_history)):]
            x = np.arange(len(recent))
            y = np.array(recent)
            if len(recent) >= 2:
                slope = (len(recent) * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                       (len(recent) * np.sum(x**2) - np.sum(x)**2 + 1e-8)
                fitness_slope = slope
        
        # Update bandit
        baseline = 0.5
        self.bandit.update_submitted(selected_model)
        self.bandit.update(selected_model, fitness, baseline)
        
        step_time_ms = (time.time() - step_start_time) * 1000
        
        # Log step results
        step_result = {
            'run_id': f"{self.config.algorithm}_{self.config.benchmark}_{self.config.seed}",
            'algo': self.config.algorithm,
            'benchmark': self.config.benchmark,
            'seed': self.config.seed,
            'step': self.current_step,
            'context': current_context or 'none',
            'fitness': fitness,
            'best_fitness': self.current_best_fitness,
            'llm_queries': 1,  # This step
            'llm_queries_cum': self.llm_queries_total,
            'time_ms': step_time_ms,
            'gen_progress': self.current_step / self.config.budget_steps,
            'no_improve_steps': self.no_improve_steps,
            'fitness_slope': fitness_slope,
            'pop_diversity': pop_diversity,
            'selected_model': selected_model,
            'improvement': improvement
        }
        
        self.context_history.append(current_context)
        self.current_step += 1
        
        return step_result


def run_single_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Run a single benchmark configuration."""
    logger.info(f"Running {config.algorithm} on {config.benchmark} (seed={config.seed})")
    
    # Create scorer and simulator
    scorer = MockLLMScorer(config.benchmark, config.seed)
    simulator = EvolutionSimulator(config, scorer)
    
    # Prepare output
    output_dir = Path(config.output_dir) / config.algorithm / config.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"run_{config.seed}.csv"
    
    # CSV logging
    fieldnames = [
        'run_id', 'algo', 'benchmark', 'seed', 'step', 'context',
        'fitness', 'best_fitness', 'llm_queries', 'llm_queries_cum',
        'time_ms', 'gen_progress', 'no_improve_steps', 'fitness_slope',
        'pop_diversity', 'selected_model', 'improvement'
    ]
    
    results = []
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Run simulation
        for step in range(config.budget_steps):
            step_result = simulator.run_step()
            writer.writerow(step_result)
            results.append(step_result)
            
            # Progress logging
            if step % 100 == 0 or step == config.budget_steps - 1:
                logger.info(f"  Step {step}: best_fitness={step_result['best_fitness']:.4f}, "
                           f"context={step_result['context']}")
    
    # Calculate final metrics
    final_metrics = {
        'run_id': results[0]['run_id'],
        'algorithm': config.algorithm,
        'benchmark': config.benchmark,
        'seed': config.seed,
        'final_best_fitness': simulator.current_best_fitness,
        'llm_queries_total': simulator.llm_queries_total,
        'llm_queries_while_stuck': simulator.llm_queries_while_stuck,
        'time_to_first_improve': simulator.time_to_first_improve,
        'no_improve_final': simulator.no_improve_steps,
        'area_under_fitness_curve': np.trapezoid(simulator.best_fitness_history),
    }
    
    # Context-specific metrics for context-aware algorithm
    if hasattr(simulator.bandit, 'get_context_stats'):
        context_stats = simulator.bandit.get_context_stats()
        final_metrics['context_switch_count'] = context_stats['context_switch_count']
        
        # Dwell time by context
        context_counts = {}
        for context in simulator.context_history:
            if context and context != 'none':
                context_counts[context] = context_counts.get(context, 0) + 1
        final_metrics['context_dwell_times'] = context_counts
    
    logger.info(f"Completed {config.algorithm}: final_fitness={final_metrics['final_best_fitness']:.4f}")
    
    return final_metrics


def analyze_results(output_dir: str = "reports/context_bandit") -> Dict[str, Any]:
    """Analyze benchmark results and generate comprehensive report."""
    logger.info("Analyzing benchmark results...")
    
    raw_dir = Path(output_dir) / "raw"
    
    # Collect all detailed results
    all_results = []
    context_switch_data = []
    oscillation_data = []
    
    for algo_dir in raw_dir.iterdir():
        if not algo_dir.is_dir():
            continue
            
        algorithm = algo_dir.name
        
        for bench_dir in algo_dir.iterdir():
            if not bench_dir.is_dir():
                continue
                
            benchmark = bench_dir.name
            
            for csv_file in bench_dir.glob("run_*.csv"):
                seed = int(csv_file.stem.split("_")[1])
                
                # Read CSV data
                df = pd.read_csv(csv_file)
                
                # Calculate comprehensive metrics
                final_fitness = df['best_fitness'].iloc[-1]
                llm_queries_total = df['llm_queries_cum'].iloc[-1]
                
                # Time to first improvement (median will be calculated across seeds)
                improvement_steps = df[df['improvement'] == True]
                time_to_first_improve = improvement_steps['step'].iloc[0] if len(improvement_steps) > 0 else len(df)
                
                # Stuck phase analysis (W=25 window)
                stuck_queries = 0
                stuck_periods = []
                window_size = 25
                
                for i in range(window_size, len(df)):
                    recent_best = df['best_fitness'].iloc[i-window_size:i]
                    recent_slope = df['fitness_slope'].iloc[i] if 'fitness_slope' in df.columns else 0
                    
                    # Stuck if no improvement OR negative slope
                    is_stuck = (recent_best.max() <= recent_best.min() + 1e-6) or (recent_slope < 0)
                    
                    if is_stuck:
                        stuck_queries += df['llm_queries'].iloc[i]
                        stuck_periods.append(i)
                
                # Area under curve (AUC)
                auc = np.trapezoid(df['best_fitness'].values)
                
                # Context-specific analysis
                context_switches = 0
                dwell_times = {}
                oscillation_count = 0
                
                if 'context' in df.columns and algorithm in ['context']:
                    contexts = df['context'].values
                    
                    # Count context switches
                    for i in range(1, len(contexts)):
                        if contexts[i] != contexts[i-1]:
                            context_switches += 1
                    
                    # Calculate dwell times by context
                    for context in df['context'].unique():
                        if context and context != 'none':
                            dwell_times[context] = np.sum(df['context'] == context) / len(df)
                    
                    # Detect oscillation (thrashing) - rapid context switches
                    if context_switches > 0:
                        switch_density = context_switches / len(df)
                        oscillation_count = switch_density if switch_density > 0.1 else 0
                
                # Variance/stability analysis
                fitness_variance = np.var(df['best_fitness'].values)
                
                result = {
                    'algorithm': algorithm,
                    'benchmark': benchmark, 
                    'seed': seed,
                    'final_fitness': final_fitness,
                    'llm_queries_total': llm_queries_total,
                    'llm_queries_while_stuck': stuck_queries,
                    'time_to_first_improve': time_to_first_improve,
                    'area_under_curve': auc,
                    'context_switches': context_switches,
                    'dwell_times': dwell_times,
                    'oscillation_count': oscillation_count,
                    'fitness_variance': fitness_variance,
                    'stuck_periods_count': len(stuck_periods)
                }
                
                # Store context switch data for analysis
                if context_switches > 0:
                    context_switch_data.append({
                        'algorithm': algorithm,
                        'benchmark': benchmark,
                        'seed': seed,
                        'switches': context_switches,
                        'dwell_times': dwell_times
                    })
                
                # Store oscillation data
                if oscillation_count > 0:
                    oscillation_data.append({
                        'algorithm': algorithm,
                        'benchmark': benchmark,
                        'seed': seed,
                        'oscillation': oscillation_count
                    })
                
                all_results.append(result)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    if results_df.empty:
        logger.warning("No results found for analysis")
        return {}
    
    # Aggregate by algorithm and benchmark with comprehensive metrics
    aggregated = results_df.groupby(['algorithm', 'benchmark']).agg({
        'final_fitness': ['mean', 'std', 'count'],
        'llm_queries_total': ['mean', 'std'],
        'llm_queries_while_stuck': ['mean', 'std'],
        'time_to_first_improve': ['median', 'std'],
        'area_under_curve': ['mean', 'std'],
        'context_switches': ['mean', 'std'],
        'oscillation_count': ['mean', 'max'],
        'fitness_variance': ['mean', 'std'],
        'stuck_periods_count': ['mean', 'std']
    }).round(4)
    
    # Save summary
    summary_file = Path(output_dir) / "summary.csv"
    aggregated.to_csv(summary_file)
    
    # Calculate comprehensive acceptance criteria with explicit deltas
    acceptance_results = {}
    delta_results = {}
    
    # Calculate deltas vs baseline for all algorithms
    for benchmark in results_df['benchmark'].unique():
        bench_df = results_df[results_df['benchmark'] == benchmark]
        
        if 'baseline' not in bench_df['algorithm'].values:
            continue
        
        # Get baseline metrics
        baseline_df = bench_df[bench_df['algorithm'] == 'baseline']
        baseline_fitness = baseline_df['final_fitness'].mean()
        baseline_auc = baseline_df['area_under_curve'].mean()
        baseline_stuck = baseline_df['llm_queries_while_stuck'].mean()
        baseline_time = baseline_df['time_to_first_improve'].median()
        baseline_variance = baseline_df['fitness_variance'].mean()
        
        delta_results[benchmark] = {}
        
        for algorithm in bench_df['algorithm'].unique():
            if algorithm == 'baseline':
                continue
                
            algo_df = bench_df[bench_df['algorithm'] == algorithm]
            
            # Calculate explicit deltas
            fitness_delta = ((algo_df['final_fitness'].mean() - baseline_fitness) / baseline_fitness) * 100 if baseline_fitness > 0 else 0
            auc_delta = ((algo_df['area_under_curve'].mean() - baseline_auc) / baseline_auc) * 100 if baseline_auc > 0 else 0
            stuck_queries_delta = ((baseline_stuck - algo_df['llm_queries_while_stuck'].mean()) / baseline_stuck) * 100 if baseline_stuck > 0 else 0
            time_delta = ((baseline_time - algo_df['time_to_first_improve'].median()) / baseline_time) * 100 if baseline_time > 0 else 0
            variance_ratio = (algo_df['fitness_variance'].mean() / baseline_variance) if baseline_variance > 0 else 1.0
            
            delta_results[benchmark][algorithm] = {
                'final_fitness_delta_pct': fitness_delta,
                'auc_delta_pct': auc_delta,
                'stuck_queries_reduction_pct': stuck_queries_delta,
                'time_to_improve_reduction_pct': time_delta,
                'variance_ratio': variance_ratio,
                'context_switches_avg': algo_df['context_switches'].mean(),
                'oscillation_avg': algo_df['oscillation_count'].mean()
            }
        
        # Context-specific acceptance criteria
        if 'context' in bench_df['algorithm'].values:
            context_metrics = delta_results[benchmark].get('context', {})
            
            # PASS/FAIL criteria (immutable)
            efficacy_fitness_pass = context_metrics.get('final_fitness_delta_pct', 0) >= 3.0
            efficacy_auc_pass = context_metrics.get('auc_delta_pct', 0) >= 3.0
            
            # For TSP and at least 1 of TOY/SYNTHETIC hard
            efficacy_pass = False
            if benchmark == 'tsp':
                efficacy_pass = efficacy_fitness_pass and efficacy_auc_pass
            elif benchmark in ['toy', 'synthetic']:
                efficacy_pass = efficacy_fitness_pass or efficacy_auc_pass
            
            efficiency_stuck_pass = context_metrics.get('stuck_queries_reduction_pct', 0) >= 10.0
            efficiency_time_pass = context_metrics.get('time_to_improve_reduction_pct', 0) >= 10.0
            efficiency_pass = efficiency_stuck_pass or efficiency_time_pass
            
            stability_pass = context_metrics.get('variance_ratio', 999) <= 2.0
            
            acceptance_results[benchmark] = {
                'efficacy_pass': efficacy_pass,
                'efficiency_pass': efficiency_pass,
                'stability_pass': stability_pass,
                'overall_pass': efficacy_pass and efficiency_pass and stability_pass,
                'metrics': context_metrics
            }
    
    logger.info(f"Analysis complete. Summary saved to {summary_file}")
    
    return {
        'aggregated_results': aggregated,
        'acceptance_results': acceptance_results,
        'delta_results': delta_results,
        'context_switch_data': context_switch_data,
        'oscillation_data': oscillation_data,
        'raw_results': results_df
    }


def generate_plots(output_dir: str = "reports/context_bandit"):
    """Generate comprehensive visualization plots."""
    logger.info("Generating extended plots...")
    
    raw_dir = Path(output_dir) / "raw"
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Collect data for all plots
    plot_data = []
    auc_data = []
    time_data = []
    
    # Collect data from all CSV files
    for algo_dir in raw_dir.iterdir():
        if not algo_dir.is_dir():
            continue
            
        algorithm = algo_dir.name
        
        for bench_dir in algo_dir.iterdir():
            if not bench_dir.is_dir():
                continue
                
            benchmark = bench_dir.name
            
            for csv_file in bench_dir.glob("run_*.csv"):
                df = pd.read_csv(csv_file)
                seed = int(csv_file.stem.split("_")[1])
                
                # Store fitness curves data
                plot_data.append({
                    'algorithm': algorithm,
                    'benchmark': benchmark,
                    'seed': seed,
                    'fitness_curve': df['best_fitness'].values,
                    'steps': df['step'].values
                })
                
                # Store AUC data
                auc = np.trapezoid(df['best_fitness'].values)
                auc_data.append({
                    'algorithm': algorithm,
                    'benchmark': benchmark,
                    'seed': seed,
                    'auc': auc
                })
                
                # Store time to first improve data
                improvement_steps = df[df['improvement'] == True]
                time_to_improve = improvement_steps['step'].iloc[0] if len(improvement_steps) > 0 else len(df)
                time_data.append({
                    'algorithm': algorithm,
                    'benchmark': benchmark,
                    'seed': seed,
                    'time_to_improve': time_to_improve
                })
    
    # 1. Fitness vs Steps plot (enhanced with all algorithms)
    algorithms = list(set([d['algorithm'] for d in plot_data]))
    benchmarks = list(set([d['benchmark'] for d in plot_data]))
    
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6*len(benchmarks), 6))
    if len(benchmarks) == 1:
        axes = [axes]
    
    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        
        for algorithm in algorithms:
            bench_algo_data = [d for d in plot_data if d['benchmark'] == benchmark and d['algorithm'] == algorithm]
            
            if not bench_algo_data:
                continue
            
            # Collect all fitness curves
            all_fitness = [d['fitness_curve'] for d in bench_algo_data]
            max_steps = max(len(curve) for curve in all_fitness)
            
            # Pad shorter runs to same length
            padded_fitness = []
            for fitness in all_fitness:
                if len(fitness) < max_steps:
                    padded = np.pad(fitness, (0, max_steps - len(fitness)), 'edge')
                    padded_fitness.append(padded)
                else:
                    padded_fitness.append(fitness)
            
            if padded_fitness:
                fitness_array = np.array(padded_fitness)
                mean_fitness = np.mean(fitness_array, axis=0)
                std_fitness = np.std(fitness_array, axis=0)
                steps = np.arange(len(mean_fitness))
                
                label = algorithm.replace('_', ' ').title()
                ax.plot(steps, mean_fitness, label=label, linewidth=2)
                ax.fill_between(steps, mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.2)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Best Fitness')
        ax.set_title(f'{benchmark.upper()} Benchmark')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'fitness_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. AUC Bar Chart
    if auc_data:
        auc_df = pd.DataFrame(auc_data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=auc_df, x='benchmark', y='auc', hue='algorithm', errorbar='sd', capsize=0.1)
        plt.title('Area Under Curve (AUC) by Algorithm and Benchmark')
        plt.ylabel('AUC (Best Fitness × Steps)')
        plt.xlabel('Benchmark')
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'auc_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Time to First Improve Boxplot
    if time_data:
        time_df = pd.DataFrame(time_data)
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=time_df, x='benchmark', y='time_to_improve', hue='algorithm')
        plt.title('Time to First Improvement Distribution')
        plt.ylabel('Steps to First Improvement')
        plt.xlabel('Benchmark')
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'time_to_improve_boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Stuck queries comparison
    results_df = []
    
    for algo_dir in raw_dir.iterdir():
        if not algo_dir.is_dir():
            continue
            
        algorithm = algo_dir.name
        
        for bench_dir in algo_dir.iterdir():
            if not bench_dir.is_dir():
                continue
                
            benchmark = bench_dir.name
            
            for csv_file in bench_dir.glob("run_*.csv"):
                df = pd.read_csv(csv_file)
                
                # Calculate stuck queries
                stuck_queries = 0
                window_size = 25
                for i in range(window_size, len(df)):
                    recent_best = df['best_fitness'].iloc[i-window_size:i]
                    if recent_best.max() <= recent_best.min() + 1e-6:
                        stuck_queries += df['llm_queries'].iloc[i]
                
                results_df.append({
                    'algorithm': algorithm,
                    'benchmark': benchmark, 
                    'stuck_queries': stuck_queries
                })
    
    if results_df:
        results_df = pd.DataFrame(results_df)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='benchmark', y='stuck_queries', hue='algorithm')
        plt.title('LLM Queries While Stuck')
        plt.ylabel('Queries')
        plt.xlabel('Benchmark')
        plt.legend(title='Algorithm')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'stuck_queries.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Plots saved to {plots_dir}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Context-Aware Thompson Sampling Benchmark")
    parser.add_argument('--benchmark', choices=['toy', 'tsp', 'synthetic', 'all'], 
                       default='toy', help='Benchmark to run')
    parser.add_argument('--algo', choices=['baseline', 'decay', 'context', 'ucb', 'epsilon', 'all'], 
                       default='all', help='Algorithm to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--budget_steps', type=int, default=1500, help='Evolution steps (full=1500, quick=300)')
    parser.add_argument('--model', default='mock', help='LLM model to use')
    parser.add_argument('--output_dir', default='reports/context_bandit/raw', 
                       help='Output directory')
    parser.add_argument('--ablation', choices=['none', 'no_gen_progress', 'no_no_improve', 
                       'no_fitness_slope', 'no_pop_diversity', '3contexts'], 
                       default='none', help='Feature ablation for context algorithm')
    parser.add_argument('--hyperparams', default='2.0,1.0,0.99', 
                       help='prior_alpha,prior_beta,decay (comma-separated)')
    parser.add_argument('--make-report', action='store_true', 
                       help='Generate analysis and plots')
    
    args = parser.parse_args()
    
    if args.make_report:
        # Generate report from existing results
        analysis = analyze_results()
        generate_plots()
        return
    
    # Determine benchmarks and algorithms to run
    benchmarks = ['toy', 'tsp', 'synthetic'] if args.benchmark == 'all' else [args.benchmark]
    algorithms = ['baseline', 'context'] if args.algo == 'both' else [args.algo]
    
    # Run benchmarks
    all_metrics = []
    
    for benchmark in benchmarks:
        for algorithm in algorithms:
            config = BenchmarkConfig(
                benchmark=benchmark,
                algorithm=algorithm,
                seed=args.seed,
                budget_steps=args.budget_steps,
                model=args.model,
                output_dir=args.output_dir
            )
            
            try:
                metrics = run_single_benchmark(config)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Failed to run {algorithm} on {benchmark}: {e}")
    
    logger.info(f"Completed {len(all_metrics)} benchmark runs")


if __name__ == "__main__":
    main()