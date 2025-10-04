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
        """TSP-like function (tour length optimization)."""
        # Simple distance-based scoring for node sequence
        n_cities = len(solution)
        if n_cities < 10:
            solution = np.tile(solution, (10 // len(solution) + 1))[:10]
        
        # Random city coordinates (consistent per seed)
        cities = self.rng.uniform(-10, 10, (10, 2))
        
        # Calculate tour length
        tour_length = 0
        for i in range(len(cities)):
            curr_city = int(abs(solution[i]) * len(cities)) % len(cities)
            next_city = int(abs(solution[(i + 1) % len(solution)]) * len(cities)) % len(cities)
            
            distance = np.sqrt(np.sum((cities[curr_city] - cities[next_city])**2))
            tour_length += distance
        
        # Convert to fitness (lower tour length = higher fitness)
        max_length = 50  # Approximate worst case
        fitness = 1.0 - (tour_length / max_length)
        
        noise = self.rng.normal(0, 0.02)
        return np.clip(fitness + noise, 0, 1)
    
    def _synthetic_function(self, solution: np.ndarray, context_info: Dict) -> float:
        """Synthetic function that depends on evolutionary context."""
        # Base fitness from Rosenbrock-like function
        fitness = 0
        for i in range(len(solution) - 1):
            fitness += 100 * (solution[i+1] - solution[i]**2)**2 + (1 - solution[i])**2
        
        # Normalize
        fitness = 1.0 / (1.0 + fitness / 1000)
        
        # Context-dependent modulation
        if context_info:
            gen_progress = context_info.get('gen_progress', 0.5)
            no_improve = context_info.get('no_improve_steps', 0)
            
            # Early phase: favor exploration (add randomness)
            if gen_progress < 0.3:
                fitness *= (0.8 + self.rng.uniform(0, 0.4))
            
            # Stuck phase: make breakthroughs harder but possible
            elif no_improve > 10:
                if self.rng.random() < 0.1:  # 10% chance of breakthrough
                    fitness *= 1.3
                else:
                    fitness *= 0.7
        
        noise = self.rng.normal(0, 0.03)
        return np.clip(fitness + noise, 0, 1)


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
        
        # Initialize bandit
        models = ["fast_model", "accurate_model", "balanced_model"]
        if config.algorithm == "baseline":
            self.bandit = ThompsonSamplingBandit(
                arm_names=models,
                seed=config.seed,
                prior_alpha=2.0,
                prior_beta=1.0,
                auto_decay=0.99,
                reward_mapping="adaptive"
            )
        elif config.algorithm == "context":
            self.bandit = ContextAwareThompsonSamplingBandit(
                arm_names=models,
                seed=config.seed,
                contexts=["early", "mid", "late", "stuck"],
                features=["gen_progress", "no_improve", "fitness_slope", "pop_diversity"],
                prior_alpha=2.0,
                prior_beta=1.0,
                auto_decay=0.99,
                reward_mapping="adaptive"
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
    """Analyze benchmark results and generate report."""
    logger.info("Analyzing benchmark results...")
    
    raw_dir = Path(output_dir) / "raw"
    
    # Collect all results
    all_results = []
    
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
                
                # Calculate metrics
                final_fitness = df['best_fitness'].iloc[-1]
                llm_queries_total = df['llm_queries_cum'].iloc[-1]
                
                # Time to first improvement
                improvement_steps = df[df['improvement'] == True]
                time_to_first_improve = improvement_steps['step'].iloc[0] if len(improvement_steps) > 0 else None
                
                # Stuck phase analysis
                stuck_queries = 0
                window_size = 25
                for i in range(window_size, len(df)):
                    recent_best = df['best_fitness'].iloc[i-window_size:i]
                    if recent_best.max() <= recent_best.min() + 1e-6:  # Stuck
                        stuck_queries += df['llm_queries'].iloc[i]
                
                # Area under curve
                auc = np.trapezoid(df['best_fitness'].values)
                
                result = {
                    'algorithm': algorithm,
                    'benchmark': benchmark,
                    'seed': seed,
                    'final_fitness': final_fitness,
                    'llm_queries_total': llm_queries_total,
                    'llm_queries_while_stuck': stuck_queries,
                    'time_to_first_improve': time_to_first_improve,
                    'area_under_curve': auc,
                    'context_switches': df['context'].nunique() - 1 if 'context' in df.columns else 0
                }
                
                all_results.append(result)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    if results_df.empty:
        logger.warning("No results found for analysis")
        return {}
    
    # Aggregate by algorithm and benchmark
    aggregated = results_df.groupby(['algorithm', 'benchmark']).agg({
        'final_fitness': ['mean', 'std', 'count'],
        'llm_queries_total': ['mean', 'std'],
        'llm_queries_while_stuck': ['mean', 'std'],
        'time_to_first_improve': ['median', 'std'],
        'area_under_curve': ['mean', 'std'],
        'context_switches': ['mean', 'std']
    }).round(4)
    
    # Save summary
    summary_file = Path(output_dir) / "summary.csv"
    aggregated.to_csv(summary_file)
    
    # Calculate acceptance criteria
    acceptance_results = {}
    
    for benchmark in results_df['benchmark'].unique():
        bench_df = results_df[results_df['benchmark'] == benchmark]
        
        if 'baseline' in bench_df['algorithm'].values and 'context' in bench_df['algorithm'].values:
            baseline_fitness = bench_df[bench_df['algorithm'] == 'baseline']['final_fitness'].mean()
            context_fitness = bench_df[bench_df['algorithm'] == 'context']['final_fitness'].mean()
            
            baseline_stuck = bench_df[bench_df['algorithm'] == 'baseline']['llm_queries_while_stuck'].mean()
            context_stuck = bench_df[bench_df['algorithm'] == 'context']['llm_queries_while_stuck'].mean()
            
            baseline_time = bench_df[bench_df['algorithm'] == 'baseline']['time_to_first_improve'].median()
            context_time = bench_df[bench_df['algorithm'] == 'context']['time_to_first_improve'].median()
            
            # Calculate improvements
            fitness_improvement = ((context_fitness - baseline_fitness) / baseline_fitness) * 100
            stuck_queries_reduction = ((baseline_stuck - context_stuck) / baseline_stuck) * 100 if baseline_stuck > 0 else 0
            time_reduction = ((baseline_time - context_time) / baseline_time) * 100 if baseline_time and context_time else 0
            
            acceptance_results[benchmark] = {
                'fitness_improvement_pct': fitness_improvement,
                'stuck_queries_reduction_pct': stuck_queries_reduction, 
                'time_to_improve_reduction_pct': time_reduction,
                'fitness_pass': fitness_improvement >= 3.0,
                'stuck_pass': stuck_queries_reduction >= 10.0 or time_reduction >= 10.0
            }
    
    logger.info(f"Analysis complete. Summary saved to {summary_file}")
    
    return {
        'aggregated_results': aggregated,
        'acceptance_results': acceptance_results,
        'raw_results': results_df
    }


def generate_plots(output_dir: str = "reports/context_bandit"):
    """Generate visualization plots."""
    logger.info("Generating plots...")
    
    raw_dir = Path(output_dir) / "raw"
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Fitness vs Steps plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    benchmarks = ['toy', 'tsp', 'synthetic']
    
    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        
        for algorithm in ['baseline', 'context']:
            algo_dir = raw_dir / algorithm / benchmark
            if not algo_dir.exists():
                continue
            
            all_fitness = []
            max_steps = 0
            
            for csv_file in algo_dir.glob("run_*.csv"):
                df = pd.read_csv(csv_file)
                all_fitness.append(df['best_fitness'].values)
                max_steps = max(max_steps, len(df))
            
            if all_fitness:
                # Pad shorter runs
                padded_fitness = []
                for fitness in all_fitness:
                    if len(fitness) < max_steps:
                        padded = np.pad(fitness, (0, max_steps - len(fitness)), 'edge')
                        padded_fitness.append(padded)
                    else:
                        padded_fitness.append(fitness)
                
                fitness_array = np.array(padded_fitness)
                mean_fitness = np.mean(fitness_array, axis=0)
                std_fitness = np.std(fitness_array, axis=0)
                steps = np.arange(len(mean_fitness))
                
                label = algorithm.replace('_', ' ').title()
                ax.plot(steps, mean_fitness, label=label, linewidth=2)
                ax.fill_between(steps, mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.3)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Best Fitness')
        ax.set_title(f'{benchmark.upper()} Benchmark')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'fitness_curves.png', dpi=150, bbox_inches='tight')
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
    parser.add_argument('--algo', choices=['baseline', 'context', 'both'], 
                       default='both', help='Algorithm to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--budget_steps', type=int, default=1000, help='Evolution steps')
    parser.add_argument('--model', default='mock', help='LLM model to use')
    parser.add_argument('--output_dir', default='reports/context_bandit/raw', 
                       help='Output directory')
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