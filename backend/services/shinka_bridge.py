"""
Bridge service to integrate DarWin Symbiont (ShinkaEvolve) backend
"""
import os
import sys
import asyncio
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add darwin-symbiont to path
DARWIN_PATH = Path("/app/darwin-symbiont")
sys.path.insert(0, str(DARWIN_PATH))

from shinka.core.runner import EvolutionConfig, EvolutionRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


class ShinkaEvolutionBridge:
    """Bridge between EMERGENT API and ShinkaEvolve backend"""
    
    def __init__(self, session_id: str, work_dir: Path):
        self.session_id = session_id
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.work_dir / "evolution.db"
        self.initial_program_path = self.work_dir / "initial.py"
        self.eval_program_path = self.work_dir / "evaluate.py"
        self.results_dir = self.work_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.runner: Optional[EvolutionRunner] = None
        self.is_running = False
        
    def create_evolution_config(self, user_config: Dict[str, Any]) -> EvolutionConfig:
        """Create ShinkaEvolve EvolutionConfig from user config"""
        # IMPORTANT: ShinkaEvolve changes cwd, so we need absolute paths
        return EvolutionConfig(
            num_generations=user_config.get("num_generations", 50),
            max_parallel_jobs=user_config.get("max_parallel_jobs", 2),
            llm_models=user_config.get("llm_models", ["emergent-gpt-4o"]),  # Use Emergent Key by default
            init_program_path=str(self.initial_program_path.absolute()),
            results_dir=str(self.results_dir.absolute()),
            patch_types=user_config.get("patch_types", ["diff"]),
            llm_cache_enabled=False,  # Disable cache to simplify
            llm_cache_path=None,
            archive_enabled=False,  # Disable archive to simplify
            archive_root=None,
        )
    
    def create_database_config(self, user_config: Dict[str, Any]) -> DatabaseConfig:
        """Create DatabaseConfig from user config"""
        return DatabaseConfig(
            db_path=str(self.db_path),
            num_islands=user_config.get("num_islands", 4),
            archive_size=user_config.get("archive_size", 100),
            migration_interval=user_config.get("migration_interval", 10),
        )
    
    def create_job_config(self, user_config: Dict[str, Any]) -> LocalJobConfig:
        """Create LocalJobConfig from user config"""
        return LocalJobConfig(
            eval_program_path=str(self.eval_program_path),
            time="00:10:00",  # 10 minutes timeout
        )
    
    def save_program(self, filename: str, code: str):
        """Save program code to file"""
        filepath = self.work_dir / filename
        filepath.write_text(code)
        print(f"✅ Saved {filename} to {filepath}")
    
    def initialize_runner(self, user_config: Dict[str, Any]):
        """Initialize EvolutionRunner with configs"""
        evo_config = self.create_evolution_config(user_config)
        db_config = self.create_database_config(user_config)
        job_config = self.create_job_config(user_config)
        
        self.runner = EvolutionRunner(
            evo_config=evo_config,
            db_config=db_config,
            job_config=job_config,
        )
        
        print(f"✅ Initialized EvolutionRunner for session {self.session_id}")
    
    async def start_evolution(self):
        """Start evolution process"""
        if not self.runner:
            raise ValueError("Runner not initialized. Call initialize_runner() first.")
        
        if self.is_running:
            raise ValueError("Evolution already running")
        
        self.is_running = True
        
        try:
            # Run evolution in background
            print(f"🧬 Starting evolution for session {self.session_id}")
            await asyncio.to_thread(self.runner.run)
            print(f"✅ Evolution completed for session {self.session_id}")
        except Exception as e:
            print(f"❌ Evolution failed: {e}")
            raise
        finally:
            self.is_running = False
    
    def _find_actual_db_path(self) -> Optional[Path]:
        """Find the actual database path (handles ShinkaEvolve path duplication)"""
        # Check expected location first
        if self.db_path.exists():
            return self.db_path
        
        # Check for nested path (ShinkaEvolve bug)
        nested_path = self.results_dir / str(self.work_dir) / "evolution.db"
        if nested_path.exists():
            return nested_path
        
        # Search recursively in results directory
        try:
            for db_file in self.results_dir.rglob("evolution.db"):
                return db_file
        except Exception:
            pass
        
        return None
    
    def get_latest_generation(self) -> Optional[int]:
        """Get latest generation from database"""
        actual_db = self._find_actual_db_path()
        if not actual_db:
            return None
        
        conn = sqlite3.connect(str(actual_db), check_same_thread=False)
        cursor = conn.execute("SELECT MAX(generation) FROM programs")
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result[0] is not None else None
    
    def get_generation_data(self, generation: int) -> Dict[str, Any]:
        """Get all data for a specific generation"""
        actual_db = self._find_actual_db_path()
        if not actual_db:
            return {}
        
        conn = sqlite3.connect(str(actual_db), check_same_thread=False)
        
        # Get programs for this generation
        cursor = conn.execute("""
            SELECT id, generation, island_idx, code, public_metrics, parent_id
            FROM programs
            WHERE generation = ?
        """, (generation,))
        
        programs = []
        for row in cursor.fetchall():
            prog_id, gen, island_idx, code, metrics_json, parent_id = row
            metrics = json.loads(metrics_json) if metrics_json else {}
            
            programs.append({
                "id": prog_id,
                "generation": gen,
                "island_id": island_idx,
                "code": code,
                "metrics": metrics,
                "parent_id": parent_id,
                "fitness": metrics.get("combined_score", 0.0)
            })
        
        # Get best fitness
        cursor = conn.execute("""
            SELECT MAX(json_extract(metrics, '$.combined_score'))
            FROM programs
            WHERE generation <= ?
        """, (generation,))
        best_fitness = cursor.fetchone()[0] or 0.0
        
        # Get average fitness for this generation
        cursor = conn.execute("""
            SELECT AVG(json_extract(metrics, '$.combined_score'))
            FROM programs
            WHERE generation = ?
        """, (generation,))
        avg_fitness = cursor.fetchone()[0] or 0.0
        
        # Get diversity (unique code count)
        cursor = conn.execute("""
            SELECT COUNT(DISTINCT code)
            FROM programs
            WHERE generation = ?
        """, (generation,))
        diversity = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "generation": generation,
            "programs": programs,
            "best_fitness": float(best_fitness),
            "avg_fitness": float(avg_fitness),
            "diversity": diversity,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_best_solution(self) -> Optional[Dict[str, Any]]:
        """Get the best solution found so far"""
        actual_db = self._find_actual_db_path()
        if not actual_db:
            return None
        
        conn = sqlite3.connect(str(actual_db), check_same_thread=False)
        cursor = conn.execute("""
            SELECT id, generation, island_id, code, metrics
            FROM programs
            ORDER BY json_extract(metrics, '$.combined_score') DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        prog_id, generation, island_id, code, metrics_json = row
        metrics = json.loads(metrics_json) if metrics_json else {}
        
        return {
            "id": prog_id,
            "generation": generation,
            "island_id": island_id,
            "code": code,
            "metrics": metrics,
            "fitness": metrics.get("combined_score", 0.0)
        }
    
    def get_island_status(self) -> Dict[int, Dict[str, Any]]:
        """Get status of all islands"""
        actual_db = self._find_actual_db_path()
        if not actual_db:
            return {}
        
        conn = sqlite3.connect(str(actual_db), check_same_thread=False)
        
        # Get latest generation per island
        cursor = conn.execute("""
            SELECT 
                island_id,
                MAX(generation) as latest_gen,
                COUNT(*) as total_programs,
                MAX(json_extract(metrics, '$.combined_score')) as best_fitness,
                AVG(json_extract(metrics, '$.combined_score')) as avg_fitness
            FROM programs
            GROUP BY island_id
        """)
        
        islands = {}
        for row in cursor.fetchall():
            island_id, latest_gen, total, best, avg = row
            islands[island_id] = {
                "island_id": island_id,
                "latest_generation": latest_gen,
                "total_programs": total,
                "best_fitness": float(best) if best else 0.0,
                "avg_fitness": float(avg) if avg else 0.0,
                "population_size": 0  # TODO: calculate from current generation
            }
        
        conn.close()
        return islands


def generate_initial_program_code(problem_type: str, params: Dict[str, Any]) -> str:
    """Generate initial.py code based on problem type"""
    
    if problem_type == "tsp":
        return """
# Initial TSP solution using nearest neighbor heuristic

def solve_tsp(cities, distance_matrix):
    \"\"\"Solve TSP using nearest neighbor heuristic\"\"\"
    n = len(cities)
    tour = [0]  # Start from city 0
    unvisited = set(range(1, n))
    
    current = 0
    while unvisited:
        # Find nearest unvisited city
        nearest = min(unvisited, key=lambda city: distance_matrix[current][city])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour


def calculate_tour_length(tour, distance_matrix):
    \"\"\"Calculate total tour length\"\"\"
    total = 0
    for i in range(len(tour) - 1):
        total += distance_matrix[tour[i]][tour[i + 1]]
    # Return to start
    total += distance_matrix[tour[-1]][tour[0]]
    return total


def run_experiment(**kwargs):
    \"\"\"Main experiment function\"\"\"
    cities = kwargs.get('cities')
    distance_matrix = kwargs.get('distance_matrix')
    
    tour = solve_tsp(cities, distance_matrix)
    distance = calculate_tour_length(tour, distance_matrix)
    
    return {
        'tour': tour,
        'distance': distance,
        'fitness': -distance  # Minimize distance
    }
"""
    
    elif problem_type == "scheduling":
        return """
# Initial scheduling solution using simple greedy approach

def schedule_tasks(tasks, resources, dependencies):
    \"\"\"Schedule tasks using greedy approach\"\"\"
    schedule = []
    available_resources = set(resources)
    completed_tasks = set()
    
    for task in tasks:
        # Check if dependencies are met
        if all(dep in completed_tasks for dep in dependencies.get(task['id'], [])):
            # Assign to first available resource
            if available_resources:
                resource = available_resources.pop()
                schedule.append({
                    'task_id': task['id'],
                    'resource': resource,
                    'start_time': len(schedule)
                })
                completed_tasks.add(task['id'])
                available_resources.add(resource)
    
    return schedule


def run_experiment(**kwargs):
    \"\"\"Main experiment function\"\"\"
    tasks = kwargs.get('tasks')
    resources = kwargs.get('resources')
    dependencies = kwargs.get('dependencies', {})
    
    schedule = schedule_tasks(tasks, resources, dependencies)
    makespan = max([s['start_time'] for s in schedule]) if schedule else 0
    
    return {
        'schedule': schedule,
        'makespan': makespan,
        'fitness': -makespan  # Minimize makespan
    }
"""
    
    else:
        # Generic template
        return """
def solve_problem(**kwargs):
    \"\"\"Generic problem solver\"\"\"
    # TODO: Implement problem-specific logic
    return {}


def run_experiment(**kwargs):
    \"\"\"Main experiment function\"\"\"
    result = solve_problem(**kwargs)
    return {
        'result': result,
        'fitness': 0.0
    }
"""


def generate_evaluate_program_code(problem_type: str, params: Dict[str, Any]) -> str:
    """Generate evaluate.py code based on problem type"""
    
    if problem_type == "tsp":
        num_cities = params.get("num_cities", 10)
        return f"""
# Evaluation program for TSP

import sys
import json
import importlib.util
import numpy as np


def load_program(program_path):
    \"\"\"Load program from path\"\"\"
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_test_cities(num_cities={num_cities}):
    \"\"\"Generate random city coordinates\"\"\"
    np.random.seed(42)
    cities = np.random.rand(num_cities, 2) * 100
    return cities


def calculate_distance_matrix(cities):
    \"\"\"Calculate Euclidean distance matrix\"\"\"
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist


def validate_tour(tour, num_cities):
    \"\"\"Validate TSP tour\"\"\"
    if len(tour) != num_cities:
        return False
    if len(set(tour)) != num_cities:
        return False
    if not all(0 <= city < num_cities for city in tour):
        return False
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <program_path> <results_dir>")
        sys.exit(1)
    
    program_path = sys.argv[1]
    results_dir = sys.argv[2]
    
    # Load program
    program = load_program(program_path)
    
    # Generate test data
    cities = generate_test_cities()
    distance_matrix = calculate_distance_matrix(cities)
    
    # Run experiments
    num_runs = 3
    results = []
    
    for run_idx in range(num_runs):
        try:
            result = program.run_experiment(
                cities=cities,
                distance_matrix=distance_matrix
            )
            
            tour = result.get('tour', [])
            distance = result.get('distance', float('inf'))
            
            # Validate
            is_valid = validate_tour(tour, len(cities))
            
            results.append({{
                'run': run_idx,
                'tour': tour,
                'distance': distance,
                'valid': is_valid,
                'fitness': -distance if is_valid else float('-inf')
            }})
        except Exception as e:
            results.append({{
                'run': run_idx,
                'error': str(e),
                'valid': False,
                'fitness': float('-inf')
            }})
    
    # Aggregate metrics
    valid_results = [r for r in results if r.get('valid', False)]
    
    if valid_results:
        distances = [r['distance'] for r in valid_results]
        best_distance = min(distances)
        avg_distance = np.mean(distances)
        
        metrics = {{
            'combined_score': -best_distance,  # Maximize negative distance
            'public': {{
                'best_distance': best_distance,
                'avg_distance': avg_distance,
                'num_valid': len(valid_results)
            }},
            'private': {{
                'all_distances': distances
            }}
        }}
        correct = True
    else:
        metrics = {{
            'combined_score': float('-inf'),
            'public': {{'error': 'No valid solutions'}},
            'private': {{}}
        }}
        correct = False
    
    # Write results
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(results_dir, 'correct.txt'), 'w') as f:
        f.write('1' if correct else '0')
    
    print(f"Metrics: {{metrics}}")
    print(f"Correct: {{correct}}")


if __name__ == '__main__':
    main()
"""
    
    elif problem_type == "scheduling":
        return """
# Evaluation program for Scheduling

import sys
import json
import importlib.util
import numpy as np


def load_program(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_test_tasks(num_tasks=20):
    np.random.seed(42)
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            'id': i,
            'duration': np.random.randint(1, 10),
            'priority': np.random.randint(1, 5)
        })
    return tasks


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <program_path> <results_dir>")
        sys.exit(1)
    
    program_path = sys.argv[1]
    results_dir = sys.argv[2]
    
    program = load_program(program_path)
    
    tasks = generate_test_tasks()
    resources = ['R1', 'R2', 'R3']
    dependencies = {}
    
    num_runs = 3
    results = []
    
    for run_idx in range(num_runs):
        try:
            result = program.run_experiment(
                tasks=tasks,
                resources=resources,
                dependencies=dependencies
            )
            
            makespan = result.get('makespan', float('inf'))
            
            results.append({
                'run': run_idx,
                'makespan': makespan,
                'valid': True,
                'fitness': -makespan
            })
        except Exception as e:
            results.append({
                'run': run_idx,
                'error': str(e),
                'valid': False,
                'fitness': float('-inf')
            })
    
    valid_results = [r for r in results if r.get('valid', False)]
    
    if valid_results:
        makespans = [r['makespan'] for r in valid_results]
        best_makespan = min(makespans)
        
        metrics = {
            'combined_score': -best_makespan,
            'public': {
                'best_makespan': best_makespan,
                'avg_makespan': np.mean(makespans)
            },
            'private': {}
        }
        correct = True
    else:
        metrics = {
            'combined_score': float('-inf'),
            'public': {'error': 'No valid solutions'},
            'private': {}
        }
        correct = False
    
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(results_dir, 'correct.txt'), 'w') as f:
        f.write('1' if correct else '0')
    
    print(f"Metrics: {metrics}")
    print(f"Correct: {correct}")


if __name__ == '__main__':
    main()
"""
    
    else:
        return """
# Generic evaluation program

import sys
import json
import importlib.util


def load_program(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <program_path> <results_dir>")
        sys.exit(1)
    
    program_path = sys.argv[1]
    results_dir = sys.argv[2]
    
    program = load_program(program_path)
    
    try:
        result = program.run_experiment()
        metrics = {
            'combined_score': result.get('fitness', 0.0),
            'public': result,
            'private': {}
        }
        correct = True
    except Exception as e:
        metrics = {
            'combined_score': float('-inf'),
            'public': {'error': str(e)},
            'private': {}
        }
        correct = False
    
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(os.path.join(results_dir, 'correct.txt'), 'w') as f:
        f.write('1' if correct else '0')


if __name__ == '__main__':
    main()
"""
