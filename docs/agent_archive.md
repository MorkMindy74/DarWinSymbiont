# Agent Archive System

The ShinkaEvolve Agent Archive provides reproducible agent archiving with export/import functionality, lineage tracking, and DGM-compatible metadata. This system enables complete reproduction of evolution runs and facilitates sharing and collaboration.

## Overview

The Agent Archive System automatically captures:
- Complete agent configuration and hyperparameters
- Git state and environment information  
- Benchmark results and performance metrics
- Artifacts (reports, plots, configurations)
- Context activity and lineage information
- DGM-compatible metadata for cross-repository analysis

## Quick Start

### Basic Usage

```bash
# List all archived agents
shinka_archive list

# Show detailed information about an agent
shinka_archive show <agent_id>

# Export an agent to ZIP file  
shinka_archive export <agent_id> -o agent.zip

# Import an agent from ZIP file
shinka_archive import agent.zip

# Reproduce an agent run (verify results)
shinka_repro <agent_id> --tolerance 1.0
```

### Automatic Archiving

Enable automatic archiving in your evolution configuration:

```yaml
# configs/evolution/with_archive.yaml
archive_enabled: true
archive_auto_save: true
archive_save_on: 
  - "on_best_fitness"  # Save when best fitness improves
  - "on_finish"        # Save final agent
```

## Configuration

### Evolution Config Integration

Add archive settings to your `EvolutionConfig`:

```python
@dataclass
class EvolutionConfig:
    # ... other fields ...
    
    # Agent Archive configuration
    archive_enabled: bool = False
    archive_auto_save: bool = True
    archive_save_on: List[str] = field(default_factory=lambda: ["on_best_fitness", "on_finish"])
    archive_root: Optional[str] = None  # Auto-generated if None
```

### Auto-save Triggers

- **`on_best_fitness`**: Save agent when fitness improves beyond previous best
- **`on_finish`**: Save final agent at end of evolution run
- **`manual`**: Only save when explicitly called programmatically

## CLI Commands

### List Agents

```bash
shinka_archive list
```

Output:
```
ID           Timestamp            Fitness    Parent       Branch          Dirty
--------------------------------------------------------------------------------
84b3873e56a2 2025-10-04 19:33:21  1.000      None         main            No   
a1b2c3d4e5f6 2025-10-04 18:45:12  0.856      84b3873e56a2 feature/new     Yes  

Total: 2 agents
```

### Show Agent Details

```bash
shinka_archive show 84b3873e56a2
```

Output includes:
- Agent ID, timestamp, lineage
- Git commit, branch, dirty status  
- Environment details (Python version, OS, dependencies)
- Seeds and hyperparameters
- Benchmark results across all test cases
- Context switching activity
- Cost estimates

### Export Agent

```bash
shinka_archive export 84b3873e56a2 -o my_agent.zip
```

Creates ZIP containing:
- `manifest.json` - Complete metadata
- `diff.patch` - Git changes (if dirty)
- `artifacts/` - Reports, plots, configurations
- `source_snapshot/` - Modified source files (optional)

### Import Agent

```bash
shinka_archive import my_agent.zip
```

Restores agent to archive with all metadata intact.

### Reproduce Agent

```bash
shinka_repro 84b3873e56a2 --tolerance 1.0
```

Reproduces the agent's benchmark run and verifies results within ±1% tolerance:

```
📊 Reproduction Results:

toy_benchmark: ✅ PASS
  ✅ final_best_fitness:
    Original:     1.0000
    Reproduction: 1.0078
    Difference:   0.78%

🎯 REPRODUCTION SUCCESS: All metrics within tolerance
```

## Archive Structure

```
archive/
├── agents/
│   ├── 20251004_193321_84b3873e56a2/
│   │   ├── manifest.json           # Complete metadata (enriched)
│   │   ├── diff.patch             # Git changes (if dirty repo)
│   │   ├── artifacts/             # Reports and outputs
│   │   │   ├── config_used.yaml   # Configuration snapshot  
│   │   │   └── benchmark_reports/ # Benchmark results & plots
│   │   ├── timeseries/            # Full time series data (CSV)
│   │   │   ├── context_toy_timeseries.csv
│   │   │   └── baseline_tsp_timeseries.csv
│   │   └── source_snapshot/       # Source files (optional)
│   └── 20251004_184512_a1b2c3d4e5f6/
└── latest_symlink -> agents/20251004_193321_84b3873e56a2/
```

## Manifest Schema

Each agent has a complete `manifest.json` with DGM-compatible metadata:

```json
{
  "agent_id": "84b3873e56a2",
  "parent_id": null,
  "timestamp": "2025-10-04T19:33:21.123456",
  "git_commit": "c52badb4798efae1426039885f13335566026f67",
  "branch": "main", 
  "dirty": false,
  
  "env": {
    "python": "3.11.13",
    "os": "Linux",
    "platform": "Linux-5.4.0-x86_64",
    "dependencies_lock": "8971cb07d52cece5"
  },
  
  "seeds": {
    "global": 42,
    "numpy": 42, 
    "torch": 42
  },
  
  "evo_config_path": "configs/evolution/context_aware.yaml",
  "evo_config_inline": { "...": "complete config" },
  
  "hyperparams": {
    "bandit": "thompson_context",
    "prior_alpha": 2.0,
    "prior_beta": 1.0,
    "decay": 0.99,
    "contexts": ["early", "mid", "late", "stuck"]
  },
  
  "benchmarks": {
    "tsp_benchmark": {
      "final_best_fitness": 0.651,
      "auc_fitness": 520.8,
      "time_to_first_improve": 5.0,
      "llm_queries_total": 766.7,
      "llm_queries_while_stuck": 676.3,
      "variance_ratio_vs_baseline": 1.45
    }
  },
  
  "cost": {
    "total_usd_est": 0.26,
    "queries_total": 2600
  },
  
  "context_activity": {
    "switch_count": 5,
    "dwell_time": {
      "early": 0.16, 
      "mid": 0.0,
      "late": 0.0, 
      "stuck": 0.84
    }
  },
  
  "dgm_compat": {
    "repo_layout_ref": "ShinkaEvolve structure (shinka/core/runner.py, bench/context_bandit_bench.py)",
    "prompts_used": [],
    "swe_bench_commit": "c52badb4798efae1426039885f13335566026f67", 
    "polyglot_prepared": true
  },
  
  "benchmarks_full": {
    "toy_baseline_timeseries": "timeseries/baseline_toy_timeseries.csv",
    "tsp_context_timeseries": "timeseries/context_tsp_timeseries.csv"
  },
  
  "complexity_metrics": {
    "lines_of_code_total": 22315,
    "cyclomatic_complexity": 3.2,
    "coupling_between_objects": 2,
    "code_coverage_pct": 85.0,
    "technical_debt_ratio": 0.0
  },
  
  "validation_levels": {
    "static_checks": {"ruff": "fail", "mypy": "unknown"},
    "unit_tests": {"status": "pass", "passed": 17, "failed": 0},
    "property_based": {"status": "not_implemented"},
    "fuzz_tests": {"status": "not_implemented"}
  },
  
  "cost_breakdown": {
    "mock_model": {"queries": 2600.0, "cost_usd": 0.0},
    "gpt_4o": {"queries": 0, "cost_usd": 78.0},
    "claude_sonnet": {"queries": 0, "cost_usd": 39.0},
    "gemini_pro": {"queries": 0, "cost_usd": 2.6}
  },
  
  "artifact_refs": {
    "benchmark_reports": "artifacts/benchmark_reports",
    "plots": "artifacts/benchmark_reports/plots", 
    "config_snapshot": "artifacts/config_used.yaml",
    "benchmark_summary": "artifacts/benchmark_reports/summary.csv"
  }
}
```

## DGM Compatibility

The archive system includes DGM (Darwin Goedel Machine) compatibility metadata to facilitate cross-repository analysis and comparison:

### Repository Layout Mapping

| DGM Structure | ShinkaEvolve Equivalent |
|---------------|------------------------|
| `DGM_outer.py` | `shinka/core/runner.py` |
| `coding_agent.py` | `shinka/llm/dynamic_sampling.py` |
| `output_dgm/` | `reports/context_bandit/` |
| `analysis/` | `docs/` + `tests/` |
| `prompts/` | `shinka/prompts/` |

### Metadata Alignment

- **`swe_bench_commit`**: Git commit for reproducibility
- **`prompts_used`**: List of prompt templates used
- **`polyglot_prepared`**: Multi-language readiness flag
- **`repo_layout_ref`**: Reference to equivalent DGM structure

## Programmatic Usage

### Archive Integration

```python
from shinka.core.runner import EvolutionRunner, EvolutionConfig
from shinka.archive import create_agent_archive

# Enable archiving in config
config = EvolutionConfig(
    archive_enabled=True,
    archive_auto_save=True,
    archive_save_on=["on_best_fitness", "on_finish"]
)

# Archives will be automatically saved during evolution
runner = EvolutionRunner(config, job_config, db_config)
runner.run()
```

### Manual Archiving

```python
from shinka.archive import create_agent_archive

archive = create_agent_archive("/path/to/archive")

# Save current state
agent_id = archive.save_agent({
    "algorithm": "context_aware",
    "seed": 42,
    "benchmark_results": {...}
})

# List all agents
agents = archive.list_agents()

# Get specific agent
manifest = archive.get_agent_manifest(agent_id)
```

### Reproduction Verification

```python  
from shinka.archive import reproduce_agent

result = reproduce_agent("84b3873e56a2", tolerance_pct=1.0)

if result["success"]:
    print("✅ Reproduction successful!")
    for benchmark, verification in result["verification_results"].items():
        if verification["passed"]:
            print(f"✅ {benchmark}")
        else:
            print(f"❌ {benchmark}")
else:
    print(f"❌ Reproduction failed: {result['error']}")
```

## Testing & Validation

### Unit Tests

```bash
# Run archive system tests
make test_archive

# Run archive sanity check  
make archive_sanity
```

### Acceptance Criteria

The system passes if:
1. **3 agents archived** in a session with list/show/export/import working
2. **Reproduction within ±1%** tolerance for key metrics
3. **DGM-compatible metadata** present in all manifests  
4. **All tests pass** in local and CI environments

### Continuous Integration

Archive testing is integrated into CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Test Archive System
  run: |
    make test_archive
    make archive_sanity
```

## Best Practices

### When to Use Archives

- **Reproducible Research**: Archive agents for papers/publications
- **Collaboration**: Share agents across teams/institutions
- **Debugging**: Preserve failing agents for analysis
- **Benchmarking**: Maintain reference implementations  
- **Production**: Archive production-quality agents

### Archive Management

- **Regular Cleanup**: Archives can grow large - prune old agents periodically
- **Naming Convention**: Use meaningful commit messages for traceability
- **Environment Consistency**: Ensure consistent dependencies for reproduction
- **Security**: Archives contain complete code - handle securely

### Reproduction Guidelines

- **Tolerance Settings**: Use 1-5% for stable algorithms, higher for stochastic ones
- **Environment Matching**: Reproduce on similar hardware/OS when possible
- **Seed Management**: Ensure all random seeds are captured and restored
- **Dependency Versions**: Archive dependency locks for long-term reproducibility

## Troubleshooting

### Common Issues

**"Agent not found" errors:**
- Verify agent ID is correct with `shinka_archive list`
- Check archive directory exists and has proper permissions

**Reproduction failures:**
- Check tolerance settings (may need higher for noisy algorithms)
- Verify environment consistency (Python version, dependencies)
- Look for hardware differences (CPU/GPU, memory)

**Large archive sizes:**
- Archives include complete benchmark results - consider pruning old data
- Use `.gitignore` style patterns to exclude large artifacts  
- Implement archive compression for long-term storage

**Import/Export issues:**
- Ensure ZIP files are complete and not corrupted
- Check file permissions in archive directories
- Verify sufficient disk space for large archives

### Debugging

Enable detailed logging:

```python
import logging
logging.getLogger('shinka.archive').setLevel(logging.DEBUG)
```

Check archive contents:
```bash
# Inspect archive structure
ls -la /app/shinka/archive/agents/

# Check manifest
cat /app/shinka/archive/agents/*/manifest.json | jq .
```

## Security Considerations

**⚠️ SECURITY WARNING**: Agent archives contain complete code snapshots and execution environments. 

- **Code Execution**: Reproduction involves executing archived code - run in sandboxed environments
- **Dependency Trust**: Archives may contain dependency snapshots - verify integrity
- **Sensitive Data**: Ensure no secrets/credentials are included in archived configurations
- **Access Control**: Limit archive access to authorized users only

## Future Enhancements

### Planned Features

- **Semantic Similarity**: LLM-based agent similarity detection
- **Performance Profiling**: Built-in benchmarking and cost analysis  
- **Archive Compression**: Automatic compression for long-term storage
- **Cloud Integration**: S3/GCS backends for distributed archives
- **Visualization**: Web UI for archive browsing and comparison

### Extensibility

The archive system is designed for extension:

- **Custom Metadata**: Add domain-specific fields to manifests
- **Alternative Formats**: Support for different serialization formats
- **Remote Archives**: Network-based archive backends
- **Integration Hooks**: Custom triggers and workflows

## Related Documentation

- **[Context-Aware Thompson Sampling](context_aware_thompson_sampling.md)**: The bandit algorithm being archived
- **[LLM Cache System](llm_cache_system.md)**: Caching layer for cost optimization
- **[Configuration Guide](configuration.md)**: Complete configuration reference
- **[Getting Started](getting_started.md)**: Basic setup and usage