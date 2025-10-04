# ThompsonSamplingBandit

## Overview

The `ThompsonSamplingBandit` is a new bandit algorithm implementation for ShinkaEvolve that provides improved handling of non-stationary rewards and enhanced exploration capabilities. It uses Bayesian inference with Beta distributions to balance exploration and exploitation in LLM model selection.

## Algorithm Description

Thompson Sampling is a probabilistic algorithm for the multi-armed bandit problem that:

1. **Maintains Beta distributions** for each arm (LLM model)
2. **Samples from distributions** to make selection decisions
3. **Updates parameters** based on observed rewards
4. **Adapts to non-stationary environments** through decay mechanisms

### Mathematical Foundation

For each arm `i`, the algorithm maintains:
- `α_i`: Success parameter of Beta distribution
- `β_i`: Failure parameter of Beta distribution

The posterior mean for arm `i` is: `μ_i = α_i / (α_i + β_i)`

## Key Features

### 🎯 **Superior Non-Stationary Handling**
- Aggressive auto-decay (default 0.99) helps adapt to changing model performance
- Exponential decay towards prior prevents getting stuck on outdated information

### 🔍 **Enhanced Exploration**
- Probabilistic sampling naturally provides exploration
- Bayesian uncertainty quantification guides exploration decisions
- Configurable prior beliefs through `prior_alpha` and `prior_beta`

### ⚙️ **Flexible Configuration**
- `prior_alpha/beta`: Initial belief about model performance
- `reward_scaling`: Controls sensitivity to reward differences  
- `auto_decay`: Adaptation rate for non-stationary environments
- Compatible with existing `shift_by_baseline` and `shift_by_parent` options

## Usage Examples

### Basic Usage

```python
from shinka.llm import ThompsonSamplingBandit

# Create bandit for LLM model selection
bandit = ThompsonSamplingBandit(
    arm_names=["gpt-4", "claude-3", "gemini-pro"],
    seed=42,
    prior_alpha=1.5,      # Slightly optimistic prior
    prior_beta=1.0,       # Standard prior
    reward_mapping="adaptive",  # Auto-adapt to reward range (default)
    reward_scaling=1.5,   # Sensitivity to differences (for adaptive/direct)
    auto_decay=0.98       # Adapt to changes
)

# Update with rewards
bandit.update_submitted("gpt-4")
bandit.update("gpt-4", reward=0.85, baseline=0.5)

# Get selection probabilities
probs = bandit.posterior(samples=1000)
print(f"Selection probabilities: {probs}")
```

### Integration with EvolutionConfig

```python
from shinka.core import EvolutionConfig
from shinka.llm import ThompsonSamplingBandit

# Method 1: Direct bandit object
thompson_bandit = ThompsonSamplingBandit(
    arm_names=["gpt-4", "claude-3", "gemini-pro"],
    prior_alpha=2.0,
    prior_beta=1.0,
    auto_decay=0.95
)

config = EvolutionConfig(
    init_program_path="initial.py",
    llm_models=["gpt-4", "claude-3", "gemini-pro"],
    llm_dynamic_selection=thompson_bandit,
    num_generations=10
)

# Method 2: String-based configuration
config = EvolutionConfig(
    init_program_path="initial.py", 
    llm_models=["gpt-4", "claude-3", "gemini-pro"],
    llm_dynamic_selection="thompson",
    llm_dynamic_selection_kwargs={
        "prior_alpha": 2.0,
        "prior_beta": 1.0,
        "reward_scaling": 2.0,
        "auto_decay": 0.99
    }
)
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_arms` | int | None | Number of arms (auto-detected from arm_names) |
| `arm_names` | List[str] | None | Names of arms (e.g., LLM model names) |
| `seed` | int | None | Random seed for reproducibility |
| `prior_alpha` | float | 1.0 | Prior success count (optimism level) |
| `prior_beta` | float | 1.0 | Prior failure count |
| `reward_mapping` | str | "adaptive" | Reward mapping method: "adaptive", "direct", "sigmoid" |
| `reward_scaling` | float | 1.0 | Scaling factor for reward sensitivity |
| `auto_decay` | float | 0.99 | Automatic decay factor for non-stationarity |
| `shift_by_baseline` | bool | True | Whether to shift rewards by baseline |
| `shift_by_parent` | bool | True | Whether to shift rewards by parent |

## Performance Comparison

Compared to existing bandit algorithms in ShinkaEvolve:

### vs AsymmetricUCB
- **Exploration**: More natural exploration through sampling vs ε-greedy
- **Non-stationary**: Better adaptation through Bayesian updating
- **Convergence**: Smoother convergence to optimal arms

### vs FixedSampler  
- **Learning**: Adapts to performance vs fixed probabilities
- **Efficiency**: Concentrates on better-performing arms over time

## Implementation Details

### Reward Processing
1. **Baseline Shifting**: Same logic as existing bandits for compatibility
2. **Smart Mapping**: Three options for converting rewards to success probabilities:
   - **Adaptive** (default): Automatically adapts to observed reward distribution
   - **Direct**: Linear mapping assuming rewards in [0,1] range
   - **Sigmoid**: Legacy mapping (not recommended for most cases)
3. **Beta Updates**: Increments α by success_prob, β by (1 - success_prob)

### Sampling Strategy
- **Single Sample**: Samples once from each Beta distribution, selects maximum
- **Multi-Sample**: Repeatedly samples to estimate selection frequencies
- **Probabilistic**: Natural exploration without explicit ε-greedy mechanism

### Decay Mechanism
- **Exponential Decay**: `α_i ← factor × α_i + (1-factor) × prior_α`
- **Towards Prior**: Gradually forgets old information
- **Automatic**: Applied after each update when auto_decay is set

## Reward Mapping Methods

The `reward_mapping` parameter controls how raw rewards are converted to success probabilities for Beta parameter updates:

### 🎯 **Adaptive Mapping (Recommended)**
```python
bandit = ThompsonSamplingBandit(reward_mapping="adaptive")  # Default
```
- **Auto-adapts** to observed reward distribution
- Uses percentile-based normalization (5th-95th percentile)
- **Best for**: Unknown reward ranges, varying scales
- **Performance**: Preserves full reward information

### 📏 **Direct Mapping**  
```python
bandit = ThompsonSamplingBandit(reward_mapping="direct")
```
- **Linear mapping** assuming rewards in [0,1] range
- **Best for**: Known reward distributions, maximum transparency
- **Performance**: Perfect information preservation

### ⚠️ **Sigmoid Mapping (Legacy)**
```python
bandit = ThompsonSamplingBandit(reward_mapping="sigmoid")  # Not recommended
```
- **Information compression**: Maps to narrow [0.39, 0.61] range
- **Use only if**: Rewards have unbounded range and you want compression
- **Performance**: Slower convergence due to information loss

### 📊 **Performance Comparison**
| Mapping | Information Preservation | Convergence Speed | Adaptability |
|---------|-------------------------|------------------|--------------|
| Adaptive | ✅ Excellent | ✅ Fast | ✅ Auto-adapts |
| Direct | ✅ Perfect | ✅ Fast | ⚠️ Assumes [0,1] |
| Sigmoid | ❌ Compressed | ⚠️ Slower | ❌ Fixed compression |

## When to Use ThompsonSamplingBandit

**Recommended for:**
- ✅ Non-stationary LLM performance (model updates, API changes)
- ✅ Scenarios requiring natural exploration
- ✅ When you have prior beliefs about model performance
- ✅ Long-running evolution experiments
- ✅ Multiple similar-quality models

**Consider alternatives for:**
- ⚠️ Very short experiments (< 20 evaluations)
- ⚠️ When you want deterministic behavior
- ⚠️ Simple two-arm scenarios

## Troubleshooting

### Common Issues

**Low exploration**: Increase `prior_beta` or decrease `reward_scaling`
```python
bandit = ThompsonSamplingBandit(prior_beta=2.0, reward_scaling=0.5)
```

**Slow adaptation**: Increase `auto_decay` rate
```python  
bandit = ThompsonSamplingBandit(auto_decay=0.95)  # More aggressive decay
```

**Too sensitive**: Decrease `reward_scaling`
```python
bandit = ThompsonSamplingBandit(reward_scaling=0.5)  # Less sensitive
```

## Testing

The implementation includes comprehensive tests:

```bash
# Run basic functionality tests
python test_thompson_bandit.py

# Run integration tests  
python test_integration_thompson.py

# Run configuration tests
python test_config_integration.py
```

## References

1. Thompson, W.R. (1933). "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples"
2. Chapelle, O. & Li, L. (2011). "An empirical evaluation of thompson sampling"
3. Agrawal, S. & Goyal, N. (2012). "Analysis of thompson sampling for the multi-armed bandit problem"

## Changelog

### v0.1.0 (2025-01-XX)
- Initial implementation of ThompsonSamplingBandit
- Integration with ShinkaEvolve bandit interface
- Support for non-stationary rewards via auto-decay
- Comprehensive test suite
- Documentation and usage examples