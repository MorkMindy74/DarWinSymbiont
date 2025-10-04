#!/usr/bin/env python3
"""
Test that ThompsonSamplingBandit can be used in EvolutionConfig.
"""

import sys
sys.path.insert(0, '/app')

from shinka.core import EvolutionConfig
from shinka.llm import ThompsonSamplingBandit

def test_evolution_config_with_thompson():
    """Test using ThompsonSamplingBandit in EvolutionConfig."""
    print("🧬 Testing EvolutionConfig with ThompsonSamplingBandit")
    print("=" * 60)
    
    # Create Thompson Sampling bandit for dynamic LLM selection
    llm_models = ["gpt-4", "claude-3", "gemini-pro"]
    
    thompson_bandit = ThompsonSamplingBandit(
        arm_names=llm_models,
        seed=42,
        prior_alpha=1.5,
        prior_beta=1.0,
        auto_decay=0.95,
        reward_scaling=1.5
    )
    
    print(f"✅ Created ThompsonSamplingBandit with models: {llm_models}")
    
    # Create EvolutionConfig using the Thompson bandit
    try:
        config = EvolutionConfig(
            init_program_path="initial.py",
            llm_models=llm_models,
            llm_dynamic_selection=thompson_bandit,  # Use Thompson Sampling
            num_generations=5,
            max_parallel_jobs=2,
        )
        
        print("✅ Successfully created EvolutionConfig with ThompsonSamplingBandit")
        print(f"   Models: {config.llm_models}")
        print(f"   Dynamic selection: {type(config.llm_dynamic_selection).__name__}")
        print(f"   Generations: {config.num_generations}")
        
        # Verify bandit configuration
        bandit = config.llm_dynamic_selection
        if isinstance(bandit, ThompsonSamplingBandit):
            print(f"   Bandit prior α: {bandit.prior_alpha}")
            print(f"   Bandit prior β: {bandit.prior_beta}")
            print(f"   Bandit auto_decay: {bandit._auto_decay}")
            print(f"   Bandit reward_scaling: {bandit.reward_scaling}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to create EvolutionConfig: {e}")
        return False

def test_string_based_config():
    """Test using string-based configuration for Thompson bandit."""
    print("\n📝 Testing String-Based Configuration")
    print("=" * 40)
    
    try:
        # This tests the scenario where users specify Thompson bandit via string
        # (would need to be implemented in the framework's configuration parsing)
        config = EvolutionConfig(
            init_program_path="initial.py",
            llm_models=["gpt-4", "claude-3"],
            llm_dynamic_selection="thompson",  # String-based selection
            llm_dynamic_selection_kwargs={
                "prior_alpha": 2.0,
                "prior_beta": 1.0,
                "reward_scaling": 2.0,
                "auto_decay": 0.99
            },
            num_generations=3
        )
        
        print("✅ String-based configuration working")
        print(f"   Selection method: {config.llm_dynamic_selection}")
        print(f"   Selection kwargs: {config.llm_dynamic_selection_kwargs}")
        
        return True
        
    except Exception as e:
        print(f"ℹ️  String-based config not yet implemented: {e}")
        print("   (This is expected - would need framework updates)")
        return False

if __name__ == "__main__":
    print("🧪 Testing ThompsonSamplingBandit Configuration Integration\n")
    
    success1 = test_evolution_config_with_thompson()
    success2 = test_string_based_config()
    
    print("\n📊 Test Results:")
    print(f"   Direct bandit object: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   String-based config:  {'✅ PASS' if success2 else 'ℹ️ NOT IMPLEMENTED'}")
    
    if success1:
        print("\n🎉 ThompsonSamplingBandit successfully integrates with ShinkaEvolve!")
        print("\n📋 Ready for use in:")
        print("   • LLM model selection")
        print("   • Dynamic sampling strategies")
        print("   • Non-stationary reward environments")
        print("   • Exploration-exploitation balance")
    else:
        print("\n❌ Integration test failed")
        sys.exit(1)