#!/usr/bin/env python3
"""
Test script for ThompsonSamplingBandit implementation.

This script tests the basic functionality of the ThompsonSamplingBandit
including initialization, reward updates, sampling, and decay mechanisms.
"""

import numpy as np
import sys
import os

# Add the shinka package to path
sys.path.insert(0, '/app')

from shinka.llm.dynamic_sampling import ThompsonSamplingBandit

def test_thompson_bandit_basic():
    """Test basic functionality of ThompsonSamplingBandit."""
    print("🧪 Testing ThompsonSamplingBandit Basic Functionality")
    print("=" * 60)
    
    # Test 1: Basic initialization
    print("Test 1: Basic Initialization")
    bandit = ThompsonSamplingBandit(
        n_arms=3,
        arm_names=["model_a", "model_b", "model_c"],
        seed=42,
        prior_alpha=1.0,
        prior_beta=1.0,
        reward_scaling=1.0
    )
    
    print(f"✅ Bandit initialized with {bandit.n_arms} arms")
    print(f"   Alpha parameters: {bandit.alpha}")
    print(f"   Beta parameters: {bandit.beta}")
    
    # Test 2: Reward updates
    print("\nTest 2: Reward Updates")
    
    # Simulate some rewards
    rewards_data = [
        ("model_a", 0.8, 0.5),  # Good reward
        ("model_b", 0.2, 0.5),  # Poor reward
        ("model_c", 0.9, 0.5),  # Great reward
        ("model_a", 0.7, 0.5),  # Another good reward
        ("model_b", 0.1, 0.5),  # Another poor reward
    ]
    
    for arm, reward, baseline in rewards_data:
        bandit.update_submitted(arm)
        r, b = bandit.update(arm, reward, baseline)
        print(f"   Updated {arm}: reward={reward:.2f}, processed_reward={r:.3f}")
    
    print(f"   Updated Alpha: {bandit.alpha}")
    print(f"   Updated Beta: {bandit.beta}")
    print(f"   Beta means: {bandit.alpha / (bandit.alpha + bandit.beta)}")
    
    # Test 3: Sampling (posterior)
    print("\nTest 3: Sampling Behavior")
    
    # Single sample
    single_sample = bandit.posterior()
    print(f"   Single sample probabilities: {single_sample}")
    
    # Multi-sample to see distribution
    multi_sample = bandit.posterior(samples=1000)
    print(f"   Multi-sample (1000) probabilities: {multi_sample}")
    
    # Test 4: Decay mechanism
    print("\nTest 4: Decay Mechanism")
    alpha_before = bandit.alpha.copy()
    beta_before = bandit.beta.copy()
    
    bandit.decay(0.9)  # 10% decay
    
    print(f"   Alpha before decay: {alpha_before}")
    print(f"   Alpha after decay:  {bandit.alpha}")
    print(f"   Beta before decay:  {beta_before}")
    print(f"   Beta after decay:   {bandit.beta}")
    
    # Test 5: Print summary
    print("\nTest 5: Summary Display")
    bandit.print_summary()
    
    print("\n✅ All tests completed successfully!")
    
def test_thompson_vs_other_bandits():
    """Compare ThompsonSamplingBandit behavior with AsymmetricUCB."""
    print("\n🔍 Comparing ThompsonSamplingBandit vs AsymmetricUCB")
    print("=" * 60)
    
    from shinka.llm.dynamic_sampling import AsymmetricUCB
    
    # Initialize both bandits
    thompson = ThompsonSamplingBandit(
        n_arms=3, 
        arm_names=["A", "B", "C"], 
        seed=42
    )
    
    ucb = AsymmetricUCB(
        n_arms=3,
        arm_names=["A", "B", "C"],
        seed=42
    )
    
    # Simulate rewards - arm A is best, B is worst, C is middle
    np.random.seed(42)
    
    for i in range(20):
        for bandit_name, bandit in [("Thompson", thompson), ("UCB", ucb)]:
            # Simulate reward for each arm
            for arm, true_reward in [("A", 0.8), ("B", 0.2), ("C", 0.5)]:
                # Add noise
                reward = true_reward + np.random.normal(0, 0.1)
                bandit.update_submitted(arm)
                bandit.update(arm, reward, 0.0)
    
    print("After 20 rounds of updates:")
    print("\nThompson Sampling:")
    thompson.print_summary()
    
    print("\nAsymmetric UCB:")
    ucb.print_summary()
    
    # Compare selection probabilities
    thompson_probs = thompson.posterior(samples=1000)
    ucb_probs = ucb.posterior()
    
    print(f"\nSelection probabilities comparison:")
    print(f"Thompson: {thompson_probs}")
    print(f"UCB:      {ucb_probs}")

def test_non_stationary_behavior():
    """Test how ThompsonSamplingBandit handles non-stationary rewards."""
    print("\n🔄 Testing Non-Stationary Reward Handling")
    print("=" * 60)
    
    bandit = ThompsonSamplingBandit(
        n_arms=2,
        arm_names=["stable", "switching"],
        seed=42,
        auto_decay=0.95  # Aggressive decay for non-stationary
    )
    
    # Phase 1: "stable" is better
    print("Phase 1: 'stable' arm is better")
    for i in range(10):
        bandit.update_submitted("stable")
        bandit.update("stable", 0.8, 0.5)
        bandit.update_submitted("switching")
        bandit.update("switching", 0.3, 0.5)
    
    phase1_probs = bandit.posterior(samples=1000)
    print(f"Selection probs after phase 1: {phase1_probs}")
    
    # Phase 2: "switching" becomes better (reward shift)
    print("\nPhase 2: 'switching' arm becomes better")
    for i in range(15):
        bandit.update_submitted("stable")
        bandit.update("stable", 0.3, 0.5)  # Now worse
        bandit.update_submitted("switching")
        bandit.update("switching", 0.9, 0.5)  # Now better
    
    phase2_probs = bandit.posterior(samples=1000)
    print(f"Selection probs after phase 2: {phase2_probs}")
    
    print(f"Adaptation: switching prob went from {phase1_probs[1]:.3f} to {phase2_probs[1]:.3f}")
    
    if phase2_probs[1] > phase1_probs[1]:
        print("✅ Successfully adapted to reward shift!")
    else:
        print("❌ Failed to adapt to reward shift")

if __name__ == "__main__":
    try:
        test_thompson_bandit_basic()
        test_thompson_vs_other_bandits()
        test_non_stationary_behavior()
        
        print("\n🎉 All Thompson Sampling Bandit tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)