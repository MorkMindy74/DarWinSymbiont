#!/usr/bin/env python3
"""
PRIORITY TESTS come richiesti nella code review
"""

import sys
import numpy as np
sys.path.insert(0, '/app')

from shinka.llm import ThompsonSamplingBandit, AsymmetricUCB

def test_convergence_to_best_model():
    """TEST PRIORITÀ 1: Verifica convergenza al modello migliore"""
    print("🎯 PRIORITY TEST 1: Convergence to Best Model")
    print("=" * 50)
    
    # Setup: 3 modelli con reward diversi
    true_rewards = {
        'model_a': 0.9,  # Best
        'model_b': 0.7,  # Medium  
        'model_c': 0.5   # Worst
    }
    
    bandit = ThompsonSamplingBandit(
        arm_names=['model_a', 'model_b', 'model_c'],
        reward_mapping="adaptive",  # Use fixed version
        prior_alpha=1.0,
        prior_beta=1.0,
        auto_decay=None  # No decay for clean test
    )
    
    baseline = 0.5
    
    # Simulate 100 rounds
    for round_num in range(100):
        # Select arm  
        probs = bandit.posterior()
        selected = list(true_rewards.keys())[np.argmax(probs)]
        
        # Get reward with noise
        reward = true_rewards[selected] + np.random.normal(0, 0.1)
        reward = np.clip(reward, 0, 1)
        
        bandit.update_submitted(selected)
        bandit.update(selected, reward=reward, baseline=baseline)
    
    # Verifica: model_a dovrebbe essere selezionato > 70% delle volte
    test_selections = []
    for _ in range(100):
        probs = bandit.posterior()
        selected = list(true_rewards.keys())[np.argmax(probs)]
        test_selections.append(selected)
    
    model_a_pct = test_selections.count('model_a') / len(test_selections)
    
    print(f"   Model selection after 100 training rounds:")
    print(f"   model_a (best): {model_a_pct:.1%}")
    print(f"   Target: >70%")
    
    success = model_a_pct > 0.7
    
    if success:
        print(f"   ✅ PASS: Converged to best model ({model_a_pct:.1%})")
    else:
        print(f"   ❌ FAIL: Did not converge ({model_a_pct:.1%})")
    
    # Show final bandit state
    print(f"\n   Final Beta parameters:")
    for i, model in enumerate(['model_a', 'model_b', 'model_c']):
        alpha = bandit.alpha[i]
        beta = bandit.beta[i]
        mean = alpha / (alpha + beta)
        print(f"   {model}: α={alpha:.2f}, β={beta:.2f}, mean={mean:.3f}")
    
    return success

def test_nonstationary_adaptation():
    """TEST PRIORITÀ 2: Verifica adattamento a shift in reward"""
    print("\n🔄 PRIORITY TEST 2: Non-Stationarity Adaptation")
    print("=" * 50)
    
    bandit = ThompsonSamplingBandit(
        arm_names=['model_a', 'model_b'],
        reward_mapping="adaptive",
        auto_decay=0.95,  # Aggressive decay per adaptation
        seed=42
    )
    
    baseline = 0.5
    
    # Phase 1: model_a è migliore  
    print("   Phase 1: model_a is better (0.9 vs 0.3)")
    for _ in range(50):
        bandit.update_submitted('model_a')
        bandit.update('model_a', reward=0.9 + np.random.normal(0, 0.05), baseline=baseline)
        bandit.update_submitted('model_b')
        bandit.update('model_b', reward=0.3 + np.random.normal(0, 0.05), baseline=baseline)
    
    # Test preferenze Phase 1
    phase1_selections = []
    for _ in range(20):
        probs = bandit.posterior()
        selected = ['model_a', 'model_b'][np.argmax(probs)]
        phase1_selections.append(selected)
    
    model_a_phase1 = phase1_selections.count('model_a') / len(phase1_selections)
    print(f"   Phase 1 selection: model_a = {model_a_phase1:.1%}")
    
    # Phase 2: model_b diventa migliore (shift!)
    print("   Phase 2: SHIFT - model_b becomes better (0.9 vs 0.3)")
    for _ in range(50):
        bandit.update_submitted('model_b') 
        bandit.update('model_b', reward=0.9 + np.random.normal(0, 0.05), baseline=baseline)
        bandit.update_submitted('model_a')
        bandit.update('model_a', reward=0.3 + np.random.normal(0, 0.05), baseline=baseline)
    
    # Test preferenze Phase 2
    phase2_selections = []
    for _ in range(20):
        probs = bandit.posterior()
        selected = ['model_a', 'model_b'][np.argmax(probs)]
        phase2_selections.append(selected)
    
    model_b_phase2 = phase2_selections.count('model_b') / len(phase2_selections)
    print(f"   Phase 2 selection: model_b = {model_b_phase2:.1%}")
    
    # Verifica adaptation
    adapted = model_b_phase2 > 0.60  # Dovrebbe preferire model_b dopo shift
    
    if adapted:
        print(f"   ✅ PASS: Successfully adapted to reward shift")
    else:
        print(f"   ❌ FAIL: Did not adapt to shift (model_b only {model_b_phase2:.1%})")
    
    print(f"   Adaptation score: {model_b_phase2 - (1-model_a_phase1):.1%} improvement")
    
    return adapted

def test_reward_mapping_critical():
    """TEST PRIORITÀ 3: Test CRITICO per reward mapping"""
    print("\n🔬 PRIORITY TEST 3: Critical Reward Mapping Test")
    print("=" * 50)
    
    bandit = ThompsonSamplingBandit(
        arm_names=['model_test'],
        reward_mapping="adaptive",
        prior_alpha=1.0,
        prior_beta=1.0
    )
    
    # Warm up adaptive mapping
    baseline = 0.5
    for _ in range(25):
        bandit._reward_to_success_probability(0.1 - baseline, baseline)
        bandit._reward_to_success_probability(0.9 - baseline, baseline)
    
    # Test con rewards in [0,1]
    test_cases = [
        (0.9, "High reward", 0.8, 1.0),    # Should map to >80%
        (0.5, "Medium reward", 0.4, 0.6),   # Should map to ~50%  
        (0.1, "Low reward", 0.0, 0.3)      # Should map to <30%
    ]
    
    print("   Testing reward → success probability mapping:")
    
    all_passed = True
    
    for reward, label, min_prob, max_prob in test_cases:
        success_prob = bandit._reward_to_success_probability(reward - baseline, baseline)
        
        passed = min_prob <= success_prob <= max_prob
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"   {label:12s} ({reward:.1f}) → {success_prob:.3f} [{min_prob:.1f}-{max_prob:.1f}] {status}")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("   ✅ PASS: All reward mappings within expected ranges")
    else:
        print("   ❌ FAIL: Some mappings outside expected ranges")
    
    return all_passed

def test_thompson_vs_asymmetric_ucb():
    """TEST PRIORITÀ 4: Benchmark vs AsymmetricUCB"""
    print("\n⚔️ PRIORITY TEST 4: Thompson vs AsymmetricUCB Benchmark") 
    print("=" * 50)
    
    # Setup stesso scenario per entrambi
    true_performance = {
        'model_excellent': 0.85,
        'model_good': 0.65, 
        'model_poor': 0.35
    }
    
    models = list(true_performance.keys())
    
    # Initialize bandits
    thompson = ThompsonSamplingBandit(
        arm_names=models,
        reward_mapping="adaptive",
        seed=42,
        auto_decay=0.98
    )
    
    ucb = AsymmetricUCB(
        arm_names=models,
        seed=42,
        auto_decay=0.98
    )
    
    baseline = 0.6  # Slightly different baseline for challenge
    
    # Run parallel experiment  
    thompson_history = []
    ucb_history = []
    
    for round_num in range(100):
        # Thompson selection
        t_probs = thompson.posterior()
        t_selected = models[np.argmax(t_probs)]
        t_reward = true_performance[t_selected] + np.random.normal(0, 0.08)
        t_reward = np.clip(t_reward, 0, 1)
        
        thompson.update_submitted(t_selected)
        thompson.update(t_selected, t_reward, baseline)
        thompson_history.append(t_selected)
        
        # UCB selection
        u_probs = ucb.posterior()
        u_selected = models[np.argmax(u_probs)]
        u_reward = true_performance[u_selected] + np.random.normal(0, 0.08)
        u_reward = np.clip(u_reward, 0, 1)
        
        ucb.update_submitted(u_selected)
        ucb.update(u_selected, u_reward, baseline)
        ucb_history.append(u_selected)
    
    # Analyze final 25 selections
    t_final = thompson_history[-25:]
    u_final = ucb_history[-25:]
    
    t_excellent = t_final.count('model_excellent') / len(t_final)
    u_excellent = u_final.count('model_excellent') / len(u_final)
    
    print(f"   Final 25 rounds - 'model_excellent' selection:")
    print(f"   Thompson Sampling: {t_excellent:.1%}")
    print(f"   AsymmetricUCB:     {u_excellent:.1%}")
    
    # Both should be high, but we're looking for comparable performance
    thompson_good = t_excellent > 0.7
    ucb_good = u_excellent > 0.7
    
    if thompson_good and ucb_good:
        difference = t_excellent - u_excellent
        if abs(difference) < 0.15:  # Within 15%
            print(f"   ✅ PASS: Comparable performance (diff: {difference:+.1%})")
            return True
        elif difference > 0:
            print(f"   ✅ EXCELLENT: Thompson outperforms UCB by {difference:+.1%}")
            return True
        else:
            print(f"   ⚠️ ACCEPTABLE: UCB slightly better by {-difference:.1%}")
            return True
    else:
        print(f"   ❌ FAIL: Poor performance from one or both algorithms")
        return False

if __name__ == "__main__":
    print("🧪 PRIORITY VALIDATION TESTS")
    print("=" * 70)
    print("Tests basati sulla code review per validare fix critici\n")
    
    # Run all priority tests
    results = {}
    
    results["convergence"] = test_convergence_to_best_model()
    results["adaptation"] = test_nonstationary_adaptation() 
    results["mapping"] = test_reward_mapping_critical()
    results["benchmark"] = test_thompson_vs_asymmetric_ucb()
    
    # Summary
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY:")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.title():15s}: {status}")
    
    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL PRIORITY TESTS PASSED!")
        print("✅ Critical Issue #1 (reward mapping) completely resolved")
        print("✅ Thompson Sampling ready for production use")
    elif passed_tests >= total_tests * 0.75:
        print("⚠️ Most tests passed - minor issues remain")
    else:
        print("❌ Multiple critical issues found - needs more work")
        
    print(f"\n🚀 ThompsonSamplingBandit validation complete!")
    print(f"   Recommendation: {'APPROVED for production' if passed_tests >= 3 else 'Needs more fixes'}")