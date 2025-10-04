#!/usr/bin/env python3
"""
CRITICAL TEST: Verifica Issue #1 - Reward Mapping Problem
"""

import sys
import numpy as np
sys.path.insert(0, '/app')

from shinka.llm import ThompsonSamplingBandit

def test_reward_mapping_issue():
    """TEST CRITICO: Verifica il problema del sigmoid mapping"""
    print("🚨 TESTING CRITICAL ISSUE #1: Sigmoid Reward Mapping")
    print("=" * 60)
    
    bandit = ThompsonSamplingBandit(
        arm_names=['model_test'],
        prior_alpha=1.0,
        prior_beta=1.0,
        reward_scaling=1.0,
        auto_decay=None  # No decay per test pulito
    )
    
    # Simula rewards tipici di ShinkaEvolve con baseline=0.5
    baseline = 0.5
    test_cases = [
        ("Excellent", 0.95, baseline),  # Ottimo risultato
        ("Good", 0.80, baseline),       # Buono 
        ("Average", 0.50, baseline),    # Media
        ("Poor", 0.20, baseline),       # Scarso
        ("Bad", 0.05, baseline),        # Pessimo
    ]
    
    print("Analisi Reward Mapping (PROBLEMA IDENTIFICATO):")
    print("Reward → (r-baseline) → sigmoid(r-baseline) → Success Prob")
    print("-" * 60)
    
    issues_found = []
    
    for label, reward, baseline in test_cases:
        # Simula il processo interno
        r = reward - baseline  # Stesso processo del codice
        success_prob = bandit._sigmoid(r)
        
        print(f"{label:10s}: {reward:4.2f} → {r:6.2f} → sigmoid({r:6.2f}) = {success_prob:.4f}")
        
        # Verifica problemi
        if reward >= 0.9 and success_prob < 0.8:
            issues_found.append(f"❌ Excellent reward {reward} → low prob {success_prob:.3f}")
        elif reward <= 0.2 and success_prob > 0.4:
            issues_found.append(f"❌ Poor reward {reward} → high prob {success_prob:.3f}")
        elif 0.4 <= reward <= 0.6 and not (0.45 <= success_prob <= 0.55):
            issues_found.append(f"⚠️ Average reward {reward} → biased prob {success_prob:.3f}")
    
    print("\n📊 ANALISI RISULTATI:")
    
    if issues_found:
        print("🚨 PROBLEMI IDENTIFICATI:")
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print("✅ Nessun problema identificato")
        
    print(f"\n🎯 PROBLEMA CORE:")
    print(f"   sigmoid(0.45) = {bandit._sigmoid(0.45):.4f}  # reward=0.95 → prob=71%! ❌")
    print(f"   sigmoid(0.30) = {bandit._sigmoid(0.30):.4f}  # reward=0.80 → prob=57%! ❌")
    print(f"   sigmoid(0.00) = {bandit._sigmoid(0.00):.4f}  # reward=0.50 → prob=50%  ⚠️")
    print(f"   sigmoid(-0.30) = {bandit._sigmoid(-0.30):.4f} # reward=0.20 → prob=43%! ❌")
    
    return len(issues_found) > 0

def test_convergence_with_broken_mapping():
    """Test: il mapping rotto causa lenta convergenza?"""
    print("\n🐌 TESTING: Convergence Speed with Broken Mapping")
    print("=" * 50)
    
    # Setup: 3 modelli con performance molto diverse
    true_performance = {
        'excellent': 0.95,   # Dovrebbe dominare
        'average': 0.50,     # Medio
        'poor': 0.05        # Dovrebbe essere evitato
    }
    
    bandit = ThompsonSamplingBandit(
        arm_names=list(true_performance.keys()),
        seed=42,
        auto_decay=None
    )
    
    # Simula 200 rounds
    selection_history = []
    baseline = 0.5
    
    for round_num in range(200):
        # Sample arm
        probs = bandit.posterior()
        selected_arm = list(true_performance.keys())[np.argmax(probs)]
        
        # Get reward
        true_reward = true_performance[selected_arm]
        noise = np.random.normal(0, 0.05)  # Small noise
        observed_reward = np.clip(true_reward + noise, 0, 1)
        
        # Update bandit
        bandit.update_submitted(selected_arm)
        bandit.update(selected_arm, observed_reward, baseline)
        
        selection_history.append(selected_arm)
        
        # Check convergence at intervals
        if round_num in [49, 99, 149, 199]:  # After 50, 100, 150, 200
            recent_selections = selection_history[-50:]
            excellent_pct = recent_selections.count('excellent') / len(recent_selections)
            print(f"   Round {round_num+1:3d}: 'excellent' selected {excellent_pct:.1%} (should be >80%)")
    
    # Final analysis
    final_50 = selection_history[-50:]
    final_excellent_pct = final_50.count('excellent') / len(final_50)
    
    print(f"\n📊 CONVERGENCE ANALYSIS:")
    print(f"   Final 50 rounds: 'excellent' = {final_excellent_pct:.1%}")
    print(f"   Expected: >90% for clearly superior model")
    
    if final_excellent_pct < 0.7:
        print(f"   ❌ SLOW CONVERGENCE: Only {final_excellent_pct:.1%} vs expected 90%+")
        return True  # Problem detected
    else:
        print(f"   ✅ Reasonable convergence: {final_excellent_pct:.1%}")
        return False

def test_with_alternative_mapping():
    """Test: come si comporta con mapping diretto (senza sigmoid)"""
    print("\n🔧 TESTING: Alternative Direct Mapping")
    print("=" * 40)
    
    class FixedThompsonBandit(ThompsonSamplingBandit):
        """Version with direct mapping invece di sigmoid"""
        
        def _direct_mapping(self, reward_diff: float) -> float:
            """Direct mapping: reward-baseline directly to probability"""
            # Assumiamo reward in [0,1] e baseline=0.5
            # reward_diff in [-0.5, 0.5] 
            # Mappiamo a [0, 1] linearmente
            
            # Normalize to [0,1]
            prob = (reward_diff + 0.5)  # [-0.5,0.5] → [0,1]
            return np.clip(prob, 0.0, 1.0)
        
        def update(self, arm, reward, baseline=None):
            """Override con mapping diretto"""
            arm_idx = self._resolve_arm(arm)
            is_real = reward is not None
            
            # Same baseline logic
            if self._shift_by_parent and self._shift_by_baseline:
                baseline = (
                    self._baseline if baseline is None else max(baseline, self._baseline)
                )
            elif self._shift_by_baseline:
                baseline = self._baseline
            elif not self._shift_by_parent:
                baseline = 0.0
            if baseline is None:
                raise ValueError("baseline required when shifting is active")
                
            # Process reward
            if is_real:
                r_raw = float(reward)
            else:
                r_raw = baseline - 1.0
                
            r = r_raw - baseline
            
            # 🔧 FIXED: Direct mapping instead of sigmoid
            success_prob = self._direct_mapping(r)
            
            # Update Beta parameters
            if is_real or r < 0:
                self.alpha[arm_idx] += success_prob
                self.beta[arm_idx] += (1.0 - success_prob)
                
            self.n_completed[arm_idx] += 1.0
            self._maybe_decay()
            
            return r, baseline
    
    # Test direct mapping
    print("Direct Mapping Results:")
    fixed_bandit = FixedThompsonBandit(arm_names=['test'], auto_decay=None)
    baseline = 0.5
    
    test_rewards = [0.95, 0.80, 0.50, 0.20, 0.05]
    
    for reward in test_rewards:
        r_diff = reward - baseline
        prob = fixed_bandit._direct_mapping(r_diff)
        print(f"   Reward {reward:.2f} → diff {r_diff:6.2f} → prob {prob:.4f}")
    
    print("\n✅ Direct mapping preserves full reward information!")
    return fixed_bandit

if __name__ == "__main__":
    print("🔍 CRITICAL ANALYSIS: Thompson Sampling Bandit Issues")
    print("=" * 70)
    
    # Test 1: Identify sigmoid mapping problem
    has_mapping_issue = test_reward_mapping_issue()
    
    # Test 2: Check if it causes slow convergence  
    has_convergence_issue = test_convergence_with_broken_mapping()
    
    # Test 3: Show fixed version
    fixed_bandit = test_with_alternative_mapping()
    
    print("\n" + "="*70)
    print("🎯 SUMMARY OF CRITICAL FINDINGS:")
    
    if has_mapping_issue:
        print("❌ ISSUE #1 CONFIRMED: Sigmoid mapping compresses reward information")
    else:
        print("✅ Reward mapping seems reasonable")
        
    if has_convergence_issue:
        print("❌ ISSUE #2 CONFIRMED: Slow convergence due to mapping problem")
    else:
        print("✅ Convergence speed acceptable")
    
    if has_mapping_issue or has_convergence_issue:
        print("\n🚨 IMMEDIATE ACTION REQUIRED:")
        print("   1. Replace sigmoid mapping with direct/adaptive mapping")
        print("   2. Re-test convergence after fix")
        print("   3. Update documentation")
    else:
        print("\n✅ No critical issues found")
        
    print("\n🔧 NEXT STEPS:")
    print("   1. Implement FixedThompsonBandit.update() as the new version")
    print("   2. Add adaptive mapping for unknown reward ranges")
    print("   3. Comprehensive testing on real ShinkaEvolve tasks")