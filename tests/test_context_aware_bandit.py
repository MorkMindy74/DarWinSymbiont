"""
Unit tests for Context-Aware Thompson Sampling Bandit.

Tests cover:
1. Context detection logic
2. Posterior management across contexts  
3. Selection consistency
4. Integration with EvolutionConfig
5. Performance improvements
"""

import unittest
import numpy as np
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, '/app')

from shinka.llm.dynamic_sampling import ContextAwareThompsonSamplingBandit
from shinka.core.runner import EvolutionConfig

class TestContextDetection(unittest.TestCase):
    """Test context detection functionality."""
    
    def setUp(self):
        """Set up test bandit."""
        self.bandit = ContextAwareThompsonSamplingBandit(
            arm_names=["gpt-4", "claude-3", "gemini-pro"],
            contexts=["early", "mid", "late", "stuck"],
            seed=42
        )
    
    def test_early_context_detection(self):
        """Test early phase detection."""
        # Early: low generation progress, recent improvements
        context = self.bandit._detect_context(
            generation=5,
            total_generations=100,
            no_improve_steps=2,
            best_fitness_history=[0.1, 0.2, 0.3, 0.4, 0.45],
            population_diversity=0.8
        )
        self.assertEqual(context, "early")
    
    def test_mid_context_detection(self):
        """Test mid phase detection."""
        # Add samples first
        for _ in range(10):
            self.bandit.update_submitted("gpt-4")
            self.bandit.update("gpt-4", reward=0.7, baseline=0.5)
        
        # Mid: ~50% progress, steady improvement
        context = self.bandit.update_context(
            generation=50,
            total_generations=100,
            no_improve_steps=3,
            best_fitness_history=[0.1, 0.3, 0.5, 0.65, 0.7],
            population_diversity=0.6
        )
        self.assertEqual(context, "mid")
    
    def test_late_context_detection(self):
        """Test late phase detection."""
        # Late: high progress, some improvement
        context = self.bandit._detect_context(
            generation=85,
            total_generations=100,
            no_improve_steps=5,
            best_fitness_history=[0.5, 0.8, 0.85, 0.87, 0.88],
            population_diversity=0.3
        )
        self.assertEqual(context, "late")
    
    def test_stuck_context_detection(self):
        """Test stuck phase detection."""
        # Stuck: no improvement for extended period
        context = self.bandit._detect_context(
            generation=60,
            total_generations=100,
            no_improve_steps=15,
            best_fitness_history=[0.7, 0.7, 0.69, 0.68, 0.67],
            population_diversity=0.2
        )
        self.assertEqual(context, "stuck")
    
    def test_context_switching_threshold(self):
        """Test context switching with threshold."""
        # Start in early context with some samples
        self.bandit.current_context = "early"
        
        # Add minimum samples to allow switching
        for _ in range(10):  # Above min_context_samples
            self.bandit.update_submitted("gpt-4")
            self.bandit.update("gpt-4", reward=0.7, baseline=0.5)
        
        # Strong change should trigger switch to stuck
        context = self.bandit.update_context(
            generation=50,
            total_generations=100,
            no_improve_steps=20,  # Clearly stuck
            best_fitness_history=[0.5, 0.5, 0.5, 0.5, 0.5],
            population_diversity=0.2
        )
        # Should switch to stuck due to strong signal
        self.assertEqual(context, "stuck")
    
    def test_fitness_slope_calculation(self):
        """Test fitness slope calculation."""
        # Improving fitness
        slope = self.bandit._calculate_fitness_slope([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertGreater(slope, 0)
        
        # Declining fitness
        slope = self.bandit._calculate_fitness_slope([0.5, 0.4, 0.3, 0.2, 0.1])
        self.assertLess(slope, 0)
        
        # Stable fitness
        slope = self.bandit._calculate_fitness_slope([0.5, 0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(slope, 0, places=5)


class TestPosteriorManagement(unittest.TestCase):
    """Test posterior management across contexts."""
    
    def setUp(self):
        """Set up test bandit."""
        self.bandit = ContextAwareThompsonSamplingBandit(
            arm_names=["model_a", "model_b", "model_c"],
            contexts=["early", "mid", "late"],
            prior_alpha=1.0,
            prior_beta=1.0,
            seed=42
        )
    
    def test_separate_posteriors_initialization(self):
        """Test that separate posteriors are initialized for each context."""
        for context in self.bandit.contexts:
            self.assertIn(context, self.bandit.context_alpha)
            self.assertIn(context, self.bandit.context_beta)
            
            # Check initial values
            np.testing.assert_array_equal(
                self.bandit.context_alpha[context],
                np.full(3, self.bandit.prior_alpha)
            )
            np.testing.assert_array_equal(
                self.bandit.context_beta[context],
                np.full(3, self.bandit.prior_beta)
            )
    
    def test_context_specific_updates(self):
        """Test that updates only affect the current context."""
        # Set context to "early"
        self.bandit.current_context = "early"
        
        # Store initial values for other contexts
        mid_alpha_before = self.bandit.context_alpha["mid"].copy()
        late_alpha_before = self.bandit.context_alpha["late"].copy()
        
        # Update in early context
        self.bandit.update_submitted("model_a")
        self.bandit.update("model_a", reward=0.8, baseline=0.5)
        
        # Check that only early context was updated
        self.assertGreater(
            self.bandit.context_alpha["early"][0],
            self.bandit.prior_alpha
        )
        
        # Check that other contexts remain unchanged
        np.testing.assert_array_equal(
            self.bandit.context_alpha["mid"],
            mid_alpha_before
        )
        np.testing.assert_array_equal(
            self.bandit.context_alpha["late"],
            late_alpha_before
        )
    
    def test_context_specific_sampling(self):
        """Test that sampling uses context-specific posteriors."""
        # Update different arms in different contexts
        self.bandit.current_context = "early"
        for _ in range(10):
            self.bandit.update_submitted("model_a")
            self.bandit.update("model_a", reward=0.9, baseline=0.5)
        
        self.bandit.current_context = "mid"  
        for _ in range(10):
            self.bandit.update_submitted("model_b")
            self.bandit.update("model_b", reward=0.9, baseline=0.5)
        
        # Sample from each context
        early_probs = self.bandit.posterior(context="early", samples=1000)
        mid_probs = self.bandit.posterior(context="mid", samples=1000)
        
        # Early should prefer model_a, mid should prefer model_b
        self.assertGreater(early_probs[0], 0.7)  # model_a
        self.assertGreater(mid_probs[1], 0.7)   # model_b
    
    def test_context_switching_updates_correctly(self):
        """Test context switching and update behavior."""
        # Start in early, update model_a
        self.bandit.current_context = "early"
        self.bandit.update_submitted("model_a")
        self.bandit.update("model_a", reward=0.8, baseline=0.5)
        
        # Switch to mid context  
        new_context = self.bandit.update_context(
            generation=50,
            total_generations=100,
            no_improve_steps=5,
            best_fitness_history=[0.1, 0.5, 0.7, 0.75, 0.76]
        )
        self.assertEqual(new_context, "mid")
        
        # Update model_b in mid context
        self.bandit.update_submitted("model_b")
        self.bandit.update("model_b", reward=0.9, baseline=0.5)
        
        # Check that both contexts have been updated appropriately
        self.assertGreater(self.bandit.context_alpha["early"][0], self.bandit.prior_alpha)
        self.assertGreater(self.bandit.context_beta["mid"][1], self.bandit.prior_beta)


class TestSelectionConsistency(unittest.TestCase):
    """Test selection consistency and performance."""
    
    def setUp(self):
        """Set up test environment."""
        self.bandit = ContextAwareThompsonSamplingBandit(
            arm_names=["fast_model", "accurate_model", "balanced_model"],
            contexts=["early", "mid", "late", "stuck"],
            seed=42
        )
    
    def test_deterministic_selection_with_seed(self):
        """Test that selection is deterministic with fixed seed."""
        # Set context and state
        self.bandit.update_context(
            generation=25,
            total_generations=100,
            no_improve_steps=3,
            best_fitness_history=[0.1, 0.3, 0.5]
        )
        
        # Multiple selections should be identical with same state
        selection1 = self.bandit.posterior()
        selection2 = self.bandit.posterior()
        
        np.testing.assert_array_equal(selection1, selection2)
    
    def test_adaptation_to_context_preferences(self):
        """Test that bandit learns context-specific preferences."""
        # Simulate early context: fast_model performs best
        self.bandit.current_context = "early"
        for _ in range(20):
            self.bandit.update_submitted("fast_model")
            self.bandit.update("fast_model", reward=0.8, baseline=0.5)
            self.bandit.update_submitted("accurate_model")  
            self.bandit.update("accurate_model", reward=0.4, baseline=0.5)
        
        # Simulate stuck context: accurate_model performs best
        self.bandit.current_context = "stuck"
        for _ in range(20):
            self.bandit.update_submitted("accurate_model")
            self.bandit.update("accurate_model", reward=0.9, baseline=0.5)
            self.bandit.update_submitted("fast_model")
            self.bandit.update("fast_model", reward=0.3, baseline=0.5)
        
        # Check context-specific preferences
        early_probs = self.bandit.posterior(context="early", samples=1000)
        stuck_probs = self.bandit.posterior(context="stuck", samples=1000)
        
        # Early should prefer fast_model, stuck should prefer accurate_model
        self.assertGreater(early_probs[0], stuck_probs[0])  # fast_model
        self.assertGreater(stuck_probs[1], early_probs[1])  # accurate_model
    
    def test_decay_across_all_contexts(self):
        """Test that decay is applied across all contexts."""
        # Update multiple contexts
        for context in ["early", "mid", "late"]:
            self.bandit.current_context = context
            self.bandit.update_submitted("fast_model")
            self.bandit.update("fast_model", reward=0.8, baseline=0.5)
        
        # Store pre-decay values
        pre_decay_alpha = {
            context: self.bandit.context_alpha[context].copy()
            for context in self.bandit.contexts
        }
        
        # Apply decay
        self.bandit.decay(0.9)
        
        # Check all contexts were decayed
        for context in self.bandit.contexts:
            for arm in range(self.bandit.n_arms):
                if pre_decay_alpha[context][arm] > self.bandit.prior_alpha:
                    self.assertLess(
                        self.bandit.context_alpha[context][arm],
                        pre_decay_alpha[context][arm]
                    )


class TestEvolutionConfigIntegration(unittest.TestCase):
    """Test integration with EvolutionConfig."""
    
    def test_thompson_context_config_creation(self):
        """Test creating EvolutionConfig with thompson_context."""
        config = EvolutionConfig(
            llm_models=["gpt-4", "claude-3", "gemini-pro"],
            llm_dynamic_selection="thompson_context",
            llm_dynamic_selection_kwargs={
                "contexts": ["early", "mid", "late", "stuck"],
                "features": ["gen_progress", "no_improve", "fitness_slope", "pop_diversity"],
                "prior_alpha": 2.0,
                "prior_beta": 1.0,
                "auto_decay": 0.99
            }
        )
        
        self.assertEqual(config.llm_dynamic_selection, "thompson_context")
        self.assertIn("contexts", config.llm_dynamic_selection_kwargs)
        self.assertIn("prior_alpha", config.llm_dynamic_selection_kwargs)
    
    def test_backward_compatibility(self):
        """Test that regular thompson still works."""
        config = EvolutionConfig(
            llm_models=["gpt-4", "claude-3"],
            llm_dynamic_selection="thompson",
            llm_dynamic_selection_kwargs={
                "prior_alpha": 1.5,
                "prior_beta": 1.0
            }
        )
        
        self.assertEqual(config.llm_dynamic_selection, "thompson")
        # Should not have context-specific kwargs
        self.assertNotIn("contexts", config.llm_dynamic_selection_kwargs)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance-related functionality."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.context_bandit = ContextAwareThompsonSamplingBandit(
            arm_names=["model_1", "model_2", "model_3"],
            contexts=["early", "mid", "late", "stuck"],
            seed=42
        )
        
        from shinka.llm.dynamic_sampling import ThompsonSamplingBandit
        self.baseline_bandit = ThompsonSamplingBandit(
            arm_names=["model_1", "model_2", "model_3"],
            seed=42
        )
    
    def test_context_statistics_tracking(self):
        """Test that context statistics are tracked correctly."""
        # Simulate some activity
        self.context_bandit.current_context = "early"
        for _ in range(5):
            self.context_bandit.update_submitted("model_1")
            self.context_bandit.update("model_1", reward=0.8, baseline=0.5)
        
        self.context_bandit.current_context = "mid"
        for _ in range(3):
            self.context_bandit.update_submitted("model_2") 
            self.context_bandit.update("model_2", reward=0.7, baseline=0.5)
        
        # Check statistics
        stats = self.context_bandit.get_context_stats()
        
        self.assertEqual(stats["context_switch_count"], 0)  # No automatic switches
        self.assertEqual(stats["contexts"]["early"]["selections"], 5)
        self.assertEqual(stats["contexts"]["mid"]["selections"], 3)
        self.assertEqual(stats["contexts"]["late"]["selections"], 0)
    
    def test_context_aware_summary_output(self):
        """Test that summary includes context information."""
        # Set some context state
        self.context_bandit.update_context(
            generation=30,
            total_generations=100,
            no_improve_steps=5,
            best_fitness_history=[0.1, 0.4, 0.6, 0.65]
        )
        
        # Update some arms
        self.context_bandit.update_submitted("model_1")
        self.context_bandit.update("model_1", reward=0.8, baseline=0.5)
        
        # This should not raise an exception and should include context info
        try:
            self.context_bandit.print_summary()
            summary_works = True
        except Exception:
            summary_works = False
        
        self.assertTrue(summary_works)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)