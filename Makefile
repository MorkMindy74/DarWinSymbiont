# ShinkaEvolve Benchmark Makefile

.PHONY: bench_quick bench_context test_context clean_reports

# Quick benchmark for CI (1 seed, 300 steps, mock model)
bench_quick:
	@echo "Running quick benchmark..."
	python -m bench.context_bandit_bench --benchmark toy --algo baseline --seed 42 --budget_steps 300 --model mock
	python -m bench.context_bandit_bench --benchmark toy --algo context --seed 42 --budget_steps 300 --model mock
	python -m bench.context_bandit_bench --benchmark tsp --algo baseline --seed 42 --budget_steps 300 --model mock
	python -m bench.context_bandit_bench --benchmark tsp --algo context --seed 42 --budget_steps 300 --model mock
	@echo "Quick benchmark complete."

# Complete extended benchmark matrix (5 seeds, 1500 steps, all algorithms)
bench_context:
	@echo "Running complete extended benchmark matrix..."
	@for seed in 42 43 44 45 46; do \
		for benchmark in toy tsp synthetic; do \
			for algo in baseline decay context ucb epsilon; do \
				echo "Running $$algo on $$benchmark with seed $$seed..."; \
				python -m bench.context_bandit_bench --benchmark $$benchmark --algo $$algo --seed $$seed --budget_steps 1500 --model mock; \
			done; \
		done; \
	done
	@echo "Generating report..."
	python -m bench.context_bandit_bench --make-report
	@echo "Complete extended benchmark matrix finished."

# Ablation study
bench_ablation:
	@echo "Running ablation study..."
	@for seed in 42 43 44; do \
		for benchmark in toy tsp synthetic; do \
			for ablation in no_gen_progress no_no_improve no_fitness_slope no_pop_diversity 3contexts; do \
				echo "Running context ($$ablation) on $$benchmark with seed $$seed..."; \
				python -m bench.context_bandit_bench --benchmark $$benchmark --algo context --seed $$seed --budget_steps 1000 --model mock --ablation $$ablation; \
			done; \
		done; \
	done
	@echo "Ablation study complete."

# Hyperparameter sensitivity
bench_hyperparam:
	@echo "Running hyperparameter sensitivity..."
	@for alpha in 1.0 2.0 3.0; do \
		for beta in 1.0 2.0; do \
			for decay in 0.97 0.99 0.995; do \
				hyperparams="$$alpha,$$beta,$$decay"; \
				echo "Testing hyperparams: $$hyperparams"; \
				python -m bench.context_bandit_bench --benchmark tsp --algo context --seed 42 --budget_steps 1000 --model mock --hyperparams $$hyperparams; \
			done; \
		done; \
	done
	@echo "Hyperparameter sensitivity complete."

# Run context-aware bandit tests
test_context:
	pytest -q tests/test_context_aware_bandit.py

# Clean benchmark reports
clean_reports:
	rm -rf reports/context_bandit/*

# Test + Quick benchmark
test_quick: test_context bench_quick

# Help
help:
	@echo "Available targets:"
	@echo "  test_context    - Run context-aware bandit unit tests"
	@echo "  bench_quick     - Quick benchmark (CI-friendly)"
	@echo "  bench_context   - Complete benchmark matrix"
	@echo "  clean_reports   - Clean benchmark reports"
	@echo "  test_quick      - Run tests + quick benchmark"