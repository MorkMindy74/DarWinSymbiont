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

# Cache determinism test
bench_cache:
	@echo "Testing cache determinism..."
	@echo "Running with cache OFF (2 runs should be identical):"
	python -m bench.context_bandit_bench --benchmark toy --algo context --seed 42 --budget_steps 200 --model mock --cache off
	python -m bench.context_bandit_bench --benchmark toy --algo context --seed 42 --budget_steps 200 --model mock --cache off
	@echo "Cache determinism test complete."

# Deduplication effectiveness test
bench_dedup:
	@echo "Testing deduplication effectiveness..."
	@echo "Running WITHOUT deduplication:"
	python -m bench.context_bandit_bench --benchmark toy --algo context --seed 42 --budget_steps 300 --model mock --dedup off
	@echo "Running WITH MinHash deduplication:"
	python -m bench.context_bandit_bench --benchmark toy --algo context --seed 42 --budget_steps 300 --model mock --dedup on --dedup-method minhash --dedup-threshold 0.8
	@echo "Running WITH SimHash deduplication:"
	python -m bench.context_bandit_bench --benchmark toy --algo context --seed 42 --budget_steps 300 --model mock --dedup on --dedup-method simhash --dedup-threshold 0.8
	@echo "Deduplication test complete."

# Complete extended suite with all features
bench_full_extended:
	@echo "Running FULL extended benchmark suite..."
	@echo "This will take ~15-20 minutes..."
	@make bench_context
	@make bench_ablation
	@make bench_hyperparam
	@make bench_cache
	@make bench_dedup
	@echo "Generating comprehensive report..."
	python -m bench.context_bandit_bench --make-report
	@echo "🎯 FULL EXTENDED BENCHMARK COMPLETE!"

# Archive system tests
test_archive:
	pytest -q tests/test_archive.py

# Archive sanity check
archive_sanity:
	@echo "Running archive sanity check..."
	@echo "1. Creating mock evolution config and running quick benchmark..."
	python -m bench.context_bandit_bench --benchmark toy --algo context --seed 42 --budget_steps 100 --model mock
	@echo "2. Testing archive CLI commands..."
	/app/shinka_archive list || echo "No agents yet (expected)"
	@echo "Archive sanity check complete."

# Test + Quick benchmark
test_quick: test_context bench_quick

# Complete validation pipeline
validate_all: test_context bench_full_extended

# Help
help:
	@echo "Available targets:"
	@echo "  test_context      - Run context-aware bandit unit tests"
	@echo "  test_archive      - Run agent archive system tests"
	@echo "  archive_sanity    - Archive system sanity check"
	@echo "  bench_quick       - Quick benchmark (CI-friendly, 300 steps)"
	@echo "  bench_context     - Complete benchmark matrix (5 algos × 5 seeds × 3 benchmarks)"
	@echo "  bench_ablation    - Feature ablation study"
	@echo "  bench_hyperparam  - Hyperparameter sensitivity analysis"
	@echo "  bench_cache       - Cache determinism test"
	@echo "  bench_dedup       - Deduplication effectiveness test"
	@echo "  bench_full_extended - Complete extended suite (15-20 min)"
	@echo "  validate_all      - Full validation pipeline"
	@echo "  clean_reports     - Clean benchmark reports"
	@echo "  test_quick        - Run tests + quick benchmark"