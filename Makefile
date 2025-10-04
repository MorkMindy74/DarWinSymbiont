# ShinkaEvolve Benchmark Makefile

.PHONY: bench_quick bench_context test_context clean_reports

# Quick benchmark for CI (1 seed, 200 steps, mock model)
bench_quick:
	@echo "Running quick benchmark..."
	python -m bench.context_bandit_bench --benchmark toy --algo both --seed 42 --budget_steps 200 --model mock
	python -m bench.context_bandit_bench --benchmark tsp --algo both --seed 42 --budget_steps 200 --model mock
	python -m bench.context_bandit_bench --benchmark synthetic --algo both --seed 42 --budget_steps 200 --model mock
	@echo "Quick benchmark complete."

# Complete benchmark matrix (3 seeds, 1000 steps, all benchmarks)
bench_context:
	@echo "Running complete benchmark matrix..."
	@for seed in 42 43 44; do \
		for benchmark in toy tsp synthetic; do \
			echo "Running $$benchmark with seed $$seed..."; \
			python -m bench.context_bandit_bench --benchmark $$benchmark --algo both --seed $$seed --budget_steps 1000 --model mock; \
		done; \
	done
	@echo "Generating report..."
	python -m bench.context_bandit_bench --make-report
	@echo "Complete benchmark matrix finished."

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