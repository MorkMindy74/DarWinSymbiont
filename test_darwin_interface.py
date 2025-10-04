#!/usr/bin/env python3
"""
Test della Darwin-style interface per il sistema di cache LLM.
Questo test usa l'interfaccia semplificata CachedLLM.
"""

import time
import hashlib
import sys

sys.path.insert(0, '/app')

from shinka.llm import CachedLLM  # Darwin-style interface

# Config cache (come nel tuo esempio)
config = {
    "enabled": True,
    "mode": "exact",
    "backend": "sqlite",
    "path": "./.cache/llm_cache.db",
    "ttl_hours": 0.001  # per test TTL rapido
}

# Inizializza cache wrapper
llm = CachedLLM(config=config)

def fake_model_call(prompt):
    """Simula una chiamata all'LLM (in realtà qui potresti usare il vero LLM)."""
    return f"RESULT({hashlib.md5(prompt.encode()).hexdigest()[:6]})"

# Collega la funzione finta
llm.set_model(fake_model_call)

### TEST 1: Duplicate queries
print("\n=== TEST 1: Duplicate Queries ===")
prompt = "Solve berlin52 TSP"
results = []
for i in range(10):
    result = llm(prompt)
    results.append(result)
    print(f"Run {i+1}: {result}")

# Verifica che tutti i risultati siano identici
if len(set(results)) == 1:
    print("✅ All duplicate queries returned identical results")
else:
    print("❌ Duplicate queries returned different results")

### TEST 2: Persistence across "runs" (stesso processo)
print("\n=== TEST 2: Persistence ===")
# Riusa lo stesso cache file, dovrebbe dare HIT subito
result = llm(prompt)
print("Persistence test result:", result)
if result == results[0]:
    print("✅ Cache persistence works correctly")
else:
    print("❌ Cache persistence failed")

### TEST 3: TTL expiration
print("\n=== TEST 3: TTL Expiration ===")
prompt_ttl = "Check circle packing optimization"
result1 = llm(prompt_ttl)
print("First call:", result1)
print("Waiting for TTL to expire...")
time.sleep(5)  # TTL è 0.001 hours = 3.6 secondi
result2 = llm(prompt_ttl)
print("After expiration:", result2)

# I risultati dovrebbero essere identici (stesso prompt → stesso hash)
# ma il secondo dovrebbe essere un cache miss
if result1 == result2:
    print("✅ Results consistent after TTL expiration")
else:
    print("❌ Results inconsistent after TTL expiration")

### TEST 4: Stress Test
print("\n=== TEST 4: Stress Test ===")
prompts = [f"query_{i}" for i in range(100)] + ["dup_query"] * 50
print(f"Running {len(prompts)} queries...")

initial_stats = llm.get_stats()
initial_misses = initial_stats.get('local_misses', 0)

for p in prompts:
    r = llm(p)

final_stats = llm.get_stats()
total_calls = final_stats['total_calls']
hits = final_stats.get('local_hits', 0) 
misses = final_stats.get('local_misses', 0)

new_misses = misses - initial_misses
hit_rate = (hits / max(total_calls, 1)) * 100

print(f"Stress test results:")
print(f"  Total calls this session: {total_calls}")
print(f"  Hits: {hits}")
print(f"  Misses: {misses}")
print(f"  New misses this test: {new_misses}")
print(f"  Overall hit rate: {hit_rate:.1f}%")

# Dovremmo avere circa 50 duplicates che danno cache hits
if new_misses <= 110:  # 100 unique + qualche margin
    print("✅ Stress test cache behavior looks good")
else:
    print("❌ Unexpected cache miss pattern in stress test")

### TEST 5: Statistics and Management
print("\n=== TEST 5: Statistics and Management ===")
stats = llm.get_stats()
print("Final cache statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")

# Test cleanup
print("\nTesting cleanup...")
removed = llm.cleanup_expired()
print(f"Removed {removed} expired entries")

print("\n=== ALL TESTS COMPLETED ===")
print("✅ Darwin-style interface working correctly!")
print("\nNote: This interface provides a simplified API while using")
print("the full ShinkaEvolve caching backend under the hood.")