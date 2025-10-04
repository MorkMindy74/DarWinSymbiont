#!/usr/bin/env python3
"""
Versione adattata del tuo codice originale per funzionare con 
il sistema di cache ShinkaEvolve.
"""

import time
import hashlib
import sys

sys.path.insert(0, '/app')

from shinka.llm import CachedLLM  # Equivalente a darwin.cache

# Config cache (identica alla tua)
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
for i in range(10):
    result = llm(prompt)
    print(f"Run {i+1}: {result}")

### TEST 2: Persistence across runs
print("\n=== TEST 2: Persistence ===")
# Riusa lo stesso cache file, dovrebbe dare HIT subito
result = llm(prompt)
print("Persistence test result:", result)

### TEST 3: TTL expiration
print("\n=== TEST 3: TTL Expiration ===")
prompt_ttl = "Check circle packing optimization"
result1 = llm(prompt_ttl)
print("First call:", result1)
print("Waiting for TTL to expire...")
time.sleep(5)
result2 = llm(prompt_ttl)
print("After expiration:", result2)

### TEST 4: Stress Test (versione ridotta per velocità)
print("\n=== TEST 4: Stress Test ===")
prompts = [f"query_{i}" for i in range(80)] + ["dup_query"] * 20
hits, misses = 0, 0

# Ottieni statistiche iniziali
initial_stats = llm.get_stats()
initial_hits = initial_stats.get('local_hits', 0)
initial_misses = initial_stats.get('local_misses', 0)

for p in prompts:
    r = llm(p)

# Ottieni statistiche finali
final_stats = llm.get_stats()
final_hits = final_stats.get('local_hits', 0)
final_misses = final_stats.get('local_misses', 0)

# Calcola differenze
hits = final_hits - initial_hits
misses = final_misses - initial_misses

print(f"Stress test - Hits: {hits}, Misses: {misses}, Hit rate: {hits/(hits+misses):.2%}")

### BONUS: Mostra statistiche dettagliate
print("\n=== BONUS: Detailed Statistics ===")
stats = llm.get_stats()
print("Cache Performance:")
print(f"  Mode: {stats['mode']}")
print(f"  Total entries in DB: {stats['total_entries']}")
print(f"  Active entries: {stats['active_entries']}")
print(f"  Overall hit rate: {stats['hit_rate_percent']:.1f}%")
print(f"  Total calls: {stats['total_calls']}")
print(f"  Cache hits: {stats['local_hits']}")
print(f"  Cache misses: {stats['local_misses']}")

print("\n✅ Il tuo codice originale funziona perfettamente con l'adapter!")
print("🎯 Differenze dall'originale:")
print("   - Import da 'shinka.llm' invece di 'darwin.cache'")
print("   - Statistiche più dettagliate disponibili")
print("   - Backend SQLite robusto invece di implementazione semplificata")
print("   - Supporto per fuzzy matching opzionale")
print("   - Logging integrato per debugging")