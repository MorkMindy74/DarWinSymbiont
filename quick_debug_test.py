#!/usr/bin/env python3
"""
Quick debug test per identificare il problema del database.
"""

import sys
import os
sys.path.insert(0, '/app')

from shinka.llm import CachedLLM

def debug_database_creation():
    """Debug della creazione del database."""
    print("🐛 DEBUG: Database Creation")
    print("-" * 40)
    
    config = {
        "enabled": True,
        "mode": "exact",
        "path": "./.cache/debug_test.db",
        "ttl_hours": 24.0
    }
    
    print(f"Config: {config}")
    
    # Rimuovi il file se esiste
    if os.path.exists(config["path"]):
        os.remove(config["path"])
        print("✅ Existing database removed")
    
    # Crea directory se necessaria
    os.makedirs(os.path.dirname(config["path"]), exist_ok=True)
    
    try:
        print("Creating CachedLLM...")
        llm = CachedLLM(config=config)
        print("✅ CachedLLM created successfully")
        
        print("Setting model function...")
        llm.set_model(lambda prompt: f"Response: {prompt}")
        print("✅ Model function set")
        
        print("Testing first query...")
        result1 = llm("Test query")
        print(f"✅ First query result: {result1}")
        
        print("Testing second query (same)...")
        result2 = llm("Test query")
        print(f"✅ Second query result: {result2}")
        
        print("Getting stats...")
        stats = llm.get_stats()
        print(f"✅ Stats: {stats}")
        
        print("✅ All operations successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_database_creation()
    print(f"\n🎯 Debug result: {'SUCCESS' if success else 'FAILED'}")