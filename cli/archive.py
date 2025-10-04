#!/usr/bin/env python3
"""
CLI commands for ShinkaEvolve Agent Archive system.

Commands:
- shinka_archive list
- shinka_archive show <id>  
- shinka_archive export <id>
- shinka_archive import <zip>
- shinka_repro <id>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add shinka to path
sys.path.insert(0, '/app')

from shinka.archive import (
    list_archived_agents,
    show_agent_manifest, 
    export_agent,
    import_agent,
    reproduce_agent
)


def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp for display."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp[:19] if len(timestamp) > 19 else timestamp


def cmd_list(args) -> int:
    """List all archived agents."""
    try:
        agents = list_archived_agents()
        
        if not agents:
            print("No agents found in archive.")
            return 0
        
        # Print header
        print(f"{'ID':<12} {'Timestamp':<20} {'Fitness':<10} {'Parent':<12} {'Branch':<15} {'Dirty':<5}")
        print("-" * 80)
        
        # Print agents
        for agent in agents:
            agent_id = agent['id'][:10] + '..' if len(agent['id']) > 12 else agent['id']
            timestamp = format_timestamp(agent['timestamp'])
            fitness = f"{agent.get('best_fitness', 0):.3f}" if agent.get('best_fitness') else "N/A"
            parent_id_raw = agent.get('parent_id')
            if parent_id_raw is None:
                parent_id = 'None'
            elif len(parent_id_raw) > 12:
                parent_id = parent_id_raw[:10] + '..'
            else:
                parent_id = parent_id_raw
            branch = agent.get('branch', 'unknown')[:14]
            dirty = "Yes" if agent.get('dirty', True) else "No"
            
            print(f"{agent_id:<12} {timestamp:<20} {fitness:<10} {parent_id:<12} {branch:<15} {dirty:<5}")
        
        print(f"\nTotal: {len(agents)} agents")
        return 0
        
    except Exception as e:
        print(f"Error listing agents: {e}")
        return 1


def cmd_show(args) -> int:
    """Show manifest for specific agent."""
    try:
        manifest_data = show_agent_manifest(args.agent_id)
        
        if not manifest_data:
            print(f"Agent {args.agent_id} not found.")
            return 1
        
        # Print formatted manifest
        if args.format == 'json':
            print(json.dumps(manifest_data, indent=2))
        else:
            # Print key information in readable format
            print(f"Agent ID: {manifest_data['agent_id']}")
            print(f"Timestamp: {format_timestamp(manifest_data['timestamp'])}")
            print(f"Parent ID: {manifest_data.get('parent_id', 'None')}")
            print(f"Git Commit: {manifest_data['git_commit']}")
            print(f"Branch: {manifest_data['branch']}")
            print(f"Dirty: {'Yes' if manifest_data['dirty'] else 'No'}")
            print()
            
            print("Environment:")
            env = manifest_data['env']
            print(f"  Python: {env.get('python', 'unknown')}")
            print(f"  OS: {env.get('os', 'unknown')}")
            print(f"  Dependencies: {env.get('dependencies_lock', 'unknown')}")
            print()
            
            print("Seeds:")
            seeds = manifest_data['seeds']
            for key, value in seeds.items():
                print(f"  {key}: {value}")
            print()
            
            print("Hyperparameters:")
            hyperparams = manifest_data['hyperparams']
            for key, value in hyperparams.items():
                print(f"  {key}: {value}")
            print()
            
            print("Benchmarks:")
            benchmarks = manifest_data['benchmarks']
            for benchmark_name, metrics in benchmarks.items():
                print(f"  {benchmark_name}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        print(f"    {metric_name}: {value:.4f}")
                    else:
                        print(f"    {metric_name}: {value}")
            print()
            
            print("Context Activity:")
            context_activity = manifest_data['context_activity']
            print(f"  Switches: {context_activity.get('switch_count', 0)}")
            print("  Dwell Times:")
            dwell_times = context_activity.get('dwell_time', {})
            for context, ratio in dwell_times.items():
                print(f"    {context}: {ratio:.2%}")
            print()
            
            print("Cost:")
            cost = manifest_data['cost']
            print(f"  Estimated USD: ${cost.get('total_usd_est', 0):.4f}")
            print(f"  Total Queries: {cost.get('queries_total', 0)}")
        
        return 0
        
    except Exception as e:
        print(f"Error showing agent: {e}")
        return 1


def cmd_export(args) -> int:
    """Export agent to ZIP file."""
    try:
        output_path = args.output or f"agent_{args.agent_id}.zip"
        
        success = export_agent(args.agent_id, output_path)
        
        if success:
            print(f"Agent {args.agent_id} exported to {output_path}")
            return 0
        else:
            print(f"Failed to export agent {args.agent_id}")
            return 1
            
    except Exception as e:
        print(f"Error exporting agent: {e}")
        return 1


def cmd_import(args) -> int:
    """Import agent from ZIP file."""
    try:
        zip_path = args.zip_path
        
        if not Path(zip_path).exists():
            print(f"ZIP file not found: {zip_path}")
            return 1
        
        agent_id = import_agent(zip_path)
        
        if agent_id:
            print(f"Agent {agent_id} imported successfully")
            return 0
        else:
            print(f"Failed to import agent from {zip_path}")
            return 1
            
    except Exception as e:
        print(f"Error importing agent: {e}")
        return 1


def cmd_reproduce(args) -> int:
    """Reproduce agent run and verify metrics."""
    try:
        print(f"Reproducing agent {args.agent_id}...")
        print(f"Tolerance: ±{args.tolerance}%")
        print()
        
        result = reproduce_agent(args.agent_id, args.tolerance)
        
        if not result['success']:
            print(f"❌ Reproduction FAILED: {result.get('error', 'Unknown error')}")
            return 1
        
        print("📊 Reproduction Results:")
        print()
        
        # Show verification results
        verification_results = result.get('verification_results', {})
        overall_passed = True
        
        for benchmark_name, verification in verification_results.items():
            status = "✅ PASS" if verification['passed'] else "❌ FAIL"
            print(f"{benchmark_name}: {status}")
            
            if 'checks' in verification:
                for metric_name, check in verification['checks'].items():
                    original = check['original']
                    reproduction = check['reproduction']
                    diff_pct = check['diff_pct']
                    check_status = "✅" if check['passed'] else "❌"
                    
                    print(f"  {check_status} {metric_name}:")
                    print(f"    Original:     {original:.4f}")
                    print(f"    Reproduction: {reproduction:.4f}")
                    print(f"    Difference:   {diff_pct:.2f}%")
            
            if not verification['passed']:
                overall_passed = False
            print()
        
        # Final result
        if overall_passed:
            print("🎯 REPRODUCTION SUCCESS: All metrics within tolerance")
            return 0
        else:
            print("💥 REPRODUCTION FAILED: Some metrics outside tolerance")
            return 1
            
    except Exception as e:
        print(f"Error reproducing agent: {e}")
        return 1


def main_archive():
    """Main entry point for shinka_archive command."""
    parser = argparse.ArgumentParser(description="ShinkaEvolve Agent Archive CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all archived agents')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show agent manifest')
    show_parser.add_argument('agent_id', help='Agent ID to show')
    show_parser.add_argument('--format', choices=['text', 'json'], default='text',
                           help='Output format')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export agent to ZIP')
    export_parser.add_argument('agent_id', help='Agent ID to export')
    export_parser.add_argument('-o', '--output', help='Output ZIP file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import agent from ZIP')
    import_parser.add_argument('zip_path', help='Path to agent ZIP file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'list':
        return cmd_list(args)
    elif args.command == 'show':
        return cmd_show(args)
    elif args.command == 'export':
        return cmd_export(args)
    elif args.command == 'import':
        return cmd_import(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


def main_repro():
    """Main entry point for shinka_repro command."""
    parser = argparse.ArgumentParser(description="ShinkaEvolve Agent Reproduction")
    parser.add_argument('agent_id', help='Agent ID to reproduce')
    parser.add_argument('--tolerance', type=float, default=1.0,
                       help='Tolerance percentage for metric verification (default: 1.0)')
    
    args = parser.parse_args()
    return cmd_reproduce(args)


if __name__ == '__main__':
    import sys
    if 'repro' in sys.argv[0]:
        sys.exit(main_repro())
    else:
        sys.exit(main_archive())