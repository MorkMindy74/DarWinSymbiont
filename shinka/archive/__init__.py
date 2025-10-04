"""
Agent Archive System for ShinkaEvolve

Provides reproducible agent archiving with DGM-compatible metadata,
export/import functionality, and reproduction capabilities.
"""

from .agent_archive import (
    AgentArchive,
    AgentManifest,
    create_agent_archive,
    list_archived_agents,
    show_agent_manifest,
    export_agent,
    import_agent,
    reproduce_agent
)

__all__ = [
    'AgentArchive',
    'AgentManifest', 
    'create_agent_archive',
    'list_archived_agents',
    'show_agent_manifest',
    'export_agent',
    'import_agent',
    'reproduce_agent'
]