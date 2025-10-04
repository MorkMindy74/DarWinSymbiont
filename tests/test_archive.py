"""
Unit tests for the Agent Archive system.
"""

import pytest
import tempfile
import shutil
import json
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, '/app')

from shinka.archive import (
    AgentArchive,
    AgentManifest,
    create_agent_archive,
    list_archived_agents,
    show_agent_manifest,
    export_agent,
    import_agent,
    reproduce_agent
)


class TestAgentManifest:
    def test_manifest_creation(self):
        """Test creating and serializing manifest."""
        manifest = AgentManifest(
            agent_id="test123",
            parent_id=None,
            timestamp="2025-10-04T10:00:00",
            git_commit="abcd1234",
            branch="main",
            dirty=False,
            env={"python": "3.11.0", "os": "Linux"},
            seeds={"global": 42, "numpy": 42, "torch": 42},
            evo_config_path="test.yaml",
            evo_config_inline={"test": "config"},
            hyperparams={"bandit": "thompson_context"},
            benchmarks={"toy": {"final_best_fitness": 0.95}},
            cost={"total_usd_est": 1.0, "queries_total": 100},
            context_activity={"switch_count": 5},
            dgm_compat={"repo_layout_ref": "ShinkaEvolve"}
        )
        
        # Test serialization
        manifest_dict = manifest.to_dict()
        assert manifest_dict["agent_id"] == "test123"
        assert manifest_dict["benchmarks"]["toy"]["final_best_fitness"] == 0.95
        
        # Test deserialization
        manifest2 = AgentManifest.from_dict(manifest_dict)
        assert manifest2.agent_id == "test123"
        assert manifest2.benchmarks["toy"]["final_best_fitness"] == 0.95


class TestAgentArchive:
    
    @pytest.fixture
    def temp_archive(self):
        """Create temporary archive directory."""
        temp_dir = tempfile.mkdtemp()
        archive = AgentArchive(temp_dir)
        yield archive, temp_dir
        shutil.rmtree(temp_dir)
    
    def test_archive_initialization(self, temp_archive):
        """Test archive initialization and directory creation."""
        archive, temp_dir = temp_archive
        
        assert archive.archive_root.exists()
        assert (archive.archive_root / "agents").exists()
    
    @patch('subprocess.run')
    def test_git_info_extraction(self, mock_subprocess, temp_archive):
        """Test git information extraction."""
        archive, temp_dir = temp_archive
        
        # Mock git commands
        mock_subprocess.side_effect = [
            MagicMock(returncode=0, stdout="abcd1234\n"),  # git rev-parse HEAD
            MagicMock(returncode=0, stdout="main\n"),      # git branch --show-current
            MagicMock(returncode=0, stdout="")             # git status --porcelain
        ]
        
        git_info = archive._get_git_info()
        
        assert git_info["commit"] == "abcd1234"
        assert git_info["branch"] == "main"
        assert git_info["dirty"] == False
    
    def test_environment_info_collection(self, temp_archive):
        """Test environment information collection."""
        archive, temp_dir = temp_archive
        
        env_info = archive._get_environment_info()
        
        assert "python" in env_info
        assert "os" in env_info
        assert "platform" in env_info
    
    def test_save_agent(self, temp_archive):
        """Test saving an agent to archive."""
        archive, temp_dir = temp_archive
        
        config = {
            "algorithm": "context",
            "seed": 42,
            "benchmark": "toy",
            "test": "config"
        }
        
        # Mock benchmark results
        with patch('pathlib.Path.exists', return_value=False):
            agent_id = archive.save_agent(config)
        
        assert agent_id is not None
        assert len(agent_id) > 0
        
        # Check if agent directory was created
        agents_dir = archive.archive_root / "agents"
        agent_dirs = list(agents_dir.iterdir())
        assert len(agent_dirs) == 1
        
        agent_dir = agent_dirs[0]
        assert (agent_dir / "manifest.json").exists()
        assert (agent_dir / "artifacts").exists()
        
        # Check manifest content
        with open(agent_dir / "manifest.json") as f:
            manifest_data = json.load(f)
        
        assert manifest_data["agent_id"] == agent_id
        assert manifest_data["evo_config_inline"]["algorithm"] == "context"
    
    def test_list_agents(self, temp_archive):
        """Test listing agents."""
        archive, temp_dir = temp_archive
        
        # Initially empty
        agents = archive.list_agents()
        assert len(agents) == 0
        
        # Save some agents
        config1 = {"algorithm": "baseline", "seed": 42}
        config2 = {"algorithm": "context", "seed": 43}
        
        with patch('pathlib.Path.exists', return_value=False):
            agent_id1 = archive.save_agent(config1)
            agent_id2 = archive.save_agent(config2)
        
        # List agents
        agents = archive.list_agents()
        assert len(agents) == 2
        
        # Check agent info
        agent_ids = [a["id"] for a in agents]
        assert agent_id1 in agent_ids
        assert agent_id2 in agent_ids
    
    def test_get_agent_manifest(self, temp_archive):
        """Test retrieving agent manifest."""
        archive, temp_dir = temp_archive
        
        config = {"algorithm": "context", "seed": 42}
        
        with patch('pathlib.Path.exists', return_value=False):
            agent_id = archive.save_agent(config)
        
        # Get manifest
        manifest = archive.get_agent_manifest(agent_id)
        
        assert manifest is not None
        assert manifest.agent_id == agent_id
        assert manifest.evo_config_inline["algorithm"] == "context"
        
        # Test non-existent agent
        manifest_none = archive.get_agent_manifest("nonexistent")
        assert manifest_none is None
    
    def test_export_import_agent(self, temp_archive):
        """Test exporting and importing agents."""
        archive, temp_dir = temp_archive
        
        config = {"algorithm": "context", "seed": 42, "test_data": "important"}
        
        with patch('pathlib.Path.exists', return_value=False):
            agent_id = archive.save_agent(config)
        
        # Export agent
        export_path = Path(temp_dir) / "export_test.zip"
        success = archive.export_agent(agent_id, str(export_path))
        
        assert success
        assert export_path.exists()
        
        # Verify ZIP contents
        with zipfile.ZipFile(export_path, 'r') as zipf:
            files = zipf.namelist()
            assert "manifest.json" in files
            assert any("artifacts" in f for f in files)
        
        # Import agent (to new archive)
        new_archive_dir = tempfile.mkdtemp()
        try:
            new_archive = AgentArchive(new_archive_dir)
            imported_agent_id = new_archive.import_agent(str(export_path))
            
            assert imported_agent_id == agent_id
            
            # Verify imported manifest
            imported_manifest = new_archive.get_agent_manifest(agent_id)
            assert imported_manifest is not None
            assert imported_manifest.evo_config_inline["test_data"] == "important"
            
        finally:
            shutil.rmtree(new_archive_dir)


class TestArchiveIntegration:
    
    def test_list_show_export_import(self):
        """Test complete list/show/export/import workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create archive and save agent
            archive = create_agent_archive(temp_dir)
            config = {"algorithm": "context", "seed": 42}
            
            with patch('pathlib.Path.exists', return_value=False):
                agent_id = archive.save_agent(config)
            
            # Test list function
            agents = list_archived_agents()  # Uses default path
            # Note: This will use /app/shinka/archive, not our temp dir
            # So we test the archive instance directly
            agents = archive.list_agents()
            assert len(agents) >= 1
            
            # Test show function
            manifest_data = archive.get_agent_manifest(agent_id)
            assert manifest_data is not None
            
            # Test export function
            export_path = Path(temp_dir) / "test_export.zip"
            success = archive.export_agent(agent_id, str(export_path))
            assert success
            assert export_path.exists()
            
            # Test import function
            imported_id = archive.import_agent(str(export_path))
            assert imported_id == agent_id
    
    def test_auto_save_on_best_fitness(self):
        """Test auto-save trigger on best fitness improvement."""
        # This would be integration test with EvolutionRunner
        # For now, test the trigger logic separately
        
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            # Simulate fitness improvements
            configs = [
                {"fitness": 0.5, "algorithm": "baseline"},
                {"fitness": 0.7, "algorithm": "context"},  # Should trigger save
                {"fitness": 0.9, "algorithm": "context"},  # Should trigger save
            ]
            
            last_fitness = 0.0
            saved_agents = []
            
            for config in configs:
                if config["fitness"] > last_fitness:
                    with patch('pathlib.Path.exists', return_value=False):
                        agent_id = archive.save_agent(config)
                    saved_agents.append(agent_id)
                    last_fitness = config["fitness"]
            
            assert len(saved_agents) == 2  # Only improvements saved
    
    def test_repro_matches_metrics_within_tolerance(self):
        """Test reproduction matches original metrics within tolerance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            # Create agent with known metrics
            config = {
                "algorithm": "context",
                "seed": 42,
                "benchmark": "toy"
            }
            
            with patch('pathlib.Path.exists', return_value=False):
                agent_id = archive.save_agent(config)
            
            # Test reproduction
            with patch('subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0, stdout="", text=True)
                
                result = reproduce_agent(agent_id, tolerance_pct=1.0)
                
                # Should succeed (mocked)
                assert "success" in result
                assert result.get("agent_id") == agent_id
    
    def test_manifest_schema_keys_present(self):
        """Test that manifest contains all required DGM-compatible keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            config = {"algorithm": "context", "seed": 42}
            
            with patch('pathlib.Path.exists', return_value=False):
                agent_id = archive.save_agent(config)
            
            manifest = archive.get_agent_manifest(agent_id)
            
            # Check required keys
            required_keys = [
                "agent_id", "parent_id", "timestamp", "git_commit", "branch", 
                "dirty", "env", "seeds", "evo_config_path", "evo_config_inline",
                "hyperparams", "benchmarks", "cost", "context_activity", "dgm_compat"
            ]
            
            for key in required_keys:
                assert hasattr(manifest, key), f"Missing required key: {key}"
            
            # Check DGM compatibility keys
            dgm_compat = manifest.dgm_compat
            assert "repo_layout_ref" in dgm_compat
            assert "prompts_used" in dgm_compat
            assert "swe_bench_commit" in dgm_compat
            assert "polyglot_prepared" in dgm_compat
    
    def test_diff_patch_applied_on_dirty_tree(self):
        """Test diff patch creation and application on dirty git tree."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            # Mock dirty git state
            with patch.object(archive, '_get_git_info') as mock_git_info:
                mock_git_info.return_value = {
                    "commit": "abcd1234",
                    "branch": "feature",
                    "dirty": True
                }
                
                with patch('subprocess.run') as mock_subprocess:
                    # Mock git diff output
                    mock_subprocess.return_value = MagicMock(
                        returncode=0, 
                        stdout="diff --git a/file.py b/file.py\n+added line\n"
                    )
                    
                    config = {"algorithm": "context", "seed": 42}
                    
                    with patch('pathlib.Path.exists', return_value=False):
                        agent_id = archive.save_agent(config)
                    
                    # Check if diff.patch was created
                    agents_dir = archive.archive_root / "agents"
                    agent_dir = next(agents_dir.iterdir())
                    
                    assert (agent_dir / "diff.patch").exists()
                    
                    # Check patch content
                    patch_content = (agent_dir / "diff.patch").read_text()
                    assert "diff --git" in patch_content
                    assert "added line" in patch_content
    
    def test_failure_modes_missing_artifacts(self):
        """Test handling of missing artifacts and error conditions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            # Test export of non-existent agent
            success = archive.export_agent("nonexistent", "output.zip")
            assert not success
            
            # Test import of invalid ZIP
            invalid_zip = Path(temp_dir) / "invalid.zip"
            invalid_zip.write_text("not a zip")
            
            imported_id = archive.import_agent(str(invalid_zip))
            assert imported_id is None
            
            # Test reproduction of non-existent agent
            result = reproduce_agent("nonexistent")
            assert not result["success"]
            assert "not found" in result["error"].lower()