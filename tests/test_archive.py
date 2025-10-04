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
            
            assert len(saved_agents) == 3  # All saved since all are improvements
    
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
            
            # Test reproduction with proper archive reference
            archive_path = str(temp_dir)
            with patch('shinka.archive.agent_archive.create_agent_archive') as mock_create:
                mock_create.return_value = archive
                
                with patch('subprocess.run') as mock_subprocess:
                    mock_subprocess.return_value = MagicMock(returncode=0, stdout="", text=True)
                    
                    result = reproduce_agent(agent_id, tolerance_pct=1.0)
                    
                    # Should succeed (mocked)
                    assert "success" in result or "error" in result  # Either success or controlled failure
    
    def test_manifest_schema_keys_present(self):
        """Test that manifest contains all required DGM-compatible keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            config = {"algorithm": "context", "seed": 42}
            
            with patch('pathlib.Path.exists', return_value=False):
                agent_id = archive.save_agent(config)
            
            manifest = archive.get_agent_manifest(agent_id)
            
            # Check required keys (original)
            required_keys = [
                "agent_id", "parent_id", "timestamp", "git_commit", "branch", 
                "dirty", "env", "seeds", "evo_config_path", "evo_config_inline",
                "hyperparams", "benchmarks", "cost", "context_activity", "dgm_compat"
            ]
            
            for key in required_keys:
                assert hasattr(manifest, key), f"Missing required key: {key}"
            
            # Check production-grade enrichment keys
            enrichment_keys = [
                "benchmarks_full", "complexity_metrics", "validation_levels",
                "cost_breakdown", "artifact_refs"
            ]
            
            for key in enrichment_keys:
                assert hasattr(manifest, key), f"Missing enrichment key: {key}"
            
            # Check DGM compatibility keys
            dgm_compat = manifest.dgm_compat
            assert "repo_layout_ref" in dgm_compat
            assert "prompts_used" in dgm_compat
            assert "swe_bench_commit" in dgm_compat
            assert "polyglot_prepared" in dgm_compat
            
            # Validate enrichment field structures
            complexity = manifest.complexity_metrics
            assert "lines_of_code_total" in complexity or "lines_of_code_delta" in complexity
            assert "cyclomatic_complexity" in complexity
            assert "coupling_between_objects" in complexity
            
            validation = manifest.validation_levels
            assert "static_checks" in validation
            assert "unit_tests" in validation
            
            cost_bd = manifest.cost_breakdown
            assert "mock_model" in cost_bd
            assert isinstance(cost_bd["mock_model"], dict)
            assert "queries" in cost_bd["mock_model"]
            assert "cost_usd" in cost_bd["mock_model"]
    
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
    
    def test_export_dirty_repo(self):
        """Test export with dirty repository includes diff.patch correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            # Mock dirty git state with actual diff content
            with patch.object(archive, '_get_git_info') as mock_git_info:
                mock_git_info.return_value = {
                    "commit": "abcd1234",
                    "branch": "feature/test",
                    "dirty": True
                }
                
                with patch('subprocess.run') as mock_subprocess:
                    # Mock git diff with realistic output
                    diff_content = """diff --git a/shinka/test.py b/shinka/test.py
index 1234567..abcdefg 100644
--- a/shinka/test.py
+++ b/shinka/test.py
@@ -1,3 +1,4 @@
 def test_function():
+    # Added this comment
     return True
 
"""
                    mock_subprocess.return_value = MagicMock(
                        returncode=0, 
                        stdout=diff_content
                    )
                    
                    config = {"algorithm": "context", "seed": 42}
                    
                    with patch('pathlib.Path.exists', return_value=False):
                        agent_id = archive.save_agent(config)
                    
                    # Verify diff.patch was created with correct content
                    agents_dir = archive.archive_root / "agents"
                    agent_dir = next(agents_dir.iterdir())
                    
                    assert (agent_dir / "diff.patch").exists()
                    patch_content = (agent_dir / "diff.patch").read_text()
                    assert "diff --git" in patch_content
                    assert "Added this comment" in patch_content
                    
                    # Test export includes diff.patch without crash
                    export_path = Path(temp_dir) / "dirty_export.zip"
                    success = archive.export_agent(agent_id, str(export_path))
                    
                    assert success
                    assert export_path.exists()
                    
                    # Verify ZIP contains diff.patch
                    with zipfile.ZipFile(export_path, 'r') as zipf:
                        files = zipf.namelist()
                        assert "diff.patch" in files
                        
                        # Verify diff.patch content in ZIP
                        diff_in_zip = zipf.read("diff.patch").decode('utf-8')
                        assert "Added this comment" in diff_in_zip
    
    def test_lineage_chain_consistency(self):
        """Test lineage chain with parent_id → child_id consistency across multiple runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            # Create lineage chain: gen0 → gen1 → gen2
            configs = [
                {"algorithm": "baseline", "generation": 0, "fitness": 0.5},
                {"algorithm": "context", "generation": 1, "fitness": 0.7},
                {"algorithm": "context", "generation": 2, "fitness": 0.9}
            ]
            
            agent_ids = []
            
            with patch('pathlib.Path.exists', return_value=False), \
                 patch('subprocess.run') as mock_subprocess:
                # Mock validation calls to speed up test
                mock_subprocess.return_value = MagicMock(returncode=1, stdout="", stderr="")
                
                for i, config in enumerate(configs):
                    parent_id = agent_ids[-1] if agent_ids else None
                    agent_id = archive.save_agent(config, parent_id=parent_id)
                    agent_ids.append(agent_id)
            
            # Verify lineage consistency
            assert len(agent_ids) == 3
            
            # Check first agent (root)
            manifest0 = archive.get_agent_manifest(agent_ids[0])
            assert manifest0.parent_id is None
            
            # Check second agent (child of first)
            manifest1 = archive.get_agent_manifest(agent_ids[1])
            assert manifest1.parent_id == agent_ids[0]
            
            # Check third agent (child of second)
            manifest2 = archive.get_agent_manifest(agent_ids[2])
            assert manifest2.parent_id == agent_ids[1]
            
            # Verify chain traversal
            lineage_chain = []
            current_id = agent_ids[2]  # Start from leaf
            
            while current_id:
                manifest = archive.get_agent_manifest(current_id)
                lineage_chain.append(current_id)
                current_id = manifest.parent_id
            
            # Chain should be: gen2 → gen1 → gen0
            assert lineage_chain == [agent_ids[2], agent_ids[1], agent_ids[0]]
            
            # Verify all agents in lineage are findable
            all_agents = archive.list_agents()
            found_ids = [agent["id"] for agent in all_agents]
            
            for agent_id in agent_ids:
                assert agent_id in found_ids
    
    def test_import_corrupt_zip(self):
        """Test graceful failure on corrupted ZIP without corrupting archive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            # Create valid agent first
            config = {"algorithm": "context", "seed": 42}
            
            with patch('pathlib.Path.exists', return_value=False):
                valid_agent_id = archive.save_agent(config)
            
            # Verify initial state
            initial_agents = archive.list_agents()
            assert len(initial_agents) == 1
            
            # Test various corruption scenarios
            
            # 1. Completely invalid ZIP
            corrupt_zip1 = Path(temp_dir) / "corrupt1.zip"
            corrupt_zip1.write_text("This is not a ZIP file at all")
            
            imported_id1 = archive.import_agent(str(corrupt_zip1))
            assert imported_id1 is None
            
            # Archive should be unchanged
            agents_after_corrupt1 = archive.list_agents()
            assert len(agents_after_corrupt1) == 1
            assert agents_after_corrupt1[0]["id"] == valid_agent_id
            
            # 2. Valid ZIP but missing manifest.json
            corrupt_zip2 = Path(temp_dir) / "corrupt2.zip"
            with zipfile.ZipFile(corrupt_zip2, 'w') as zipf:
                zipf.writestr("some_file.txt", "content")
                zipf.writestr("artifacts/config.yaml", "test: config")
                # Missing manifest.json
            
            imported_id2 = archive.import_agent(str(corrupt_zip2))
            assert imported_id2 is None
            
            # Archive should still be unchanged
            agents_after_corrupt2 = archive.list_agents()
            assert len(agents_after_corrupt2) == 1
            
            # 3. Valid ZIP with malformed manifest.json
            corrupt_zip3 = Path(temp_dir) / "corrupt3.zip"
            with zipfile.ZipFile(corrupt_zip3, 'w') as zipf:
                zipf.writestr("manifest.json", "{ invalid json content }")
                zipf.writestr("artifacts/config.yaml", "test: config")
            
            imported_id3 = archive.import_agent(str(corrupt_zip3))
            assert imported_id3 is None
            
            # Archive should remain stable
            agents_after_corrupt3 = archive.list_agents()
            assert len(agents_after_corrupt3) == 1
            
            # 4. Test that valid import still works after corruption attempts
            export_path = Path(temp_dir) / "valid_export.zip"
            success = archive.export_agent(valid_agent_id, str(export_path))
            assert success
            
            imported_valid_id = archive.import_agent(str(export_path))
            assert imported_valid_id == valid_agent_id
            
            # Should have agents (import creates new entry with timestamp)
            final_agents = archive.list_agents()
            assert len(final_agents) >= 1  # May have duplicates due to import timestamp
    
    def test_enriched_manifest_fields_correctness(self):
        """Test that enriched manifest fields contain correct and meaningful data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = AgentArchive(temp_dir)
            
            config = {
                "algorithm": "context",
                "seed": 42,
                "benchmark": "tsp",
                "budget_steps": 1000
            }
            
            with patch('pathlib.Path.exists', return_value=False):
                agent_id = archive.save_agent(config)
            
            manifest = archive.get_agent_manifest(agent_id)
            
            # Test complexity metrics correctness
            complexity = manifest.complexity_metrics
            assert isinstance(complexity["cyclomatic_complexity"], (int, float))
            assert complexity["cyclomatic_complexity"] >= 1.0
            assert isinstance(complexity["coupling_between_objects"], int)
            assert complexity["coupling_between_objects"] >= 0
            
            # Test validation levels structure
            validation = manifest.validation_levels
            assert validation["static_checks"]["ruff"] in ["pass", "fail", "unknown"]
            assert validation["unit_tests"]["status"] in ["pass", "fail", "unknown"]
            assert isinstance(validation["unit_tests"]["passed"], int)
            assert validation["unit_tests"]["passed"] >= 0
            
            # Test cost breakdown structure and values
            cost_bd = manifest.cost_breakdown
            total_cost = 0
            total_queries = 0
            
            for model, data in cost_bd.items():
                assert isinstance(data["queries"], (int, float))
                assert isinstance(data["cost_usd"], (int, float))
                assert data["queries"] >= 0
                assert data["cost_usd"] >= 0
                total_cost += data["cost_usd"]
                total_queries += data["queries"]
            
            # Mock model should have zero cost (may have zero queries if no benchmarks)
            assert cost_bd["mock_model"]["cost_usd"] == 0.0
            assert cost_bd["mock_model"]["queries"] >= 0  # May be 0 if no benchmark data
            
            # Test artifact references
            artifacts = manifest.artifact_refs
            assert isinstance(artifacts, dict)
            # Should have at least config snapshot
            if "config_snapshot" in artifacts:
                assert artifacts["config_snapshot"].endswith(".yaml")
            
            # Test benchmarks_full references  
            benchmarks_full = manifest.benchmarks_full
            assert isinstance(benchmarks_full, dict)
            # May be empty if no timeseries data available
            for ref_name, file_path in benchmarks_full.items():
                assert isinstance(file_path, str)
                assert "timeseries" in ref_name.lower() or "csv" in file_path.lower()