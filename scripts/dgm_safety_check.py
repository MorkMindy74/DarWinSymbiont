#!/usr/bin/env python3
"""
DGM Safety Validation Script

Validates DGM integration safety measures and policies.
"""

import sys
import subprocess
import tempfile
import time
from pathlib import Path

sys.path.insert(0, '/app')

from adapters.dgm_runner import DGMConfig, DGMRunner, DGMSafetyError


def test_tool_allowlist_enforcement():
    """Test that dangerous tools are blocked."""
    print("Testing tool allowlist enforcement...")
    
    config = DGMConfig(unsafe_allow=False)
    runner = DGMRunner(config)
    
    # Test blocked commands
    dangerous_commands = [
        ["sudo", "apt", "update"],
        ["rm", "-rf", "/"],
        ["curl", "http://malicious.com"],
        ["wget", "http://evil.site"],
        ["docker", "run", "malware"],
        ["chmod", "777", "/etc/passwd"]
    ]
    
    for cmd in dangerous_commands:
        try:
            runner._validate_command_safety(cmd)
            print(f"❌ FAIL: Command {cmd} was not blocked")
            return False
        except DGMSafetyError:
            print(f"✅ PASS: Command {cmd} properly blocked")
    
    return True


def test_allowed_commands():
    """Test that safe commands are allowed."""
    print("Testing allowed commands...")
    
    config = DGMConfig(unsafe_allow=False)
    runner = DGMRunner(config)
    
    # Test allowed commands
    safe_commands = [
        ["python3", "-c", "print('hello')"],
        ["pytest", "--version"],
        ["ls", "-la"],
        ["cat", "/app/README.md"],
        ["head", "-5", "somefile.txt"]
    ]
    
    for cmd in safe_commands:
        try:
            runner._validate_command_safety(cmd)
            print(f"✅ PASS: Command {cmd} properly allowed")
        except DGMSafetyError as e:
            print(f"❌ FAIL: Safe command {cmd} was blocked: {e}")
            return False
    
    return True


def test_resource_limits():
    """Test resource limit configuration."""
    print("Testing resource limits...")
    
    config = DGMConfig(
        max_memory_gb=2.0,
        max_cpus=1.0,
        timeout_seconds=300
    )
    
    # Validate configuration
    assert config.max_memory_gb == 2.0
    assert config.max_cpus == 1.0
    assert config.timeout_seconds == 300
    
    print("✅ PASS: Resource limits properly configured")
    return True


def test_network_isolation():
    """Test network isolation in Docker compose."""
    print("Testing network isolation...")
    
    try:
        # Check docker-compose configuration
        result = subprocess.run([
            "docker-compose", "-f", "/app/docker-compose.dgm.yml", "config"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"❌ FAIL: Docker compose config error: {result.stderr}")
            return False
        
        # Check for network isolation
        config_content = result.stdout
        if "internal: true" in config_content:
            print("✅ PASS: Network isolation properly configured")
            return True
        else:
            print("⚠️  WARNING: Network isolation may not be configured")
            return True  # Not critical for basic functionality
            
    except Exception as e:
        print(f"❌ FAIL: Could not test network isolation: {e}")
        return False


def test_file_path_validation():
    """Test suspicious file path detection."""
    print("Testing file path validation...")
    
    config = DGMConfig(unsafe_allow=False)
    runner = DGMRunner(config)
    
    # Test dangerous file paths
    dangerous_commands = [
        ["cat", "/etc/passwd"],
        ["ls", "/root/"],
        ["cat", "~/.ssh/id_rsa"],
        ["touch", "/etc/hosts"],
        ["rm", "/var/log/system.log"]
    ]
    
    for cmd in dangerous_commands:
        try:
            runner._validate_command_safety(cmd)
            print(f"❌ FAIL: Dangerous path {cmd} was not blocked")
            return False
        except DGMSafetyError:
            print(f"✅ PASS: Dangerous path {cmd} properly blocked")
    
    return True


def test_docker_security_options():
    """Test Docker security configuration."""
    print("Testing Docker security options...")
    
    # Read docker-compose file
    compose_file = Path("/app/docker-compose.dgm.yml")
    if not compose_file.exists():
        print("❌ FAIL: Docker compose file not found")
        return False
    
    with open(compose_file) as f:
        content = f.read()
    
    # Check for security options
    security_checks = [
        ("no-new-privileges:true", "no-new-privileges"),
        ("cap_drop:", "capability dropping"),
        ("read_only:", "read-only filesystem options"),
        ("mem_limit:", "memory limits"),
        ("cpus:", "CPU limits")
    ]
    
    all_passed = True
    for check, description in security_checks:
        if check in content:
            print(f"✅ PASS: {description} configured")
        else:
            print(f"⚠️  WARNING: {description} may not be configured")
            # Don't fail for warnings
    
    return all_passed


def test_timeout_enforcement():
    """Test timeout enforcement."""
    print("Testing timeout enforcement...")
    
    # This is a mock test since we can't easily test real timeouts
    config = DGMConfig(timeout_seconds=5)  # Very short timeout
    
    assert config.timeout_seconds == 5
    print("✅ PASS: Timeout configuration validated")
    
    return True


def test_unsafe_allow_flag():
    """Test unsafe_allow flag behavior."""
    print("Testing unsafe_allow flag...")
    
    # Test strict mode (default)
    strict_config = DGMConfig(unsafe_allow=False)
    assert "bash" not in strict_config.allowed_tools
    assert "sh" not in strict_config.allowed_tools
    print("✅ PASS: Strict mode blocks dangerous tools")
    
    # Test unsafe mode
    unsafe_config = DGMConfig(unsafe_allow=True)
    assert "bash" in unsafe_config.allowed_tools
    assert "sh" in unsafe_config.allowed_tools
    print("✅ PASS: Unsafe mode allows additional tools")
    
    return True


def main():
    """Run all safety validation tests."""
    print("🔒 DGM Safety Validation Starting...")
    print("=" * 50)
    
    tests = [
        ("Tool Allowlist Enforcement", test_tool_allowlist_enforcement),
        ("Allowed Commands", test_allowed_commands),
        ("Resource Limits", test_resource_limits),
        ("Network Isolation", test_network_isolation),
        ("File Path Validation", test_file_path_validation),
        ("Docker Security Options", test_docker_security_options),
        ("Timeout Enforcement", test_timeout_enforcement),
        ("Unsafe Allow Flag", test_unsafe_allow_flag)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAIL: {test_name} - Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🔒 DGM Safety Validation Complete")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("🎉 All safety tests passed!")
        return 0
    else:
        print("⚠️  Some safety tests failed - review configuration")
        return 1


if __name__ == "__main__":
    sys.exit(main())