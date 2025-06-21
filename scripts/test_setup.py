#!/usr/bin/env python3
"""
Test script to verify project setup and configuration
"""
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config

def test_config():
    """Test configuration setup"""
    print("=== Testing Configuration ===")
    
    config = Config()
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Models directory: {config.MODELS_DIR}")
    print(f"Outputs directory: {config.OUTPUTS_DIR}")
    print(f"Device: {config.DEVICE}")
    
    # Test directory creation
    config.create_directories()
    print("✅ Directories created successfully")
    
    return True

def test_pytorch():
    """Test PyTorch installation"""
    print("\n=== Testing PyTorch ===")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available - will use CPU")
    
    # Test basic tensor operations
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = x + y
    print(f"✅ Basic tensor operations work: {z.shape}")
    
    return True

def test_project_structure():
    """Test project structure"""
    print("\n=== Testing Project Structure ===")
    
    required_files = [
        "config.py",
        "requirements.txt", 
        "README.md",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "scripts/test_job.slurm",
        "scripts/test_pytorch_job.py",
        "scripts/train_anonymization.slurm"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all tests"""
    print("=== Project Setup Verification ===\n")
    
    tests = [
        test_config,
        test_pytorch,
        test_project_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    print(f"\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Project setup is complete.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 