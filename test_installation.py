"""Quick test script to verify installation and basic functionality."""

import sys
import importlib


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "PIL",
        "cv2",
        "decord",
        "tqdm",
        "pandas",
        "tabulate",
        "nltk",
    ]
    
    failed = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                importlib.import_module("PIL")
            elif package == "cv2":
                importlib.import_module("cv2")
            else:
                importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    try:
        import torch
        
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️  CUDA not available - will run on CPU (slower)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error testing CUDA: {e}")
        return False


def test_dataset_loading():
    """Test loading a small dataset."""
    print("\nTesting dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Try loading a tiny dataset
        print("  Loading test dataset...")
        ds = load_dataset("Anthropic/hh-rlhf", split="test[:1]")
        print(f"  ✓ Loaded {len(ds)} sample(s)")
        return True
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return False


def test_model_interface():
    """Test model base class."""
    print("\nTesting model interface...")
    
    try:
        from models.base_vlm import BaseVLM
        from models.example_vlm import ExampleVLM
        from PIL import Image
        
        # Test example model
        model = ExampleVLM(model_id="test")
        img = Image.new("RGB", (224, 224))
        response = model.ask_images([img], "Test question")
        
        print(f"  ✓ Model interface working")
        print(f"  Response: {response[:50]}...")
        return True
    except Exception as e:
        print(f"  ✗ Error testing model: {e}")
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("\nTesting metrics...")
    
    try:
        from metrics import Metrics, format_mcq_prompt
        
        # Test choice extraction
        text = "The answer is B"
        choice = Metrics.extract_choice_letter(text)
        assert choice == "B", f"Expected 'B', got '{choice}'"
        
        # Test accuracy
        preds = ["A", "B", "C"]
        gts = ["A", "B", "D"]
        acc = Metrics.mcq_accuracy(preds, gts)
        assert abs(acc - 0.6667) < 0.01, f"Expected 0.6667, got {acc}"
        
        # Test prompt formatting
        prompt = format_mcq_prompt("What is 2+2?", ["3", "4", "5"])
        assert "(A)" in prompt and "(B)" in prompt
        
        print("  ✓ All metric tests passed")
        return True
    except Exception as e:
        print(f"  ✗ Error testing metrics: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("VLM Evaluation Framework - Installation Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Dataset Loading", test_dataset_loading),
        ("Model Interface", test_model_interface),
        ("Metrics", test_metrics),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    print("="*60)
    if all_passed:
        print("✓ All tests passed! System is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
