import sys
import os
from src.cli import main

# Mocking sys.argv to simulate CLI usage
def test_cli_fix_osd():
    print("🧪 Testing CLI fix-osd with dynamic bounding box...")
    
    # Arguments for test
    test_args = [
        "aforos", 
        "fix-osd", 
        "input/videos/boulevard_oriente.mp4",
        "--date", "2026-01-07",
        "--top", "60",
        "--right", "388",
        "--bottom", "93",
        "--left", "50",
        "--minutes", "0.01", # Very short
        "--debug",
        "--output", "results/cli_test_output.mp4",
        "--quiet"
    ]
    
    # Save original sys.argv
    original_argv = sys.argv
    
    try:
        sys.argv = test_args
        exit_code = main()
        
        if exit_code == 0:
            print("✅ CLI test successful!")
            if os.path.exists("results/cli_test_output.mp4"):
                 print("✅ Output file created: results/cli_test_output.mp4")
            else:
                 print("❌ Output file NOT created.")
        else:
            print(f"❌ CLI test failed with exit code: {exit_code}")
            
    except Exception as e:
        print(f"❌ CLI test raised exception: {e}")
    finally:
        # Restore sys.argv
        sys.argv = original_argv

if __name__ == "__main__":
    test_cli_fix_osd()
