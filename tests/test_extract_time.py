import os
from mglon_vehicle_counter import extract_time

VIDEO_INPUT = "input/videos/boulevard_oriente.mp4"
# No hay output dir de video en extract_time, pero sí CSV opcional
OUTPUT_DIR = "results_tests/extract_time"

def setup_module():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_extract_time_clahe():
    """Test extract_time with CLAHE preprocessing."""
    print("Testing extract_time (CLAHE)...")
    results = extract_time(
        videos=[VIDEO_INPUT], # Lista
        model="easyocr",
        preprocess="clahe",
        quiet=True
    )
    assert isinstance(results, list)
    assert len(results) > 0
    print(f"  Result: {results[0]}")

def test_extract_time_binary():
    """Test extract_time with Binary preprocessing."""
    print("Testing extract_time (Binary)...")
    results = extract_time(
        videos=VIDEO_INPUT, # String único
        model="easyocr",
        preprocess="binary",
        quiet=True
    )
    assert isinstance(results, list)
    assert len(results) > 0
    print(f"  Result: {results[0]}")

if __name__ == "__main__":
    setup_module()
    try:
        test_extract_time_clahe()
        test_extract_time_binary()
        print("✅ test_extract_time passed!")
    except Exception as e:
        print(f"❌ test_extract_time failed: {e}")
        exit(1)
