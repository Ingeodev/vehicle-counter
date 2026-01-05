import os
import pytest
from mglon_vehicle_counter import fix_osd

VIDEO_INPUT = "input/videos/boulevard_oriente.mp4"
OUTPUT_DIR = "results_tests/fix_osd"
DURATION = 0.05  # minutos

def setup_module():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_fix_osd_h264():
    """Test fix_osd with H.264 codec."""
    output = os.path.join(OUTPUT_DIR, "test_h264.mp4")
    result = fix_osd(
        video=VIDEO_INPUT,
        date="01-01-2026",
        output=output,
        max_minutes=DURATION,
        codec="h264",
        quiet=True
    )
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0

def test_fix_osd_h265():
    """Test fix_osd with H.265 codec."""
    output = os.path.join(OUTPUT_DIR, "test_h265.mp4")
    result = fix_osd(
        video=VIDEO_INPUT,
        date="02-01-2026",
        output=output,
        max_minutes=DURATION,
        codec="h265",
        quiet=True
    )
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0

def test_fix_osd_copy():
    """Test fix_osd with copy codec (fastest)."""
    output = os.path.join(OUTPUT_DIR, "test_copy.mp4")
    result = fix_osd(
        video=VIDEO_INPUT,
        date="03-01-2026",
        output=output,
        max_minutes=DURATION,
        codec="copy",
        quiet=True
    )
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0

if __name__ == "__main__":
    setup_module()
    test_fix_osd_h264()
    test_fix_osd_h265()
    test_fix_osd_copy()
    print("✅ test_fix_osd passed!")
