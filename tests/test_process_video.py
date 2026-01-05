import os
import shutil
from mglon_vehicle_counter import process_video

VIDEO_INPUT = "input/videos/boulevard_oriente.mp4"
ZONES_INPUT = "zones.json"  # Asumimos que existe en la raíz
OUTPUT_DIR = "results_tests/process_video"
DURATION = 0.05  # minutos

def setup_module():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_process_box_cpu():
    """Test process_video with Box tracking on CPU."""
    print("Testing process_video (Box/CPU)...")
    output_path = os.path.join(OUTPUT_DIR, "box_cpu")
    result = process_video(
        video=VIDEO_INPUT,
        zones=ZONES_INPUT,
        output=output_path,
        device="cpu",
        strategy="box",
        max_minutes=DURATION,
        quiet=True
    )
    # Verificar salidas
    video_out = os.path.join(output_path, "boulevard_oriente_processed.mp4")
    assert os.path.exists(video_out), f"Output video not found: {video_out}"
    print(f"  Detections: {result.total_detections}")

def test_process_seg_cpu():
    """Test process_video with Segmentation on CPU."""
    print("Testing process_video (Seg/CPU)...")
    output_path = os.path.join(OUTPUT_DIR, "seg_cpu")
    result = process_video(
        video=VIDEO_INPUT,
        zones=ZONES_INPUT,
        output=output_path,
        device="cpu",
        strategy="seg",
        max_minutes=DURATION,
        quiet=True
    )
    video_out = os.path.join(output_path, "boulevard_oriente_processed.mp4")
    assert os.path.exists(video_out), f"Output video not found: {video_out}"
    print(f"  Detections: {result.total_detections}")

if __name__ == "__main__":
    setup_module()
    try:
        test_process_box_cpu()
        test_process_seg_cpu()
        print("✅ test_process_video passed!")
    except Exception as e:
        print(f"❌ test_process_video failed: {e}")
        exit(1)
