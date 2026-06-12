"""
Aforos - Vehicle Counter Package

Sistema modular para conteo de vehículos usando YOLO.

Capas:
    - storage: Abstracción de almacenamiento (local, Google Drive)
    - data: Modelos de datos, video source, directory scanner
    - processing: Detector YOLO, tracker, zone manager, counter
    - output: Video writer, CSV exporter, visualizer
    - config: Configuración centralizada
    - utils: Utilidades compartidas (OCR, OSD modifier)

Example - Pipeline:
    >>> from mglon_vehicle_counter import VideoPipeline, PipelineConfig
    >>>
    >>> config = PipelineConfig()
    >>> config.detector.device = "cuda"
    >>>
    >>> pipeline = VideoPipeline(config)
    >>> result = pipeline.process_video("video.mp4", zones_path="zones.json")

Example - OSD Date Fix:
    >>> from mglon_vehicle_counter import OSDModifier
    >>>
    >>> modifier = OSDModifier()
    >>> fixed_frame = modifier.process_frame(frame, "05-01-2026 Mon")

Example - Extract Time:
    >>> from mglon_vehicle_counter import OSDReader
    >>>
    >>> reader = OSDReader(ocr_engine="easyocr")
    >>> date, time = reader.read_frame(frame)
"""

from .config import PipelineConfig, VideoConfig, DetectorConfig, OutputConfig
from .storage import (
    StorageReader,
    StorageWriter,
    LocalStorageReader,
    LocalStorageWriter,
)
from .data import VideoInfo, VideoSource, DirectoryScanner, ProcessingResult
from .processing import YOLODetector, VehicleTracker, ZoneManager, VehicleCounter
from .pipeline import VideoPipeline
from .utils.osd_modifier import OSDModifier
from .utils.osd_reader import OSDReader
from .api import fix_osd, process_video, extract_time

try:
    from importlib.metadata import version as _version

    __version__ = _version("mglon-vehicle-counter")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    # High-level API (recommended)
    "fix_osd",
    "process_video",
    "extract_time",
    # Config
    "PipelineConfig",
    "VideoConfig",
    "DetectorConfig",
    "OutputConfig",
    # Storage
    "StorageReader",
    "StorageWriter",
    "LocalStorageReader",
    "LocalStorageWriter",
    # Data
    "VideoInfo",
    "VideoSource",
    "DirectoryScanner",
    "ProcessingResult",
    # Processing
    "YOLODetector",
    "VehicleTracker",
    "ZoneManager",
    "VehicleCounter",
    # Pipeline
    "VideoPipeline",
    # Utils - OSD
    "OSDModifier",
    "OSDReader",
]
