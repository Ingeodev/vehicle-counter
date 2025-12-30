"""
Aforos - Vehicle Counter Package

Sistema modular para conteo de vehículos usando YOLO.

Capas:
    - storage: Abstracción de almacenamiento (local, Google Drive)
    - data: Modelos de datos, video source, directory scanner
    - processing: Detector YOLO, tracker, zone manager, counter
    - output: Video writer, CSV exporter, visualizer
    - config: Configuración centralizada
    - utils: Utilidades compartidas

Example:
    >>> from src import VideoPipeline, PipelineConfig
    >>> 
    >>> config = PipelineConfig()
    >>> config.detector.device = "cuda"
    >>> 
    >>> pipeline = VideoPipeline(config)
    >>> result = pipeline.process_video("video.mp4", zones_path="zones.json")
"""

from .config import PipelineConfig, VideoConfig, DetectorConfig, OutputConfig
from .storage import StorageReader, StorageWriter, LocalStorageReader, LocalStorageWriter
from .data import VideoInfo, VideoSource, DirectoryScanner, ProcessingResult
from .processing import YOLODetector, VehicleTracker, ZoneManager, VehicleCounter
from .pipeline import VideoPipeline

__version__ = "1.0.0"

__all__ = [
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
]
