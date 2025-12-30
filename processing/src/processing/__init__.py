"""
Processing layer - Detección, tracking y conteo de vehículos.
"""

from .detector import YOLODetector, DetectorConfig
from .tracker import VehicleTracker
from .zone_manager import ZoneManager
from .mask_manager import MaskManager
from .track_deduplicator import TrackDeduplicator
from .counter import VehicleCounter, CounterConfig

__all__ = [
    "YOLODetector",
    "DetectorConfig",
    "VehicleTracker",
    "ZoneManager",
    "MaskManager",
    "TrackDeduplicator",
    "VehicleCounter",
    "CounterConfig",
]
