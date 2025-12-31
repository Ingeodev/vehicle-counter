"""
Base Detector - Interface para estrategias de detección.
"""

from typing import Protocol, runtime_checkable
import numpy as np
from dataclasses import dataclass

from ..data.schemas import Detection

@dataclass
class DetectorConfig:
    """Configuración del detector YOLO."""
    model_path: str = "yolov8s.pt"
    device: str = "cpu"
    vehicle_classes: dict[int, str] | None = None
    confidence_threshold: float = 0.5
    tracker_config: str | None = None
    
    # Nuevo: configuración específica
    use_segmentation: bool = False
    
    def __post_init__(self):
        # Clases de vehículos por defecto (COCO dataset)
        if self.vehicle_classes is None:
            self.vehicle_classes = {
                1: "bicycle",
                2: "car", 
                3: "motorcycle",
                5: "bus",
                7: "truck"
            }


@runtime_checkable
class BaseDetector(Protocol):
    """Interface común para detectores."""
    
    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        """Detecta y trackea vehículos en un frame."""
        ...
    
    def detect_only(self, frame: np.ndarray) -> list[Detection]:
        """Solo detecta (sin tracking)."""
        ...
        
    def reset_tracking(self) -> None:
        """Reinicia el tracker."""
        ...
        
    @property
    def class_names(self) -> dict[int, str]:
        """Retorna mapeo de IDs a nombres."""
        ...
