"""
YOLODetector - Wrapper para detección con YOLO.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from ultralytics import YOLO

from ..data.schemas import Detection, BoundingBox


# Clases de vehículos por defecto (COCO dataset)
DEFAULT_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car", 
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


@dataclass
class DetectorConfig:
    """Configuración del detector YOLO."""
    model_path: str = "yolov8s.pt"
    device: str = "cpu"
    vehicle_classes: dict[int, str] | None = None
    confidence_threshold: float = 0.5
    tracker_config: str | None = None  # Ruta a config de tracker (e.g. botsort.yaml)
    
    def __post_init__(self):
        if self.vehicle_classes is None:
            self.vehicle_classes = DEFAULT_VEHICLE_CLASSES.copy()


class YOLODetector:
    """
    Wrapper para detección y tracking con YOLO.
    
    Example:
        >>> config = DetectorConfig(model_path="yolov8s.pt", device="cuda")
        >>> detector = YOLODetector(config)
        >>> 
        >>> detections = detector.detect_and_track(frame)
        >>> for det in detections:
        ...     print(f"ID: {det.track_id}, Type: {det.class_name}")
    """
    
    def __init__(self, config: DetectorConfig | None = None):
        """
        Inicializa el detector.
        
        Args:
            config: Configuración del detector. Si es None, usa valores por defecto.
        """
        self.config = config or DetectorConfig()
        
        # Cargar modelo YOLO
        self.model = YOLO(self.config.model_path)
        self.device = self.config.device
        
        # Mapeo de clases
        self.vehicle_classes = self.config.vehicle_classes or DEFAULT_VEHICLE_CLASSES
        self.allowed_class_ids = list(self.vehicle_classes.keys())
    
    @classmethod
    def from_path(
        cls, 
        model_path: str, 
        device: str = "cpu",
        vehicle_classes: dict[int, str] | None = None
    ) -> "YOLODetector":
        """
        Crea un detector desde una ruta de modelo.
        
        Args:
            model_path: Ruta al modelo YOLO
            device: Dispositivo ('cpu', 'cuda', 'mps')
            vehicle_classes: Diccionario de clases permitidas
            
        Returns:
            YOLODetector configurado
        """
        config = DetectorConfig(
            model_path=model_path,
            device=device,
            vehicle_classes=vehicle_classes
        )
        return cls(config)
    
    def detect_and_track(
        self, 
        frame: np.ndarray,
        persist: bool = True
    ) -> list[Detection]:
        """
        Detecta y trackea vehículos en un frame.
        
        Args:
            frame: Frame de video (numpy array BGR)
            persist: Si True, mantiene el tracking entre frames
            
        Returns:
            Lista de detecciones con IDs de tracking
        """
        # Preparar argumentos de tracking
        track_kwargs = {
            "persist": persist,
            "device": self.device,
            "classes": self.allowed_class_ids,
            "verbose": False
        }
        
        # Usar tracker config si está definido
        if self.config.tracker_config:
            # Verificar si es ruta absoluta o relativa
            tracker_path = self.config.tracker_config
            if not os.path.isabs(tracker_path):
                # Buscar en el directorio de config del paquete
                pkg_dir = Path(__file__).parent.parent / "config"
                local_path = pkg_dir / tracker_path
                if local_path.exists():
                    tracker_path = str(local_path)
            track_kwargs["tracker"] = tracker_path
        
        # Ejecutar inferencia con tracking
        results = self.model.track(frame, **track_kwargs)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        
        # Verificar que hay detecciones con IDs
        if result.boxes.id is None:
            return []
        
        # Extraer datos
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        
        # Crear lista de detecciones
        detections: list[Detection] = []
        
        for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
            # Verificar clase permitida
            if cls_id not in self.vehicle_classes:
                continue
            
            # Verificar confianza
            if conf < self.config.confidence_threshold:
                continue
            
            x1, y1, x2, y2 = box.astype(int)
            
            detection = Detection(
                track_id=int(track_id),
                class_id=int(cls_id),
                class_name=self.vehicle_classes[int(cls_id)],
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=float(conf)
            )
            detections.append(detection)
        
        return detections
    
    def detect_only(self, frame: np.ndarray) -> list[Detection]:
        """
        Solo detecta (sin tracking).
        
        Args:
            frame: Frame de video
            
        Returns:
            Lista de detecciones (sin IDs de tracking)
        """
        results = self.model(
            frame,
            device=self.device,
            classes=self.allowed_class_ids,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        
        detections: list[Detection] = []
        
        for i, (box, cls_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
            if cls_id not in self.vehicle_classes:
                continue
            
            if conf < self.config.confidence_threshold:
                continue
            
            x1, y1, x2, y2 = box.astype(int)
            
            detection = Detection(
                track_id=i,  # Usar índice como ID temporal
                class_id=int(cls_id),
                class_name=self.vehicle_classes[int(cls_id)],
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=float(conf)
            )
            detections.append(detection)
        
        return detections
    
    def reset_tracking(self) -> None:
        """Reinicia el estado del tracker."""
        # YOLO automáticamente reinicia con un nuevo video
        # Forzar reinicio recargando el modelo
        self.model = YOLO(self.config.model_path)
    
    @property
    def class_names(self) -> dict[int, str]:
        """Retorna el mapeo de IDs a nombres de clases."""
        return self.vehicle_classes.copy()
    
    def get_device_info(self) -> str:
        """Retorna información del dispositivo."""
        return f"Device: {self.device}"
