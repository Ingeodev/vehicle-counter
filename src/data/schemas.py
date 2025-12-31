"""
Modelos de datos (schemas) para el sistema de conteo de vehículos.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ZoneConfig:
    """Configuración de una zona de interés."""
    label: str
    points: np.ndarray  # Puntos del polígono en formato (N, 2)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneConfig":
        """Crea un ZoneConfig desde un diccionario."""
        return cls(
            label=data["label"],
            points=np.array(data["points"], dtype=np.int32)
        )
    
    def scale(self, scale_x: float, scale_y: float) -> "ZoneConfig":
        """Retorna una nueva zona con coordenadas escaladas."""
        scaled_points = self.points.copy().astype(float)
        scaled_points[:, 0] *= scale_x
        scaled_points[:, 1] *= scale_y
        return ZoneConfig(
            label=self.label,
            points=scaled_points.astype(np.int32)
        )


@dataclass
class VideoInfo:
    """Información de un video a procesar."""
    path: str
    hora_inicial: str  # Formato HH:MM o HH:MM:SS
    mask_path: str | None = None
    zones_path: str | None = None
    output_folder: str | None = None
    output_name: str | None = None
    context: str | None = None  # Contexto/carpeta del video
    video_id: str | None = None
    
    # Metadatos del video (se llenan después de abrir)
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    total_frames: int | None = None
    
    @property
    def duration_seconds(self) -> float | None:
        """Duración del video en segundos."""
        if self.fps and self.total_frames:
            return self.total_frames / self.fps
        return None
    
    @property
    def duration_minutes(self) -> float | None:
        """Duración del video en minutos."""
        if self.duration_seconds:
            return self.duration_seconds / 60
        return None


@dataclass
class BoundingBox:
    """Caja delimitadora de una detección."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def center(self) -> tuple[int, int]:
        """Centro de la caja."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class Detection:
    """Una detección de vehículo en un frame."""
    track_id: int
    class_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float = 1.0
    mask: np.ndarray | None = None  # Máscara binaria (segmentación)
    mask_polygon: np.ndarray | None = None  # Polígono (N, 2)
    mask_center: tuple[int, int] | None = None  # Centroide calculado desde la máscara
    
    @property
    def center(self) -> tuple[int, int]:
        """Centro de la detección (prioriza centroide de máscara si existe)."""
        if self.mask_center is not None:
            return self.mask_center
        return self.bbox.center


@dataclass
class ZoneEntry:
    """Registro de entrada de un vehículo a una zona."""
    vehicle_id: int
    vehicle_type: str
    zone: str
    date: str
    exact_time: str
    timestamp_formatted: str
    # timestamp_seconds eliminado a petición del usuario



@dataclass
class ProcessingResult:
    """Resultado del procesamiento de un video."""
    video_path: str
    output_video_path: str | None
    csv_path: str | None
    total_detections: int
    zone_counts: dict[str, int]
    processing_time_seconds: float
    frames_processed: int
    detection_log: list[ZoneEntry] = field(default_factory=list)
    
    @property
    def processing_time_minutes(self) -> float:
        """Tiempo de procesamiento en minutos."""
        return self.processing_time_seconds / 60


@dataclass 
class TrackHistory:
    """Historial de tracking de un vehículo."""
    track_id: int
    positions: list[tuple[int, int]] = field(default_factory=list)
    zones_visited: list[ZoneEntry] = field(default_factory=list)
    vehicle_type: str | None = None
    
    def add_position(self, x: int, y: int) -> None:
        """Agrega una posición al historial."""
        self.positions.append((x, y))
    
    @property
    def last_position(self) -> tuple[int, int] | None:
        """Última posición registrada."""
        return self.positions[-1] if self.positions else None
    
    @property
    def previous_position(self) -> tuple[int, int] | None:
        """Penúltima posición registrada."""
        return self.positions[-2] if len(self.positions) >= 2 else None
