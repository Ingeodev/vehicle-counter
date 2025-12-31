"""
ZoneManager - Gestión de zonas de interés para conteo.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..data.schemas import ZoneConfig, ZoneEntry
from ..storage.base import StorageReader
from .zone_deduplicator import ZoneDeduplicator


@dataclass
class ZoneCheckResult:
    """Resultado de verificar entrada a zonas."""
    entered_zone: str | None  # Label de la zona si entró a una nueva
    is_new_entry: bool  # True si es primera vez en esta zona
    all_zones: list[str]  # Todas las zonas donde ha estado el vehículo


class ZoneManager:
    """
    Gestiona zonas de interés para conteo de vehículos.
    
    Carga zonas desde archivos JSON, escala coordenadas según resolución,
    y verifica si los vehículos entran a las zonas.
    
    Example:
        >>> zones = ZoneManager()
        >>> zones.load_from_json(reader, "zones.json")
        >>> zones.scale_to_resolution(1920, 1080, 960, 540)
        >>> 
        >>> result = zones.check_entry(track_id=1, x=100, y=200, 
        ...                            timestamp=1.5, vehicle_type="car")
        >>> if result.is_new_entry:
        ...     print(f"Vehicle entered zone: {result.entered_zone}")
    """
    
    def __init__(self, enable_deduplication: bool = True, dedup_spatial_threshold: float = 150.0, dedup_temporal_threshold: float = 5.0):
        """
        Inicializa el ZoneManager.
        
        Args:
            enable_deduplication: Si activar detección de duplicados por proximidad
            dedup_spatial_threshold: Distancia máxima para considerar duplicado (píxeles)
            dedup_temporal_threshold: Tiempo máximo para considerar duplicado (segundos)
        """
        self._zones_original: list[ZoneConfig] = []
        self._zones_scaled: list[ZoneConfig] = []
        self._vehicle_zone_history: dict[int, list[str]] = {}
        self._detection_log: list[ZoneEntry] = []
        
        # Deduplicador de zonas
        self._enable_dedup = enable_deduplication
        self._zone_dedup = ZoneDeduplicator(
            spatial_threshold=dedup_spatial_threshold,
            temporal_threshold=dedup_temporal_threshold,
            same_type_only=False  # Detectar duplicados incluso si el tipo cambia
        ) if enable_deduplication else None
        
        self._frame_num = 0  # Contador de frames para deduplicador
    
    def load_from_json(self, storage: StorageReader, path: str) -> int:
        """
        Carga zonas desde un archivo JSON.
        
        El JSON debe tener formato:
        {
            "zones": [
                {"label": "A", "points": [[x1,y1], [x2,y2], ...]},
                ...
            ]
        }
        
        Args:
            storage: Reader de storage
            path: Ruta al archivo JSON
            
        Returns:
            Número de zonas cargadas
        """
        data = storage.read_json(path)
        zones_data = data.get("zones", [])
        
        self._zones_original = [
            ZoneConfig.from_dict(z) for z in zones_data
        ]
        self._zones_scaled = self._zones_original.copy()
        
        return len(self._zones_original)
    
    def load_from_dict(self, data: dict[str, Any]) -> int:
        """
        Carga zonas desde un diccionario.
        
        Args:
            data: Diccionario con zonas
            
        Returns:
            Número de zonas cargadas
        """
        zones_data = data.get("zones", [])
        
        self._zones_original = [
            ZoneConfig.from_dict(z) for z in zones_data
        ]
        self._zones_scaled = self._zones_original.copy()
        
        return len(self._zones_original)
    
    def scale_to_resolution(
        self, 
        original_width: int, 
        original_height: int,
        new_width: int, 
        new_height: int
    ) -> None:
        """
        Escala las coordenadas de las zonas a una nueva resolución.
        
        Args:
            original_width, original_height: Resolución original
            new_width, new_height: Nueva resolución
        """
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        
        self._zones_scaled = [
            zone.scale(scale_x, scale_y)
            for zone in self._zones_original
        ]
    
    def point_in_zone(self, point: tuple[int, int], zone: ZoneConfig) -> bool:
        """
        Verifica si un punto está dentro de una zona.
        
        Args:
            point: Coordenadas (x, y)
            zone: Configuración de la zona
            
        Returns:
            True si el punto está dentro
        """
        # OpenCV requiere tupla de floats
        pt = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(zone.points, pt, False) >= 0
    
    def check_entry(
        self,
        track_id: int,
        x: int,
        y: int,
        timestamp_seconds: float,
        vehicle_type: str,
        date: str,
        exact_time: str
    ) -> ZoneCheckResult:
        """
        Verifica si un vehículo entró a una zona nueva.
        
        Args:
            track_id: ID del vehículo
            x, y: Coordenadas del centro
            timestamp_seconds: Tiempo actual en segundos (para deduplicación)
            vehicle_type: Tipo de vehículo
            date: Fecha del video (YYYY-MM-DD)
            exact_time: Hora exacta (HH:MM:SS)
            
        Returns:
            ZoneCheckResult con información de la entrada
        """
        point = (x, y)
        
        # Inicializar historial si es nuevo
        if track_id not in self._vehicle_zone_history:
            self._vehicle_zone_history[track_id] = []
        
        entered_zone = None
        is_new_entry = False
        
        # Verificar cada zona
        for zone in self._zones_scaled:
            if self.point_in_zone(point, zone):
                # Verificar si es primera vez en esta zona para este track_id
                if zone.label not in self._vehicle_zone_history[track_id]:
                    
                    # DEDUPLICACIÓN: Verificar si hay otro vehículo cercano que ya entró
                    if self._enable_dedup and self._zone_dedup:
                        is_valid = self._zone_dedup.should_count_entry(
                            track_id=track_id,
                            zone=zone.label,
                            position=point,
                            timestamp=timestamp_seconds,
                            vehicle_type=vehicle_type,
                            frame_num=self._frame_num
                        )
                        if not is_valid:
                            # Es duplicado - marcar como visitado para no re-intentar
                            # pero NO registrar en log
                            self._vehicle_zone_history[track_id].append(zone.label)
                            continue  # Revisar siguiente zona
                    
                    # Nueva entrada válida
                    self._vehicle_zone_history[track_id].append(zone.label)
                    entered_zone = zone.label
                    is_new_entry = True
                    
                    # Formatear timestamp relativo (info extra)
                    minutes = int(timestamp_seconds // 60)
                    secs = int(timestamp_seconds % 60)
                    timestamp_formatted = f"{minutes:02d}:{secs:02d}"
                    
                    # Registrar en log
                    entry = ZoneEntry(
                        vehicle_id=track_id,
                        vehicle_type=vehicle_type,
                        zone=zone.label,
                        date=date,
                        exact_time=exact_time,
                        timestamp_formatted=timestamp_formatted
                    )
                    self._detection_log.append(entry)
                    
                    break  # Solo registrar primera zona nueva por update
        
        self._frame_num += 1
        
        return ZoneCheckResult(
            entered_zone=entered_zone,
            is_new_entry=is_new_entry,
            all_zones=self._vehicle_zone_history.get(track_id, []).copy()
        )
    
    def get_zones_for_vehicle(self, track_id: int) -> list[str]:
        """
        Obtiene las zonas visitadas por un vehículo.
        
        Args:
            track_id: ID del vehículo
            
        Returns:
            Lista de labels de zonas visitadas
        """
        return self._vehicle_zone_history.get(track_id, []).copy()
    
    def get_detection_log(self) -> list[ZoneEntry]:
        """Retorna el log de todas las detecciones."""
        return self._detection_log.copy()
        
    def get_log_count(self) -> int:
        """Retorna cantidad de detecciones en log."""
        return len(self._detection_log)
    
    def get_zone_counts(self) -> dict[str, int]:
        """
        Obtiene el conteo de entradas por zona.
        
        Returns:
            Diccionario {label: count}
        """
        counts: dict[str, int] = {}
        for entry in self._detection_log:
            counts[entry.zone] = counts.get(entry.zone, 0) + 1
        return counts
    
    def clear_history(self) -> None:
        """Limpia el historial de zonas y detecciones."""
        self._vehicle_zone_history.clear()
        self._detection_log.clear()
    
    @property
    def zones(self) -> list[ZoneConfig]:
        """Retorna las zonas escaladas."""
        return self._zones_scaled
    
    @property
    def zone_labels(self) -> list[str]:
        """Retorna los labels de todas las zonas."""
        return [z.label for z in self._zones_scaled]
    
    def __len__(self) -> int:
        """Número de zonas."""
        return len(self._zones_scaled)
    
    def __bool__(self) -> bool:
        """True si hay zonas definidas."""
        return len(self._zones_scaled) > 0
