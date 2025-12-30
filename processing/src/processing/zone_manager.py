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
    
    def __init__(self):
        """Inicializa el ZoneManager."""
        self._zones_original: list[ZoneConfig] = []
        self._zones_scaled: list[ZoneConfig] = []
        self._vehicle_zone_history: dict[int, list[str]] = {}
        self._detection_log: list[ZoneEntry] = []
    
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
        exact_time: str | None = None
    ) -> ZoneCheckResult:
        """
        Verifica si un vehículo entró a una zona nueva.
        
        Args:
            track_id: ID del vehículo
            x, y: Coordenadas del centro
            timestamp_seconds: Tiempo actual en segundos
            vehicle_type: Tipo de vehículo
            exact_time: Hora exacta opcional (formato HH:MM:SS)
            
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
                # Verificar si es primera vez en esta zona
                if zone.label not in self._vehicle_zone_history[track_id]:
                    # Nueva entrada
                    self._vehicle_zone_history[track_id].append(zone.label)
                    entered_zone = zone.label
                    is_new_entry = True
                    
                    # Formatear timestamp
                    minutes = int(timestamp_seconds // 60)
                    secs = int(timestamp_seconds % 60)
                    timestamp_formatted = f"{minutes:02d}:{secs:02d}"
                    
                    # Registrar en log
                    entry = ZoneEntry(
                        vehicle_id=track_id,
                        vehicle_type=vehicle_type,
                        zone=zone.label,
                        timestamp_seconds=timestamp_seconds,
                        timestamp_formatted=timestamp_formatted,
                        exact_time=exact_time
                    )
                    self._detection_log.append(entry)
                    
                    break  # Solo registrar primera zona nueva por update
        
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
