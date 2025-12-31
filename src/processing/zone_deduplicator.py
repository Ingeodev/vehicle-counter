"""
ZoneDeduplicator - Deduplicación de entradas a zonas por proximidad espacial.

Detecta cuando múltiples IDs de tracking corresponden al mismo vehículo físico
analizando la proximidad espacial y temporal de las entradas a zonas.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ZoneEntryRecord:
    """Registro de una entrada a zona para deduplicación."""
    track_id: int
    vehicle_type: str
    zone: str
    position: tuple[int, int]
    timestamp: float
    frame_num: int
    is_duplicate: bool = False


class ZoneDeduplicator:
    """
    Detecta entradas duplicadas a zonas basándose en proximidad espacial y temporal.
    
    Cuando un vehículo entra a una zona pero su ID cambia (fragmentación del tracker),
    este sistema detecta que la nueva entrada está muy cerca espacial y temporalmente
    de una entrada reciente y la marca como duplicado.
    
    Example:
        >>> dedup = ZoneDeduplicator(spatial_threshold=100, temporal_threshold=3.0)
        >>> 
        >>> # Para cada posible entrada a zona
        >>> is_valid = dedup.should_count_entry(
        ...     track_id=5, zone="B", position=(500, 300),
        ...     timestamp=3.5, vehicle_type="bus", frame_num=105
        ... )
        >>> if is_valid:
        ...     count_vehicle()
    """
    
    def __init__(
        self,
        spatial_threshold: float = 150.0,
        temporal_threshold: float = 5.0,
        same_type_only: bool = True
    ):
        """
        Inicializa el deduplicador de zonas.
        
        Args:
            spatial_threshold: Distancia máxima para considerar mismo vehículo (píxeles)
            temporal_threshold: Tiempo máximo entre entradas para considerar duplicado (segundos)
            same_type_only: Si solo considerar duplicados del mismo tipo de vehículo
        """
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.same_type_only = same_type_only
        
        # Historial de entradas por zona
        self._zone_entries: dict[str, list[ZoneEntryRecord]] = {}
        
        # Mapeo de IDs que ya fueron contados como duplicados
        self._duplicate_ids: set[int] = set()
        
        # Estadísticas
        self._stats = {
            "total_attempts": 0,
            "duplicates_detected": 0,
            "valid_entries": 0
        }
    
    def should_count_entry(
        self,
        track_id: int,
        zone: str,
        position: tuple[int, int],
        timestamp: float,
        vehicle_type: str,
        frame_num: int
    ) -> bool:
        """
        Determina si una entrada a zona debe contarse o es duplicado.
        
        Args:
            track_id: ID del tracker
            zone: Label de la zona
            position: Posición (x, y) del centro del vehículo
            timestamp: Tiempo en segundos
            vehicle_type: Tipo de vehículo
            frame_num: Número de frame
            
        Returns:
            True si debe contarse, False si es duplicado
        """
        self._stats["total_attempts"] += 1
        
        # Inicializar lista de la zona si no existe
        if zone not in self._zone_entries:
            self._zone_entries[zone] = []
        
        # Buscar entradas recientes en la misma zona que sean muy cercanas
        is_duplicate = self._check_duplicate(
            zone, position, timestamp, vehicle_type
        )
        
        # Registrar la entrada
        entry = ZoneEntryRecord(
            track_id=track_id,
            vehicle_type=vehicle_type,
            zone=zone,
            position=position,
            timestamp=timestamp,
            frame_num=frame_num,
            is_duplicate=is_duplicate
        )
        self._zone_entries[zone].append(entry)
        
        if is_duplicate:
            self._stats["duplicates_detected"] += 1
            self._duplicate_ids.add(track_id)
            return False
        
        self._stats["valid_entries"] += 1
        return True
    
    def _check_duplicate(
        self,
        zone: str,
        position: tuple[int, int],
        timestamp: float,
        vehicle_type: str
    ) -> bool:
        """
        Verifica si hay una entrada reciente cercana (posible duplicado).
        """
        if zone not in self._zone_entries:
            return False
        
        recent_entries = self._zone_entries[zone]
        
        for entry in reversed(recent_entries):
            # Verificar ventana temporal
            time_diff = timestamp - entry.timestamp
            if time_diff > self.temporal_threshold:
                break  # Las entradas están ordenadas por tiempo
            
            if time_diff < 0:
                continue  # Entrada futura (no debería pasar)
            
            # Verificar tipo si es requerido
            if self.same_type_only and entry.vehicle_type != vehicle_type:
                continue
            
            # Calcular distancia espacial
            dist = self._distance(position, entry.position)
            
            # Si está muy cerca, es duplicado
            if dist < self.spatial_threshold:
                return True
        
        return False
    
    def _distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Calcula distancia euclidiana."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_stats(self) -> dict[str, Any]:
        """Retorna estadísticas de deduplicación."""
        return self._stats.copy()
    
    def get_duplicate_ids(self) -> set[int]:
        """Retorna IDs marcados como duplicados."""
        return self._duplicate_ids.copy()
    
    def clear(self) -> None:
        """Limpia todo el estado."""
        self._zone_entries.clear()
        self._duplicate_ids.clear()
        self._stats = {
            "total_attempts": 0,
            "duplicates_detected": 0,
            "valid_entries": 0
        }
    
    def cleanup_old_entries(self, current_timestamp: float) -> None:
        """
        Limpia entradas muy antiguas para liberar memoria.
        
        Args:
            current_timestamp: Tiempo actual en segundos
        """
        cutoff = current_timestamp - (self.temporal_threshold * 3)
        
        for zone in self._zone_entries:
            self._zone_entries[zone] = [
                e for e in self._zone_entries[zone]
                if e.timestamp > cutoff
            ]
