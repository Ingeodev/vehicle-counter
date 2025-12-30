"""
VehicleTracker - Gestión del historial de tracking.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..data.schemas import Detection, TrackHistory


class VehicleTracker:
    """
    Gestiona el historial de posiciones de vehículos.
    
    Mantiene un registro de las posiciones de cada vehículo trackeado
    para análisis de trayectorias y detección de cruces.
    
    Example:
        >>> tracker = VehicleTracker(max_history=30)
        >>> 
        >>> for detection in detections:
        ...     tracker.update(detection)
        ...     trajectory = tracker.get_trajectory(detection.track_id)
        ...     if trajectory:
        ...         draw_trajectory(frame, trajectory)
    """
    
    def __init__(self, max_history: int = 50):
        """
        Inicializa el tracker.
        
        Args:
            max_history: Número máximo de posiciones a mantener por vehículo
        """
        self.max_history = max_history
        self._tracks: dict[int, TrackHistory] = {}
    
    def update(self, detection: Detection) -> TrackHistory:
        """
        Actualiza el historial con una nueva detección.
        
        Args:
            detection: Detección a registrar
            
        Returns:
            TrackHistory actualizado del vehículo
        """
        track_id = detection.track_id
        cx, cy = detection.center
        
        # Crear nuevo historial si no existe
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackHistory(
                track_id=track_id,
                vehicle_type=detection.class_name
            )
        
        track = self._tracks[track_id]
        
        # Agregar posición
        track.add_position(cx, cy)
        
        # Limitar historial
        if len(track.positions) > self.max_history:
            track.positions = track.positions[-self.max_history:]
        
        # Actualizar tipo de vehículo si no estaba definido
        if track.vehicle_type is None:
            track.vehicle_type = detection.class_name
        
        return track
    
    def update_batch(self, detections: list[Detection]) -> None:
        """
        Actualiza el historial con múltiples detecciones.
        
        Args:
            detections: Lista de detecciones a registrar
        """
        for detection in detections:
            self.update(detection)
    
    def get_track(self, track_id: int) -> TrackHistory | None:
        """
        Obtiene el historial de un vehículo.
        
        Args:
            track_id: ID del vehículo
            
        Returns:
            TrackHistory o None si no existe
        """
        return self._tracks.get(track_id)
    
    def get_trajectory(self, track_id: int, last_n: int | None = None) -> list[tuple[int, int]]:
        """
        Obtiene la trayectoria de un vehículo.
        
        Args:
            track_id: ID del vehículo
            last_n: Número de últimos puntos a retornar (None = todos)
            
        Returns:
            Lista de posiciones (x, y)
        """
        track = self._tracks.get(track_id)
        if track is None:
            return []
        
        positions = track.positions
        if last_n is not None and len(positions) > last_n:
            return positions[-last_n:]
        return positions
    
    def get_trajectory_array(self, track_id: int, last_n: int | None = None) -> np.ndarray:
        """
        Obtiene la trayectoria como array numpy para dibujo.
        
        Args:
            track_id: ID del vehículo
            last_n: Número de últimos puntos
            
        Returns:
            Array numpy de forma (N, 1, 2) para cv2.polylines
        """
        positions = self.get_trajectory(track_id, last_n)
        if not positions:
            return np.array([])
        return np.array(positions, dtype=np.int32).reshape((-1, 1, 2))
    
    def has_previous_position(self, track_id: int) -> bool:
        """
        Verifica si el vehículo tiene al menos 2 posiciones registradas.
        
        Útil para detectar cruces de líneas.
        """
        track = self._tracks.get(track_id)
        return track is not None and len(track.positions) >= 2
    
    def get_movement(self, track_id: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """
        Obtiene el movimiento reciente (posición anterior y actual).
        
        Args:
            track_id: ID del vehículo
            
        Returns:
            Tupla ((prev_x, prev_y), (curr_x, curr_y)) o None
        """
        track = self._tracks.get(track_id)
        if track is None or len(track.positions) < 2:
            return None
        
        return (track.positions[-2], track.positions[-1])
    
    def clear(self) -> None:
        """Limpia todo el historial de tracking."""
        self._tracks.clear()
    
    def remove_track(self, track_id: int) -> None:
        """Elimina el historial de un vehículo."""
        self._tracks.pop(track_id, None)
    
    def get_active_tracks(self) -> list[int]:
        """Retorna los IDs de todos los tracks activos."""
        return list(self._tracks.keys())
    
    def __len__(self) -> int:
        """Número de tracks activos."""
        return len(self._tracks)
    
    def __contains__(self, track_id: int) -> bool:
        """Verifica si un track existe."""
        return track_id in self._tracks
    
    def __iter__(self) -> Iterator[TrackHistory]:
        """Itera sobre todos los tracks."""
        return iter(self._tracks.values())
