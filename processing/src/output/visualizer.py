"""
Visualizer - Dibujo de anotaciones en frames.
"""

from typing import Sequence, Callable, Any

import cv2
import numpy as np

from ..data.schemas import Detection, ZoneConfig


# Colores predefinidos para zonas
ZONE_COLORS = [
    (255, 0, 0),    # Azul
    (0, 255, 0),    # Verde
    (0, 0, 255),    # Rojo
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Amarillo
    (128, 0, 128),  # Púrpura
    (255, 128, 0),  # Naranja
    (128, 255, 0),  # Lima
]


class Visualizer:
    """
    Dibuja anotaciones en frames de video.
    
    Example:
        >>> viz = Visualizer()
        >>> viz.draw_zones(frame, zones)
        >>> viz.draw_detections(frame, detections, tracker)
        >>> viz.draw_stats(frame, {"Total": 10})
    """
    
    def __init__(
        self,
        box_color: tuple[int, int, int] = (0, 255, 0),
        box_color_with_zone: tuple[int, int, int] = (0, 255, 0),
        box_color_no_zone: tuple[int, int, int] = (255, 0, 0),
        trajectory_color: tuple[int, int, int] = (255, 0, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 2,
        font_scale: float = 0.5
    ):
        """
        Inicializa el visualizador.
        
        Args:
            box_color: Color por defecto para cajas
            box_color_with_zone: Color para vehículos que han entrado a zonas
            box_color_no_zone: Color para vehículos sin zona visitada
            trajectory_color: Color para trayectorias
            text_color: Color para textos
            line_thickness: Grosor de líneas
            font_scale: Escala de fuente
        """
        self.box_color = box_color
        self.box_color_with_zone = box_color_with_zone
        self.box_color_no_zone = box_color_no_zone
        self.trajectory_color = trajectory_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_zones(
        self,
        frame: np.ndarray,
        zones: Sequence[ZoneConfig],
        alpha: float = 0.3,
        draw_labels: bool = True
    ) -> np.ndarray:
        """
        Dibuja las zonas de interés en el frame.
        
        Args:
            frame: Frame de video (se modifica in-place)
            zones: Lista de zonas
            alpha: Transparencia del relleno
            draw_labels: Si dibujar las etiquetas
            
        Returns:
            Frame con zonas dibujadas
        """
        overlay = frame.copy()
        
        for i, zone in enumerate(zones):
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            
            # Dibujar polígono relleno
            cv2.fillPoly(overlay, [zone.points], color)
            
            # Dibujar contorno
            cv2.polylines(frame, [zone.points], True, color, self.line_thickness)
            
            # Dibujar etiqueta
            if draw_labels:
                centroid = zone.points.mean(axis=0).astype(int)
                cv2.putText(
                    frame, 
                    zone.label, 
                    tuple(centroid),
                    self.font, 
                    1.0, 
                    (255, 255, 255), 
                    2
                )
        
        # Mezclar overlay con frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        zones_visited: list[str] | None = None,
        trajectory: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Dibuja una detección individual.
        
        Args:
            frame: Frame de video
            detection: Detección a dibujar
            zones_visited: Zonas visitadas por el vehículo
            trajectory: Array de trayectoria para dibujar
            
        Returns:
            Frame con detección dibujada
        """
        bbox = detection.bbox
        has_zones = zones_visited and len(zones_visited) > 0
        
        # Color según estado de zonas
        color = self.box_color_with_zone if has_zones else self.box_color_no_zone
        
        # Dibujar polígono si existe (segmentación)
        if detection.mask_polygon is not None:
            # Crear overlay para transparencia
            overlay = frame.copy()
            cv2.fillPoly(overlay, [detection.mask_polygon], color)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Borde del polígono
            cv2.polylines(frame, [detection.mask_polygon], True, color, 2)
            
            # Caja más sutil
            cv2.rectangle(
                frame,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                color,
                1
            )
        else:
            # Dibujar caja normal
            cv2.rectangle(
                frame,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                color,
                self.line_thickness
            )
        
        # Dibujar centro
        cx, cy = detection.center
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        
        # Etiqueta con ID y tipo
        label = f"{detection.class_name}-{detection.track_id}"
        cv2.putText(
            frame,
            label,
            (bbox.x1, bbox.y1 - 5),
            self.font,
            self.font_scale,
            self.text_color,
            2
        )
        
        # Mostrar zonas visitadas
        if has_zones:
            zones_text = f"Zonas: {','.join(zones_visited)}"
            cv2.putText(
                frame,
                zones_text,
                (bbox.x1, bbox.y2 + 15),
                self.font,
                self.font_scale * 0.8,
                self.box_color_with_zone,
                1
            )
        
        # Dibujar trayectoria
        if trajectory is not None and len(trajectory) > 0:
            cv2.polylines(
                frame,
                [trajectory],
                False,
                self.trajectory_color,
                self.line_thickness
            )
        
        return frame
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        get_zones: Callable[[int], list[str] | None] | None = None,
        get_trajectory: Callable[[int], np.ndarray | None] | None = None
    ) -> np.ndarray:
        """
        Dibuja múltiples detecciones.
        
        Args:
            frame: Frame de video
            detections: Lista de detecciones
            get_zones: Función que recibe track_id y retorna zonas visitadas
            get_trajectory: Función que recibe track_id y retorna trayectoria
            
        Returns:
            Frame con detecciones dibujadas
        """
        for det in detections:
            zones = get_zones(det.track_id) if get_zones else None
            trajectory = get_trajectory(det.track_id) if get_trajectory else None
            
            self.draw_detection(frame, det, zones, trajectory)
        
        return frame
    
    def draw_stats(
        self,
        frame: np.ndarray,
        stats: dict[str, int | str],
        position: tuple[int, int] = (10, 30),
        line_height: int = 30
    ) -> np.ndarray:
        """
        Dibuja estadísticas en el frame.
        
        Args:
            frame: Frame de video
            stats: Diccionario de estadísticas {label: value}
            position: Posición inicial (x, y)
            line_height: Altura entre líneas
            
        Returns:
            Frame con estadísticas
        """
        x, y = position
        
        for label, value in stats.items():
            text = f"{label}: {value}"
            cv2.putText(
                frame,
                text,
                (x, y),
                self.font,
                0.7,
                self.text_color,
                2
            )
            y += line_height
        
        return frame
    
    def draw_timestamp(
        self,
        frame: np.ndarray,
        timestamp: str,
        position: tuple[int, int] | None = None
    ) -> np.ndarray:
        """
        Dibuja el timestamp en el frame.
        
        Args:
            frame: Frame de video
            timestamp: Texto del timestamp
            position: Posición (x, y), si None usa esquina superior derecha
            
        Returns:
            Frame con timestamp
        """
        if position is None:
            h, w = frame.shape[:2]
            position = (w - 150, 30)
        
        cv2.putText(
            frame,
            timestamp,
            position,
            self.font,
            0.7,
            self.text_color,
            2
        )
        
        return frame
