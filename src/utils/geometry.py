"""
geometry - Utilidades geométricas.
"""

import cv2
import numpy as np


def point_in_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
    """
    Verifica si un punto está dentro de un polígono.
    
    Args:
        point: Coordenadas (x, y)
        polygon: Array numpy de puntos del polígono, forma (N, 2)
        
    Returns:
        True si el punto está dentro del polígono
        
    Example:
        >>> polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        >>> point_in_polygon((50, 50), polygon)
        True
    """
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def scale_points(
    points: np.ndarray,
    scale_x: float,
    scale_y: float
) -> np.ndarray:
    """
    Escala coordenadas de puntos.
    
    Args:
        points: Array numpy de puntos, forma (N, 2)
        scale_x: Factor de escala en X
        scale_y: Factor de escala en Y
        
    Returns:
        Array numpy de puntos escalados
        
    Example:
        >>> points = np.array([[100, 200], [300, 400]])
        >>> scale_points(points, 0.5, 0.5)
        array([[50, 100], [150, 200]])
    """
    scaled = points.copy().astype(float)
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled.astype(np.int32)


def calculate_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """
    Calcula el centro de un rectángulo.
    
    Args:
        x1, y1: Esquina superior izquierda
        x2, y2: Esquina inferior derecha
        
    Returns:
        Tupla (cx, cy) con el centro
    """
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def line_intersection(
    line1_start: tuple[int, int],
    line1_end: tuple[int, int],
    line2_start: tuple[int, int],
    line2_end: tuple[int, int]
) -> tuple[float, float] | None:
    """
    Calcula el punto de intersección de dos líneas.
    
    Args:
        line1_start, line1_end: Puntos de la primera línea
        line2_start, line2_end: Puntos de la segunda línea
        
    Returns:
        Punto de intersección (x, y) o None si son paralelas
    """
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None  # Líneas paralelas
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def crossed_line(
    prev_y: int,
    curr_y: int,
    line_y: int
) -> str | None:
    """
    Detecta si un punto cruzó una línea horizontal.
    
    Args:
        prev_y: Posición Y anterior
        curr_y: Posición Y actual
        line_y: Posición Y de la línea
        
    Returns:
        'down' si cruzó hacia abajo, 'up' si cruzó hacia arriba, None si no cruzó
    """
    if prev_y < line_y <= curr_y:
        return "down"
    elif prev_y > line_y >= curr_y:
        return "up"
    return None


def polygon_area(points: np.ndarray) -> float:
    """
    Calcula el área de un polígono usando la fórmula del shoelace.
    
    Args:
        points: Array numpy de puntos, forma (N, 2)
        
    Returns:
        Área del polígono
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def polygon_centroid(points: np.ndarray) -> tuple[int, int]:
    """
    Calcula el centroide de un polígono.
    
    Args:
        points: Array numpy de puntos, forma (N, 2)
        
    Returns:
        Tupla (x, y) del centroide
    """
    centroid = points.mean(axis=0).astype(int)
    return (int(centroid[0]), int(centroid[1]))
