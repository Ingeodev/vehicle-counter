"""
MaskManager - Gestión de máscaras para filtrado de detecciones.
"""

import cv2
import numpy as np

from ..storage.base import StorageReader


class MaskManager:
    """
    Gestiona máscaras para filtrar regiones de interés en el video.
    
    La máscara es una imagen en escala de grises donde:
    - Píxeles blancos (255) = área de interés (se procesan detecciones)
    - Píxeles negros (0) = área ignorada (se descartan detecciones)
    
    Example:
        >>> mask_mgr = MaskManager()
        >>> mask_mgr.load(reader, "mask.png")
        >>> mask_mgr.resize_to_frame(960, 540)
        >>> 
        >>> # Filtrar detecciones
        >>> if mask_mgr.is_point_valid(cx, cy):
        ...     process_detection(det)
    """
    
    def __init__(self):
        """Inicializa el MaskManager."""
        self._mask_original: np.ndarray | None = None
        self._mask_resized: np.ndarray | None = None
        self._current_size: tuple[int, int] | None = None
    
    def load(self, storage: StorageReader, path: str) -> bool:
        """
        Carga una máscara desde archivo.
        
        Args:
            storage: Reader de storage
            path: Ruta a la imagen de máscara
            
        Returns:
            True si se cargó correctamente
        """
        try:
            # Obtener ruta local
            local_path = storage.get_video_path(path)  # Funciona para cualquier archivo
            
            # Cargar como escala de grises
            mask = cv2.imread(local_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"⚠️ No se pudo cargar la máscara: {path}")
                return False
            
            self._mask_original = mask
            self._mask_resized = mask.copy()
            self._current_size = (mask.shape[1], mask.shape[0])
            
            print(f"✅ Máscara cargada: {mask.shape[1]}x{mask.shape[0]}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error al cargar máscara: {e}")
            return False
    
    def load_from_path(self, path: str) -> bool:
        """
        Carga una máscara directamente desde una ruta local.
        
        Args:
            path: Ruta local a la imagen
            
        Returns:
            True si se cargó correctamente
        """
        try:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"⚠️ No se pudo cargar la máscara: {path}")
                return False
            
            self._mask_original = mask
            self._mask_resized = mask.copy()
            self._current_size = (mask.shape[1], mask.shape[0])
            
            print(f"✅ Máscara cargada: {mask.shape[1]}x{mask.shape[0]}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error al cargar máscara: {e}")
            return False
    
    def resize_to_frame(self, width: int, height: int) -> None:
        """
        Redimensiona la máscara al tamaño del frame.
        
        Usa interpolación NEAREST para preservar los bordes binarios.
        
        Args:
            width: Ancho del frame
            height: Alto del frame
        """
        if self._mask_original is None:
            return
        
        # Solo redimensionar si el tamaño es diferente
        if self._current_size == (width, height):
            return
        
        self._mask_resized = cv2.resize(
            self._mask_original,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )
        self._current_size = (width, height)
        
        print(f"🔧 Máscara redimensionada a: {width}x{height}")
    
    def is_point_valid(self, x: int, y: int) -> bool:
        """
        Verifica si un punto está dentro del área válida de la máscara.
        
        Args:
            x, y: Coordenadas del punto
            
        Returns:
            True si el punto está en área blanca (válida)
        """
        if self._mask_resized is None:
            return True  # Sin máscara = todo válido
        
        # Verificar límites
        h, w = self._mask_resized.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        
        # Verificar valor del píxel (> 0 = válido)
        return self._mask_resized[y, x] > 0
    
    def filter_detections(self, detections: list) -> list:
        """
        Filtra detecciones que están fuera de la máscara.
        
        Args:
            detections: Lista de detecciones con atributo .center
            
        Returns:
            Lista filtrada de detecciones válidas
        """
        if self._mask_resized is None:
            return detections
        
        valid = []
        for det in detections:
            cx, cy = det.center
            if self.is_point_valid(cx, cy):
                valid.append(det)
        
        return valid
    
    def apply_to_frame(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Aplica overlay visual de la máscara sobre el frame (para debug).
        
        Args:
            frame: Frame de video
            alpha: Transparencia del overlay
            
        Returns:
            Frame con overlay de máscara
        """
        if self._mask_resized is None:
            return frame
        
        # Crear overlay rojo para áreas ignoradas
        overlay = frame.copy()
        mask_inv = cv2.bitwise_not(self._mask_resized)
        overlay[mask_inv > 0] = [0, 0, 255]  # Rojo para áreas ignoradas
        
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    @property
    def is_loaded(self) -> bool:
        """True si hay una máscara cargada."""
        return self._mask_original is not None
    
    @property
    def original_size(self) -> tuple[int, int] | None:
        """Tamaño original de la máscara (width, height)."""
        if self._mask_original is None:
            return None
        return (self._mask_original.shape[1], self._mask_original.shape[0])
    
    @property
    def current_size(self) -> tuple[int, int] | None:
        """Tamaño actual de la máscara redimensionada."""
        return self._current_size
    
    def __bool__(self) -> bool:
        """True si hay máscara cargada."""
        return self.is_loaded
