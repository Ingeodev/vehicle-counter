"""
AnnotatedVideoWriter - Escritura de video con anotaciones.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Iterator

import cv2
import numpy as np

from ..storage.base import StorageWriter


@dataclass
class VideoWriterConfig:
    """Configuración del video writer."""
    fps: float = 30.0
    codec: str = "mp4v"
    width: int = 1920
    height: int = 1080


class AnnotatedVideoWriter:
    """
    Escribe videos con anotaciones usando OpenCV.
    
    Example:
        >>> config = VideoWriterConfig(fps=30, width=960, height=540)
        >>> with AnnotatedVideoWriter.open(writer, "output.mp4", config) as vw:
        ...     for frame in frames:
        ...         vw.write(annotated_frame)
    """
    
    def __init__(
        self,
        cv_writer: cv2.VideoWriter,
        output_path: str,
        storage: StorageWriter | None = None,
        temp_path: str | None = None
    ):
        """
        Inicializa el writer.
        
        Args:
            cv_writer: VideoWriter de OpenCV
            output_path: Ruta final del video
            storage: Writer de storage (opcional)
            temp_path: Ruta temporal si es diferente de output_path
        """
        self._writer = cv_writer
        self._output_path = output_path
        self._storage = storage
        self._temp_path = temp_path
        self._frame_count = 0
    
    @classmethod
    @contextmanager
    def open(
        cls,
        storage: StorageWriter,
        path: str,
        config: VideoWriterConfig
    ) -> Iterator["AnnotatedVideoWriter"]:
        """
        Abre un video writer como context manager.
        
        Args:
            storage: Writer de storage
            path: Ruta del video de salida
            config: Configuración del video
            
        Yields:
            AnnotatedVideoWriter configurado
        """
        # Obtener ruta para escribir
        write_path = storage.get_video_writer_path(path)
        
        # Crear VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*config.codec)
        writer = cv2.VideoWriter(
            write_path,
            fourcc,
            config.fps,
            (config.width, config.height)
        )
        
        if not writer.isOpened():
            raise ValueError(f"No se pudo crear VideoWriter: {path}")
        
        video_writer = cls(writer, path, storage, write_path)
        
        try:
            yield video_writer
        finally:
            writer.release()
            
            # Finalizar si es necesario (upload para storage remoto)
            if write_path != path:
                storage.finalize_video(write_path, path)
    
    @classmethod
    def from_path(
        cls,
        path: str,
        config: VideoWriterConfig
    ) -> "AnnotatedVideoWriter":
        """
        Crea un writer directamente desde una ruta local.
        
        Nota: El caller es responsable de llamar close().
        
        Args:
            path: Ruta del video
            config: Configuración
            
        Returns:
            AnnotatedVideoWriter
        """
        fourcc = cv2.VideoWriter_fourcc(*config.codec)
        writer = cv2.VideoWriter(
            path,
            fourcc,
            config.fps,
            (config.width, config.height)
        )
        
        if not writer.isOpened():
            raise ValueError(f"No se pudo crear VideoWriter: {path}")
        
        return cls(writer, path)
    
    def write(self, frame: np.ndarray) -> None:
        """
        Escribe un frame al video.
        
        Args:
            frame: Frame a escribir (BGR)
        """
        self._writer.write(frame)
        self._frame_count += 1
    
    def close(self) -> None:
        """Cierra el video writer y libera recursos."""
        self._writer.release()
        
        if self._storage and self._temp_path and self._temp_path != self._output_path:
            self._storage.finalize_video(self._temp_path, self._output_path)
    
    @property
    def frame_count(self) -> int:
        """Número de frames escritos."""
        return self._frame_count
    
    @property
    def output_path(self) -> str:
        """Ruta del video de salida."""
        return self._output_path
