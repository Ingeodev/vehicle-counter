"""
VideoSource - Abstracción para lectura de video con OpenCV.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Iterator, ContextManager
from contextlib import contextmanager

from ..storage.base import StorageReader


@dataclass
class FrameData:
    """Datos de un frame leído."""

    frame: np.ndarray
    index: int
    timestamp_seconds: float


class VideoSource:
    """
    Fuente de video que abstrae la lectura con OpenCV.

    Soporta diferentes backends de storage mediante la interfaz StorageReader.

    Example:
        >>> from mglon_vehicle_counter.storage import LocalStorageReader
        >>> reader = LocalStorageReader()
        >>>
        >>> with VideoSource.open(reader, "/path/to/video.mp4") as video:
        ...     print(f"Resolution: {video.width}x{video.height}")
        ...     print(f"FPS: {video.fps}")
        ...     for frame_data in video.frames():
        ...         process(frame_data.frame)
    """

    def __init__(self, cap: cv2.VideoCapture, video_path: str):
        """
        Inicializa el VideoSource.

        Args:
            cap: Objeto VideoCapture de OpenCV
            video_path: Ruta del video (para referencia)
        """
        self._cap = cap
        self._video_path = video_path
        self._frame_index = 0

        # Extraer metadatos
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @classmethod
    @contextmanager
    def open(cls, storage: StorageReader, path: str) -> Iterator["VideoSource"]:
        """
        Abre un video como context manager.

        Args:
            storage: Reader de storage para obtener la ruta del video
            path: Ruta del video en el storage

        Yields:
            VideoSource configurado

        Example:
            >>> with VideoSource.open(reader, "video.mp4") as video:
            ...     for frame in video.frames():
            ...         process(frame)
        """
        # Obtener ruta accesible para OpenCV
        video_path = storage.get_video_path(path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {path}")

        try:
            yield cls(cap, video_path)
        finally:
            cap.release()

    @classmethod
    def from_path(cls, path: str) -> "VideoSource":
        """
        Abre un video directamente desde una ruta local.

        Nota: El caller es responsable de llamar close().
        Para uso con context manager, usar open().

        Args:
            path: Ruta local del video

        Returns:
            VideoSource configurado
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {path}")
        return cls(cap, path)

    def close(self) -> None:
        """Cierra el video y libera recursos."""
        if self._cap.isOpened():
            self._cap.release()

    @property
    def fps(self) -> float:
        """Frames por segundo del video."""
        return self._fps

    @property
    def width(self) -> int:
        """Ancho del video en píxeles."""
        return self._width

    @property
    def height(self) -> int:
        """Alto del video en píxeles."""
        return self._height

    @property
    def total_frames(self) -> int:
        """Número total de frames en el video."""
        return self._total_frames

    @property
    def duration_seconds(self) -> float:
        """Duración del video en segundos."""
        return self._total_frames / self._fps if self._fps > 0 else 0

    @property
    def current_frame_index(self) -> int:
        """Índice del frame actual."""
        return self._frame_index

    @property
    def current_timestamp(self) -> float:
        """Timestamp actual en segundos."""
        return self._frame_index / self._fps if self._fps > 0 else 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Lee el siguiente frame.

        Returns:
            Tupla (success, frame) donde frame es None si no hay más frames
        """
        ret, frame = self._cap.read()
        if ret:
            self._frame_index += 1
        return ret, frame if ret else None

    def read_resized(self, width: int, height: int) -> tuple[bool, np.ndarray | None]:
        """
        Lee el siguiente frame y lo redimensiona.

        Args:
            width: Nuevo ancho
            height: Nueva altura

        Returns:
            Tupla (success, frame_resized)
        """
        ret, frame = self.read()
        if ret and frame is not None:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return ret, frame

    def frames(
        self,
        max_frames: int | None = None,
        resize: tuple[int, int] | None = None,
        skip_rate: int = 1,
    ) -> Iterator[FrameData]:
        """
        Generador que itera sobre los frames del video.

        Args:
            max_frames: Número máximo de frames a leer (None = todos)
            resize: Tupla (width, height) para redimensionar (None = original)
            skip_rate: Leer cada N frames (1 = todos, 2 = cada 2, etc.)

        Yields:
            FrameData con el frame y metadatos

        Example:
            >>> for frame_data in video.frames(max_frames=1000, resize=(640, 480)):
            ...     process(frame_data.frame)
            ...     print(f"Frame {frame_data.index}, time: {frame_data.timestamp_seconds:.2f}s")
        """
        frames_read = 0

        while True:
            # Verificar límite de frames
            if max_frames is not None and frames_read >= max_frames:
                break

            # Aplicar skip rate: usar grab() para saltar sin decodificar
            if skip_rate > 1 and frames_read > 0:
                for _ in range(skip_rate - 1):
                    if not self._cap.grab():
                        break
                    self._frame_index += 1

            # Leer frame (solo cuando realmente vamos a procesarlo)
            ret, frame = self._cap.read()
            if not ret:
                break

            self._frame_index += 1

            # Redimensionar si es necesario
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

            frames_read += 1

            yield FrameData(
                frame=frame,
                index=self._frame_index - 1,
                timestamp_seconds=self.current_timestamp,
            )

    def seek(self, frame_index: int) -> bool:
        """
        Salta a un frame específico.

        Args:
            frame_index: Índice del frame al que saltar

        Returns:
            True si tuvo éxito
        """
        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if success:
            self._frame_index = frame_index
        return success

    def get_sample_frame(
        self, resize: tuple[int, int] | None = None
    ) -> np.ndarray | None:
        """
        Obtiene un frame de muestra (el primero) y vuelve al inicio.

        Args:
            resize: Tupla (width, height) opcional para redimensionar

        Returns:
            Frame de muestra o None si no se pudo leer
        """
        # Guardar posición actual
        current_pos = self._frame_index

        # Ir al inicio
        self.seek(0)

        # Leer frame
        ret, frame = self.read()

        # Volver a la posición original
        self.seek(current_pos)

        if ret and frame is not None and resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

        return frame
