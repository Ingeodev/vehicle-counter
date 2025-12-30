"""
VehicleCounter - Orquestador de detección, tracking y conteo.
"""

import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..data.schemas import Detection, VideoInfo, ProcessingResult, ZoneEntry
from ..data.video_source import VideoSource, FrameData
from ..storage.base import StorageReader
from .detector import YOLODetector, DetectorConfig
from .tracker import VehicleTracker
from .zone_manager import ZoneManager


@dataclass
class CounterConfig:
    """Configuración del contador de vehículos."""
    reduce_factor: int = 1
    max_minutes: float | None = None
    progress_interval: int = 100
    verbose: bool = True
    draw_zones: bool = True
    draw_trajectories: bool = True
    max_trajectory_points: int = 20


class VehicleCounter:
    """
    Contador de vehículos que integra detector, tracker y zone manager.
    
    Example:
        >>> detector = YOLODetector.from_path("yolov8s.pt", device="cuda")
        >>> counter = VehicleCounter(detector)
        >>> 
        >>> with VideoSource.open(reader, "video.mp4") as video:
        ...     result = counter.process(video, video_info, storage_writer)
        ...     print(f"Detections: {result.total_detections}")
    """
    
    def __init__(
        self,
        detector: YOLODetector,
        config: CounterConfig | None = None,
        zone_manager: ZoneManager | None = None
    ):
        """
        Inicializa el contador.
        
        Args:
            detector: Detector YOLO configurado
            config: Configuración del contador
            zone_manager: Gestor de zonas (opcional)
        """
        self.detector = detector
        self.config = config or CounterConfig()
        self.tracker = VehicleTracker()
        self.zone_manager = zone_manager or ZoneManager()
        
        # Callbacks opcionales
        self._on_detection: Callable[[Detection, FrameData], None] | None = None
        self._on_zone_entry: Callable[[ZoneEntry], None] | None = None
    
    def setup_zones(self, storage: StorageReader, zones_path: str) -> int:
        """
        Configura las zonas de interés.
        
        Args:
            storage: Reader de storage
            zones_path: Ruta al archivo JSON de zonas
            
        Returns:
            Número de zonas cargadas
        """
        return self.zone_manager.load_from_json(storage, zones_path)
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_seconds: float,
        base_time: str | None = None
    ) -> list[Detection]:
        """
        Procesa un frame individual.
        
        Args:
            frame: Frame de video
            timestamp_seconds: Tiempo actual en segundos
            base_time: Hora base del video
            
        Returns:
            Lista de detecciones
        """
        # Detectar y trackear
        detections = self.detector.detect_and_track(frame)
        
        # Actualizar tracker
        self.tracker.update_batch(detections)
        
        # Verificar zonas para cada detección
        if self.zone_manager:
            for det in detections:
                cx, cy = det.center
                result = self.zone_manager.check_entry(
                    track_id=det.track_id,
                    x=cx,
                    y=cy,
                    timestamp_seconds=timestamp_seconds,
                    vehicle_type=det.class_name,
                    exact_time=base_time  # TODO: calcular hora exacta
                )
                
                if result.is_new_entry and self._on_zone_entry:
                    # Obtener última entrada del log
                    entries = self.zone_manager.get_detection_log()
                    if entries:
                        self._on_zone_entry(entries[-1])
        
        return detections
    
    def process(
        self,
        video: VideoSource,
        video_info: VideoInfo,
        base_time: str | None = None,
        on_progress: Callable[[int, int, float], None] | None = None
    ) -> ProcessingResult:
        """
        Procesa un video completo.
        
        Args:
            video: Fuente de video
            video_info: Información del video
            base_time: Hora base del video
            on_progress: Callback para progreso (frame_index, total, elapsed_time)
            
        Returns:
            ProcessingResult con estadísticas
        """
        config = self.config
        
        # Calcular resolución de procesamiento
        new_width = video.width // config.reduce_factor
        new_height = video.height // config.reduce_factor
        
        # Escalar zonas si es necesario
        if self.zone_manager and config.reduce_factor > 1:
            self.zone_manager.scale_to_resolution(
                video.width, video.height,
                new_width, new_height
            )
        
        # Calcular frames a procesar
        if config.max_minutes is not None and config.max_minutes > 0:
            max_frames = int(config.max_minutes * 60 * video.fps)
        else:
            max_frames = video.total_frames
        
        # Resetear estado
        self.tracker.clear()
        self.zone_manager.clear_history()
        
        # Variables de control
        frames_processed = 0
        start_time = time.time()
        
        # Info inicial
        if config.verbose:
            print("\n" + "=" * 70)
            print("🎬 PROCESANDO VIDEO")
            print("=" * 70)
            print(f"📹 {video_info.path}")
            print(f"📊 Resolución: {video.width}x{video.height} → {new_width}x{new_height}")
            print(f"🎯 Frames: {min(max_frames, video.total_frames)}")
            print(f"⚡ FPS: {video.fps:.1f}")
            print(f"🔷 Zonas: {len(self.zone_manager)}")
            print("=" * 70)
        
        # Procesar frames
        resize = (new_width, new_height) if config.reduce_factor > 1 else None
        
        for frame_data in video.frames(max_frames=max_frames, resize=resize):
            # Procesar frame
            self.process_frame(
                frame_data.frame,
                frame_data.timestamp_seconds,
                base_time
            )
            
            frames_processed += 1
            
            # Callback de progreso
            if on_progress and frames_processed % config.progress_interval == 0:
                elapsed = time.time() - start_time
                on_progress(frames_processed, max_frames, elapsed)
            
            # Log de progreso
            if config.verbose and frames_processed % config.progress_interval == 0:
                elapsed = time.time() - start_time
                percent = (frames_processed / max_frames) * 100
                fps_actual = frames_processed / elapsed if elapsed > 0 else 0
                remaining = (max_frames - frames_processed) / fps_actual if fps_actual > 0 else 0
                
                print(f"⏳ {frames_processed}/{max_frames} | "
                      f"{percent:.1f}% | "
                      f"{fps_actual:.1f} FPS | "
                      f"Restante: ~{int(remaining)}s")
        
        # Resultados
        processing_time = time.time() - start_time
        detection_log = self.zone_manager.get_detection_log()
        zone_counts = self.zone_manager.get_zone_counts()
        
        if config.verbose:
            print("\n" + "=" * 70)
            print("✅ PROCESO COMPLETADO")
            print("=" * 70)
            print(f"⏱️ Tiempo: {processing_time / 60:.2f} min")
            print(f"📊 Detecciones: {len(detection_log)}")
            print(f"\n📍 POR ZONA:")
            for zone, count in sorted(zone_counts.items()):
                print(f"   Zona {zone}: {count}")
            print("=" * 70)
        
        return ProcessingResult(
            video_path=video_info.path,
            output_video_path=None,
            csv_path=None,
            total_detections=len(detection_log),
            zone_counts=zone_counts,
            processing_time_seconds=processing_time,
            frames_processed=frames_processed,
            detection_log=detection_log
        )
    
    def on_detection(self, callback: Callable[[Detection, FrameData], None]) -> None:
        """Registra callback para cada detección."""
        self._on_detection = callback
    
    def on_zone_entry(self, callback: Callable[[ZoneEntry], None]) -> None:
        """Registra callback para entrada a zonas."""
        self._on_zone_entry = callback
