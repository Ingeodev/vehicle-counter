"""
Pipeline - Orquestador principal del procesamiento de videos.
"""

import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread
from tqdm import tqdm

from typing import Callable
from .config import PipelineConfig, OutputConfig
from .storage import (
    StorageReader,
    StorageWriter,
    LocalStorageReader,
    LocalStorageWriter,
)
from .data import VideoSource, VideoInfo, DirectoryScanner, ProcessingResult
from .processing import (
    YOLODetector,
    DetectorConfig,
    VehicleTracker,
    ZoneManager,
    MaskManager,
    TrackDeduplicator,
    VehicleCounter,
    CounterConfig,
)
from .processing.segmenter import YOLOSegmenter
from .output import AnnotatedVideoWriter, VideoWriterConfig, CSVExporter, Visualizer
from .utils import FrameDeblurrer, format_timestamp


class VideoPipeline:
    """
    Pipeline completo para procesamiento de videos de conteo vehicular.

    Integra todas las capas: storage, data, processing, output.

    Example:
        >>> # Configuración básica
        >>> config = PipelineConfig()
        >>> config.detector.device = "cuda"
        >>>
        >>> # Crear pipeline
        >>> pipeline = VideoPipeline(config)
        >>>
        >>> # Procesar un video
        >>> result = pipeline.process_video(
        ...     video_path="/path/to/video.mp4",
        ...     zones_path="/path/to/zones.json",
        ...     output_folder="/path/to/output",
        ...     base_time="14:30"
        ... )
        >>> print(f"Detecciones: {result.total_detections}")
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        storage_reader: StorageReader | None = None,
        storage_writer: StorageWriter | None = None,
    ):
        """
        Inicializa el pipeline.

        Args:
            config: Configuración del pipeline
            storage_reader: Reader de storage (default: LocalStorageReader)
            storage_writer: Writer de storage (default: LocalStorageWriter)
        """
        self.config = config or PipelineConfig()

        # Storage
        self.reader = storage_reader or LocalStorageReader()
        self.writer = storage_writer or LocalStorageWriter()

        # Detector (lazy initialization)
        self._detector: YOLODetector | None = None

        # Componentes
        self.visualizer = Visualizer()
        self.csv_exporter = CSVExporter(self.writer)

    @property
    def detector(self) -> YOLODetector | YOLOSegmenter:
        """Obtiene o crea el detector (Box o Seg)."""
        if self._detector is None:
            detector_config = DetectorConfig(
                model_path=self.config.detector.model_path,
                device=self.config.detector.device,
                vehicle_classes=self.config.detector.vehicle_classes,
                confidence_threshold=self.config.detector.confidence_threshold,
                strategy=self.config.detector.strategy,
                class_thresholds=self.config.detector.class_thresholds,
                tracker_config="botsort_vehicles.yaml",
            )

            if self.config.detector.strategy == "seg":
                self._detector = YOLOSegmenter(detector_config)
            else:
                self._detector = YOLODetector(detector_config)

        return self._detector

    def process_video(
        self,
        video_path: str,
        zones_path: str | None = None,
        mask_path: str | None = None,
        output_folder: str | None = None,
        output_name: str | None = None,
        base_time: str = "00:00",
        date: str = "Unknown",
        on_progress: Callable[[int, int, float], None] | None = None,
        enable_deblurring: bool = False,
        enable_night_enhance: bool = False,
    ) -> ProcessingResult:
        """
        Procesa un video completo.

        Args:
            video_path: Ruta al video de entrada
            zones_path: Ruta al archivo JSON de zonas (opcional)
            mask_path: Ruta a la máscara (no implementado aún)
            output_folder: Carpeta de salida
            output_name: Nombre del archivo de salida
            base_time: Hora base del video
            date: Fecha del video (YYYY-MM-DD)
            on_progress: Callback para progreso

        Returns:
            ProcessingResult con estadísticas
        """
        cfg = self.config
        out_cfg = cfg.output

        # Configurar output
        output_folder = output_folder or out_cfg.output_folder
        self.writer.makedirs(output_folder)

        # Crear zone manager si hay zonas
        zone_manager = ZoneManager()
        if zones_path and self.reader.exists(zones_path):
            zone_manager.load_from_json(self.reader, zones_path)

        # Crear mask manager si hay máscara
        mask_manager = MaskManager()
        if mask_path and self.reader.exists(mask_path):
            mask_manager.load_from_path(self.reader.get_video_path(mask_path))

        # Crear video info
        video_info = VideoInfo(
            path=video_path,
            hora_inicial=base_time,
            mask_path=mask_path,
            zones_path=zones_path,
            output_folder=output_folder,
            output_name=output_name,
        )

        # Abrir video
        video_path_local = self.reader.get_video_path(video_path)
        video = VideoSource.from_path(video_path_local)

        # Resetear tracker si es necesario (para limpiar estado de video anterior)
        if hasattr(self.detector, "reset_tracking"):
            self.detector.reset_tracking()

        try:
            # Actualizar video info con metadatos
            video_info.fps = video.fps
            video_info.width = video.width
            video_info.height = video.height
            video_info.total_frames = video.total_frames

            # Calcular resolución de procesamiento
            reduce_factor = cfg.video.reduce_factor
            new_width = video.width // reduce_factor
            new_height = video.height // reduce_factor

            # Escalar zonas
            if zone_manager and reduce_factor > 1:
                zone_manager.scale_to_resolution(
                    video.width, video.height, new_width, new_height
                )

            # Redimensionar máscara al tamaño del frame de procesamiento
            if mask_manager.is_loaded:
                mask_manager.resize_to_frame(new_width, new_height)

            # Calcular frames a procesar
            if cfg.video.max_minutes is not None and cfg.video.max_minutes > 0:
                max_frames = int(cfg.video.max_minutes * 60 * video.fps)
            else:
                max_frames = video.total_frames

            # Crear counter
            counter_config = CounterConfig(
                reduce_factor=reduce_factor,
                max_minutes=cfg.video.max_minutes,
                progress_interval=out_cfg.progress_interval,
                verbose=out_cfg.verbose,
                draw_zones=out_cfg.draw_zones,
                draw_trajectories=out_cfg.draw_trajectories,
                max_trajectory_points=out_cfg.max_trajectory_points,
            )
            counter = VehicleCounter(self.detector, counter_config, zone_manager)

            # Crear deduplicador para IDs estables
            deduplicator = TrackDeduplicator(
                max_distance=60.0,
                lookback_frames=int(video.fps * 3),  # 3 segundos
                min_positions_to_compare=5,
            )

            # Configurar video writer
            output_video_path = None
            video_writer = None

            if out_cfg.save_video:
                if output_name is None:
                    import os

                    base_name = os.path.basename(video_path)
                    output_name = base_name.replace(".mp4", "_processed.mp4")

                output_video_path = f"{output_folder}/{output_name}"

                writer_config = VideoWriterConfig(
                    fps=cfg.video.output_fps,
                    codec=cfg.video.codec,
                    width=new_width,
                    height=new_height,
                )
                video_writer = AnnotatedVideoWriter.from_path(
                    output_video_path, writer_config
                )

            # Variables de control
            frames_processed = 0
            start_time = time.time()

            # Info inicial
            if out_cfg.verbose:
                self._print_start_info(
                    video_info, new_width, new_height, max_frames, zone_manager
                )

            # Inicializar barra de progreso
            pbar = tqdm(
                total=max_frames,
                unit="frames",
                desc=f"Procesando {output_name or 'video'}",
                disable=not out_cfg.verbose,
            )

            # Procesar frames
            resize = (new_width, new_height) if reduce_factor > 1 else None

            # Preparar mejora de imagen
            enhancer = None
            if enable_night_enhance:
                enhancer = FrameDeblurrer.create_night_enhance()
                if out_cfg.verbose:
                    tqdm.write("🌙 Night Enhance activado (Gamma + CLAHE)")
            elif enable_deblurring:
                enhancer = FrameDeblurrer.create_aggressive()
                if out_cfg.verbose:
                    tqdm.write("🌃 Deblurring activado (Agresivo)")

            # ─── Threaded reader: lee frames en background ────────────────
            input_queue: Queue = Queue(maxsize=4)

            def _reader_worker():
                try:
                    for fd in video.frames(
                        max_frames=max_frames,
                        resize=resize,
                        skip_rate=cfg.video.skip_rate,
                    ):
                        input_queue.put((fd.frame, fd.timestamp_seconds))
                finally:
                    input_queue.put(None)

            reader = Thread(target=_reader_worker, daemon=True)
            reader.start()

            # ─── Threaded writer: escribe frames en background ───────────
            output_queue: Queue | None = None
            writer: Thread | None = None

            if video_writer:
                output_queue = Queue(maxsize=4)

                def _writer_worker():
                    try:
                        while True:
                            f = output_queue.get()
                            if f is None:
                                break
                            video_writer.write(f)
                    finally:
                        video_writer.close()

                writer = Thread(target=_writer_worker, daemon=True)
                writer.start()

            # ─── Main loop: solo inferencia + visualización ────────────
            for item in iter(input_queue.get, None):
                frame, timestamp = item

                # Aplicar mejora si está habilitada
                if enhancer:
                    frame = enhancer.process(frame)

                # Detectar y trackear
                detections = counter.process_frame(frame, timestamp, base_time, date)

                # Filtrar detecciones con máscara
                if mask_manager.is_loaded:
                    detections = mask_manager.filter_detections(detections)

                # Aplicar deduplicación para IDs estables
                for det in detections:
                    original_id = deduplicator.get_original_id(
                        det.track_id, det.center, frames_processed, det.class_name
                    )
                    if original_id != det.track_id:
                        det.track_id = original_id

                # Visualizar (main thread) y encolar para escritura
                if video_writer and output_queue is not None:
                    if out_cfg.draw_zones and zone_manager:
                        self.visualizer.draw_zones(frame, zone_manager.zones)

                    for det in detections:
                        zones_visited = (
                            zone_manager.get_zones_for_vehicle(det.track_id)
                            if zone_manager
                            else None
                        )
                        trajectory = counter.tracker.get_trajectory_array(
                            det.track_id, out_cfg.max_trajectory_points
                        )
                        self.visualizer.draw_detection(
                            frame, det, zones_visited, trajectory
                        )

                    from .utils import format_timestamp

                    stats = {
                        "Detecciones": len(zone_manager.get_detection_log())
                        if zone_manager
                        else 0,
                        "Tiempo": format_timestamp(timestamp),
                    }
                    self.visualizer.draw_stats(frame, stats)

                    output_queue.put(frame)

                frames_processed += 1

                if on_progress and frames_processed % out_cfg.progress_interval == 0:
                    elapsed = time.time() - start_time
                    on_progress(frames_processed, max_frames, elapsed)

                if frames_processed % 10 == 0:
                    postfix = {}
                    if zone_manager:
                        postfix["detections"] = zone_manager.get_log_count()
                    pbar.set_postfix(postfix)

                pbar.update(1)

            pbar.close()

            # Cerrar writer thread
            if writer is not None and output_queue is not None:
                output_queue.put(None)
                writer.join(timeout=10)

            # Exportar CSV
            csv_path = None
            if out_cfg.save_csv and zone_manager:
                base_csv = (
                    output_video_path.replace(".mp4", "")
                    if output_video_path
                    else f"{output_folder}/result"
                )
                csv_path = f"{base_csv}_detections.csv"
                self.csv_exporter.export_detections(
                    zone_manager.get_detection_log(), csv_path
                )

            # Resultados
            processing_time = time.time() - start_time
            detection_log = zone_manager.get_detection_log() if zone_manager else []
            zone_counts = zone_manager.get_zone_counts() if zone_manager else {}

            if out_cfg.verbose:
                self._print_end_info(
                    processing_time,
                    detection_log,
                    zone_counts,
                    output_video_path,
                    csv_path,
                )

            return ProcessingResult(
                video_path=video_path,
                output_video_path=output_video_path,
                csv_path=csv_path,
                total_detections=len(detection_log),
                zone_counts=zone_counts,
                processing_time_seconds=processing_time,
                frames_processed=frames_processed,
                detection_log=detection_log,
            )

        finally:
            video.close()

    def scan_directory(self, root_path: str, recursive: bool = True):
        """
        Escanea un directorio buscando videos.

        Args:
            root_path: Ruta raíz a escanear
            recursive: Si escanear subdirectorios

        Returns:
            ScanResult con lista de videos
        """
        scanner = DirectoryScanner(self.reader)
        return scanner.scan(root_path, recursive)

    def process_directory(
        self,
        root_path: str,
        output_base: str | None = None,
        recursive: bool = True,
        on_video_complete: Callable[[VideoInfo, ProcessingResult], None] | None = None,
        mask_path: str | None = None,
        zones_path: str | None = None,
        enable_deblurring: bool = False,
        enable_night_enhance: bool = False,
        date: str = "Unknown",
    ) -> list[ProcessingResult]:
        """
        Procesa todos los videos en un directorio.

        Args:
            root_path: Ruta raíz a escanear
            output_base: Carpeta base de salida
            recursive: Si escanear subdirectorios
            on_video_complete: Callback al completar cada video
            date: Fecha común para todos los videos

        Returns:
            Lista de ProcessingResult
        """
        # Escanear directorio
        scan_result = self.scan_directory(root_path, recursive)

        if self.config.output.verbose:
            print(f"\n📹 Encontrados {len(scan_result.videos)} videos")

        results: list[ProcessingResult] = []

        for video_info in scan_result.videos:
            try:
                output_folder = output_base or video_info.output_folder

                result = self.process_video(
                    video_path=video_info.path,
                    zones_path=zones_path or video_info.zones_path,
                    mask_path=mask_path or video_info.mask_path,
                    output_folder=output_folder,
                    output_name=video_info.output_name,
                    base_time=video_info.hora_inicial,
                    date=date,
                    enable_deblurring=enable_deblurring,
                    enable_night_enhance=enable_night_enhance,
                )

                results.append(result)

                if on_video_complete:
                    on_video_complete(video_info, result)

            except Exception as e:
                if self.config.output.verbose:
                    print(f"❌ Error procesando {video_info.path}: {e}")

        return results

    def _print_start_info(self, video_info, new_w, new_h, max_frames, zone_manager):
        """Imprime información inicial."""
        print("\n" + "=" * 70)
        print("🎬 PROCESANDO VIDEO")
        print("=" * 70)
        print(f"📹 {video_info.path}")
        print(
            f"📊 Resolución: {video_info.width}x{video_info.height} → {new_w}x{new_h}"
        )
        print(f"🎯 Frames: {max_frames}")
        print(f"⚡ FPS: {video_info.fps:.1f}")
        print(f"🔷 Zonas: {len(zone_manager) if zone_manager else 0}")
        print("=" * 70)

    def _print_progress(self, frames, max_frames, start_time, zone_manager):
        """Imprime progreso."""
        elapsed = time.time() - start_time
        percent = (frames / max_frames) * 100
        fps_actual = frames / elapsed if elapsed > 0 else 0
        remaining = (max_frames - frames) / fps_actual if fps_actual > 0 else 0
        detections = len(zone_manager.get_detection_log()) if zone_manager else 0

        print(
            f"⏳ {frames}/{max_frames} | "
            f"{percent:.1f}% | "
            f"{fps_actual:.1f} FPS | "
            f"Restante: ~{int(remaining)}s | "
            f"Det: {detections}"
        )

    def _print_end_info(
        self, processing_time, detection_log, zone_counts, video_path, csv_path
    ):
        """Imprime información final."""
        print("\n" + "=" * 70)
        print("✅ PROCESO COMPLETADO")
        print("=" * 70)
        print(f"⏱️ Tiempo: {processing_time / 60:.2f} min")
        if video_path:
            print(f"📁 Video: {video_path}")
        if csv_path:
            print(f"📊 CSV: {csv_path}")
        print(f"\n🚗 Detecciones totales: {len(detection_log)}")
        if zone_counts:
            print("\n📍 POR ZONA:")
            for zone, count in sorted(zone_counts.items()):
                print(f"   Zona {zone}: {count}")
        print("=" * 70)
