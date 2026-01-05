"""
DirectoryScanner - Escaneo de directorios para encontrar videos y archivos asociados.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timedelta

from ..storage.base import StorageReader, FileInfo
from .schemas import VideoInfo


@dataclass
class ScanResult:
    """Resultado del escaneo de un directorio."""
    videos: list[VideoInfo]
    total_duration_minutes: float
    summary: dict[str, Any] = field(default_factory=dict)


class DirectoryScanner:
    """
    Escanea directorios para encontrar videos y sus archivos asociados.
    
    Busca videos (.mp4) junto con sus máscaras (.png) y zonas (.json).
    
    Example:
        >>> from mglon_vehicle_counter.storage import LocalStorageReader
        >>> reader = LocalStorageReader()
        >>> scanner = DirectoryScanner(reader)
        >>> result = scanner.scan("/path/to/videos")
        >>> for video in result.videos:
        ...     print(f"{video.path} - {video.hora_inicial}")
    """
    
    # Extensiones reconocidas
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
    MASK_EXTENSIONS = {".png", ".jpg", ".jpeg"}
    ZONES_EXTENSION = ".json"
    
    def __init__(
        self, 
        storage: StorageReader,
        video_duration_minutes: float = 24.0
    ):
        """
        Inicializa el scanner.
        
        Args:
            storage: Reader de storage para acceso a archivos
            video_duration_minutes: Duración estimada de cada video en minutos
        """
        self.storage = storage
        self.video_duration_minutes = video_duration_minutes
    
    def scan(
        self, 
        root_path: str,
        recursive: bool = True,
        output_base_path: str | None = None
    ) -> ScanResult:
        """
        Escanea un directorio buscando videos y archivos asociados.
        
        Args:
            root_path: Ruta raíz a escanear
            recursive: Si True, escanea subdirectorios recursivamente
            output_base_path: Ruta base para outputs (si None, usa root_path)
            
        Returns:
            ScanResult con la lista de videos encontrados
        """
        videos: list[VideoInfo] = []
        
        if recursive:
            videos = self._scan_recursive(root_path, output_base_path or root_path)
        else:
            videos = self._scan_directory(root_path, output_base_path or root_path)
        
        # Calcular duración total
        total_duration = len(videos) * self.video_duration_minutes
        
        return ScanResult(
            videos=videos,
            total_duration_minutes=total_duration,
            summary={
                "total_videos": len(videos),
                "total_hours": total_duration / 60,
                "total_days": total_duration / 1440
            }
        )
    
    def _scan_recursive(
        self, 
        path: str, 
        output_base: str,
        current_mask: str | None = None,
        current_zones: str | None = None
    ) -> list[VideoInfo]:
        """Escanea recursivamente un directorio."""
        videos: list[VideoInfo] = []
        
        # Buscar mask y zones en el nivel actual
        mask_path = current_mask
        zones_path = current_zones
        
        # Buscar archivos en el directorio actual
        files = self.storage.list_files(path, "*")
        
        for file_info in files:
            ext = file_info.extension.lower()
            
            # Detectar máscara
            if ext in self.MASK_EXTENSIONS and "mask" in file_info.name.lower():
                mask_path = file_info.path
            
            # Detectar zonas
            if ext == self.ZONES_EXTENSION:
                zones_path = file_info.path
        
        # Procesar videos en el directorio actual
        for file_info in files:
            ext = file_info.extension.lower()
            
            if ext in self.VIDEO_EXTENSIONS:
                video_info = self._create_video_info(
                    file_info, 
                    path,
                    output_base,
                    mask_path,
                    zones_path
                )
                videos.append(video_info)
        
        # Escanear subdirectorios
        subdirs = self.storage.list_directories(path)
        for subdir in subdirs:
            sub_videos = self._scan_recursive(
                subdir, 
                output_base,
                mask_path, 
                zones_path
            )
            videos.extend(sub_videos)
        
        return videos
    
    def _scan_directory(
        self, 
        path: str, 
        output_base: str
    ) -> list[VideoInfo]:
        """Escanea un solo directorio (sin recursión)."""
        videos: list[VideoInfo] = []
        mask_path: str | None = None
        zones_path: str | None = None
        
        files = self.storage.list_files(path, "*")
        
        # Primera pasada: encontrar mask y zones
        for file_info in files:
            ext = file_info.extension.lower()
            
            if ext in self.MASK_EXTENSIONS and "mask" in file_info.name.lower():
                mask_path = file_info.path
            
            if ext == self.ZONES_EXTENSION:
                zones_path = file_info.path
        
        # Segunda pasada: procesar videos
        for file_info in files:
            ext = file_info.extension.lower()
            
            if ext in self.VIDEO_EXTENSIONS:
                video_info = self._create_video_info(
                    file_info,
                    path,
                    output_base,
                    mask_path,
                    zones_path
                )
                videos.append(video_info)
        
        return videos
    
    def _create_video_info(
        self,
        file_info: FileInfo,
        source_path: str,
        output_base: str,
        mask_path: str | None,
        zones_path: str | None
    ) -> VideoInfo:
        """Crea un VideoInfo a partir de la información del archivo."""
        # Extraer hora inicial del nombre del archivo
        # Formato esperado: algo_HH:MM_algo.mp4
        hora_inicial = self._extract_time_from_filename(file_info.name)
        
        # Extraer ID del video del nombre
        video_id = self._extract_video_id(file_info.name)
        
        # Generar nombre de salida
        output_name = file_info.name.replace(".mp4", "_processed.mp4")
        
        # Calcular carpeta de salida (preservar estructura)
        relative_path = source_path.replace(output_base, "").lstrip("/\\")
        output_folder = os.path.join(output_base, "output", relative_path)
        
        return VideoInfo(
            path=file_info.path,
            hora_inicial=hora_inicial,
            mask_path=mask_path,
            zones_path=zones_path,
            output_folder=output_folder,
            output_name=output_name,
            context=relative_path,
            video_id=video_id
        )
    
    def _extract_time_from_filename(self, filename: str) -> str:
        """
        Extrae la hora del nombre del archivo.
        
        Busca patrones como:
        - video_14:30_001.mp4 -> 14:30
        - 2024-01-01_08-00_video.mp4 -> 08:00
        """
        # Patrón para HH:MM
        pattern = r"(\d{1,2})[:\-](\d{2})"
        match = re.search(pattern, filename)
        
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
        
        # Si no encuentra patrón, retornar valor por defecto
        return "00:00"
    
    def _extract_video_id(self, filename: str) -> str:
        """Extrae el ID del video del nombre del archivo."""
        # Buscar números en el nombre
        numbers = re.findall(r"\d+", filename)
        if numbers:
            # Usar el último número como ID
            return numbers[-1]
        return "0"
    
    @staticmethod
    def calculate_video_time(
        hora_inicial: str, 
        video_id: int, 
        duration_minutes: float = 24
    ) -> str:
        """
        Calcula la hora de inicio de un video dado su ID.
        
        Args:
            hora_inicial: Hora del primer video en formato HH:MM
            video_id: ID del video (1-indexed)
            duration_minutes: Duración de cada video en minutos
            
        Returns:
            Hora calculada en formato HH:MM
        """
        try:
            base_time = datetime.strptime(hora_inicial, "%H:%M")
            offset = timedelta(minutes=(video_id - 1) * duration_minutes)
            result_time = base_time + offset
            return result_time.strftime("%H:%M")
        except ValueError:
            return hora_inicial
    
    def print_summary(self, result: ScanResult) -> None:
        """Imprime un resumen del escaneo."""
        print("=" * 70)
        print("📊 RESUMEN DE VIDEOS A PROCESAR")
        print("=" * 70)
        print(f"📹 Cantidad de videos: {result.summary['total_videos']}")
        print(f"⏱️  Duración por video: {self.video_duration_minutes} minutos")
        print(f"⏰ Tiempo total: {result.summary['total_hours']:.2f} horas "
              f"({result.summary['total_days']:.2f} días)")
        print("=" * 70)
