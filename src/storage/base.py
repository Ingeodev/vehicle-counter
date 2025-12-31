"""
Interfaces abstractas para la capa de almacenamiento.

Define los contratos que deben implementar los diferentes backends
de almacenamiento (local, Google Drive, etc.).
"""

from typing import Protocol, Iterator, Any, runtime_checkable
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FileInfo:
    """Información básica de un archivo."""
    path: str
    name: str
    extension: str
    size_bytes: int | None = None


@dataclass
class VideoMetadata:
    """Metadatos de un archivo de video."""
    path: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_seconds: float


@runtime_checkable
class StorageReader(Protocol):
    """
    Interface para lectura de archivos desde cualquier backend.
    
    Implementaciones:
        - LocalStorageReader: Sistema de archivos local
        - GDriveStorageReader: Google Drive
    """
    
    def list_files(self, directory: str, pattern: str = "*") -> list[FileInfo]:
        """
        Lista archivos en un directorio que coincidan con el patrón.
        
        Args:
            directory: Ruta del directorio a escanear
            pattern: Patrón glob para filtrar archivos (ej: "*.mp4")
            
        Returns:
            Lista de FileInfo con información de cada archivo
        """
        ...
    
    def list_directories(self, directory: str) -> list[str]:
        """
        Lista subdirectorios en un directorio.
        
        Args:
            directory: Ruta del directorio padre
            
        Returns:
            Lista de rutas de subdirectorios
        """
        ...
    
    def read_file(self, path: str) -> bytes:
        """
        Lee el contenido completo de un archivo.
        
        Args:
            path: Ruta del archivo
            
        Returns:
            Contenido del archivo como bytes
        """
        ...
    
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """
        Lee el contenido de un archivo de texto.
        
        Args:
            path: Ruta del archivo
            encoding: Codificación del texto
            
        Returns:
            Contenido del archivo como string
        """
        ...
    
    def read_json(self, path: str) -> dict[str, Any]:
        """
        Lee y parsea un archivo JSON.
        
        Args:
            path: Ruta del archivo JSON
            
        Returns:
            Diccionario con el contenido JSON
        """
        ...
    
    def exists(self, path: str) -> bool:
        """
        Verifica si un archivo o directorio existe.
        
        Args:
            path: Ruta a verificar
            
        Returns:
            True si existe, False en caso contrario
        """
        ...
    
    def is_file(self, path: str) -> bool:
        """Verifica si la ruta corresponde a un archivo."""
        ...
    
    def is_directory(self, path: str) -> bool:
        """Verifica si la ruta corresponde a un directorio."""
        ...
    
    def get_video_path(self, path: str) -> str:
        """
        Obtiene la ruta accesible para abrir un video con OpenCV.
        
        Para almacenamiento local, retorna la misma ruta.
        Para almacenamiento remoto, puede descargar a un archivo temporal.
        
        Args:
            path: Ruta del video en el storage
            
        Returns:
            Ruta local accesible para cv2.VideoCapture
        """
        ...


@runtime_checkable
class StorageWriter(Protocol):
    """
    Interface para escritura de archivos a cualquier backend.
    
    Implementaciones:
        - LocalStorageWriter: Sistema de archivos local
        - GDriveStorageWriter: Google Drive
    """
    
    def write_file(self, path: str, content: bytes) -> None:
        """
        Escribe contenido binario a un archivo.
        
        Args:
            path: Ruta destino del archivo
            content: Contenido a escribir
        """
        ...
    
    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """
        Escribe contenido de texto a un archivo.
        
        Args:
            path: Ruta destino del archivo
            content: Texto a escribir
            encoding: Codificación del texto
        """
        ...
    
    def write_json(self, path: str, data: dict[str, Any], indent: int = 2) -> None:
        """
        Escribe un diccionario como JSON.
        
        Args:
            path: Ruta destino del archivo
            data: Diccionario a serializar
            indent: Indentación del JSON
        """
        ...
    
    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """
        Crea un directorio y sus padres si no existen.
        
        Args:
            path: Ruta del directorio a crear
            exist_ok: Si True, no lanza error si ya existe
        """
        ...
    
    def get_video_writer_path(self, path: str) -> str:
        """
        Obtiene la ruta para escribir un video con OpenCV.
        
        Para almacenamiento local, retorna la misma ruta.
        Para almacenamiento remoto, puede retornar una ruta temporal
        que luego se sube al destino final.
        
        Args:
            path: Ruta destino del video en el storage
            
        Returns:
            Ruta local donde cv2.VideoWriter escribirá
        """
        ...
    
    def finalize_video(self, temp_path: str, final_path: str) -> None:
        """
        Finaliza la escritura de un video.
        
        Para almacenamiento local, puede ser un no-op o renombrar.
        Para almacenamiento remoto, sube el archivo temporal al destino.
        
        Args:
            temp_path: Ruta temporal donde se escribió el video
            final_path: Ruta destino final
        """
        ...
    
    def copy_file(self, source: str, destination: str) -> None:
        """
        Copia un archivo de una ubicación a otra.
        
        Args:
            source: Ruta origen
            destination: Ruta destino
        """
        ...
    
    def delete_file(self, path: str) -> None:
        """
        Elimina un archivo.
        
        Args:
            path: Ruta del archivo a eliminar
        """
        ...
