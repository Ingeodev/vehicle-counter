"""
Implementación de storage para sistema de archivos local.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Any

from .base import FileInfo, StorageReader, StorageWriter


class LocalStorageReader:
    """
    Implementación de StorageReader para sistema de archivos local.
    
    Example:
        >>> reader = LocalStorageReader()
        >>> videos = reader.list_files("/path/to/videos", "*.mp4")
        >>> for video in videos:
        ...     print(video.name)
    """
    
    def __init__(self, base_path: str | None = None):
        """
        Inicializa el reader.
        
        Args:
            base_path: Ruta base opcional. Si se proporciona, todas las rutas
                      serán relativas a esta base.
        """
        self.base_path = Path(base_path) if base_path else None
    
    def _resolve_path(self, path: str) -> Path:
        """Resuelve una ruta considerando la base."""
        p = Path(path)
        if self.base_path and not p.is_absolute():
            return self.base_path / p
        return p
    
    def list_files(self, directory: str, pattern: str = "*") -> list[FileInfo]:
        """Lista archivos en un directorio que coincidan con el patrón."""
        dir_path = self._resolve_path(directory)
        
        if not dir_path.exists():
            return []
        
        files: list[FileInfo] = []
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                files.append(FileInfo(
                    path=str(file_path),
                    name=file_path.name,
                    extension=file_path.suffix.lower(),
                    size_bytes=file_path.stat().st_size
                ))
        
        return files
    
    def list_directories(self, directory: str) -> list[str]:
        """Lista subdirectorios en un directorio."""
        dir_path = self._resolve_path(directory)
        
        if not dir_path.exists():
            return []
        
        return [
            str(p) for p in dir_path.iterdir() 
            if p.is_dir()
        ]
    
    def read_file(self, path: str) -> bytes:
        """Lee el contenido completo de un archivo."""
        file_path = self._resolve_path(path)
        return file_path.read_bytes()
    
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Lee el contenido de un archivo de texto."""
        file_path = self._resolve_path(path)
        return file_path.read_text(encoding=encoding)
    
    def read_json(self, path: str) -> dict[str, Any]:
        """Lee y parsea un archivo JSON."""
        content = self.read_text(path)
        return json.loads(content)
    
    def exists(self, path: str) -> bool:
        """Verifica si un archivo o directorio existe."""
        return self._resolve_path(path).exists()
    
    def is_file(self, path: str) -> bool:
        """Verifica si la ruta corresponde a un archivo."""
        return self._resolve_path(path).is_file()
    
    def is_directory(self, path: str) -> bool:
        """Verifica si la ruta corresponde a un directorio."""
        return self._resolve_path(path).is_dir()
    
    def get_video_path(self, path: str) -> str:
        """
        Obtiene la ruta accesible para abrir un video con OpenCV.
        
        Para almacenamiento local, simplemente retorna la ruta resuelta.
        """
        return str(self._resolve_path(path))


class LocalStorageWriter:
    """
    Implementación de StorageWriter para sistema de archivos local.
    
    Example:
        >>> writer = LocalStorageWriter()
        >>> writer.makedirs("/path/to/output")
        >>> writer.write_text("/path/to/output/results.csv", csv_content)
    """
    
    def __init__(self, base_path: str | None = None):
        """
        Inicializa el writer.
        
        Args:
            base_path: Ruta base opcional. Si se proporciona, todas las rutas
                      serán relativas a esta base.
        """
        self.base_path = Path(base_path) if base_path else None
    
    def _resolve_path(self, path: str) -> Path:
        """Resuelve una ruta considerando la base."""
        p = Path(path)
        if self.base_path and not p.is_absolute():
            return self.base_path / p
        return p
    
    def write_file(self, path: str, content: bytes) -> None:
        """Escribe contenido binario a un archivo."""
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
    
    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Escribe contenido de texto a un archivo."""
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=encoding)
    
    def write_json(self, path: str, data: dict[str, Any], indent: int = 2) -> None:
        """Escribe un diccionario como JSON."""
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        self.write_text(path, content)
    
    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        """Crea un directorio y sus padres si no existen."""
        dir_path = self._resolve_path(path)
        dir_path.mkdir(parents=True, exist_ok=exist_ok)
    
    def get_video_writer_path(self, path: str) -> str:
        """
        Obtiene la ruta para escribir un video con OpenCV.
        
        Para almacenamiento local, retorna la ruta directamente
        después de asegurar que el directorio padre existe.
        """
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return str(file_path)
    
    def finalize_video(self, temp_path: str, final_path: str) -> None:
        """
        Finaliza la escritura de un video.
        
        Para almacenamiento local, es un no-op si las rutas son iguales,
        o renombra/mueve si son diferentes.
        """
        if temp_path != final_path:
            src = self._resolve_path(temp_path)
            dst = self._resolve_path(final_path)
            shutil.move(str(src), str(dst))
    
    def copy_file(self, source: str, destination: str) -> None:
        """Copia un archivo de una ubicación a otra."""
        src = self._resolve_path(source)
        dst = self._resolve_path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    
    def delete_file(self, path: str) -> None:
        """Elimina un archivo."""
        file_path = self._resolve_path(path)
        if file_path.exists():
            file_path.unlink()
