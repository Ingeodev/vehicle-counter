"""
Storage layer - Abstracciones para acceso a archivos.

Provee interfaces y implementaciones para leer/escribir archivos
desde diferentes backends (local, Google Drive).
"""

from .base import StorageReader, StorageWriter
from .local import LocalStorageReader, LocalStorageWriter

__all__ = [
    "StorageReader",
    "StorageWriter",
    "LocalStorageReader",
    "LocalStorageWriter",
]
