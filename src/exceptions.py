"""
Excepciones personalizadas para el paquete aforos.

Todas las excepciones heredan de AforosError para facilitar el catch general.

Example:
    >>> from src.exceptions import VideoNotFoundError, ConfigurationError
    >>> 
    >>> try:
    ...     fix_osd("video_que_no_existe.mp4", date="2026-01-05")
    ... except VideoNotFoundError as e:
    ...     print(f"Video no encontrado: {e.path}")
    ... except AforosError as e:
    ...     print(f"Error general: {e}")
"""

from pathlib import Path
from typing import Optional


class AforosError(Exception):
    """Excepción base para todos los errores del paquete aforos."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.full_message)
    
    @property
    def full_message(self) -> str:
        if self.details:
            return f"{self.message}\n  Detalles: {self.details}"
        return self.message


class VideoNotFoundError(AforosError):
    """El archivo de video especificado no existe."""
    
    def __init__(self, path: str):
        self.path = path
        super().__init__(
            f"Video no encontrado: {path}",
            details="Verifica que la ruta sea correcta y el archivo exista."
        )


class VideoOpenError(AforosError):
    """No se pudo abrir el archivo de video."""
    
    def __init__(self, path: str, reason: Optional[str] = None):
        self.path = path
        details = reason or "El archivo puede estar corrupto o en un formato no soportado."
        super().__init__(
            f"No se pudo abrir el video: {path}",
            details=details
        )


class ConfigurationError(AforosError):
    """Error en la configuración o archivo de configuración."""
    
    def __init__(self, message: str, config_path: Optional[str] = None):
        self.config_path = config_path
        details = f"Archivo: {config_path}" if config_path else None
        super().__init__(message, details=details)


class FileNotFoundError(AforosError):
    """Archivo requerido no encontrado (zonas, máscara, config, etc.)."""
    
    def __init__(self, path: str, file_type: str = "archivo"):
        self.path = path
        self.file_type = file_type
        super().__init__(
            f"{file_type.capitalize()} no encontrado: {path}",
            details="Verifica que la ruta sea correcta."
        )


class OutputDirectoryError(AforosError):
    """Error con el directorio de salida."""
    
    def __init__(self, path: str, reason: str):
        self.path = path
        super().__init__(
            f"Error con directorio de salida: {path}",
            details=reason
        )


class ProcessingError(AforosError):
    """Error durante el procesamiento de video."""
    
    def __init__(self, message: str, video_path: Optional[str] = None):
        self.video_path = video_path
        details = f"Video: {video_path}" if video_path else None
        super().__init__(message, details=details)


class OCRError(AforosError):
    """Error en el reconocimiento óptico de caracteres."""
    
    def __init__(self, message: str, engine: Optional[str] = None):
        self.engine = engine
        details = f"Motor OCR: {engine}" if engine else None
        super().__init__(message, details=details)


class FFmpegError(AforosError):
    """Error al ejecutar ffmpeg."""
    
    def __init__(self, message: str, stderr: Optional[str] = None):
        self.stderr = stderr
        details = stderr[:200] if stderr and len(stderr) > 200 else stderr
        super().__init__(message, details=details)


class FFmpegNotFoundError(FFmpegError):
    """ffmpeg no está instalado o no está en el PATH."""
    
    def __init__(self):
        super().__init__(
            "ffmpeg no encontrado en el sistema",
            stderr="Instala ffmpeg: 'apt install ffmpeg' (Linux) o descarga desde ffmpeg.org"
        )


class ModelNotFoundError(AforosError):
    """Modelo YOLO no encontrado."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        super().__init__(
            f"Modelo YOLO no encontrado: {model_path}",
            details="Verifica la ruta o usa un modelo estándar como 'yolov8s.pt'"
        )


class InvalidDateFormatError(AforosError):
    """Formato de fecha inválido."""
    
    def __init__(self, date_string: str):
        self.date_string = date_string
        super().__init__(
            f"Formato de fecha inválido: {date_string}",
            details="Usa formato YYYY-MM-DD (ej: 2026-01-05) o DD-MM-YYYY (ej: 05-01-2026)"
        )
