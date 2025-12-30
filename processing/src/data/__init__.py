"""
Data layer - Modelos de datos y abstracción de video.
"""

from .schemas import VideoInfo, ZoneConfig, Detection, ProcessingResult
from .video_source import VideoSource
from .directory_scanner import DirectoryScanner

__all__ = [
    "VideoInfo",
    "ZoneConfig", 
    "Detection",
    "ProcessingResult",
    "VideoSource",
    "DirectoryScanner",
]
