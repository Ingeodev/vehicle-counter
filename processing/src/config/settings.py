"""
settings - Dataclasses de configuración.
"""

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import yaml


# Clases de vehículos por defecto (COCO dataset)
DEFAULT_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


@dataclass
class VideoConfig:
    """Configuración de procesamiento de video."""
    reduce_factor: int = 1
    max_minutes: float | None = None
    skip_rate: int = 1
    output_fps: float = 15.0
    codec: str = "mp4v"


@dataclass
class DetectorConfig:
    """Configuración del detector YOLO."""
    model_path: str = "yolov8s.pt"
    device: str = "cpu"
    confidence_threshold: float = 0.5
    strategy: str = "box"  # 'box' o 'seg'
    tracker_config: str | None = None
    vehicle_classes: dict[int, str] = field(default_factory=lambda: DEFAULT_VEHICLE_CLASSES.copy())


@dataclass
class OutputConfig:
    """Configuración de salida."""
    output_folder: str = "./output"
    save_video: bool = True
    save_csv: bool = True
    draw_zones: bool = True
    draw_trajectories: bool = True
    max_trajectory_points: int = 20
    progress_interval: int = 100
    verbose: bool = True


@dataclass
class PipelineConfig:
    """Configuración completa del pipeline."""
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """
        Carga configuración desde archivo YAML.
        
        Args:
            path: Ruta al archivo YAML
            
        Returns:
            PipelineConfig con valores del archivo
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """
        Carga configuración desde diccionario.
        
        Args:
            data: Diccionario con configuración
            
        Returns:
            PipelineConfig con valores del diccionario
        """
        video_data = data.get("video", {})
        detector_data = data.get("detector", {})
        output_data = data.get("output", {})
        
        return cls(
            video=VideoConfig(**video_data),
            detector=DetectorConfig(**detector_data),
            output=OutputConfig(**output_data)
        )
    
    def to_yaml(self, path: str) -> None:
        """
        Guarda configuración a archivo YAML.
        
        Args:
            path: Ruta del archivo
        """
        data = self.to_dict()
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "video": {
                "reduce_factor": self.video.reduce_factor,
                "max_minutes": self.video.max_minutes,
                "skip_rate": self.video.skip_rate,
                "output_fps": self.video.output_fps,
                "codec": self.video.codec,
            },
            "detector": {
                "model_path": self.detector.model_path,
                "device": self.detector.device,
                "confidence_threshold": self.detector.confidence_threshold,
                "strategy": self.detector.strategy,
                "tracker_config": self.detector.tracker_config,
                "vehicle_classes": self.detector.vehicle_classes,
            },
            "output": {
                "output_folder": self.output.output_folder,
                "save_video": self.output.save_video,
                "save_csv": self.output.save_csv,
                "draw_zones": self.output.draw_zones,
                "draw_trajectories": self.output.draw_trajectories,
                "max_trajectory_points": self.output.max_trajectory_points,
                "progress_interval": self.output.progress_interval,
                "verbose": self.output.verbose,
            }
        }


def get_default_config() -> PipelineConfig:
    """Retorna configuración por defecto."""
    return PipelineConfig()


def detect_device() -> str:
    """
    Detecta el mejor dispositivo disponible.
    
    Returns:
        'cuda' si hay GPU NVIDIA, 'mps' si hay GPU Apple, 'cpu' en otro caso
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    
    return "cpu"
