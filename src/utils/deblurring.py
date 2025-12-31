"""
Deblurring - Reducción de motion blur para videos nocturnos.

Versión optimizada: usa bilateral filter en lugar de fastNlMeans
para mayor velocidad y mejor preservación de bordes.
"""

import cv2
import numpy as np


class FrameDeblurrer:
    """
    Aplica técnicas de deblurring/sharpening a frames de video.
    
    Diseñado para ser rápido y preservar las características
    necesarias para detección con YOLO.
    
    Example:
        >>> deblurrer = FrameDeblurrer(mode="night")
        >>> enhanced = deblurrer.process(frame)
    """
    
    # Kernels de sharpening (más suaves para no distorsionar)
    KERNELS = {
        "soft": np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ], dtype=np.float32),
        
        "medium": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32),
        
        "strong": np.array([
            [-0.5, -1, -0.5],
            [-1, 7, -1],
            [-0.5, -1, -0.5]
        ], dtype=np.float32),
    }
    
    def __init__(
        self,
        mode: str = "night",
        use_bilateral: bool = True,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 50,
        bilateral_sigma_space: float = 50,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        sharpen_strength: str = "medium",
        gamma_correction: float = 1.2
    ):
        """
        Inicializa el deblurrer.
        
        Args:
            mode: Modo predefinido ("night", "fast", "quality")
            use_bilateral: Si usar filtro bilateral (preserva bordes)
            bilateral_d: Diámetro del filtro bilateral
            bilateral_sigma_color: Sigma para color
            bilateral_sigma_space: Sigma para espacio
            use_clahe: Si usar mejora de contraste
            clahe_clip_limit: Límite de CLAHE
            sharpen_strength: Intensidad de sharpening ("soft", "medium", "strong")
            gamma_correction: Corrección gamma para iluminación (1.0 = sin cambio)
        """
        # Aplicar preset según modo
        if mode == "night":
            use_clahe = True
            clahe_clip_limit = 2.5
            sharpen_strength = "medium"
            gamma_correction = 1.3
        elif mode == "fast":
            use_bilateral = False
            use_clahe = True
            clahe_clip_limit = 2.0
            sharpen_strength = "soft"
        elif mode == "quality":
            use_bilateral = True
            bilateral_d = 7
            use_clahe = True
            clahe_clip_limit = 3.0
            use_clahe = True
            clahe_clip_limit = 3.0
            sharpen_strength = "strong"
        elif mode == "enhance_only":
            # Modo solo mejora de luz/contraste (sin deblurring/sharpening agresivo)
            use_bilateral = False
            use_clahe = True
            clahe_clip_limit = 3.0
            sharpen_strength = "soft"  # Muy suave o ninguno
            gamma_correction = 1.4
        
        self.use_bilateral = use_bilateral
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.use_clahe = use_clahe
        self.gamma_correction = gamma_correction
        
        # Kernel de sharpening
        self.kernel = self.KERNELS.get(sharpen_strength, self.KERNELS["medium"])
        
        # CLAHE
        if use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=(8, 8)
            )
        else:
            self.clahe = None
        
        # Tabla de gamma precalculada
        if gamma_correction != 1.0:
            inv_gamma = 1.0 / gamma_correction
            self.gamma_table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype(np.uint8)
        else:
            self.gamma_table = None
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame aplicando mejoras para visión nocturna.
        
        Pipeline optimizado:
        1. Corrección gamma (ilumina zonas oscuras)
        2. Bilateral filter (suaviza ruido, preserva bordes)
        3. CLAHE (mejora contraste local)
        4. Sharpening suave (refuerza bordes)
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            Frame procesado
        """
        result = frame
        
        # 1. Corrección gamma (rápido, mejora luminosidad)
        if self.gamma_table is not None:
            result = cv2.LUT(result, self.gamma_table)
        
        # 2. Bilateral filter (preserva bordes, reduce ruido)
        if self.use_bilateral:
            result = cv2.bilateralFilter(
                result,
                self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space
            )
        
        # 3. CLAHE para mejorar contraste
        if self.clahe is not None:
            result = self._apply_clahe(result)
        
        # 4. Sharpening suave
        result = cv2.filter2D(result, -1, self.kernel)
        
        return result
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Aplica CLAHE al canal L en espacio LAB."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @classmethod
    def create_aggressive(cls) -> "FrameDeblurrer":
        """Config para videos nocturnos con motion blur."""
        return cls(mode="night")
    
    @classmethod
    def create_fast(cls) -> "FrameDeblurrer":
        """Config rápida con mínimo impacto en FPS."""
        return cls(mode="fast")
    
    @classmethod
    def create_balanced(cls) -> "FrameDeblurrer":
        """Config balanceada calidad/velocidad."""
        return cls(
            mode="night",
            bilateral_d=3,
            clahe_clip_limit=2.0,
            sharpen_strength="soft"
        )
    
    @classmethod
    def create_night_enhance(cls) -> "FrameDeblurrer":
        """Config para mejorar visibilidad nocturna sin distorsionar (para segmentación)."""
        return cls(mode="enhance_only")


def deblur_frame(frame: np.ndarray, mode: str = "night") -> np.ndarray:
    """
    Función de conveniencia para deblurring de un frame.
    
    Args:
        frame: Frame BGR
        mode: Modo ("night", "fast", "quality")
        
    Returns:
        Frame procesado
    """
    deblurrer = FrameDeblurrer(mode=mode)
    return deblurrer.process(frame)
