"""
OSD Reader - Utilidad para leer fecha/hora del OSD de videos mediante OCR.
Implementa el patrón Strategy para soportar múltiples motores OCR (Tesseract, EasyOCR, TrOCR).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, NamedTuple, Literal, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Intentar importar dependencias opcionales
try:
    from PIL import Image
except ImportError:
    Image = None

class VideoTimeInfo(NamedTuple):
    """Información de tiempo extraída de un video."""
    path: str
    date: str  # DD-MM-YYYY
    start_time: str  # HH:MM:SS
    end_time: str  # HH:MM:SS
    duration: timedelta  # Duración calculada desde frames/fps


class OCRStrategy(ABC):
    """Clase base abstracta para estrategias de OCR."""
    
    @abstractmethod
    def recognize_text(self, image: np.ndarray) -> str:
        """
        Reconoce texto de una imagen.
        
        Args:
            image: Imagen en escala de grises o color (numpy array)
            
        Returns:
            Texto reconocido
        """
        pass


class TesseractStrategy(OCRStrategy):
    """Estrategia OCR usando Tesseract."""
    
    def __init__(self):
        try:
            import pytesseract
            self._tesseract = pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract no está instalado. "
                "Instala con: pip install pytesseract\n"
                "También necesitas instalar Tesseract OCR en tu sistema."
            )

    def recognize_text(self, image: np.ndarray) -> str:
        # Configuración optimizada para texto de fecha/hora
        config = '--psm 7 -c tessedit_char_whitelist=0123456789:-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        text = self._tesseract.image_to_string(image, config=config)
        return text.strip()


class EasyOCRStrategy(OCRStrategy):
    """Estrategia OCR usando EasyOCR."""
    
    def __init__(self):
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except ImportError:
                raise ImportError(
                    "easyocr no está instalado. "
                    "Instala con: pip install easyocr"
                )
        return self._reader

    def recognize_text(self, image: np.ndarray) -> str:
        reader = self._get_reader()
        results = reader.readtext(image, detail=0)
        return ' '.join(results).strip()


class TrOCRStrategy(OCRStrategy):
    """
    Estrategia OCR usando Vision Transformer (TrOCR).
    Requiere 'transformers' y 'torch'.
    """
    
    # Class-level cache to avoid reloading model on every instance
    _cached_processor = None
    _cached_model = None
    _cached_device = None

    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        self.model_name = model_name
        
    def _load_model(self):
        # Check class-level cache
        if TrOCRStrategy._cached_model is None:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                import torch
                
                print(f"⏳ Cargando modelo TrOCR ({self.model_name})... esto puede tardar la primera vez.")
                TrOCRStrategy._cached_processor = TrOCRProcessor.from_pretrained(self.model_name)
                TrOCRStrategy._cached_model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                
                # Mover a GPU si está disponible
                TrOCRStrategy._cached_device = "cuda" if torch.cuda.is_available() else "cpu"
                TrOCRStrategy._cached_model.to(TrOCRStrategy._cached_device)
                print(f"✅ Modelo TrOCR cargado en {TrOCRStrategy._cached_device}")
                
            except ImportError:
                raise ImportError(
                    "Librerías de TrOCR no instaladas. "
                    "Instala con: pip install transformers torch torchvision"
                )
        
        # Assign local references for convenience
        self._processor = TrOCRStrategy._cached_processor
        self._model = TrOCRStrategy._cached_model
        self.device = TrOCRStrategy._cached_device
    
    def recognize_text(self, image: np.ndarray) -> str:
        self._load_model()
        
        # TrOCR espera imagen PIL RGB
        if len(image.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocesar y generar texto
        pixel_values = self._processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        generated_ids = self._model.generate(pixel_values, max_new_tokens=20)
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()


class OSDReader:
    """
    Lee texto del OSD (On-Screen Display) en videos usando OCR.
    Usa el patrón Strategy para delegar el reconocimiento de texto.
    """
    
    # Coordenadas de referencia para 1920x1080
    REF_WIDTH = 1920.0
    REF_HEIGHT = 1080.0
    
    # Coordenadas relativas (porcentajes) - ROI expandido
    REL_X1 = 30 / REF_WIDTH    # 0.0156
    REL_Y1 = 40 / REF_HEIGHT   # 0.0370
    REL_X2 = 650 / REF_WIDTH   # 0.3385
    REL_Y2 = 110 / REF_HEIGHT  # 0.1019
    
    def __init__(self, ocr_engine: Literal["tesseract", "easyocr", "trocr"] = "easyocr",
                 preprocess: Literal["clahe", "binary", "color"] = "clahe"):
        """
        Inicializa el lector OSD.
        
        Args:
            ocr_engine: Motor OCR a usar ("tesseract", "easyocr", "trocr")
            preprocess: Modo de preprocesamiento ("clahe", "binary", "color")
        """
        self.preprocess_mode = preprocess
        self.strategy = self._get_strategy(ocr_engine)
        
    def _get_strategy(self, engine_name: str) -> OCRStrategy:
        if engine_name == "tesseract":
            return TesseractStrategy()
        elif engine_name == "easyocr":
            return EasyOCRStrategy()
        elif engine_name == "trocr":
            return TrOCRStrategy()
        else:
            raise ValueError(f"Motor OCR no soportado: {engine_name}")

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extrae la región de interés (ROI) del OSD."""
        height, width = frame.shape[:2]
        x1 = int(width * self.REL_X1)
        y1 = int(height * self.REL_Y1)
        x2 = int(width * self.REL_X2)
        y2 = int(height * self.REL_Y2)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        return frame[y1:y2, x1:x2].copy()
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """Pre-procesa el ROI para mejorar la precisión del OCR."""
        # Escalar x2 para mejorar OCR
        scaled = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        if self.preprocess_mode == "color":
            return scaled
        
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        
        if self.preprocess_mode == "clahe":
            # CLAHE: Contraste adaptativo
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(gray)
            result = cv2.copyMakeBorder(result, 10, 10, 10, 10, 
                                        cv2.BORDER_CONSTANT, value=255)
            return result
        
        elif self.preprocess_mode == "binary":
            # Binarización basada en histograma
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            dark_sum = np.sum(hist[:50])
            light_sum = np.sum(hist[200:])
            
            if light_sum > dark_sum:
                _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            else:
                _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            
            # Morfología
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Borde
            binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, 
                                        cv2.BORDER_CONSTANT, value=255)
            return binary
        
        return gray
    
    def _parse_datetime(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parsea el texto OCR para extraer fecha y hora."""
        import re
        
        # Limpieza básica
        text = text.replace('\n', ' ').strip()
        text = text.replace(';', ':').replace(',', ':')
        
        # Patrón fecha: DD-MM-YYYY
        date_match = re.search(r'(\d{2}[-/]\d{2}[-/]\d{4})', text)
        
        # Patrón hora: HH:MM:SS
        time_match = re.search(r'(\d{1,2}:\d{2}:\d{2})', text)
        
        date_str = date_match.group(1).replace('/', '-') if date_match else None
        time_str = time_match.group(1) if time_match else None
        
        # Intentos de recuperación de hora malformada
        if not time_str:
            # Caso "05401:18" -> (\d+):(\d{2})
            match1 = re.search(r'(\d{4,5}):(\d{2})', text)
            if match1:
                digits = match1.group(1)
                seconds = match1.group(2)
                if len(digits) >= 4:
                    time_str = f"{digits[:2]}:{digits[2:4]}:{seconds}"
            
            # Caso "054018" -> 6 dígitos
            if not time_str:
                digits_match = re.search(r'(\d{6})', text)
                if digits_match:
                    raw = digits_match.group(1)
                    # Validación básica de hora
                    if int(raw[0:2]) < 24 and int(raw[2:4]) < 60 and int(raw[4:6]) < 60:
                        time_str = f"{raw[0:2]}:{raw[2:4]}:{raw[4:6]}"
        
        # Normalizar H:MM:SS -> 0H:MM:SS
        if time_str and len(time_str.split(':')[0]) == 1:
            time_str = '0' + time_str
            
        return date_str, time_str

    def read_frame(self, frame: np.ndarray, export_roi_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Lee la fecha y hora del OSD en un frame."""
        roi = self._extract_roi(frame)
        processed = self._preprocess_roi(roi)
        
        if export_roi_path:
            cv2.imwrite(export_roi_path, processed)
        
        text = self.strategy.recognize_text(processed)
        return self._parse_datetime(text)
    
    def extract_video_times(self, video_path: str, export_roi_dir: Optional[str] = None) -> VideoTimeInfo:
        """Extrae información de tiempo de un video."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el video: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            # Preparar rutas de exportación
            export_start, export_end = None, None
            if export_roi_dir:
                export_dir = Path(export_roi_dir)
                export_dir.mkdir(parents=True, exist_ok=True)
                export_start = str(export_dir / f"{path.stem}_start.png")
                export_end = str(export_dir / f"{path.stem}_end.png")
            
            # Leer inicio
            ret, first_frame = cap.read()
            if not ret:
                raise IOError("No se pudo leer el primer frame")
            start_date, start_time = self.read_frame(first_frame, export_start)
            
            # Leer final con fallback
            end_date, end_time = None, None
            for offset in [1, 10, 50, 100, 500]:
                target_frame = max(1, total_frames - offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, last_frame = cap.read()
                if ret:
                    end_date, end_time = self.read_frame(last_frame, export_end if offset == 1 else None)
                    if end_time:
                        if offset != 1 and export_end:
                            self.read_frame(last_frame, export_end)
                        break
            
            date = start_date or end_date or "Unknown"
            
            return VideoTimeInfo(
                path=str(video_path),
                date=date,
                start_time=start_time or "Unknown",
                end_time=end_time or "Unknown",
                duration=timedelta(seconds=duration_seconds)
            )
            
        finally:
            cap.release()

    def extract_multiple(self, video_paths: list, export_roi_dir: Optional[str] = None) -> list:
        """Extrae información de tiempo de múltiples videos."""
        results = []
        for path in video_paths:
            try:
                info = self.extract_video_times(path, export_roi_dir)
                results.append(info)
            except Exception as e:
                results.append(VideoTimeInfo(
                    path=str(path),
                    date="Error",
                    start_time="Error",
                    end_time="Error",
                    duration=timedelta(0)
                ))
                print(f"⚠️ Error procesando {path}: {e}")
        return results

def format_duration(td: timedelta) -> str:
    """Formatea un timedelta como HH:MM:SS."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
