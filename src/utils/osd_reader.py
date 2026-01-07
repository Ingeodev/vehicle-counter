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
    duration: timedelta
    roi_coords: dict  # {'x1': int, 'y1': int, 'x2': int, 'y2': int}


class OCRStrategy(ABC):
    """Clase base abstracta para estrategias de OCR."""
    
    @abstractmethod
    def recognize_text(self, image: np.ndarray) -> Tuple[str, Optional[dict]]:
        """
        Reconoce texto de una imagen.
        
        Args:
            image: Imagen en escala de grises o color (numpy array)
            
        Returns:
            Tuple[text, bbox_dict]
            bbox_dict: {'x1': int, 'y1': int, 'x2': int, 'y2': int} o None
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

    def recognize_text(self, image: np.ndarray) -> Tuple[str, Optional[dict]]:
        # Configuración optimizada para texto de fecha/hora
        config = '--psm 7 -c tessedit_char_whitelist=0123456789:-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        text = self._tesseract.image_to_string(image, config=config)
        return text.strip(), None


import threading

class EasyOCRStrategy(OCRStrategy):
    """Estrategia OCR usando EasyOCR."""
    
    # Cache a nivel de clase y Lock de inicialización
    _cached_reader = None
    _init_lock = threading.Lock()

    def __init__(self):
        pass

    def _get_reader(self):
        # Doble verificación (Double-Checked Locking) para eficiencia
        if EasyOCRStrategy._cached_reader is None:
            with EasyOCRStrategy._init_lock:
                if EasyOCRStrategy._cached_reader is None:
                    try:
                        import easyocr
                        import torch
                        use_gpu = torch.cuda.is_available()
                        print(f"⏳ Cargando modelo EasyOCR (GPU={use_gpu})...")
                        EasyOCRStrategy._cached_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                        if use_gpu:
                            print("🚀 EasyOCR cargado en GPU")
                    except ImportError:
                        raise ImportError(
                            "easyocr no está instalado. "
                            "Instala con: pip install easyocr"
                        )
        return EasyOCRStrategy._cached_reader

    def recognize_text(self, image: np.ndarray) -> Tuple[str, Optional[dict]]:
        import re
        reader = self._get_reader()
        # detail=1 devuelve (bbox, text, prob)
        results = reader.readtext(image, detail=1)
        
        if not results:
            return "", None
        
        # Patrones para identificar texto de fecha/hora
        datetime_pattern = re.compile(r'[\d:/-]')  # Contiene dígitos, :, /, -
        day_pattern = re.compile(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', re.IGNORECASE)
            
        text_parts = []
        relevant_boxes = []
        
        for bbox, text, prob in results:
            if not text.strip():
                continue
            
            text_clean = text.strip()
            
            # Solo incluir cajas que parecen fecha/hora:
            # - Contienen dígitos Y (: o - o /)
            # - O son días de la semana
            has_datetime_chars = bool(re.search(r'\d', text_clean) and datetime_pattern.search(text_clean))
            is_day_name = bool(day_pattern.search(text_clean))
            
            if has_datetime_chars or is_day_name:
                text_parts.append(text_clean)
                relevant_boxes.append(bbox)
        
        final_text = ' '.join(text_parts).strip()
        
        if not relevant_boxes:
            return final_text, None
        
        # Calcular union solo de cajas relevantes
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for bbox in relevant_boxes:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            min_x = min(min_x, min(xs))
            min_y = min(min_y, min(ys))
            max_x = max(max_x, max(xs))
            max_y = max(max_y, max(ys))
        
        # Ajuste fino: encontrar límites exactos del texto usando análisis de píxeles
        x1, y1, x2, y2 = int(min_x), int(min_y), int(max_x), int(max_y)
        
        # Asegurar límites dentro de la imagen
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            # Extraer región y convertir a escala de grises
            roi = image[y1:y2, x1:x2]
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Binarizar para encontrar texto (oscuro sobre claro o claro sobre oscuro)
            mean_val = np.mean(gray)
            if mean_val > 127:
                # Fondo claro, texto oscuro
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                # Fondo oscuro, texto claro
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Encontrar filas y columnas con contenido
            row_sums = np.sum(binary, axis=1)
            col_sums = np.sum(binary, axis=0)
            
            # Umbral mínimo para considerar que hay contenido (evitar ruido)
            threshold = binary.shape[1] * 2  # Al menos 2 píxeles por fila en promedio
            
            rows_with_content = np.where(row_sums > threshold)[0]
            cols_with_content = np.where(col_sums > 0)[0]
            
            if len(rows_with_content) > 0 and len(cols_with_content) > 0:
                # Ajustar límites verticales (más estrictos)
                tight_y1 = y1 + rows_with_content[0]
                tight_y2 = y1 + rows_with_content[-1] + 1
                
                # Ajustar límites horizontales
                tight_x1 = x1 + cols_with_content[0]
                tight_x2 = x1 + cols_with_content[-1] + 1
                
                x1, y1, x2, y2 = tight_x1, tight_y1, tight_x2, tight_y2
            
        bbox_dict = {
            'x1': int(x1), 
            'y1': int(y1), 
            'x2': int(x2), 
            'y2': int(y2)
        }
        return final_text, bbox_dict


class TrOCRStrategy(OCRStrategy):
    """
    Estrategia OCR usando Vision Transformer (TrOCR).
    Requiere 'transformers' y 'torch'.
    """
    
    # Class-level cache and Lock
    _cached_processor = None
    _cached_model = None
    _cached_device = None
    _init_lock = threading.Lock()

    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        self.model_name = model_name
        
    def _load_model(self):
        # Double-Checked Locking
        if TrOCRStrategy._cached_model is None:
            with TrOCRStrategy._init_lock:
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
    
    def recognize_text(self, image: np.ndarray) -> Tuple[str, Optional[dict]]:
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
        
        return generated_text.strip(), None


class OSDReader:
    """
    Lee texto del OSD (On-Screen Display) en videos usando OCR.
    Usa el patrón Strategy para delegar el reconocimiento de texto.
    """
    
    # Coordenadas de referencia para 1920x1080
    REF_WIDTH = 1920.0
    REF_HEIGHT = 1080.0
    
    # Valores por defecto (si no se especifican otros)
    # Cubre desde la esquina hasta mitad (0.5) del ancho y un séptimo (0.143) de la altura
    DEFAULT_X_PCT = 0.5
    DEFAULT_Y_PCT = 1.0 / 7.0
    
    def __init__(self, ocr_engine: Literal["tesseract", "easyocr", "trocr"] = "easyocr",
                 preprocess: Literal["clahe", "binary", "color"] = "clahe",
                 roi_x: float = DEFAULT_X_PCT,
                 roi_y: float = DEFAULT_Y_PCT,
                 corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "top_left"):
        """
        Inicializa el lector OSD con opciones de ROI dinámico.
        
        Args:
            ocr_engine: Motor OCR a usar.
            preprocess: Estrategia de preprocesamiento de imagen.
            roi_x: Porcentaje del ancho a capturar (0.0 a 1.0). Default 0.5.
            roi_y: Porcentaje de la altura a capturar (0.0 a 1.0). Default 1/7.
            corner: Esquina de anclaje ("top_left", "top_right", "bottom_left", "bottom_right").
        """
        self.preprocess_mode = preprocess
        self.strategy = self._get_strategy(ocr_engine)
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.corner = corner
        
    def _get_strategy(self, engine_name: str) -> OCRStrategy:
        if engine_name == "tesseract":
            return TesseractStrategy()
        elif engine_name == "easyocr":
            return EasyOCRStrategy()
        elif engine_name == "trocr":
            return TrOCRStrategy()
        else:
            raise ValueError(f"Motor OCR no soportado: {engine_name}")

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Extrae la región de interés (ROI) basada en configuración dinámica.
        Returns:
            Tuple[imagen_recortada, dict_coords]
        """
        h, w = frame.shape[:2]
        
        # Calcular dimensiones del recorte
        crop_w = int(w * self.roi_x)
        crop_h = int(h * self.roi_y)
        
        # Calcular coordenadas segun la esquina
        if self.corner == "top_left":
            x1, y1 = 0, 0
            x2, y2 = crop_w, crop_h
        elif self.corner == "top_right":
            x1, y1 = w - crop_w, 0
            x2, y2 = w, crop_h
        elif self.corner == "bottom_left":
            x1, y1 = 0, h - crop_h
            x2, y2 = crop_w, h
        elif self.corner == "bottom_right":
            x1, y1 = w - crop_w, h - crop_h
            x2, y2 = w, h
        else:
            x1, y1 = 0, 0
            x2, y2 = crop_w, crop_h
            
        # Asegurar límites
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        coords = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        return frame[y1:y2, x1:x2].copy(), coords

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """Pre-procesa el ROI para mejorar la precisión del OCR."""
        if roi.size == 0: return roi
        
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

    def read_frame(self, frame: np.ndarray, export_roi_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str], dict]:
        """Lee la fecha y hora del OSD en un frame. Retorna (date, time, coords)."""
        roi, coords = self._extract_roi(frame)
        processed = self._preprocess_roi(roi)
        
        if export_roi_path:
            cv2.imwrite(export_roi_path, processed)
        
        text, text_bbox = self.strategy.recognize_text(processed)
        d, t = self._parse_datetime(text)
        
        # Calcular coordenadas reales del texto si es posible
        final_coords = coords  # Default: ROI de búsqueda
        
        if text_bbox:
            # Revertir las transformaciones de _preprocess_roi
            # 1. Quitar borde (si aplica)
            # 2. Deshacer escalado (x2)
            # 3. Sumar offset del ROI en el frame
            
            border_x, border_y = 0, 0
            if self.preprocess_mode in ["clahe", "binary"]:
                border_x, border_y = 10, 10
                
            scale = 2.0
            
            # Coordenadas en 'roi' (antes de preproceso)
            x1_roi = int((text_bbox['x1'] - border_x) / scale)
            y1_roi = int((text_bbox['y1'] - border_y) / scale)
            x2_roi = int((text_bbox['x2'] - border_x) / scale)
            y2_roi = int((text_bbox['y2'] - border_y) / scale)
            
            # Clamping a 0 (por si el borde negativo)
            x1_roi = max(0, x1_roi)
            y1_roi = max(0, y1_roi)
            
            # Coordenadas en 'frame'
            roi_x_offset = coords['x1']
            roi_y_offset = coords['y1']
            
            final_coords = {
                'x1': roi_x_offset + x1_roi,
                'y1': roi_y_offset + y1_roi,
                'x2': roi_x_offset + x2_roi,
                'y2': roi_y_offset + y2_roi
            }
            
            # Ajuste fino: refinar límites usando análisis de píxeles del frame original
            x1, y1 = final_coords['x1'], final_coords['y1']
            x2, y2 = final_coords['x2'], final_coords['y2']
            
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                text_roi = frame[y1:y2, x1:x2]
                if len(text_roi.shape) == 3:
                    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray = text_roi
                
                # Binarizar para encontrar texto
                mean_val = np.mean(gray)
                if mean_val > 127:
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                else:
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Análisis por columnas: encontrar primer y último píxel blanco por columna
                # Luego usar la moda para determinar límites superior e inferior
                top_pixels = []
                bottom_pixels = []
                
                for col in range(binary.shape[1]):
                    column = binary[:, col]
                    white_indices = np.where(column > 0)[0]
                    
                    if len(white_indices) > 0:
                        top_pixels.append(white_indices[0])
                        bottom_pixels.append(white_indices[-1])
                
                if len(top_pixels) > 5 and len(bottom_pixels) > 5:  # Suficientes columnas
                    from scipy import stats
                    
                    # Calcular moda de los límites
                    top_mode = stats.mode(top_pixels, keepdims=True)[0][0]
                    bottom_mode = stats.mode(bottom_pixels, keepdims=True)[0][0]
                    
                    # Aplicar ajuste
                    tight_y1 = y1 + int(top_mode)
                    tight_y2 = y1 + int(bottom_mode) + 1
                    
                    # Validar que los límites son razonables
                    if tight_y2 > tight_y1:
                        final_coords = {
                            'x1': int(x1),
                            'y1': int(tight_y1),
                            'x2': int(x2),
                            'y2': int(tight_y2)
                        }
            
        return d, t, final_coords
    
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
            
            # Leer inicio con fallback - si primer frame falla, intentar otros y calcular hacia atrás
            start_date, start_time, start_coords = None, None, {}
            start_frame_offsets = [0, 10, 30, 60, 150, 300]  # Frames a intentar
            
            for frame_offset in start_frame_offsets:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                export_path_for_frame = export_start if frame_offset == 0 else None
                date, time_str, coords = self.read_frame(frame, export_path_for_frame)
                
                if time_str and time_str != "Unknown":
                    # Si no es el primer frame, calcular la hora hacia atrás
                    if frame_offset > 0 and fps > 0:
                        try:
                            from datetime import datetime as dt
                            # Parsear la hora encontrada
                            time_parts = time_str.split(':')
                            if len(time_parts) == 3:
                                h, m, s = int(time_parts[0]), int(time_parts[1]), int(time_parts[2])
                                frame_time = dt(2000, 1, 1, h, m, s)
                                
                                # Restar el tiempo del offset de frames
                                seconds_offset = frame_offset / fps
                                adjusted_time = frame_time - timedelta(seconds=seconds_offset)
                                time_str = adjusted_time.strftime("%H:%M:%S")
                        except Exception:
                            pass  # Si falla el cálculo, usar la hora sin ajustar
                    
                    start_date = date
                    start_time = time_str
                    start_coords = coords
                    
                    # Exportar ROI si corresponde y no se grabó en el primer intento
                    if export_start and frame_offset > 0:
                        self.read_frame(frame, export_start)
                    break
            
            # Leer final con fallback
            end_date, end_time = None, None
            for offset in [1, 10, 50, 100, 500]:
                target_frame = max(1, total_frames - offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, last_frame = cap.read()
                if ret:
                    end_date, end_time, _ = self.read_frame(last_frame, export_end if offset == 1 else None)
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
                duration=timedelta(seconds=duration_seconds),
                roi_coords=start_coords
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
                    duration=timedelta(0),
                    roi_coords={}
                ))
                print(f"⚠️ Error procesando {path}: {e}")
        return results

def format_duration(td: timedelta) -> str:
    """Formatea un timedelta como HH:MM:SS."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
