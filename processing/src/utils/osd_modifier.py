"""
OSD Modifier - Utilidad para modificar texto en pantalla (On-Screen Display).
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple

class OSDModifier:
    """
    Modifica elementos del OSD en frames de video.
    Específicamente diseñado para reemplazar la fecha manteniendo el estilo Hikvision.
    """
    
    def __init__(self, font_path: str = None):
        # Coordenadas relativas basadas en frame 1920x1080
        # [[20, 48], [397, 103]]
        # x=20, y=48, w=377, h=55
        self.ref_w = 1920.0
        self.ref_h = 1080.0
        
        self.rel_x = 20 / self.ref_w
        self.rel_y = 48 / self.ref_h
        self.rel_w = 377 / self.ref_w
        self.rel_h = 55 / self.ref_h
        
        # Cache para máscara
        self._last_dims = (0, 0)
        self._mask = None
        
        # Configurar fuente
        if font_path:
             self.font_path = font_path
             print(f"✅ Usando fuente especificada: {self.font_path}")
        else:
            # Búsqueda automática
            self.font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Fallback
            
            # Prioridad: 1. input/assets/*.ttf, 2. Root *.ttf
            import os
            from pathlib import Path
            
            search_paths = [
                Path("input/assets"),
                Path(".")
            ]
            
            for path in search_paths:
                if path.exists():
                    fonts = list(path.glob("*.ttf"))
                    if fonts:
                        self.font_path = str(fonts[0])
                        print(f"✅ Fuente personalizada encontrada: {self.font_path}")
                        break
        
    def _create_mask(self, width: int, height: int) -> np.ndarray:
        """Crea la máscara para inpainting basada en la resolución actual."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        x = int(width * self.rel_x)
        y = int(height * self.rel_y)
        w = int(width * self.rel_w)
        h = int(height * self.rel_h)
        
        # Dibujar rectángulo blanco en la zona a borrar
        # Expandimos ligeramente para asegurar cobertura
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        return mask

    def process_frame(self, frame: np.ndarray, new_date_text: str) -> np.ndarray:
        """
        Procesa un frame: borra la fecha antigua y escribe la nueva.
        """
        height, width = frame.shape[:2]
        
        # Actualizar máscara si cambia la resolución
        if (width, height) != self._last_dims:
            self._mask = self._create_mask(width, height)
            self._last_dims = (width, height)
        
        # 1. Inpainting (Borrar fecha antigua)
        clean_frame = cv2.inpaint(frame, self._mask, 3, cv2.INPAINT_TELEA)
        
        # 2. Convertir a PIL para dibujar texto Unicode
        img_pil = Image.fromarray(cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Calcular tamaño de fuente dinámico
        # Altura del área de texto: ~56px en 576p
        zone_h = int(height * self.rel_h)
        font_size = int(zone_h * 0.92) # 92% de la altura de la zona (antes 80%)
        
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
            
        # Posición
        x = int(width * self.rel_x)
        y = int(height * self.rel_y)
        
        # Ajuste fino: centrar verticalmente
        # En PIL, draw.text((x, y)) dibuja con la esquina superior izquierda
        # Vamos a añadir un pequeño margen a la izquierda
        x_text = x + int(width * 0.005)
        # Y centrar verticalmente en la zona
        # Ajuste visual (subido ~5% respecto al 0.15 anterior para alinear con la hora)
        y_text = y + (zone_h - font_size) // 2 + int(font_size * 0.10)
        
        # IMPORTANTE: Extraer brillo del frame ORIGINAL (no inpainted)
        gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- ENFOQUE OPTIMIZADO: Color POR CARÁCTER con ROIs pequeños ---
        # Pre-calcular métricas de caracteres una sola vez
        char_metrics = []
        total_width = 0
        for char in new_date_text:
            try:
                bbox = font.getbbox(char)
                char_w = bbox[2] - bbox[0]
                char_h = bbox[3] - bbox[1]
            except:
                char_w = font_size // 2
                char_h = font_size
            # Estirar horizontalmente (1.166 * 1.1 = 1.28)
            char_w_stretched = int(char_w * 1.28)
            char_metrics.append((char, char_w_stretched, char_h))
            total_width += char_w_stretched
        
        # Dibujar cada carácter con su color óptimo
        current_x = x_text
        char_index = 0
        day_start_index = len(new_date_text) - 3  # Últimos 3 caracteres son el día (Wed, Mon, etc.)
        
        for char, char_w, char_h in char_metrics:
            # Definir ROI pequeño alrededor del carácter (solo lo necesario)
            roi_x1 = max(0, current_x)
            roi_y1 = max(0, y_text)
            roi_x2 = min(width, current_x + char_w + 2)
            roi_y2 = min(height, y_text + font_size + 2)
            
            # Extraer brillo promedio del ROI pequeño (muy rápido)
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi_gray = gray_original[roi_y1:roi_y2, roi_x1:roi_x2]
                avg_brightness = np.mean(roi_gray)
                
                # Elegir color
                if avg_brightness > 127:
                    char_color = (0, 0, 0)  # Negro sobre fondo claro
                else:
                    char_color = (255, 255, 255)  # Blanco sobre fondo oscuro
            else:
                char_color = (255, 255, 255)
            
            # Dibujar este carácter con NEGRITA simulada (para todo el texto)
            # Bold simulation: dibujar en posiciones +1px horizontalmente
            for dx in [0, 1]:
                draw.text((current_x + dx, y_text), char, font=font, fill=char_color)
            
            # Avanzar posición X
            current_x += char_w
            char_index += 1
        
        # Convertir vuelta a BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
