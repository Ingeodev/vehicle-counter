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

    def _find_optimal_font_size(
        self, 
        text: str, 
        target_width: int, 
        target_height: int,
        min_size: int = 8,
        max_size: int = 200
    ) -> int:
        """
        Encuentra el tamaño de fuente óptimo para que el texto quepa en el área especificada.
        
        Args:
            text: Texto a renderizar
            target_width: Ancho disponible en píxeles
            target_height: Alto disponible en píxeles
            min_size: Tamaño mínimo de fuente a probar
            max_size: Tamaño máximo de fuente a probar
            
        Returns:
            Tamaño de fuente óptimo
        """
        optimal_size = min_size
        
        for size in range(min_size, max_size + 1):
            try:
                font = ImageFont.truetype(self.font_path, size)
            except IOError:
                font = ImageFont.load_default()
                break
            
            # Calcular ancho total con el factor de estiramiento (1.28)
            total_width = 0
            for char in text:
                try:
                    bbox = font.getbbox(char)
                    char_w = bbox[2] - bbox[0]
                except:
                    char_w = size // 2
                total_width += int(char_w * 1.28)
            
            # Verificar si cabe en el área
            if total_width <= target_width and size <= target_height:
                optimal_size = size
            else:
                # Ya no cabe, el anterior era el óptimo
                break
        
        return optimal_size

    def process_frame(
        self, 
        frame: np.ndarray, 
        new_date_text: str,
        top: int = None,
        right: int = None,
        bottom: int = None,
        left: int = None,
        debug: bool = False
    ) -> np.ndarray:
        """
        Procesa un frame: borra la fecha antigua y escribe la nueva.
        
        Args:
            frame: Frame de video (numpy array BGR)
            new_date_text: Texto de la nueva fecha a escribir
            top: Coordenada Y superior del bounding box (opcional)
            right: Coordenada X derecha del bounding box (opcional)
            bottom: Coordenada Y inferior del bounding box (opcional)
            left: Coordenada X izquierda del bounding box (opcional)
            debug: Si es True, dibuja el bounding box en rojo para visualización
            
        Si se proporcionan los 4 parámetros de bounding box, el tamaño de fuente
        se ajusta dinámicamente para ocupar todo el área especificada.
        Si no se proporcionan, se usa el comportamiento estático original.
        """
        height, width = frame.shape[:2]
        
        # Determinar si usar bounding box dinámico o estático
        use_dynamic_bbox = all(v is not None for v in [top, right, bottom, left])
        
        if use_dynamic_bbox:
            # Validar y ajustar límites
            left = max(0, min(left, width - 1))
            right = max(left + 1, min(right, width))
            top = max(0, min(top, height - 1))
            bottom = max(top + 1, min(bottom, height))
            
            # Crear máscara dinámica para el bounding box especificado
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
            
            # Calcular dimensiones del área
            zone_w = right - left
            zone_h = bottom - top
            
            # Encontrar tamaño de fuente óptimo para el bounding box
            font_size = self._find_optimal_font_size(
                new_date_text, 
                zone_w, 
                int(zone_h * 0.92)  # 92% de altura para margen
            )
            
            # Posición de texto
            x = left
            y = top
        else:
            # Comportamiento original: usar coordenadas relativas
            # Actualizar máscara si cambia la resolución
            if (width, height) != self._last_dims:
                self._mask = self._create_mask(width, height)
                self._last_dims = (width, height)
            
            mask = self._mask
            
            # Calcular tamaño de fuente dinámico basado en proporciones
            zone_h = int(height * self.rel_h)
            zone_w = int(width * self.rel_w)
            font_size = int(zone_h * 0.92)
            
            # Posición
            x = int(width * self.rel_x)
            y = int(height * self.rel_y)
        
        # 1. Inpainting (Borrar fecha antigua)
        clean_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        
        # Extraer brillo del frame LIMPIO (después del inpainting) para determinar color correcto
        gray_clean = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
        
        if use_dynamic_bbox:
            # === ENFOQUE ESCALADO: Renderizar texto y escalarlo al bounding box ===
            zone_w = right - left
            zone_h = bottom - top
            
            # Determinar color basado en brillo promedio del bounding box LIMPIO
            roi_gray = gray_clean[top:bottom, left:right]
            avg_brightness = np.mean(roi_gray)
            
            if avg_brightness > 127:
                text_color = (0, 0, 0)  # Negro sobre fondo claro
            else:
                text_color = (255, 255, 255)  # Blanco sobre fondo oscuro
            
            # Usar un tamaño de fuente grande para mejor calidad al escalar
            render_font_size = 100
            try:
                font = ImageFont.truetype(self.font_path, render_font_size)
            except IOError:
                font = ImageFont.load_default()
            
            # Calcular dimensiones del texto renderizado
            # Crear imagen dummy para medir
            dummy_img = Image.new('RGBA', (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)
            
            # Medir el texto completo
            text_bbox = dummy_draw.textbbox((0, 0), new_date_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            # Crear imagen del tamaño del texto con fondo transparente
            text_img = Image.new('RGBA', (text_w + 4, text_h + 4), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_img)
            
            # Dibujar texto con negrita simulada
            for dx in [0, 1]:
                text_draw.text((2 - text_bbox[0] + dx, 2 - text_bbox[1]), 
                              new_date_text, font=font, fill=(*text_color, 255))
            
            # Escalar la imagen del texto al tamaño exacto del bounding box
            text_img_scaled = text_img.resize((zone_w, zone_h), Image.Resampling.LANCZOS)
            
            # Convertir frame limpio a PIL
            img_pil = Image.fromarray(cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB))
            img_pil = img_pil.convert('RGBA')
            
            # Pegar el texto escalado en la posición del bounding box
            img_pil.paste(text_img_scaled, (left, top), text_img_scaled)
            
            # Convertir vuelta a BGR
            result = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        else:
            # === COMPORTAMIENTO ORIGINAL: para modo estático ===
            img_pil = Image.fromarray(cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except IOError:
                font = ImageFont.load_default()
            
            # Ajuste fino original: centrar verticalmente
            zone_h = int(height * self.rel_h)
            x_text = x + int(width * 0.005)
            y_text = y + (zone_h - font_size) // 2 + int(font_size * 0.10)
            
            # Pre-calcular métricas de caracteres
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
                char_w_stretched = int(char_w * 1.28)
                char_metrics.append((char, char_w_stretched, char_h))
                total_width += char_w_stretched
            
            # Dibujar cada carácter con su color óptimo
            current_x = x_text
            
            for char, char_w, char_h in char_metrics:
                roi_x1 = max(0, current_x)
                roi_y1 = max(0, y_text)
                roi_x2 = min(width, current_x + char_w + 2)
                roi_y2 = min(height, y_text + font_size + 2)
                
                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    roi_gray = gray_clean[roi_y1:roi_y2, roi_x1:roi_x2]
                    avg_brightness = np.mean(roi_gray)
                    
                    if avg_brightness > 127:
                        char_color = (0, 0, 0)
                    else:
                        char_color = (255, 255, 255)
                else:
                    char_color = (255, 255, 255)
                
                for dx in [0, 1]:
                    draw.text((current_x + dx, y_text), char, font=font, fill=char_color)
                
                current_x += char_w
            
            result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Debug: dibujar bounding box en rojo
        if debug and use_dynamic_bbox:
            cv2.rectangle(result, (left, top), (right, bottom), (0, 0, 255), 2)
        elif debug:
            # Dibujar bounding box estático en modo debug
            x = int(width * self.rel_x)
            y = int(height * self.rel_y)
            w = int(width * self.rel_w)
            h = int(height * self.rel_h)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        return result
