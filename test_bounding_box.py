"""
Script de prueba para la funcionalidad de bounding box dinámico en OSDModifier.
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import locale

# Importar el modifier desde el paquete
from src.utils.osd_modifier import OSDModifier

def test_bounding_box(
    video_path: str,
    output_path: str,
    date: str,
    top: int,
    right: int,
    bottom: int,
    left: int,
    max_minutes: float = None,
    font_path: str = None,
    debug: bool = False
):
    """
    Prueba la funcionalidad de bounding box dinámico.
    """
    # Verificar video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    
    # Crear directorio de salida
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {video_path}")
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frames totales: {total_frames}")
    print(f"📍 Bounding box: top={top}, right={right}, bottom={bottom}, left={left}")
    
    # Limitar frames si es necesario
    max_frames = total_frames
    if max_minutes is not None and max_minutes > 0:
        max_frames_limit = int(max_minutes * 60 * fps)
        max_frames = min(total_frames, max_frames_limit)
        print(f"⏱️ Limitado a {max_minutes} minutos ({max_frames} frames)")
    
    # Buscar fuente por defecto
    if font_path is None:
        default_font = Path(__file__).parent / "src" / "assets" / "AcPlus_IBM_VGA_8x16.ttf"
        if default_font.exists():
            font_path = str(default_font)
            print(f"🔤 Usando fuente: {font_path}")
    
    # Procesar fecha
    try:
        original_locale = locale.getlocale(locale.LC_TIME)
        locale.setlocale(locale.LC_TIME, 'C')
        
        dt = None
        display_date = date
        
        # Formato YYYY-MM-DD
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            display_date = dt.strftime("%d-%m-%Y")
        except ValueError:
            pass
        
        # Formato DD-MM-YYYY
        if dt is None:
            try:
                dt = datetime.strptime(date, "%d-%m-%Y")
                display_date = date
            except ValueError:
                pass
        
        if dt:
            day_name = dt.strftime("%a")
            new_date = f"{display_date} {day_name}"
        else:
            new_date = date
            
        locale.setlocale(locale.LC_TIME, original_locale)
    except Exception:
        new_date = date
    
    print(f"📅 Texto OSD final: '{new_date}'")
    
    # Writer y modifier
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    modifier = OSDModifier(font_path=font_path)
    
    # Procesar frames
    try:
        pbar = tqdm(total=max_frames, unit="frames")
        processed = 0
        
        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Usar el bounding box dinámico
            fixed_frame = modifier.process_frame(
                frame, 
                new_date,
                top=top,
                right=right,
                bottom=bottom,
                left=left,
                debug=debug
            )
            out.write(fixed_frame)
            processed += 1
            pbar.update(1)
        
        pbar.close()
    finally:
        cap.release()
        out.release()
    
    print(f"✅ Video guardado: {output_path}")
    return output_path


if __name__ == "__main__":
    # Parámetros de prueba según el usuario
    test_bounding_box(
        video_path="input/videos/boulevard_oriente.mp4",
        output_path="results/bounding_boxes/boulevard_oriente_fixed.mp4",
        date="2026-01-07",  # Fecha de hoy
        top=60,
        right=381,
        bottom=93,
        left=50,
        max_minutes=0.1,
        debug=True
    )
