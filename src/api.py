"""
API de alto nivel para el paquete aforos.

Funciones que exponen la misma funcionalidad que el CLI pero como llamadas Python.
Cada función acepta los mismos parámetros que su equivalente en línea de comandos.

Example:
    >>> from src.api import fix_osd, process_video, extract_time
    >>> 
    >>> # Cambiar fecha en video (igual que: aforos fix-osd video.mp4 --date 2026-01-05)
    >>> fix_osd("video.mp4", date="2026-01-05", output="fixed.mp4", convert_h264=True)
    >>> 
    >>> # Procesar video con detección (igual que: aforos process video.mp4 --device cuda)
    >>> result = process_video("video.mp4", zones="zones.json", device="cuda")
    >>> 
    >>> # Extraer fecha/hora de videos (igual que: aforos extract-time video.mp4)
    >>> times = extract_time(["video1.mp4", "video2.mp4"], model="easyocr")
"""

import cv2
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Literal
from tqdm import tqdm
from datetime import datetime
import locale

from .utils.osd_modifier import OSDModifier
from .utils.osd_reader import OSDReader
from .config import PipelineConfig
from .pipeline import VideoPipeline


def fix_osd(
    video: str,
    date: str,
    output: Optional[str] = None,
    max_minutes: Optional[float] = None,
    font: Optional[str] = None,
    convert_h264: bool = False,
    quiet: bool = False
) -> str:
    """
    Corregir la fecha en el OSD de un video.
    
    Args:
        video: Ruta al video de entrada.
        date: Nueva fecha en formato YYYY-MM-DD o DD-MM-YYYY.
        output: Ruta de salida (default: video_fixed.mp4).
        max_minutes: Límite de minutos a procesar.
        font: Ruta a fuente TTF personalizada.
        convert_h264: Convertir salida a H.264 usando ffmpeg.
        quiet: Modo silencioso sin mensajes de progreso.
    
    Returns:
        Ruta al video de salida.
    
    Example:
        >>> fix_osd("video.mp4", date="2026-01-05", convert_h264=True)
        'video_fixed.mp4'
    """
    import os
    
    # Determinar ruta de salida
    if output is None:
        base_name = os.path.basename(video)
        output = base_name.replace(".mp4", "_fixed.mp4")
        if output == base_name:
            output = "fixed_" + base_name
    
    if not quiet:
        print(f"🔧 Corrigiendo OSD en: {video}")
        print(f"📅 Nueva fecha: {date}")
        print(f"💾 Salida: {output}")
    
    # Abrir video
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limitar frames si es necesario
    max_frames = total_frames
    if max_minutes is not None and max_minutes > 0:
        max_frames_limit = int(max_minutes * 60 * fps)
        max_frames = min(total_frames, max_frames_limit)
    
    # Usar fuente por defecto si no se especifica
    if font is None:
        default_font = Path(__file__).parent / "assets" / "AcPlus_IBM_VGA_8x16.ttf"
        if default_font.exists():
            font = str(default_font)
    
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
            
    except Exception:
        new_date = date
    
    if not quiet:
        print(f"📅 Texto OSD final: '{new_date}'")
    
    # Writer y modifier
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    modifier = OSDModifier(font_path=font)
    
    # Procesar frames
    try:
        pbar = tqdm(total=max_frames, unit="frames", disable=quiet)
        processed = 0
        
        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            fixed_frame = modifier.process_frame(frame, new_date)
            out.write(fixed_frame)
            processed += 1
            pbar.update(1)
        
        pbar.close()
    finally:
        cap.release()
        out.release()
    
    if not quiet:
        print("✅ Corrección completada")
    
    # Convertir a H.264 si se solicita
    if convert_h264:
        temp_path = output
        final_path = output.replace(".mp4", "_h264.mp4")
        if final_path == temp_path:
            final_path = output.replace(".mp4", "") + "_h264.mp4"
        
        if not quiet:
            print("🔄 Convirtiendo a H.264 con ffmpeg...")
        
        try:
            result = subprocess.run([
                "ffmpeg", "-i", temp_path,
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "copy",
                "-y", final_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                import os
                os.remove(temp_path)
                os.rename(final_path, output)
                if not quiet:
                    print(f"✅ Video H.264 guardado: {output}")
            else:
                if not quiet:
                    print(f"⚠️ Error en ffmpeg: {result.stderr}", file=sys.stderr)
        except FileNotFoundError:
            if not quiet:
                print("⚠️ ffmpeg no encontrado. Manteniendo video mp4v.", file=sys.stderr)
    
    return output


def process_video(
    video: str,
    zones: Optional[str] = None,
    mask: Optional[str] = None,
    output: str = "./output",
    base_time: str = "00:00",
    date: str = "2025-01-01",
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    model: str = "yolov8s.pt",
    reduce: int = 1,
    max_minutes: Optional[float] = None,
    strategy: Literal["box", "seg"] = "box",
    night_enhance: bool = False,
    model_config: Optional[str] = None,
    deblurring: bool = False,
    no_video: bool = False,
    quiet: bool = False
) -> "ProcessingResult":
    """
    Procesar un video con detección de vehículos.
    
    Args:
        video: Ruta al video de entrada.
        zones: Ruta al archivo JSON de zonas de conteo.
        mask: Ruta a la máscara de región de interés.
        output: Carpeta de salida.
        base_time: Hora base del video (HH:MM).
        date: Fecha del video (YYYY-MM-DD).
        device: Dispositivo para YOLO (cpu, cuda, mps).
        model: Modelo YOLO a usar.
        reduce: Factor de reducción de resolución.
        max_minutes: Límite de minutos a procesar.
        strategy: Estrategia de detección ('box' o 'seg').
        night_enhance: Mejorar visibilidad nocturna.
        model_config: Ruta a archivo YAML con parámetros del modelo.
        deblurring: Aplicar deblurring para videos nocturnos.
        no_video: No guardar video de salida.
        quiet: Modo silencioso.
    
    Returns:
        ProcessingResult con estadísticas de detección.
    
    Example:
        >>> result = process_video("video.mp4", zones="zones.json", device="cuda")
        >>> print(f"Detecciones: {result.total_detections}")
    """
    import yaml
    
    config = PipelineConfig()
    config.detector.device = device
    config.detector.model_path = model
    config.video.reduce_factor = reduce
    config.video.max_minutes = max_minutes
    config.output.save_video = not no_video
    config.output.verbose = not quiet
    config.output.output_folder = output
    config.detector.strategy = strategy
    
    # Cargar config de modelo si existe
    if model_config:
        with open(model_config, "r") as f:
            model_params = yaml.safe_load(f)
        
        if "default_threshold" in model_params:
            config.detector.confidence_threshold = float(model_params["default_threshold"])
        
        if "class_thresholds" in model_params:
            config.detector.class_thresholds = model_params["class_thresholds"]
        
        if "vehicle_classes" in model_params:
            config.detector.vehicle_classes = {int(k): str(v) for k, v in model_params["vehicle_classes"].items()}
    
    pipeline = VideoPipeline(config)
    
    result = pipeline.process_video(
        video_path=video,
        zones_path=zones,
        mask_path=mask,
        output_folder=output,
        base_time=base_time,
        date=date,
        enable_deblurring=deblurring,
        enable_night_enhance=night_enhance
    )
    
    if not quiet:
        print(f"\n✅ Completado: {result.total_detections} detecciones")
    
    return result


def extract_time(
    videos: Union[str, List[str]],
    model: Literal["tesseract", "easyocr", "trocr"] = "easyocr",
    preprocess: Literal["clahe", "binary", "color"] = "clahe",
    output_csv: Optional[str] = None,
    export_roi: Optional[str] = None,
    quiet: bool = False
) -> List[Dict]:
    """
    Extraer fecha/hora del OSD de uno o más videos.
    
    Args:
        videos: Ruta a video o lista de rutas.
        model: Motor OCR a usar (tesseract, easyocr, trocr).
        preprocess: Modo de preprocesamiento (clahe, binary, color).
        output_csv: Archivo CSV de salida (opcional).
        export_roi: Directorio para exportar ROIs procesados (debug).
        quiet: Modo silencioso.
    
    Returns:
        Lista de diccionarios con información de tiempo por video.
    
    Example:
        >>> times = extract_time(["video1.mp4", "video2.mp4"])
        >>> for t in times:
        ...     print(f"{t['video']}: {t['start_date']} {t['start_time']}")
    """
    import os
    import csv
    from .utils.osd_reader import format_duration
    
    if isinstance(videos, str):
        videos = [videos]
    
    if not quiet:
        print(f"🔍 Extrayendo información de tiempo con OCR ({model}, {preprocess})...")
        if len(videos) > 1:
            print(f"   Procesando {len(videos)} videos...")
    
    reader = OSDReader(ocr_engine=model, preprocess=preprocess)
    results = []
    
    for video_path in videos:
        if not quiet:
            print(f"\n📹 {os.path.basename(video_path)}:")
        
        export_path = None
        if export_roi:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            export_path = os.path.join(export_roi, base_name)
        
        try:
            time_info = reader.extract_video_times(video_path, export_roi_prefix=export_path)
            
            result = {
                "video": video_path,
                "start_date": time_info.get("start_date", "Unknown"),
                "start_time": time_info.get("start_time", "Unknown"),
                "end_date": time_info.get("end_date", "Unknown"),
                "end_time": time_info.get("end_time", "Unknown"),
                "duration": time_info.get("duration", "Unknown"),
            }
            results.append(result)
            
            if not quiet:
                print(f"   Inicio: {result['start_date']} {result['start_time']}")
                print(f"   Fin:    {result['end_date']} {result['end_time']}")
                print(f"   Duración: {result['duration']}")
                
        except Exception as e:
            result = {
                "video": video_path,
                "start_date": "Error",
                "start_time": str(e),
                "end_date": "",
                "end_time": "",
                "duration": "",
            }
            results.append(result)
            if not quiet:
                print(f"   ❌ Error: {e}")
    
    # Guardar CSV si se solicita
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["video", "start_date", "start_time", 
                                                    "end_date", "end_time", "duration"])
            writer.writeheader()
            writer.writerows(results)
        if not quiet:
            print(f"\n💾 CSV guardado: {output_csv}")
    
    return results
