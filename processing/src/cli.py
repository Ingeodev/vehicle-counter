#!/usr/bin/env python
"""
CLI - Interfaz de línea de comandos para el contador de vehículos.
"""

import argparse
import sys
import yaml
from pathlib import Path

from .config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Contador de vehículos con YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Procesar un video
  python -m src.cli process video.mp4 --zones zonas.json --output ./output

  # Procesar directorio completo
  python -m src.cli scan /path/to/videos --output ./output --recursive

  # Usar GPU
  python -m src.cli process video.mp4 --device cuda
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Comando: process
    process_parser = subparsers.add_parser("process", help="Procesar un video")
    process_parser.add_argument("video", help="Ruta al video de entrada")
    process_parser.add_argument("--zones", "-z", help="Ruta al archivo JSON de zonas")
    process_parser.add_argument("--mask", "-m", help="Ruta a la máscara")
    process_parser.add_argument("--output", "-o", default="./output", help="Carpeta de salida")
    process_parser.add_argument("--base-time", "-t", default="00:00", help="Hora base del video (HH:MM)")
    process_parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda", "mps"],
                               help="Dispositivo para YOLO")
    process_parser.add_argument("--model", default="yolov8s.pt", help="Modelo YOLO a usar")
    process_parser.add_argument("--reduce", "-r", type=int, default=1,
                               help="Factor de reducción de resolución")
    process_parser.add_argument("--max-minutes", type=float, help="Límite de minutos a procesar")
    process_parser.add_argument("--strategy",choices=["box", "seg"], default="box",
                               help="Estrategia de detección: 'box' (cajas) o 'seg' (segmentación)")
    process_parser.add_argument("--night-enhance", action="store_true",
                               help="Mejorar visibilidad nocturna (Gamma + CLAHE) sin distorsión")
    process_parser.add_argument("--date", default="2025-01-01",  # Default provisional
                               help="Fecha del video (YYYY-MM-DD)")
    process_parser.add_argument("--model-config", type=str,
                               help="Ruta a archivo YAML con parámetros del modelo (thresholds por clase)")

    process_parser.add_argument("--no-video", action="store_true", help="No guardar video de salida")
    process_parser.add_argument("--deblurring", action="store_true", 
                               help="Aplicar deblurring agresivo (para videos nocturnos con motion blur)")
    process_parser.add_argument("--quiet", "-q", action="store_true", help="Modo silencioso")
    
    # Comando: scan
    scan_parser = subparsers.add_parser("scan", help="Procesar directorio de videos")
    scan_parser.add_argument("directory", help="Directorio raíz a escanear")
    scan_parser.add_argument("--output", "-o", default="./output", help="Carpeta base de salida")
    scan_parser.add_argument("--recursive", "-R", action="store_true", help="Escanear subdirectorios")
    scan_parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda", "mps"])
    scan_parser.add_argument("--model", default="yolov8s.pt", help="Modelo YOLO")
    scan_parser.add_argument("--reduce", "-r", type=int, default=1)
    scan_parser.add_argument("--max-minutes", type=float)
    scan_parser.add_argument("--zones", "-z", help="Ruta global al archivo JSON de zonas (override)")
    scan_parser.add_argument("--mask", "-m", help="Ruta global a la máscara (override)")
    scan_parser.add_argument("--strategy",choices=["box", "seg"], default="box",
                               help="Estrategia de detección: 'box' (cajas) o 'seg' (segmentación)")
    scan_parser.add_argument("--night-enhance", action="store_true",
                               help="Mejorar visibilidad nocturna")
    scan_parser.add_argument("--date", default="2025-01-01",
                               help="Fecha común para todos los videos (YYYY-MM-DD)")
    scan_parser.add_argument("--model-config", type=str,
                               help="Ruta a archivo YAML con parámetros del modelo")
    scan_parser.add_argument("--deblurring", action="store_true", help="Aplicar deblurring")
    scan_parser.add_argument("--yes", "-y", action="store_true", help="Confirmar automáticamente sin preguntar")
    scan_parser.add_argument("--quiet", "-q", action="store_true")
    
    # Comando: fix-osd
    fix_parser = subparsers.add_parser("fix-osd", help="Corregir fecha en OSD (Inpainting)")
    fix_parser.add_argument("video", help="Video de entrada")
    fix_parser.add_argument("--date", required=True, help="Nueva fecha (YYYY-MM-DD)")
    fix_parser.add_argument("--output", "-o", help="Archivo de salida (default: video_fixed.mp4)")
    fix_parser.add_argument("--max-minutes", type=float, help="Límite de minutos a procesar")
    fix_parser.add_argument("--font", help="Ruta fuente .ttf personalizada")
    fix_parser.add_argument("--quiet", "-q", action="store_true")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Ejecutar comando
    if args.command == "process":
        return cmd_process(args)
    elif args.command == "scan":
        return cmd_scan(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "fix-osd":
        return cmd_fix_osd(args)
    
    return 0


def cmd_process(args):
    """Comando: procesar un video."""
    from .config import PipelineConfig
    from .pipeline import VideoPipeline
    
    # Configurar
    config = PipelineConfig()
    config.detector.device = args.device
    config.detector.model_path = args.model
    config.video.reduce_factor = args.reduce
    config.video.max_minutes = args.max_minutes
    config.output.save_video = not args.no_video
    config.output.verbose = not args.quiet
    config.output.output_folder = args.output
    config.output.output_folder = args.output
    config.detector.strategy = args.strategy
    
    # Sobrescribir modelo si se especifica
    if args.model:
        config.detector.model_path = args.model
    
    # Cargar config de modelo si existe
    if args.model_config:
        try:
            with open(args.model_config, "r") as f:
                model_params = yaml.safe_load(f)
                
            if "default_threshold" in model_params:
                config.detector.confidence_threshold = float(model_params["default_threshold"])
                
            if "class_thresholds" in model_params:
                config.detector.class_thresholds = model_params["class_thresholds"]
                
            if not args.quiet:
                print(f"✅ Configuración de modelo cargada desde: {args.model_config}")
                
        except Exception as e:
            print(f"❌ Error al cargar config de modelo: {e}")
            sys.exit(1)
    
    # Crear pipeline
    pipeline = VideoPipeline(config)
    
    # Procesar
    try:
        result = pipeline.process_video(
            video_path=args.video,
            zones_path=args.zones,
            mask_path=args.mask,
            output_folder=args.output,
            base_time=args.base_time,
            date=args.date,
            enable_deblurring=args.deblurring,
            enable_night_enhance=args.night_enhance
        )
        
        if not args.quiet:
            print(f"\n✅ Completado: {result.total_detections} detecciones")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def cmd_scan(args):
    """Comando: escanear y procesar directorio."""
    from .config import PipelineConfig
    from .pipeline import VideoPipeline
    
    # Configurar
    config = PipelineConfig()
    config.detector.device = args.device
    config.detector.model_path = args.model
    config.video.reduce_factor = args.reduce
    config.video.max_minutes = args.max_minutes
    config.output.verbose = not args.quiet
    config.output.output_folder = args.output
    config.detector.strategy = args.strategy
    
    # Sobrescribir modelo si se especifica
    if args.model:
        config.detector.model_path = args.model
    
    # Cargar config de modelo si existe
    if args.model_config:
        try:
            with open(args.model_config, "r") as f:
                model_params = yaml.safe_load(f)
                
            if "default_threshold" in model_params:
                config.detector.confidence_threshold = float(model_params["default_threshold"])
                
            if "class_thresholds" in model_params:
                config.detector.class_thresholds = model_params["class_thresholds"]
                
            if not args.quiet:
                print(f"✅ Configuración de modelo cargada desde: {args.model_config}")
                
        except Exception as e:
            print(f"❌ Error al cargar config de modelo: {e}")
            sys.exit(1)
    
    # Crear pipeline
    pipeline = VideoPipeline(config)
    
    # Escanear primero
    scan_result = pipeline.scan_directory(args.directory, args.recursive)
    
    if not args.quiet:
        print(f"\n📹 Encontrados {len(scan_result.videos)} videos")
        print(f"⏱️ Duración total estimada: {scan_result.total_duration_minutes / 60:.1f} horas")
        
        if not args.yes:
            response = input("\n¿Procesar todos? [y/N] ")
            if response.lower() != "y":
                print("Cancelado.")
                return 0
    
    # Procesar
    try:
        results = pipeline.process_directory(
            root_path=args.directory,
            output_base=args.output,
            recursive=args.recursive,
            mask_path=args.mask,
            zones_path=args.zones,
            enable_deblurring=args.deblurring,
            enable_night_enhance=args.night_enhance,
            date=args.date
        )
        
        if not args.quiet:
            total_detections = sum(r.total_detections for r in results)
            print(f"\n✅ Completado: {len(results)} videos, {total_detections} detecciones")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def cmd_info(args):
    """Comando: mostrar información de un video."""
    import cv2
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {args.video}", file=sys.stderr)
        return 1
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    
    cap.release()
    
    print(f"\n📹 Video: {args.video}")
    print(f"   Resolución: {width}x{height}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Frames: {frames}")
    print(f"   Duración: {duration / 60:.1f} minutos")
    
    return 0


def cmd_fix_osd(args):
    """Comando: corregir OSD en video."""
    import cv2
    from tqdm import tqdm
    from .utils.osd_modifier import OSDModifier
    
    input_path = args.video
    output_path = args.output
    
    if not output_path:
        import os
        base_name = os.path.basename(input_path)
        output_path = base_name.replace(".mp4", "_fixed.mp4")
        if output_path == base_name: # Prevent replacement
            output_path = "fixed_" + base_name
            
    if not args.quiet:
        print(f"🔧 Corrigiendo OSD en: {input_path}")
        print(f"📅 Nueva fecha: {args.date}")
        print(f"💾 Salida: {output_path}")
        
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Error abriendo video", file=sys.stderr)
        return 1
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limitar frames si se especifica max-minutes
    max_frames = total_frames
    if args.max_minutes is not None and args.max_minutes > 0:
        max_frames_limit = int(args.max_minutes * 60 * fps)
        max_frames = min(total_frames, max_frames_limit)
    
    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    modifier = OSDModifier(font_path=args.font)
    
    # Procesar fecha para añadir día de la semana (Ej: 31-12-2025 -> 31-12-2025 Wed)
    from datetime import datetime
    import locale
    
    try:
        # Forzar locale inglés para el día de la semana
        original_locale = locale.getlocale(locale.LC_TIME)
        locale.setlocale(locale.LC_TIME, 'C') 
        
        # Asumimos formato DD-MM-YYYY como en el input
        dt = datetime.strptime(args.date, "%d-%m-%Y")
        day_name = dt.strftime("%a") # Mon, Tue, Wed...
        
        new_date = f"{args.date} {day_name}"
        
        # Restaurar locale (buena práctica)
        # locale.setlocale(locale.LC_TIME, original_locale) 
    except ValueError:
        # Si falla el parseo (otro formato), usar el string original
        print(f"⚠️ Formato de fecha no reconocido (esperado DD-MM-YYYY), usando texto original: {args.date}")
        new_date = args.date
        
    if not args.quiet:
        print(f"📅 Texto OSD final: '{new_date}'")
    
    processed_count = 0
    try:
        pbar = tqdm(total=max_frames, unit="frames", disable=args.quiet)
        
        while processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            fixed_frame = modifier.process_frame(frame, new_date)
            out.write(fixed_frame)
            processed_count += 1
            pbar.update(1)
            
        pbar.close()
        
    finally:
        cap.release()
        out.release()
        
    if not args.quiet:
        print("✅ Corrección completada")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
