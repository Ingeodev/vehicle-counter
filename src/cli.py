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
  python -m mglon_vehicle_counter.cli process video.mp4 --zones zonas.json --output ./output
  
  # Escanear directorio
  python -m mglon_vehicle_counter.cli scan /path/to/videos --output ./output --recursive
  
  # Usar CUDA
  python -m mglon_vehicle_counter.cli process video.mp4 --device cuda
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
    
    # Comando:    # Fix OSD
    fix_parser = subparsers.add_parser("fix-osd", help="Corregir fecha en OSD")
    fix_parser.add_argument("video", help="Video de entrada")
    fix_parser.add_argument("--date", required=True, help="Nueva fecha (YYYY-MM-DD o DD-MM-YYYY)")
    fix_parser.add_argument("--output", help="Video de salida")
    fix_parser.add_argument("--minutes", type=float, help="Limitar minutos")
    fix_parser.add_argument("--font", help="Ruta a fuente TTF")
    fix_parser.add_argument("--convert-h264", action="store_true", help="(Deprecado) Usar --codec h264")
    fix_parser.add_argument("--codec", choices=["copy", "h264", "h265"], default="copy", help="Codec de salida")
    fix_parser.add_argument("--quiet", "-q", action="store_true", help="Modo silencioso")
    fix_parser.set_defaults(func=cmd_fix_osd)
    # Comando: extract-time
    extract_parser = subparsers.add_parser("extract-time", help="Extraer fecha/hora del OSD de videos")
    extract_parser.add_argument("video", nargs="+", help="Video(s) de entrada")
    extract_parser.add_argument("--model", "-m", choices=["tesseract", "easyocr", "trocr"], default="easyocr",
                               help="Motor OCR a usar (default: easyocr)")
    extract_parser.add_argument("--preprocess", "-p", choices=["clahe", "binary", "color"], default="clahe",
                               help="Modo de preprocesamiento (default: clahe)")
    extract_parser.add_argument("--output", "-o", help="Archivo CSV de salida (opcional)")
    extract_parser.add_argument("--export-roi", help="Directorio para guardar ROIs procesados (debug)")
    extract_parser.add_argument("--quiet", "-q", action="store_true")
    
    
    
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
    elif args.command == "extract-time":
        return cmd_extract_time(args)
    
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

            if "vehicle_classes" in model_params:
                config.detector.vehicle_classes = {int(k): str(v) for k, v in model_params["vehicle_classes"].items()}
                
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

            if "vehicle_classes" in model_params:
                config.detector.vehicle_classes = {int(k): str(v) for k, v in model_params["vehicle_classes"].items()}
                
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
    """
    Función envoltorio para llamar a la API de fix_osd desde el CLI.
    """
    from mglon_vehicle_counter.api import fix_osd
    from mglon_vehicle_counter.exceptions import AforosError
    
    try:
        fix_osd(
            video=args.video,
            date=args.date,
            output=args.output,
            max_minutes=args.minutes,
            font=args.font,
            convert_h264=args.convert_h264,
            codec=args.codec,
            quiet=args.quiet
        )
    except AforosError as e:
        print(f"❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        exit(1)


def cmd_extract_time(args):
    """Comando: extraer fecha/hora del OSD de videos."""
    from .utils.osd_reader import OSDReader, format_duration
    import csv
    
    if not args.quiet:
        print(f"🔍 Extrayendo información de tiempo con OCR ({args.model}, {args.preprocess})...")
        if len(args.video) > 1:
            print(f"   Procesando {len(args.video)} videos...")
        if args.export_roi:
            print(f"   Exportando ROIs a: {args.export_roi}")
    
    # Crear reader con el motor y preprocesamiento seleccionados
    try:
        reader = OSDReader(ocr_engine=args.model, preprocess=args.preprocess)
    except ImportError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    # Procesar videos
    results = reader.extract_multiple(args.video, export_roi_dir=args.export_roi)
    
    
    # Mostrar resultados
    if not args.quiet:
        print("\n" + "=" * 80)
        print(f"{'Video':<40} {'Fecha':<12} {'Inicio':<10} {'Fin':<10} {'Duración':<10}")
        print("=" * 80)
        
        for info in results:
            # Truncar nombre de video si es muy largo
            video_name = Path(info.path).name
            if len(video_name) > 38:
                video_name = video_name[:35] + "..."
            
            duration_str = format_duration(info.duration)
            
            print(f"{video_name:<40} {info.date:<12} {info.start_time:<10} {info.end_time:<10} {duration_str:<10}")
        
        print("=" * 80)
    
    # Exportar a CSV si se especifica
    if args.output:
        try:
            with open(args.output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['video', 'date', 'start_time', 'end_time', 'duration'])
                
                for info in results:
                    writer.writerow([
                        info.path,
                        info.date,
                        info.start_time,
                        info.end_time,
                        format_duration(info.duration)
                    ])
            
            if not args.quiet:
                print(f"\n💾 Resultados exportados a: {args.output}")
                
        except IOError as e:
            print(f"❌ Error escribiendo CSV: {e}", file=sys.stderr)
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

