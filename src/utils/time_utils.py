"""
time_utils - Utilidades para manejo de tiempos.
"""

from datetime import datetime, timedelta


def format_timestamp(seconds: float) -> str:
    """
    Convierte segundos a formato MM:SS.
    
    Args:
        seconds: Tiempo en segundos
        
    Returns:
        String en formato MM:SS
        
    Example:
        >>> format_timestamp(125.5)
        '02:05'
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def calculate_exact_time(base_time: str, elapsed_seconds: float) -> str | None:
    """
    Calcula la hora exacta sumando tiempo base + segundos transcurridos.
    
    Args:
        base_time: Hora base en formato HH:MM o HH:MM:SS
        elapsed_seconds: Segundos transcurridos desde el inicio
        
    Returns:
        Hora exacta en formato HH:MM:SS o None si hay error
        
    Example:
        >>> calculate_exact_time("14:30", 125)
        '14:32:05'
    """
    try:
        # Determinar formato
        parts = base_time.split(":")
        if len(parts) == 3:
            fmt = "%H:%M:%S"
        elif len(parts) == 2:
            fmt = "%H:%M"
        else:
            return None
        
        # Parsear hora base con fecha ficticia
        base_str = f"2000-01-01 {base_time}"
        base_dt = datetime.strptime(base_str, f"%Y-%m-%d {fmt}")
        
        # Sumar segundos
        result_dt = base_dt + timedelta(seconds=elapsed_seconds)
        
        return result_dt.strftime("%H:%M:%S")
        
    except ValueError:
        return None


def hora_video(hora_inicial: str, video_id: int, duracion_min: float = 24) -> str:
    """
    Calcula la hora de inicio de un video dado su ID secuencial.
    
    Args:
        hora_inicial: Hora del primer video en formato HH:MM
        video_id: ID del video (1-indexed)
        duracion_min: Duración de cada video en minutos
        
    Returns:
        Hora calculada en formato HH:MM
        
    Example:
        >>> hora_video("08:00", 3, 24)
        '08:48'  # 08:00 + 2*24min
    """
    try:
        base = datetime.strptime(hora_inicial, "%H:%M")
        offset = timedelta(minutes=(video_id - 1) * duracion_min)
        result = base + offset
        return result.strftime("%H:%M")
    except ValueError:
        return hora_inicial


def parse_time(time_str: str) -> tuple[int, int, int] | None:
    """
    Parsea un string de tiempo a tupla (hour, minute, second).
    
    Args:
        time_str: Tiempo en formato HH:MM o HH:MM:SS
        
    Returns:
        Tupla (hour, minute, second) o None si es inválido
    """
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        elif len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        return None
    except ValueError:
        return None


def seconds_to_hms(seconds: float) -> str:
    """
    Convierte segundos a formato HH:MM:SS.
    
    Args:
        seconds: Tiempo en segundos
        
    Returns:
        String en formato HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
