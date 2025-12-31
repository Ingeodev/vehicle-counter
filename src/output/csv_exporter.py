"""
CSVExporter - Exportación de resultados a CSV.
"""

import csv
from dataclasses import asdict
from typing import Sequence, Any

from ..data.schemas import ZoneEntry
from ..storage.base import StorageWriter


class CSVExporter:
    """
    Exporta resultados de detección a archivos CSV.
    
    Example:
        >>> exporter = CSVExporter(writer)
        >>> exporter.export_detections(entries, "detections.csv")
        >>> exporter.export_summary(zone_counts, "summary.csv")
    """
    
    def __init__(self, storage: StorageWriter):
        """
        Inicializa el exportador.
        
        Args:
            storage: Writer de storage
        """
        self.storage = storage
    
    def export_detections(
        self,
        entries: Sequence[ZoneEntry],
        path: str,
        include_header: bool = True
    ) -> str:
        """
        Exporta log de detecciones a CSV.
        
        Args:
            entries: Lista de entradas de zona
            path: Ruta del archivo CSV
            include_header: Si incluir encabezados
            
        Returns:
            Ruta del archivo creado
        """
        if not entries:
            return path
        
        # Convertir a lista de diccionarios
        rows = [
            {
                "vehicle_id": e.vehicle_id,
                "vehicle_type": e.vehicle_type,
                "zone": e.zone,
                "date": e.date,
                "exact_time": e.exact_time,
                "timestamp_formatted": e.timestamp_formatted
            }
            for e in entries
        ]
        
        # Generar CSV como texto
        lines = []
        
        if include_header:
            headers = list(rows[0].keys())
            lines.append(",".join(headers))
        
        for row in rows:
            values = [str(v) for v in row.values()]
            lines.append(",".join(values))
        
        content = "\n".join(lines)
        self.storage.write_text(path, content)
        
        return path
    
    def export_summary(
        self,
        zone_counts: dict[str, int],
        path: str,
        include_total: bool = True
    ) -> str:
        """
        Exporta resumen de conteos por zona.
        
        Args:
            zone_counts: Diccionario {zona: conteo}
            path: Ruta del archivo CSV
            include_total: Si incluir fila con total
            
        Returns:
            Ruta del archivo creado
        """
        lines = ["zone,count"]
        
        total = 0
        for zone, count in sorted(zone_counts.items()):
            lines.append(f"{zone},{count}")
            total += count
        
        if include_total:
            lines.append(f"TOTAL,{total}")
        
        content = "\n".join(lines)
        self.storage.write_text(path, content)
        
        return path
    
    def export_vehicle_types(
        self,
        entries: Sequence[ZoneEntry],
        path: str
    ) -> str:
        """
        Exporta conteo por tipo de vehículo.
        
        Args:
            entries: Lista de entradas
            path: Ruta del archivo CSV
            
        Returns:
            Ruta del archivo creado
        """
        # Contar por tipo
        type_counts: dict[str, int] = {}
        for entry in entries:
            vtype = entry.vehicle_type
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        lines = ["vehicle_type,count"]
        for vtype, count in sorted(type_counts.items()):
            lines.append(f"{vtype},{count}")
        
        content = "\n".join(lines)
        self.storage.write_text(path, content)
        
        return path
    
    def export_all(
        self,
        entries: Sequence[ZoneEntry],
        zone_counts: dict[str, int],
        base_path: str
    ) -> dict[str, str]:
        """
        Exporta todos los reportes.
        
        Args:
            entries: Lista de entradas
            zone_counts: Conteos por zona
            base_path: Ruta base (sin extensión)
            
        Returns:
            Diccionario con rutas de archivos creados
        """
        paths = {}
        
        # Detecciones
        det_path = f"{base_path}_detections.csv"
        self.export_detections(entries, det_path)
        paths["detections"] = det_path
        
        # Resumen por zona
        summary_path = f"{base_path}_summary.csv"
        self.export_summary(zone_counts, summary_path)
        paths["summary"] = summary_path
        
        # Por tipo de vehículo
        types_path = f"{base_path}_types.csv" 
        self.export_vehicle_types(entries, types_path)
        paths["types"] = types_path
        
        return paths
