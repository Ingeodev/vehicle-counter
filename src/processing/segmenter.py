"""
YOLOSegmenter - Implementación de detección basada en segmentación de instancias.
"""

import os
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

from ..data.schemas import Detection, BoundingBox
from ..config.settings import DetectorConfig, DEFAULT_VEHICLE_CLASSES


class YOLOSegmenter:
    """
    Detector que usa segmentación de instancias (YOLOv8-Seg).

    Ventajas:
    - Más preciso para detectar la forma real del vehículo.
    - Permite calcular el centroide real (masa) en lugar del centro de la caja.
    - Más robusto a oclusiones y formas irregulares.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

        # Asegurar que el modelo sea de segmentación si no se especifica
        model_path = self.config.model_path
        if not model_path.endswith("-seg.pt") and "seg" not in model_path:
            # Intentar cambiar a versión seg si es estándar
            if model_path == "yolov8s.pt":
                model_path = "yolov8s-seg.pt"
            elif model_path == "yolov8n.pt":
                model_path = "yolov8n-seg.pt"

        self.model = YOLO(model_path)
        self.device = self.config.device
        self.vehicle_classes = self.config.vehicle_classes or DEFAULT_VEHICLE_CLASSES
        self.allowed_class_ids = list(self.vehicle_classes.keys())

    def detect_and_track(
        self, frame: np.ndarray, persist: bool = True
    ) -> list[Detection]:
        """
        Ejecuta tracking con segmentación.
        """
        # Preparar argumentos
        track_kwargs = {
            "persist": persist,
            "device": self.device,
            "classes": self.allowed_class_ids,
            "verbose": False,
            "retina_masks": True,
            "half": self.device != "cpu",
        }

        # Tracker config
        if self.config.tracker_config:
            tracker_path = self.config.tracker_config
            if not os.path.isabs(tracker_path):
                # Buscar en config del paquete
                # Asumiendo estructura similar a detector.py
                pkg_dir = Path(__file__).parent.parent / "config"
                local_path = pkg_dir / tracker_path
                if local_path.exists():
                    tracker_path = str(local_path)
            track_kwargs["tracker"] = tracker_path

        results = self.model.track(frame, **track_kwargs)

        if not results or len(results) == 0:
            return []

        result = results[0]

        if result.boxes.id is None:
            return []

        # Extraer datos
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        # Extraer máscaras (si hay)
        masks = None
        if result.masks is not None:
            # Solo guardamos referencia, no descargamos los bitmaps pesados (.data)
            # Usaremos .xy (polígonos) que es mucho más ligero
            masks = result.masks

        track_detections: list[Detection] = []

        for i, (box, track_id, cls_id, conf) in enumerate(
            zip(boxes, track_ids, class_ids, confidences)
        ):
            if cls_id not in self.vehicle_classes:
                continue

            # Obtener umbral específico o general
            cls_name = self.vehicle_classes[int(cls_id)]
            threshold = self.config.confidence_threshold

            if (
                self.config.class_thresholds
                and cls_name in self.config.class_thresholds
            ):
                threshold = self.config.class_thresholds[cls_name]

            if conf < threshold:
                continue

            x1, y1, x2, y2 = box.astype(int)
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

            # Procesar máscara
            mask_binary = None
            mask_center = None
            mask_polygon = None

            if masks is not None and i < len(masks):
                # Obtener máscara binaria
                # La máscara puede venir en tamaño reducido, hay que escalar a la caja original o frame
                # Ultralytics handle this usually inside result object utility

                # Forma robusta: obtener polígono y dibujar máscara
                poly = result.masks.xy[i]  # Lista de puntos (N, 2)
                if len(poly) > 0:
                    # Calcular centroide usando momentos
                    # poly es float32
                    M = cv2.moments(poly.astype(np.float32))
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        mask_center = (cx, cy)
                        mask_polygon = poly.astype(np.int32)

                    # Opcional: Crear máscara binaria completa si se necesita para visualización
                    # mask_binary = ... (costoso generar full frame mask cada vez)

            detection = Detection(
                track_id=int(track_id),
                class_id=int(cls_id),
                class_name=self.vehicle_classes[int(cls_id)],
                bbox=bbox,
                confidence=float(conf),
                mask=None,  # No guardamos bitmap pesado por ahora
                mask_polygon=mask_polygon,
                mask_center=mask_center,
            )
            track_detections.append(detection)

        return track_detections

    def detect_only(self, frame: np.ndarray) -> list[Detection]:
        # Implementación simple sin tracking
        results = self.model(
            frame, device=self.device, classes=self.allowed_class_ids, verbose=False
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        detections = []
        for i, (box, cls_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
            cls_name = self.vehicle_classes.get(int(cls_id), "unknown")
            threshold = self.config.confidence_threshold
            if (
                self.config.class_thresholds
                and cls_name in self.config.class_thresholds
            ):
                threshold = self.config.class_thresholds[cls_name]

            if conf < threshold:
                continue
            x1, y1, x2, y2 = box.astype(int)

            mask_center = None
            mask_polygon = None
            if result.masks is not None:
                poly = result.masks.xy[i]
                if len(poly) > 0:
                    M = cv2.moments(poly.astype(np.float32))
                    if M["m00"] > 0:
                        mask_center = (
                            int(M["m10"] / M["m00"]),
                            int(M["m01"] / M["m00"]),
                        )
                        mask_polygon = poly.astype(np.int32)

            detections.append(
                Detection(
                    track_id=i,
                    class_id=int(cls_id),
                    class_name=self.vehicle_classes.get(int(cls_id), "unknown"),
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=float(conf),
                    mask_polygon=mask_polygon,
                    mask_center=mask_center,
                )
            )
        return detections

    def reset_tracking(self) -> None:
        pass

    @property
    def class_names(self) -> dict[int, str]:
        return self.vehicle_classes.copy()
