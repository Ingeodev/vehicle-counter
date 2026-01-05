#!/usr/bin/env python
"""
Ejemplo de uso de la librería aforos para cambiar fecha en video.
Simula cómo se usaría en Google Colab.
"""
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Importar desde la librería
from src import OSDModifier

# Configuración
input_video = "input/videos/boulevard_oriente.mp4"
output_video = "results/test_library_api.mp4"
new_date = "05-01-2026 Mon"  # Formato DD-MM-YYYY + día
max_seconds = 6  # Solo 6 segundos de prueba

# Abrir video
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames = int(max_seconds * fps)

print(f"🎬 Video: {input_video}")
print(f"📅 Nueva fecha: {new_date}")
print(f"📹 Procesando {max_frames} frames ({max_seconds}s)")

# Crear writer y modifier
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
modifier = OSDModifier()  # Usa fuente por defecto de src/assets

# Procesar
for i in tqdm(range(max_frames), desc="Procesando"):
    ret, frame = cap.read()
    if not ret:
        break
    fixed_frame = modifier.process_frame(frame, new_date)
    out.write(fixed_frame)

cap.release()
out.release()

print(f"✅ Video guardado: {output_video}")
