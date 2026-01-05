#!/usr/bin/env python
"""
Test de H.265 para la API de aforos.
"""
from src import fix_osd
import os

# Configuración
input_video = "input/videos/boulevard_oriente.mp4"
output_video = "results/test_h265.mp4"
new_date = "05-01-2026"

print(f"🧪 Iniciando test de H.265...")
print(f"🎬 Input: {input_video}")

try:
    output = fix_osd(
        video=input_video,
        date=new_date,
        output=output_video,
        max_minutes=0.5, # 30 segundos suficiente
        codec="h265",    # Probando codec nuevo
        quiet=False
    )
    print(f"\n✅ Test Exitoso!")
    print(f"📁 Video generado: {output}")
    
    # Verificar existencia y tamaño
    if os.path.exists(output):
        size = os.path.getsize(output) / (1024*1024)
        print(f"📊 Tamaño: {size:.2f} MB")
        
        # Verificar codec (opcional, visual)
        print("ℹ️  Verifica el codec con: ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 " + output)
        
    else:
        print("❌ Error: El archivo de salida no existe")

except Exception as e:
    print(f"\n❌ Test Fallido:")
    print(f"{e}")
