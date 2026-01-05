#!/usr/bin/env python
"""
Test de regresión para la API de alto nivel aforos.
"""
from src import fix_osd
import os

# Configuración
input_video = "input/videos/boulevard_oriente.mp4"
output_video = "results/test_api_refactor.mp4"
new_date = "05-01-2026" # Probamos formato sin día, el sistema debería añadirlo si la lógica está bien

print(f"🧪 Iniciando test de regresión con API de alto nivel...")
print(f"🎬 Input: {input_video}")

try:
    output = fix_osd(
        video=input_video,
        date=new_date,
        output=output_video,
        max_minutes=1.0, # Primer minuto
        convert_h264=True, # Validar conversión y manejo de errores ffmpeg
        quiet=False
    )
    print(f"\n✅ Test Exitoso!")
    print(f"📁 Video generado: {output}")
    
    # Verificar que existe
    if os.path.exists(output):
        size = os.path.getsize(output) / (1024*1024)
        print(f"📊 Tamaño: {size:.2f} MB")
    else:
        print("❌ Error: El archivo de salida no existe")

except Exception as e:
    print(f"\n❌ Test Fallido:")
    print(f"{e}")
    # Print traceback for debugging
    import traceback
    traceback.print_exc()
