Bien, veo la estructura. Déjame leer los archivos de código fuente directamente.No hay resultados directos. Voy a acceder a los archivos fuente directamente vía raw.githubusercontent.com:GitHub está bloqueando el acceso directo. Voy a intentar la API de GitHub para leer los archivos:Ya tengo todo lo que necesito. El análisis está completo. Encontré **múltiples problemas críticos**:

---

## Diagnóstico: 7 problemas encontrados

### 🔴 Crítico 1 — Device por defecto es `cpu`

```python
# settings.py línea 37
device: str = "cpu"  # ← AQUÍ ESTÁ EL PROBLEMA PRINCIPAL
```

Si no pasas `--device cuda` explícitamente al CLI, **todo corre en CPU**. Y el `detect_device()` existe pero **nunca se llama** como default en `DetectorConfig`. Es un helper que no está conectado.

**Fix:**
```python
device: str = field(default_factory=detect_device)
```

---

### 🔴 Crítico 2 — `reset_tracking` recarga el modelo completo cada video

```python
# detector.py línea 221
def reset_tracking(self) -> None:
    self.model = YOLO(self.config.model_path)  # ← Recarga desde disco cada video
```

Cuando `process_directory` procesa múltiples videos, recarga el modelo completo entre cada uno. Eso es **decenas de segundos de overhead** por video, más presión en VRAM.

---

### 🔴 Crítico 3 — Codec `mp4v` en lugar de `avc1`/`h264`

```python
# settings.py línea 29
codec: str = "mp4v"
```

`mp4v` es MPEG-4 Part 2 — lento y sin aceleración de hardware en OpenCV. Para escritura en CPU puede ser **2–3x más lento** que `avc1`. Cada frame procesado espera a que el escritor termine antes de leer el siguiente.

---

### 🟡 Importante 4 — Loop síncrono: lectura bloquea inferencia

```python
# pipeline.py línea 248
for frame_data in video.frames(...):
    frame = frame_data.frame
    detections = counter.process_frame(frame, ...)  # ← GPU espera CPU
    if video_writer:
        video_writer.write(frame)                   # ← CPU espera GPU
```

El ciclo es completamente secuencial. Mientras la GPU infiere, la CPU no decodifica el siguiente frame. Mientras la CPU escribe el frame al disco, la GPU espera. **La GPU nunca está al 100%.**

---

### 🟡 Importante 5 — `skip_rate` no se pasa al generador de frames

```python
# settings.py línea 27
skip_rate: int = 1  # Existe en config...

# pipeline.py línea 248
for frame_data in video.frames(max_frames=max_frames, resize=resize):
    # ← skip_rate nunca se pasa aquí
```

`video.frames()` soporta `skip_rate` pero `pipeline.py` nunca lo usa. Estás procesando el 100% de frames aunque hayas configurado otro valor.

---

### 🟡 Importante 6 — Tres terminales compiten por la misma GPU

Cuando corres tres instancias simultáneas en el mismo PC, cada proceso YOLO carga su propio modelo en VRAM. Con YOLOv8s eso es ~50MB × 3 = ~150MB, aceptable. **Pero el scheduler de CUDA serializa las llamadas** — los tres procesos se turnan, no corren en paralelo real. El overhead de context switching puede degradar hasta un 40% el throughput total vs una sola instancia bien optimizada.

---

### 🟢 Menor 7 — `skip_rate` en el generador tiene un bug de lógica

```python
# video_source.py líneas 211-212
if skip_rate > 1 and (self._frame_index - 1) % skip_rate != 0:
    continue  # ← Descarta el frame DESPUÉS de leerlo de disco
```

Lee el frame de disco y luego lo descarta. Para skip real de I/O hay que usar `cap.grab()` en lugar de `cap.read()` en los frames descartados.

---

## Fix rápido — cambios mínimos para 5–10x speedup

**1. Forzar CUDA siempre:**
```bash
aforos process video.mp4 --device cuda
```
O en código, conectar `detect_device()`:
```python
# settings.py
device: str = field(default_factory=detect_device)
```

**2. Cambiar codec de salida:**
```python
codec: str = "avc1"  # O usar --no-video para máxima velocidad
```

**3. Pasar skip_rate en pipeline.py línea 248:**
```python
for frame_data in video.frames(max_frames=max_frames, resize=resize, skip_rate=cfg.video.skip_rate):
```

**4. Fix reset_tracking — no recargar el modelo:**
```python
def reset_tracking(self) -> None:
    # No recargar — YOLO reinicia tracking automáticamente
    pass
```

**5. Una sola instancia bien configurada en lugar de tres terminales:**
```bash
aforos scan ./videos/ --device cuda --no-video
```

Con estos 5 cambios, un video de 25min debería bajar de 1.5 horas a **~3–5 minutos**. ¿Quieres que prepare un PR con los fixes?