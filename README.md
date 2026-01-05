# 🚗 Vehicle Counter CLI

Herramienta de línea de comandos para el conteo automático de vehículos, procesamiento de video y utilidades de OSD (On-Screen Display).

## 📋 Tabla de Contenidos
- [Instalación](#instalación)
- [Uso General](#uso-general)
- [Comandos](#comandos)
    - [process](#1-process)
    - [scan](#2-scan)
    - [extract-time](#3-extract-time)
    - [fix-osd](#4-fix-osd)
    - [info](#5-info)

## Instalación

### Opción 1: Instalar como paquete (Recomendado)
Para usar tanto el CLI como la librería en tus scripts:

```bash
# Desde el directorio raíz
pip install .

# O construir y usar el wheel (ideal para Colab)
python -m build
pip install dist/aforos_cli-1.0.0-py3-none-any.whl
```

### Opción 2: Solo dependencias
Si solo vas a ejecutar el código fuente:

```bash
pip install -r requirements.txt
```

Si planeas usar el motor OCR avanzado (TrOCR), necesitarás dependencias adicionales:
```bash
pip install transformers torch torchvision
```

## Uso como Librería (Python API)

El paquete expone funciones de alto nivel para usar en tus propios scripts o notebooks (Google Colab). Todas las funciones tienen tipado fuerte y autocompletado.

```python
from mglon_vehicle_counter import fix_osd, process_video, extract_time

# 1. Corregir fecha en video (con conversión H.264/H.265)
output = fix_osd(
    video="input.mp4", 
    date="2026-01-05", 
    codec="h265"  # Opciones: 'copy', 'h264', 'h265'
)

# 2. Procesar video (Pipeline completo)
result = process_video(
    video="input.mp4",
    zones="zones.json",
    device="cuda",  # 'cpu', 'cuda', 'mps'
    strategy="seg"  # 'box' o 'seg'
)
print(f"Total detectado: {result.total_detections}")

# 3. Extraer tiempo
times = extract_time(["video1.mp4"], model="trocr")
```

---

## Uso CLI (Línea de Comandos)

Una vez instalado el paquete, puedes usar el comando `aforos` directamente:

```bash
aforos <comando> [argumentos]
```

O ejecutar como módulo:
```bash
python -m src.cli <comando> [argumentos]
```

### 1. `process`
Procesa un único archivo de video para detectar y contar vehículos.

**Uso:**
```bash
aforos process input/video.mp4 --zones zonas.json [opciones]
```

#### 📥 Inputs (Argumentos)
| Argumento | Alias | Requerido | Descripción | Valor por defecto |
|-----------|-------|-----------|-------------|-------------------|
| `video` | - | **Sí** | Ruta al archivo de video a procesar. | - |
| `--zones` | `-z` | No | Archivo JSON con la definición de zonas de conteo. | - |
| `--mask` | `-m` | No | Imagen de máscara binaria para ignorar regiones. | - |
| `--output` | `-o` | No | Carpeta donde se guardarán los resultados. | `./output` |
| `--model` | - | No | Modelo YOLO a utilizar (`.pt`). | `yolov8s.pt` |
| `--device` | `-d` | No | Dispositivo de computación (`cpu`, `cuda`, `mps`). | `cpu` |
| `--strategy` | - | No | Estrategia: `box` (bounding box) o `seg` (segmentación). | `box` |
| `--model-config` | - | No | Archivo YAML para configurar tresholds por clase. | - |
| `--night-enhance` | - | No | Activa mejora de imagen para videos nocturnos (CLAHE). | `False` |
| `--deblurring` | - | No | Aplica filtro agresivo para corregir motion blur. | `False` |
| `--no-video` | - | No | Si se activa, NO genera el video de salida (solo CSVs). | `False` |

#### ⚙️ Configuración del Modelo (`--model-config`)
Puedes personalizar las clases a detectar y sus umbrales creando un archivo YAML (ej. `config.yml`):

```yaml
# Umbral global por defecto
default_threshold: 0.5

# Umbrales específicos por clase
class_thresholds:
  car: 0.6
  person: 0.4

# Clases a detectar (ID COCO: Nombre)
# Si se define, sobrescribe las clases por defecto
vehicle_classes:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  5: bus
  7: truck
```

#### 📤 Outputs (Salida)
| Archivo | Descripción |
|---------|-------------|
| `*_processed.mp4` | Video resultante con las detecciones y conteos renderizados (si no se usa `--no-video`). |
| `*_detections.csv` | Registro detallado de cada vehículo detectado (ID, clase, tiempo, zona). |
| `*_summary.csv` | Resumen total de conteos por clase y zona. |

---

### 2. `scan`
Escanéa recursivamente un directorio y procesa todos los videos encontrados.

**Uso:**
```bash
aforos scan input/videos/ --recursive --output ./results
```

#### 📥 Inputs (Argumentos)
| Argumento | Alias | Requerido | Descripción |
|-----------|-------|-----------|-------------|
| `directory` | - | **Sí** | Carpeta raíz para buscar videos. |
| `--recursive` | `-R` | No | Buscar también en subcarpetas. |
| `--output` | `-o` | No | Carpeta base para los resultados (mantiene estructura). |
| `--zones` | `-z` | No | Archivo de zonas global para todos los videos. |
| `--mask` | `-m` | No | Máscara global para todos los videos. |
| `--date` | - | No | Forzar una fecha específica para todos los videos (YYYY-MM-DD). |
| *Opciones* | | | Acepta opciones de `process` (`--model`, `--device`, etc.) |

#### 📤 Outputs (Salida)
Genera la misma estructura de archivos que `process` para cada video encontrado, replicando la estructura de carpetas dentro de `--output`.

---

### 3. `extract-time`
Extrae la fecha y hora impresa en el OSD (On-Screen Display) del video utilizando OCR. Útil para sincronizar tiempos reales.

**Uso:**
```bash
aforos extract-time input/video.mp4 --model trocr
```

#### 📥 Inputs (Argumentos)
| Argumento | Alias | Requerido | Descripción | Opciones |
|-----------|-------|-----------|-------------|----------|
| `video` | - | **Sí** | Uno o más videos para analizar. | - |
| `--model` | `-m` | No | Motor OCR a utilizar. `trocr` es recomendado para baja calidad. | `easyocr` (default), `tesseract`, `trocr` |
| `--preprocess` | `-p` | No | Filtro de preprocesamiento de imagen antes del OCR. | `clahe` (default), `binary`, `color` |
| `--output` | `-o` | No | Archivo CSV para exportar los resultados. | - |
| `--export-roi` | - | No | Carpeta para guardar las imágenes recortadas del OSD (debug). | - |

#### 📤 Outputs (Salida)
- **Consola**: Tabla con Fecha, Hora Inicio, Hora Fin y Duración de cada video.
- **CSV** (opcional): Archivo con los datos tabulados.

---

### 4. `fix-osd`
Corrige o reemplaza el texto de la fecha en el OSD del video mediante técnicas de Inpainting (borrado) y superposición de texto nuevo.

**Uso:**
```bash
aforos fix-osd video.mp4 --date 31-12-2025
```

#### 📥 Inputs (Argumentos)
| Argumento | Alias | Requerido | Descripción |
|-----------|-------|-----------|-------------|
| `video` | - | **Sí** | Video a corregir. |
| `--date` | - | **Sí** | Nueva fecha a estampar (`DD-MM-YYYY`). |
| `--output` | `-o` | No | Ruta del video de salida. Default: `*_fixed.mp4`. |
| `--codec` | - | No | Codec de salida: `copy` (default), `h264`, `h265`. |
| `--font` | - | No | Ruta a una fuente `.ttf` personalizada. |
| `--max-minutes` | - | No | Limitar la duración del video corregido. |

#### 📤 Outputs (Salida)
| Archivo | Descripción |
|---------|-------------|
| `*_fixed.mp4` | Nuevo archivo de video con la fecha antigua borrada y la nueva escrita encima. |

---

### 5. `info`
Muestra metadatos técnicos rápidos de un archivo de video.

**Uso:**
```bash
aforos info video.mp4
```

#### 📤 Salida (Consola)
```text
📹 Video: video.mp4
   Resolución: 1920x1080
   FPS: 30.0
   Frames: 18000
   Duración: 10.0 minutos
```
