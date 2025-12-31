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

Asegúrate de tener instaladas las dependencias:

```bash
pip install -r requirements.txt
```

Si planeas usar el motor OCR avanzado (TrOCR), necesitarás dependencias adicionales que se instalarán automáticamente o puedes instalar manualmente:
```bash
pip install transformers torch torchvision
```

## Uso General

La interfaz se ejecuta a través del módulo `src.cli`:

```bash
python -m src.cli <comando> [argumentos]
```

Para ver ayuda de cualquier comando:
```bash
python -m src.cli <comando> --help
```

---

## Comandos

### 1. `process`
Procesa un único archivo de video para detectar y contar vehículos.

**Uso:**
```bash
python -m src.cli process input/video.mp4 --zones zonas.json [opciones]
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
python -m src.cli scan input/videos/ --recursive --output ./results
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
python -m src.cli extract-time input/video.mp4 --model trocr
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
python -m src.cli fix-osd video.mp4 --date 31-12-2025
```

#### 📥 Inputs (Argumentos)
| Argumento | Alias | Requerido | Descripción |
|-----------|-------|-----------|-------------|
| `video` | - | **Sí** | Video a corregir. |
| `--date` | - | **Sí** | Nueva fecha a estampar (`DD-MM-YYYY`). |
| `--output` | `-o` | No | Ruta del video de salida. Default: `*_fixed.mp4`. |
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
python -m src.cli info video.mp4
```

#### 📤 Salida (Consola)
```text
📹 Video: video.mp4
   Resolución: 1920x1080
   FPS: 30.0
   Frames: 18000
   Duración: 10.0 minutos
```
