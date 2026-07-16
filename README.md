# Plataforma Modular para Entrenamiento de Modelos de ML y Redes Neuronales

Proyecto Final — Fundamentos de Redes Neuronales

Plataforma modular en Python que permite entrenar y evaluar modelos de Machine Learning clásico, redes neuronales y modelos de detección de objetos desde línea de comandos, con configuración vía YAML y argumentos CLI.

## Características

- **9 modelos implementados** con la misma interfaz (train, predict, save, load)
- **Datos tabulares:** Random Forest, XGBoost, MLP, TabNet, FT-Transformer, TabPFN
- **Detección de objetos:** YOLOv8, Faster R-CNN, SSD
- Configuración híbrida: CLI + YAML (la consola tiene prioridad)
- Reproducibilidad garantizada (semilla fija)
- Logging detallado + guardado automático de métricas, gráficas y modelos
- Validación temprana de argumentos y rutas
- Soporte automático de GPU (CUDA) o CPU

---

## Requisitos

- **Python 3.10+** (probado en 3.12)
- **Ubuntu / Linux** (recomendado)
- **GPU NVIDIA con CUDA** (recomendado; funciona en CPU pero más lento)
- **~5 GB de espacio libre** para dependencias

---

## Instalación

### 1. Clonar / descomprimir el proyecto
```bash
cd /ruta/al/proyecto/Proyecto_fundamentos
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. (Opcional) Instalar PyTorch con soporte GPU

Si tu GPU es NVIDIA, instala PyTorch con CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 4. Verificar instalación
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

---

## Uso rápido

### Ver argumentos disponibles
```bash
python main.py --help
```

### Ejemplo 1: Random Forest desde consola
```bash
python main.py \
  --task classification \
  --model random_forest \
  --dataset data/mi_dataset.csv \
  --target columna_objetivo \
  --name experimento_rf \
  --seed 42
```

### Ejemplo 2: MLP en GPU con arquitectura personalizada
```bash
python main.py \
  --task classification \
  --model mlp \
  --dataset data/mi_dataset.csv \
  --target columna_objetivo \
  --name experimento_mlp \
  --device 0 \
  --hidden_dims 256 128 64 \
  --dropout 0.3 \
  --epochs 100 \
  --patience 15
```

### Ejemplo 3: YOLO para detección
```bash
python main.py \
  --task detection \
  --model yolo \
  --data /ruta/a/data.yaml \
  --name deteccion_rostros \
  --device 0 \
  --epochs 50 \
  --batch 16
```

### Ejemplo 4: Configuración desde YAML (recomendado)
```bash
python main.py --config configs/mlp.yaml --name mi_experimento
```

Los argumentos de consola sobrescriben los del YAML. Ejemplo:
```bash
# Usa la config del YAML pero cambia epochs y device
python main.py --config configs/mlp.yaml --epochs 200 --device 0
```

---

## Modelos disponibles

| Modelo | Tipo | Tarea | Librería |
|--------|------|-------|----------|
| `random_forest` | ML clásico | Clasificación / Regresión | scikit-learn |
| `xgboost` | ML clásico (boosting) | Clasificación / Regresión | xgboost |
| `mlp` | Red neuronal simple | Clasificación / Regresión | PyTorch |
| `tabnet` | Red con atención secuencial | Clasificación / Regresión | pytorch-tabnet |
| `ft_transformer` | Transformer para tabular | Clasificación / Regresión | pytorch-tabular |
| `tabpfn` | Foundation model | Solo Clasificación | tabpfn |
| `yolo` | Detección one-stage | Detección de objetos | ultralytics |
| `faster_rcnn` | Detección two-stage | Detección de objetos | torchvision |
| `ssd` | Detección one-stage | Detección de objetos | torchvision |

---

## Estructura del proyecto
Proyecto_fundamentos/
├── main.py                     # Punto de entrada (CLI + argparse)
├── requirements.txt            # Dependencias
├── configs/                    # YAMLs de ejemplo (uno por modelo)
├── data/                       # Datasets
├── runs/                       # Salidas de experimentos (auto)
├── utils/                      # Utilidades compartidas
│   ├── config.py               # Fusión YAML + consola
│   ├── seed.py                 # Reproducibilidad
│   ├── logger.py               # Sistema de logs
│   ├── validate.py             # Validaciones de argumentos
│   └── prepare_yolo_dataset.py # Herramienta para preparar datasets YOLO
├── models/
│   ├── tabular/                # 6 modelos tabulares
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   ├── metrics.py
│   │   ├── trainer.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── mlp.py
│   │   ├── tabnet_model.py
│   │   ├── ft_transformer.py
│   │   └── tabpfn_model.py
│   └── detection/              # 3 modelos de detección
│       ├── trainer.py
│       ├── yolo_to_torch_dataset.py
│       ├── yolo_model.py
│       ├── faster_rcnn_model.py
│       └── ssd_model.py
└── README.md

---

## Formato de datos esperado

### Tabular (CSV o XLSX)

Debe ser un archivo con encabezados, incluyendo la columna objetivo:

```csv
feat_1,feat_2,categoria,target
1.5,2.3,A,0
3.1,0.8,B,1
```

Especifica la columna objetivo con `--target target`. El sistema detecta automáticamente si es clasificación o regresión.

### Detección de objetos (formato YOLO)

Estructura requerida:
dataset/
├── images/
│   ├── train/  (fotos .jpg / .png)
│   └── val/
├── labels/
│   ├── train/  (una .txt por imagen)
│   └── val/
└── data.yaml

Cada archivo `.txt` tiene una línea por objeto:
clase_id x_centro y_centro ancho alto
Todos los valores normalizados entre 0 y 1.

**Herramienta auxiliar:** si tienes las imágenes y labels en carpetas planas sin split, usa:
```bash
python utils/prepare_yolo_dataset.py \
  --source /ruta/dataset_original \
  --output data/mi_dataset \
  --val_split 0.2
```

---

## Interpretación de resultados

Cada experimento genera una carpeta única en `runs/<nombre>/` con:

**Comunes a todos los experimentos:**
- `run.log` — log completo de la ejecución
- `config_usada.yaml` — configuración exacta usada (para reproducir)
- `metrics.json` — métricas finales en JSON
- `model.pkl` / `model.pt` — modelo entrenado

**Tabular (clasificación):**
- `confusion_matrix.png`
- `roc_curve.png` (solo binaria)
- `classification_report.txt`
- `feature_importances.json`

**Tabular (regresión):**
- `predictions_vs_real.png`

**Detección (YOLO):**
- `results.png` — losses por época
- `confusion_matrix.png`
- `PR_curve.png`, `F1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`
- `train_batch*.jpg`, `val_batch*_pred.jpg` — visualizaciones

---

## Reproducibilidad

Todos los experimentos usan `--seed 42` por defecto. La misma configuración corrida dos veces produce resultados idénticos gracias a la fijación de semillas en `random`, `numpy` y `torch`.

---

## Créditos

Proyecto final del curso **Fundamentos de Redes Neuronales**.

Desarrollado por:
- Diego Gutiérrez Hernández
- Yassed Meneses Fontecha
- Angel Arroyo 

---

## Licencia y uso

Este proyecto es únicamente para fines académicos.