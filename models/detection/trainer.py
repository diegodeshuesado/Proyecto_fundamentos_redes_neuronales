"""
Trainer para modelos de deteccion.
Orquesta: carga config -> entrena -> evalua -> guarda.
"""
import time
import json
import yaml
import shutil
from pathlib import Path

from models.detection.yolo_model import YOLOModel
from models.detection.faster_rcnn_model import FasterRCNNModel
from models.detection.ssd_model import SSDModel


# Mapeo nombre -> clase del modelo
DETECTION_REGISTRY = {
    "yolo": YOLOModel,
    "faster_rcnn": FasterRCNNModel,
    "ssd": SSDModel,
    # En siguientes pasos: ssd
}


def build_detection_model(model_name: str, weights: str, device: str, seed: int):
    """Factory para modelos de deteccion."""
    if model_name not in DETECTION_REGISTRY:
        raise ValueError(
            f"Modelo de deteccion '{model_name}' no implementado. "
            f"Disponibles: {list(DETECTION_REGISTRY.keys())}"
        )
    ModelClass = DETECTION_REGISTRY[model_name]
    return ModelClass(weights=weights, device=device, seed=seed)


def train_detection(config: dict, run_dir: Path, logger) -> dict:
    """Pipeline completo de entrenamiento para deteccion."""
    logger.info("=" * 60)
    logger.info("Iniciando pipeline DETECCION")
    logger.info("=" * 60)

    # ===== 1. Validar config =====
    data_yaml = config.get("data")
    if not data_yaml or not Path(data_yaml).exists():
        raise FileNotFoundError(f"No se encontro data.yaml en: {data_yaml}")

    # El campo 'model' puede ser: nombre simbolico (yolo) o ruta a .pt
    model_arg = config["model"]
    if model_arg.endswith(".pt"):
        # Es una ruta a pesos: detectamos el tipo por el nombre
        weights = model_arg
        model_name = "yolo"  # por ahora todos los .pt son YOLO
    else:
        # Es nombre simbolico, usar pesos por defecto
        weights = config.get("weights", "yolov8n.pt")
        model_name = model_arg

    if model_name == "yolo":
        logger.info(f"[1/4] Modelo: {model_name} | weights: {weights}")
    else:
        logger.info(f"[1/4] Modelo: {model_name}")

    # ===== 2. Crear modelo =====
    model = build_detection_model(
        model_name=model_name,
        weights=weights,
        device=config.get("device", "cpu"),
        seed=config.get("seed", 42),
    )

    # ===== 3. Entrenar =====
    logger.info("[2/4] Entrenando modelo...")
    epochs = config.get("epochs") or 50
    batch = config.get("batch") or 16
    imgsz = config.get("imgsz") or 640

    t_start = time.time()
    model.train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project_dir=(run_dir / "yolo_train").resolve(),
        run_name="train",
    )
    train_time = time.time() - t_start
    logger.info(f"Tiempo de entrenamiento: {train_time:.2f} segundos")

    # ===== 4. Evaluar =====
    logger.info("[3/4] Evaluando en conjunto de validacion...")
    metrics = model.validate(data_yaml=data_yaml)
    metrics["train_time_seconds"] = round(train_time, 2)

    # Guardar metricas en JSON
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metricas guardadas en: {metrics_path}")

    # ===== 5. Guardar mejor modelo =====
    logger.info("[4/4] Guardando modelo entrenado...")
    model.save(run_dir / "model.pt")

    # Copiar tambien las graficas que YOLO genera automaticamente
    if model_name == "yolo":
        yolo_train_dir = (run_dir / "yolo_train" / "train").resolve()
        if yolo_train_dir.exists():
            for ext in ("*.png", "*.jpg"):
                for plot_file in yolo_train_dir.glob(ext):
                    shutil.copy2(plot_file, run_dir / plot_file.name)
            logger.info("Graficas de YOLO copiadas a la carpeta del experimento")
        else:
            logger.warning(f"No se encontraron graficas en: {yolo_train_dir}")

    # Log final de metricas
    logger.info("=" * 60)
    logger.info("METRICAS FINALES:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    return metrics