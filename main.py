"""
Punto de entrada principal de la plataforma.
Ejecuta: python main.py --help para ver todos los argumentos.
"""
import argparse
import sys
from pathlib import Path

from utils.config import merge_config, save_config
from utils.seed import set_seed
from utils.logger import create_run_dir, setup_logger
from utils.validate import validate_all


def parse_args():
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Plataforma modular para entrenar modelos de ML y redes neuronales."
    )

    # --- Configuración general ---
    parser.add_argument("--config", type=str, default=None,
                        help="Ruta a archivo YAML de configuración (opcional)")
    parser.add_argument("--task", type=str, default=None,
                        choices=["classification", "regression", "detection"],
                        help="Tipo de tarea")
    parser.add_argument("--model", type=str, default=None,
                        help="Modelo a utilizar (ej: xgboost, yolo, weight_y12/yolov12l.pt)")
    parser.add_argument("--name", type=str, default="experimento",
                        help="Nombre del experimento (carpeta en runs/)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Dispositivo: 'cpu' o número de GPU (ej: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")

    # --- Datos tabulares ---
    parser.add_argument("--dataset", type=str, default=None,
                        help="Ruta del dataset tabular (.csv o .xlsx)")
    parser.add_argument("--target", type=str, default=None,
                        help="Columna objetivo (solo tabular)")

    # --- Datos de detección ---
    parser.add_argument("--data", type=str, default=None,
                        help="Archivo YAML del dataset (solo detección)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Tamaño de imagen (solo detección)")

# --- Hiperparámetros comunes ---
    parser.add_argument("--epochs", type=int, default=None,
                        help="Número de épocas")
    parser.add_argument("--batch", type=int, default=None,
                        help="Tamaño de lote")

# --- Hiperparámetros redes neuronales (MLP, TabNet, FT-T) ---
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=None,
                        help="Lista de neuronas por capa, ej: --hidden_dims 128 64 32")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Tasa de dropout (0.0 a 1.0)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Tasa de aprendizaje (default 0.001)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Épocas sin mejora para early stopping (default 10)")
    return parser.parse_args()

def main():
    # 1. Leer argumentos
    args = parse_args()
    args_dict = vars(args)  # convierte Namespace a dict
    yaml_path = args_dict.pop("config", None)  # extraer --config aparte

    # 2. Fusionar YAML + consola
    config = merge_config(args_dict, yaml_path)

    # 3. Validar
    validate_all(config)

    # 4. Fijar semilla
    set_seed(config.get("seed", 42))

    # 5. Crear carpeta del experimento
    run_dir = create_run_dir(name=config["name"])

    # 6. Configurar logger
    logger = setup_logger(run_dir, name=config["name"])
    logger.info(f"Configuración final: {config}")

    # 7. Guardar config usada (reproducibilidad)
    save_config(config, run_dir / "config_usada.yaml")

    # 8. Despachar según la tarea
    task = config["task"]
    if task in {"classification", "regression"}:
        from models.tabular.trainer import train_tabular
        train_tabular(config, run_dir, logger)
    elif task == "detection":
        from models.detection.trainer import train_detection
        train_detection(config, run_dir, logger)
    logger.info("Ejecución finalizada correctamente ✓")


if __name__ == "__main__":
    main()