"""
Módulo para validar argumentos antes de entrenar.
Detiene la ejecución temprano si algo está mal configurado.
"""
from pathlib import Path


# Modelos soportados por la plataforma
TABULAR_MODELS = {
    "random_forest", "xgboost", "mlp", "tabnet",
    "ft_transformer", "tabpfn", "logistic_regression",
    "svm", "knn", "decision_tree"
}
DETECTION_MODELS = {"yolo", "ssd", "faster_rcnn"}

# Tareas soportadas
VALID_TASKS = {"classification", "regression", "detection"}

# Extensiones válidas
TABULAR_EXT = {".csv", ".xlsx"}
DETECTION_EXT = {".yaml", ".yml", ".pt"}  # .pt para pesos preentrenados


def validate_task(task: str) -> None:
    """Verifica que la tarea sea válida."""
    if task not in VALID_TASKS:
        raise ValueError(
            f"Tarea '{task}' no válida. Opciones: {VALID_TASKS}"
        )


def validate_model(model: str, task: str) -> None:
    """Verifica que el modelo sea válido para la tarea indicada."""
    if task == "detection":
        # En detección el "modelo" puede ser una ruta a pesos .pt
        if not (model in DETECTION_MODELS or model.endswith(".pt")):
            raise ValueError(
                f"Modelo '{model}' no válido para detección. "
                f"Opciones: {DETECTION_MODELS} o archivo .pt"
            )
    else:
        if model not in TABULAR_MODELS:
            raise ValueError(
                f"Modelo '{model}' no válido para tabular. "
                f"Opciones: {TABULAR_MODELS}"
            )


def validate_path(path: str, valid_extensions: set = None) -> Path:
    """Verifica que un archivo exista y tenga la extensión correcta."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    if valid_extensions and p.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Extensión '{p.suffix}' no válida. Esperadas: {valid_extensions}"
        )
    return p


def validate_hyperparams(epochs: int = None, batch: int = None) -> None:
    """Verifica que los hiperparámetros numéricos sean positivos."""
    if epochs is not None and epochs <= 0:
        raise ValueError(f"epochs debe ser > 0, recibido: {epochs}")
    if batch is not None and batch <= 0:
        raise ValueError(f"batch debe ser > 0, recibido: {batch}")


def validate_all(config: dict) -> None:
    """
    Ejecuta todas las validaciones en orden.
    Llamar esto desde main.py antes de empezar a entrenar.
    """
    task = config.get("task")
    validate_task(task)
    validate_model(config.get("model"), task)

    # Validar rutas según el tipo de tarea
    if task in {"classification", "regression"}:
        validate_path(config.get("dataset"), TABULAR_EXT)
    elif task == "detection":
        validate_path(config.get("data"), {".yaml", ".yml"})

    validate_hyperparams(
        epochs=config.get("epochs"),
        batch=config.get("batch")
    )
    print("[validate] Todas las validaciones pasaron ✓")