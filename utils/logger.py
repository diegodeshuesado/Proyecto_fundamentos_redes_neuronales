"""
Módulo para crear carpetas de experimento y configurar logging.
"""
import logging
from pathlib import Path
from datetime import datetime


def create_run_dir(base_dir: str = "runs", name: str = "experimento") -> Path:
    """
    Crea una carpeta única para el experimento dentro de runs/.
    Si ya existe, agrega sufijo numérico: experimento_2, experimento_3...
    """
    base = Path(base_dir)
    base.mkdir(exist_ok=True)  # crea runs/ si no existe

    run_dir = base / name
    counter = 2
    while run_dir.exists():
        run_dir = base / f"{name}_{counter}"
        counter += 1

    run_dir.mkdir()
    print(f"[logger] Carpeta de experimento creada: {run_dir}")
    return run_dir


def setup_logger(run_dir: Path, name: str = "experimento") -> logging.Logger:
    """
    Configura un logger que escribe tanto en consola como en run.log.
    """
    log_file = run_dir / "run.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # evita duplicados si se llama varias veces

    # Formato de los mensajes
    formato = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler 1: escribir al archivo run.log
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formato)
    logger.addHandler(file_handler)

    # Handler 2: imprimir en consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formato)
    logger.addHandler(console_handler)

    logger.info(f"Experimento iniciado: {name}")
    logger.info(f"Fecha y hora: {datetime.now().isoformat()}")
    return logger