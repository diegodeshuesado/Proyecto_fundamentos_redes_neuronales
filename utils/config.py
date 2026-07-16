import yaml
from pathlib import Path


def load_yaml(path: str) -> dict:
    """Lee un archivo .yaml y devuelve su contenido como diccionario."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo YAML: {path}")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_config(args_dict: dict, yaml_path: str = None) -> dict:
    """
    Fusiona configuración del YAML con argumentos de consola.

    Args:S
        args_dict: diccionario con argumentos de consola (de argparse).
        yaml_path: ruta opcional a un archivo .yaml.

    Returns:
        Diccionario unificado. Consola tiene prioridad sobre YAML.
    """
    config = {}

    # 1. Cargar YAML si se proporcionó
    if yaml_path:
        config.update(load_yaml(yaml_path))
        print(f"[config] YAML cargado: {yaml_path}")

    # 2. Sobrescribir con argumentos de consola (solo los que no son None)
    for key, value in args_dict.items():
        if value is not None:
            config[key] = value

    return config


def save_config(config: dict, output_path: str) -> None:
    """Guarda la configuración usada en un archivo .yaml (reproducibilidad)."""
    with open(output_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[config] Configuración guardada en: {output_path}")