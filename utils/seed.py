"""
Módulo para fijar la semilla aleatoria y garantizar reproducibilidad.
"""
import random
import os
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Fija la semilla en todas las librerías que usan aleatoriedad.

    Args:
        seed: número entero que se usará como semilla. Default 42.
    """
    random.seed(seed)              # módulo random de Python
    np.random.seed(seed)           # NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)  # hash de Python

    # Cuando agreguemos PyTorch (Fase 2-3) descomentar:
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    print(f"[seed] Semilla fijada en: {seed}")