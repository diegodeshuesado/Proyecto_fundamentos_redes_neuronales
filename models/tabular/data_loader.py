"""
Módulo para cargar y preparar datasets tabulares.
Soporta .csv y .xlsx
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga un dataset desde .csv o .xlsx y devuelve un DataFrame.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(p)
    elif ext == ".xlsx":
        df = pd.read_excel(p)
    else:
        raise ValueError(f"Extensión no soportada: {ext}")

    print(f"[data_loader] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def split_features_target(df: pd.DataFrame, target: str):
    """
    Separa el DataFrame en X (features) e y (target).
    """
    if target not in df.columns:
        raise ValueError(
            f"Columna objetivo '{target}' no encontrada. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    X = df.drop(columns=[target])
    y = df[target]
    print(f"[data_loader] Features: {X.shape[1]}, Target: '{target}'")
    return X, y


def detect_task_type(y: pd.Series) -> str:
    """
    Detecta automáticamente si es clasificación o regresión.
    - Si el target es numérico continuo (float) → regresión.
    - Si es entero/categórico con pocos valores únicos → clasificación.
    """
    n_unique = y.nunique()

    # Si es float o tiene muchos valores únicos → regresión
    if y.dtype.kind == "f" or n_unique > 20:
        return "regression"
    else:
        return "classification"


def split_train_test(X, y, test_size: float = 0.2, seed: int = 42, task: str = "classification"):
    """
    Divide los datos en train/test.
    Para clasificación usa stratify para mantener proporción de clases.
    """
    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify
    )

    print(f"[data_loader] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def load_and_prepare(path: str, target: str, test_size: float = 0.2, seed: int = 42):
    """
    Pipeline completo: carga + separa + divide.
    Esta es la función "maestra" que llamarás desde trainer.py.
    """
    df = load_dataset(path)
    X, y = split_features_target(df, target)
    task_detected = detect_task_type(y)
    print(f"[data_loader] Tipo de tarea detectado: {task_detected}")

    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, seed=seed, task=task_detected
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "task_detected": task_detected,
        "feature_names": list(X.columns),
    }