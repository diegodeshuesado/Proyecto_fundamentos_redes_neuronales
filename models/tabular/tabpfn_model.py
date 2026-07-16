"""
TabPFN para clasificacion tabular.
Es un foundation model pre-entrenado: no entrena con tus datos, los usa como contexto.
Limitaciones: max ~10K filas, ~100 features, ~10 clases. Solo clasificacion estable.
"""
import numpy as np
import joblib
from pathlib import Path

from tabpfn import TabPFNClassifier


class TabPFNModel:
    """Wrapper de TabPFN con la misma interfaz que los otros modelos."""

    def __init__(self, task="classification", seed=42, device="cpu",
                 input_dim=None, num_classes=None, **kwargs):
        self.task = task
        self.seed = seed
        self.input_dim = input_dim
        self.num_classes = num_classes

        if task != "classification":
            raise ValueError("TabPFN solo soporta clasificacion en esta version")

        # Detectar device
        if device == "cpu":
            self.device = "cpu"
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        self.model = TabPFNClassifier(device=self.device, random_state=seed)

        print(f"[tabpfn] Modelo cargado | device={self.device}")
        print(f"[tabpfn] NOTA: TabPFN no entrena, usa los datos como contexto (foundation model)")

    def train(self, X_train, y_train):
        """En TabPFN, 'train' solo guarda los datos como contexto (no hay backprop)."""
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train)

        # Validaciones por limitaciones de TabPFN
        n, d = X_train.shape
        if n > 10000:
            print(f"[tabpfn] ADVERTENCIA: {n} filas excede el limite recomendado (10K). "
                  f"Puede ser lento o fallar.")
        if d > 100:
            print(f"[tabpfn] ADVERTENCIA: {d} features excede el limite recomendado (100).")

        print(f"[tabpfn] Ajustando contexto con {n} muestras y {d} features...")
        self.model.fit(X_train, y_train)
        print(f"[tabpfn] Contexto cargado")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names):
        # TabPFN no expone feature importances
        return {}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix not in {".pkl", ".joblib"}:
            path = path.with_suffix(".pkl")
        joblib.dump(self.model, path)
        print(f"[tabpfn] Modelo guardado en: {path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"[tabpfn] Modelo cargado desde: {path}")
        return self