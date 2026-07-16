"""
TabNet para clasificación y regresión tabular.
Usa la librería pytorch-tabnet (implementación oficial).
"""
import numpy as np
import torch
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor


class TabNetModel:
    """
    Wrapper de TabNet con la misma interfaz que los otros modelos del proyecto.
    Soporta clasificación y regresión.
    """

    def __init__(self, task: str = "classification", seed: int = 42,
                 device: str = "cpu", input_dim: int = None, num_classes: int = None,
                 n_d: int = 8, n_a: int = 8, n_steps: int = 3, gamma: float = 1.3,
                 learning_rate: float = 0.02, epochs: int = 100,
                 batch: int = 32, patience: int = 10,
                 **kwargs):
        """
        Args principales:
            task: 'classification' o 'regression'
            n_d: dimensión de las features de decisión (default 8)
            n_a: dimensión de las features de atención (default 8)
            n_steps: número de pasos de decisión secuencial (default 3)
            gamma: coeficiente de relajación (default 1.3)
            learning_rate: tasa de aprendizaje
            epochs: máximo de épocas
            batch: tamaño de batch
            patience: épocas sin mejora para early stopping
        """
        self.task = task
        self.seed = seed
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs if epochs is not None else 100
        self.batch_size = batch if batch is not None else 32
        self.patience = patience if patience is not None else 10

        # Configurar dispositivo
        if device == "cpu":
            device_name = "cpu"
        else:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        # Fijar semillas
        torch.manual_seed(seed)
        if device_name == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Parámetros comunes para TabNet
        tabnet_params = {
            "n_d": n_d,
            "n_a": n_a,
            "n_steps": n_steps,
            "gamma": gamma,
            "seed": seed,
            "device_name": device_name,
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=learning_rate),
            "scheduler_params": {"step_size": 50, "gamma": 0.9},
            "scheduler_fn": torch.optim.lr_scheduler.StepLR,
            "verbose": 10,  # imprime cada 10 épocas
        }

        # Instanciar según la tarea
        if task == "classification":
            self.model = TabNetClassifier(**tabnet_params)
        elif task == "regression":
            self.model = TabNetRegressor(**tabnet_params)
        else:
            raise ValueError(f"Tarea no válida para TabNet: {task}")

        print(f"[tabnet] Modelo creado para {task} | device={device_name}")
        print(f"[tabnet] Arquitectura: n_d={n_d}, n_a={n_a}, n_steps={n_steps}, gamma={gamma}")
        print(f"[tabnet] Hiperparámetros: lr={learning_rate}, epochs={self.epochs}, "
              f"batch={self.batch_size}, patience={self.patience}")

    def train(self, X_train, y_train):
        """Entrena TabNet con early stopping interno usando un split de validación."""
        # Convertir a numpy
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train)

        # Separar val interno para early stopping
        n = len(X_train)
        val_n = int(n * 0.15)
        idx = np.arange(n)
        np.random.seed(self.seed)
        np.random.shuffle(idx)
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # En regresión, y debe tener shape (n, 1)
        if self.task == "regression":
            y_tr = y_tr.astype(np.float32).reshape(-1, 1)
            y_val = y_val.astype(np.float32).reshape(-1, 1)

        print(f"[tabnet] Train interno: {len(X_tr)} | Val interno: {len(X_val)}")

        eval_metric = ["accuracy"] if self.task == "classification" else ["rmse"]

        self.model.fit(
            X_train=X_tr,
            y_train=y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_metric,
            max_epochs=self.epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=min(128, self.batch_size),
        )

        print(f"[tabnet] Entrenamiento completado")
        return self

    def predict(self, X):
        """Devuelve predicciones (etiquetas para clasificación, valores para regresión)."""
        X = np.asarray(X, dtype=np.float32)
        preds = self.model.predict(X)
        # En regresión, TabNet devuelve shape (n, 1) → aplanamos
        if self.task == "regression":
            preds = preds.flatten()
        return preds

    def predict_proba(self, X):
        """Devuelve probabilidades (solo clasificación)."""
        if self.task != "classification":
            raise ValueError("predict_proba solo aplica en clasificación")
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names: list) -> dict:
        """TabNet expone feature_importances_ nativamente (ventaja propia)."""
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances.tolist()))

    def save(self, path: str):
        """Guarda el modelo entrenado."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # TabNet guarda como .zip internamente; ajustamos extensión
        if path.suffix in {".pkl", ".pt"}:
            path = path.with_suffix("")  # quita extensión, TabNet le pone .zip
        self.model.save_model(str(path))
        print(f"[tabnet] Modelo guardado en: {path}.zip")

    def load(self, path: str):
        """Carga un modelo previamente guardado."""
        self.model.load_model(str(path))
        print(f"[tabnet] Modelo cargado desde: {path}")
        return self