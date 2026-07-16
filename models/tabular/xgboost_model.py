"""
Modelo XGBoost para clasificación y regresión.
"""
import joblib
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor


class XGBoostModel:
    """
    Wrapper de XGBoost que soporta clasificación y regresión.
    Mantiene la misma interfaz que RandomForestModel.
    """

    def __init__(self, task: str = "classification", seed: int = 42,
                 device: str = "cpu", **hyperparams):
        """
        Args:
            task: 'classification' o 'regression'
            seed: semilla para reproducibilidad
            device: 'cpu' o 'cuda' (GPU)
            **hyperparams: hiperparámetros adicionales
        """
        self.task = task
        self.seed = seed
        self.device = device
        self.model = None

        # Hiperparámetros por defecto (sobrescribibles desde consola/YAML)
        self.params = {
            "n_estimators": hyperparams.get("n_estimators", 100),
            "max_depth": hyperparams.get("max_depth", 6),
            "learning_rate": hyperparams.get("learning_rate", 0.1),
            "subsample": hyperparams.get("subsample", 1.0),
            "random_state": seed,
            "n_jobs": -1,
            "device": "cuda" if device != "cpu" else "cpu",
        }

        # Instanciar el modelo según la tarea
        if task == "classification":
            self.params["eval_metric"] = "logloss"
            self.model = XGBClassifier(**self.params)
        elif task == "regression":
            self.params["eval_metric"] = "rmse"
            self.model = XGBRegressor(**self.params)
        else:
            raise ValueError(f"Tarea no válida para XGBoost: {task}")

        print(f"[xgboost] Modelo creado para {task} con params: {self.params}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrena el modelo. Si se pasa conjunto de validación, XGBoost lo usa
        para monitorear el desempeño durante el entrenamiento.
        """
        print(f"[xgboost] Entrenando con {len(X_train)} muestras...")

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        print(f"[xgboost] Entrenamiento completado ✓")
        return self

    def predict(self, X):
        """Predice etiquetas (clasificación) o valores (regresión)."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predice probabilidades (solo clasificación)."""
        if self.task != "classification":
            raise ValueError("predict_proba solo aplica en clasificación")
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names: list) -> dict:
        """Devuelve la importancia de cada feature."""
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))

    def save(self, path: str):
        """Guarda el modelo entrenado en disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[xgboost] Modelo guardado en: {path}")

    def load(self, path: str):
        """Carga un modelo previamente guardado."""
        self.model = joblib.load(path)
        print(f"[xgboost] Modelo cargado desde: {path}")
        return self