"""
Modelo Random Forest para clasificación y regresión.
"""
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForestModel:
    """
    Wrapper de Random Forest que soporta clasificación y regresión.
    """

    def __init__(self, task: str = "classification", seed: int = 42, **hyperparams):
        """
        Args:
            task: 'classification' o 'regression'
            seed: semilla para reproducibilidad
            **hyperparams: hiperparámetros adicionales (n_estimators, max_depth, etc.)
        """
        self.task = task
        self.seed = seed
        self.model = None

        # Hiperparámetros por defecto (sobrescribibles desde consola/YAML)
        self.params = {
            "n_estimators": hyperparams.get("n_estimators", 100),
            "max_depth": hyperparams.get("max_depth", None),
            "min_samples_split": hyperparams.get("min_samples_split", 2),
            "random_state": seed,
            "n_jobs": -1,  # usa todos los cores disponibles
        }

        # Instanciar el modelo según la tarea
        if task == "classification":
            self.model = RandomForestClassifier(**self.params)
        elif task == "regression":
            self.model = RandomForestRegressor(**self.params)
        else:
            raise ValueError(f"Tarea no válida para Random Forest: {task}")

        print(f"[random_forest] Modelo creado para {task} con params: {self.params}")

    def train(self, X_train, y_train):
        """Entrena el modelo con los datos de entrenamiento."""
        print(f"[random_forest] Entrenando con {len(X_train)} muestras...")
        self.model.fit(X_train, y_train)
        print(f"[random_forest] Entrenamiento completado ✓")
        return self

    def predict(self, X):
        """Predice etiquetas (clasificación) o valores (regresión)."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predice probabilidades (solo clasificación).
        Necesario para calcular curvas ROC.
        """
        if self.task != "classification":
            raise ValueError("predict_proba solo aplica en clasificación")
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names: list) -> dict:
        """
        Devuelve la importancia de cada feature.
        Útil para entender qué variables son más relevantes.
        """
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))

    def save(self, path: str):
        """Guarda el modelo entrenado en disco (.pkl)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[random_forest] Modelo guardado en: {path}")

    def load(self, path: str):
        """Carga un modelo previamente guardado."""
        self.model = joblib.load(path)
        print(f"[random_forest] Modelo cargado desde: {path}")
        return self