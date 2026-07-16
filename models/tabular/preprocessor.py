"""
Módulo de preprocesamiento para datos tabulares.
Maneja valores nulos, encoding de categóricas, escalado y encoding del target.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class TabularPreprocessor:
    """
    Preprocesador para datos tabulares.
    Aprende del conjunto de entrenamiento y aplica las mismas transformaciones a test.
    """

    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features

        # Estos se "aprenden" en fit() y se reutilizan en transform()
        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.label_encoders = {}      # uno por cada columna categórica
        self.target_encoder = None    # para el target si es texto
        self.num_cols = []
        self.cat_cols = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None, task: str = "classification"):
        """
        Aprende los parámetros del preprocesamiento usando los datos de entrenamiento.
        """
        # 1. Detectar columnas numéricas vs categóricas
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        print(f"[preprocessor] Numéricas: {len(self.num_cols)} | Categóricas: {len(self.cat_cols)}")

        # 2. Imputadores para nulos
        if self.num_cols:
            self.num_imputer = SimpleImputer(strategy="mean")
            self.num_imputer.fit(X[self.num_cols])

        if self.cat_cols:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            self.cat_imputer.fit(X[self.cat_cols])

        # 3. Label encoders para categóricas (uno por columna)
        for col in self.cat_cols:
            le = LabelEncoder()
            # rellenar nulos antes de encodear para evitar errores
            le.fit(X[col].fillna("__missing__").astype(str))
            self.label_encoders[col] = le

        # 4. Scaler para columnas numéricas
        if self.scale_features and self.num_cols:
            self.scaler = StandardScaler()
            # entrenar scaler con los numéricos ya imputados
            X_num_imputed = self.num_imputer.transform(X[self.num_cols])
            self.scaler.fit(X_num_imputed)

        # 5. Encoder del target (solo si es clasificación con texto)
        if y is not None and task == "classification" and y.dtype == "object":
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(y.astype(str))
            print(f"[preprocessor] Target encodeado: {dict(zip(self.target_encoder.classes_, range(len(self.target_encoder.classes_))))}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica las transformaciones aprendidas a un DataFrame (train o test).
        """
        X = X.copy()

        # 1. Imputar nulos
        if self.num_cols and self.num_imputer:
            X[self.num_cols] = self.num_imputer.transform(X[self.num_cols])

        if self.cat_cols and self.cat_imputer:
            X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])

        # 2. Encodear categóricas
        for col in self.cat_cols:
            le = self.label_encoders[col]
            # manejar valores nuevos no vistos en entrenamiento
            X[col] = X[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )

        # 3. Escalar numéricas
        if self.scale_features and self.num_cols and self.scaler:
            X[self.num_cols] = self.scaler.transform(X[self.num_cols])

        return X

    def transform_target(self, y: pd.Series) -> pd.Series:
        """Transforma el target si tiene encoder (solo clasificación con texto)."""
        if self.target_encoder is not None:
            return pd.Series(
                self.target_encoder.transform(y.astype(str)),
                index=y.index
            )
        return y

    def inverse_transform_target(self, y_pred):
        """Revierte el encoding del target (útil para reportes legibles)."""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y_pred)
        return y_pred