"""
FT-Transformer para clasificacion y regresion tabular.
Usa pytorch-tabular (que internamente usa PyTorch Lightning).
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)


class FTTransformerModel:
    """Wrapper de FT-Transformer con la misma interfaz que los otros modelos."""

    def __init__(self, task="classification", seed=42, device="cpu",
                 input_dim=None, num_classes=None,
                 input_embed_dim=32, num_heads=8, num_attn_blocks=3,
                 attn_dropout=0.1, ff_dropout=0.1,
                 learning_rate=0.001, epochs=100,
                 batch=64, patience=10, **kwargs):
        self.task = task
        self.seed = seed
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs if epochs is not None else 100
        self.batch_size = batch if batch is not None else 64
        self.patience = patience if patience is not None else 10
        self.learning_rate = learning_rate

        self.input_embed_dim = input_embed_dim
        self.num_heads = num_heads
        self.num_attn_blocks = num_attn_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if device == "cpu":
            self.accelerator = "cpu"
        else:
            self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        self.model = None
        self._feature_names = None
        self._target_name = "target"

        print(f"[ft_transformer] Configurado para {task} | accelerator={self.accelerator}")
        print(f"[ft_transformer] Arquitectura: embed={input_embed_dim}, heads={num_heads}, blocks={num_attn_blocks}")
        print(f"[ft_transformer] Hiperparametros: lr={learning_rate}, epochs={self.epochs}, batch={self.batch_size}, patience={self.patience}")

    def _build_df(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self._feature_names = list(df.columns)
        else:
            X = np.asarray(X, dtype=np.float32)
            if self._feature_names is None:
                self._feature_names = [f"feat_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=self._feature_names)

        if y is not None:
            df[self._target_name] = np.asarray(y)
        return df

    def train(self, X_train, y_train):
        df_train = self._build_df(X_train, y_train)

        n = len(df_train)
        val_n = int(n * 0.15)
        idx = np.arange(n)
        np.random.seed(self.seed)
        np.random.shuffle(idx)
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        df_tr = df_train.iloc[train_idx].reset_index(drop=True)
        df_val = df_train.iloc[val_idx].reset_index(drop=True)

        print(f"[ft_transformer] Train interno: {len(df_tr)} | Val interno: {len(df_val)}")

        task_str = "classification" if self.task == "classification" else "regression"

        data_config = DataConfig(
            target=[self._target_name],
            continuous_cols=self._feature_names,
            categorical_cols=[],
        )

        trainer_config = TrainerConfig(
            batch_size=self.batch_size,
            max_epochs=self.epochs,
            early_stopping="valid_loss",
            early_stopping_patience=self.patience,
            checkpoints=None,
            accelerator=self.accelerator,
            seed=self.seed,
            progress_bar="none",
        )

        optimizer_config = OptimizerConfig(
            optimizer="Adam",
            optimizer_params={"weight_decay": 1e-5},
        )

        model_config = FTTransformerConfig(
            task=task_str,
            input_embed_dim=self.input_embed_dim,
            num_heads=self.num_heads,
            num_attn_blocks=self.num_attn_blocks,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            learning_rate=self.learning_rate,
        )

        self.model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        self.model.fit(train=df_tr, validation=df_val)
        print(f"[ft_transformer] Entrenamiento completado")
        return self

    def predict(self, X):
        df = self._build_df(X)
        preds_df = self.model.predict(df)
        col = f"{self._target_name}_prediction"
        return preds_df[col].values

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba solo aplica en clasificacion")
        df = self._build_df(X)
        preds_df = self.model.predict(df)
        prefix = f"{self._target_name}_"
        prob_cols = [c for c in preds_df.columns if c.endswith("_probability")]
        prob_cols = sorted(
            prob_cols,
            key=lambda c: int(c.replace(prefix, "").replace("_probability", ""))
        )
        return preds_df[prob_cols].values

    def get_feature_importances(self, feature_names):
        return {}

    def save(self, path):
        path = Path(path)
        if path.suffix in {".pkl", ".pt", ".zip"}:
            path = path.with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        print(f"[ft_transformer] Modelo guardado en: {path}/")

    def load(self, path):
        self.model = TabularModel.load_model(str(path))
        print(f"[ft_transformer] Modelo cargado desde: {path}")
        return self
