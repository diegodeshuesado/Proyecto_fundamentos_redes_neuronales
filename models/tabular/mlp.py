"""
Multi-Layer Perceptron (MLP) para clasificación y regresión tabular.
Implementado con PyTorch puro.
Incluye early stopping y arquitectura flexible.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ============================================================
# 1. Dataset wrapper para PyTorch
# ============================================================

class TabularDataset(Dataset):
    """Envuelve arrays de NumPy/Pandas en formato PyTorch."""

    def __init__(self, X, y=None, task="classification"):
        X = np.asarray(X, dtype=np.float32)
        self.X = torch.from_numpy(X)

        if y is not None:
            y_arr = np.asarray(y)
            if task == "classification":
                self.y = torch.from_numpy(y_arr).long()
            else:
                self.y = torch.from_numpy(y_arr.astype(np.float32))
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ============================================================
# 2. Arquitectura de la red (flexible)
# ============================================================

class MLPNet(nn.Module):
    """
    Red MLP con número y tamaño de capas configurables.
    Construye dinámicamente: input → [hidden_dims] → output
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. Wrapper con interfaz estándar
# ============================================================

class MLPModel:
    """Wrapper de MLP con la misma interfaz que RandomForestModel y XGBoostModel."""

    def __init__(self, task: str = "classification", seed: int = 42,
                 device: str = "cpu", input_dim: int = None, num_classes: int = None,
                 hidden_dims: list = None, dropout: float = 0.2,
                 learning_rate: float = 0.001, epochs: int = 100,
                 batch: int = 32, patience: int = 10, val_split: float = 0.15,
                 **kwargs):
        self.task = task
        self.seed = seed
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs if epochs is not None else 100
        self.batch_size = batch if batch is not None else 32
        self.patience = patience if patience is not None else 10
        self.val_split = val_split

        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        output_dim = num_classes if task == "classification" else 1

        self.net = MLPNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout=self.dropout,
        ).to(self.device)

        if task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        print(f"[mlp] Modelo creado para {task} | device={self.device}")
        print(f"[mlp] Arquitectura: input={input_dim} -> {self.hidden_dims} -> output={output_dim}")
        print(f"[mlp] Hiperparámetros: lr={self.learning_rate}, epochs={self.epochs}, "
              f"batch={self.batch_size}, dropout={self.dropout}, patience={self.patience}")

    def train(self, X_train, y_train):
        n = len(X_train)
        val_n = int(n * self.val_split)
        idx = np.arange(n)
        np.random.seed(self.seed)
        np.random.shuffle(idx)
        val_idx, train_idx = idx[:val_n], idx[val_n:]

        X_tr = np.asarray(X_train)[train_idx]
        y_tr = np.asarray(y_train)[train_idx]
        X_val = np.asarray(X_train)[val_idx]
        y_val = np.asarray(y_train)[val_idx]

        train_ds = TabularDataset(X_tr, y_tr, task=self.task)
        val_ds = TabularDataset(X_val, y_val, task=self.task)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        print(f"[mlp] Train interno: {len(X_tr)} | Val interno: {len(X_val)}")

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.net(xb)
                if self.task == "regression":
                    logits = logits.squeeze(-1)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.net(xb)
                    if self.task == "regression":
                        logits = logits.squeeze(-1)
                    loss = self.criterion(logits, yb)
                    val_loss += loss.item() * len(xb)
            val_loss /= len(val_ds)

            if epoch % 10 == 0 or epoch == 1:
                print(f"[mlp] Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"[mlp] Early stopping en época {epoch} (mejor val_loss={best_val_loss:.4f})")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        print(f"[mlp] Entrenamiento completado")
        return self

    def predict(self, X):
        self.net.eval()
        ds = TabularDataset(X, task=self.task)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                logits = self.net(xb)
                if self.task == "classification":
                    preds.append(logits.argmax(dim=1).cpu().numpy())
                else:
                    preds.append(logits.squeeze(-1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba solo aplica en clasificación")

        self.net.eval()
        ds = TabularDataset(X, task=self.task)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        probas = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                logits = self.net(xb)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probas.append(p)
        return np.concatenate(probas, axis=0)

    def get_feature_importances(self, feature_names: list) -> dict:
        first_layer = None
        for module in self.net.net:
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        if first_layer is None:
            return {}

        weights = first_layer.weight.detach().cpu().numpy()
        importances = np.abs(weights).mean(axis=0)
        return dict(zip(feature_names, importances.tolist()))

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".pkl":
            path = path.with_suffix(".pt")
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": {
                "task": self.task,
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
        }, path)
        print(f"[mlp] Modelo guardado en: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        print(f"[mlp] Modelo cargado desde: {path}")
        return self
