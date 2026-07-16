"""
Módulo de métricas y gráficas para evaluación de modelos tabulares.
Soporta clasificación y regresión.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)


# ============================================================
# CLASIFICACIÓN
# ============================================================

def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calcula métricas estándar para clasificación."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # AUC solo aplica para clasificación binaria
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            metrics["auc"] = float(auc(fpr, tpr))
        except Exception as e:
            print(f"[metrics] No se pudo calcular AUC: {e}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, output_path: Path, class_names=None):
    """Genera y guarda matriz de confusión como PNG."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names if class_names is not None else "auto",
        yticklabels=class_names if class_names is not None else "auto"
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[metrics] Matriz de confusión guardada en: {output_path}")


def plot_roc_curve(y_true, y_proba, output_path: Path):
    """Genera y guarda curva ROC (solo clasificación binaria)."""
    if len(np.unique(y_true)) != 2:
        print("[metrics] ROC solo aplica para clasificación binaria, se omite.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[metrics] Curva ROC guardada en: {output_path}")


# ============================================================
# REGRESIÓN
# ============================================================

def compute_regression_metrics(y_true, y_pred) -> dict:
    """Calcula métricas estándar para regresión."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_predictions_vs_real(y_true, y_pred, output_path: Path):
    """Gráfica de dispersión predicciones vs valores reales (regresión)."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", s=40)

    # Línea diagonal ideal (predicción perfecta)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Predicción perfecta")

    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title("Predicciones vs Valores Reales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[metrics] Gráfica de predicciones guardada en: {output_path}")


# ============================================================
# UTILIDADES
# ============================================================

def save_metrics(metrics: dict, output_path: Path):
    """Guarda las métricas como JSON."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] Métricas guardadas en: {output_path}")


def save_classification_report(y_true, y_pred, output_path: Path, class_names=None):
    """Guarda el reporte de clasificación completo como texto."""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )
    with open(output_path, "w") as f:
        f.write(report)
    print(f"[metrics] Reporte de clasificación guardado en: {output_path}")


def evaluate_and_save(y_true, y_pred, y_proba, task: str, run_dir: Path, class_names=None) -> dict:
    """
    Función maestra: calcula métricas, genera gráficas y guarda todo en run_dir.
    """
    if task == "classification":
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        plot_confusion_matrix(y_true, y_pred, run_dir / "confusion_matrix.png", class_names)
        if y_proba is not None:
            plot_roc_curve(y_true, y_proba, run_dir / "roc_curve.png")
        save_classification_report(y_true, y_pred, run_dir / "classification_report.txt", class_names)

    elif task == "regression":
        metrics = compute_regression_metrics(y_true, y_pred)
        plot_predictions_vs_real(y_true, y_pred, run_dir / "predictions_vs_real.png")

    else:
        raise ValueError(f"Tarea no soportada para métricas: {task}")

    save_metrics(metrics, run_dir / "metrics.json")
    return metrics