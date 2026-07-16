"""
Trainer para modelos tabulares.
Orquesta: carga, preprocesa, entrena, evalua, guarda.
"""
import time
import json
from pathlib import Path

from models.tabular.data_loader import load_and_prepare
from models.tabular.preprocessor import TabularPreprocessor
from models.tabular.metrics import evaluate_and_save
from models.tabular.random_forest import RandomForestModel
from models.tabular.xgboost_model import XGBoostModel
from models.tabular.mlp import MLPModel
from models.tabular.tabnet_model import TabNetModel
from models.tabular.ft_transformer import FTTransformerModel
from models.tabular.tabpfn_model import TabPFNModel


MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "mlp": MLPModel,
    "tabnet": TabNetModel,
    "ft_transformer": FTTransformerModel,
    "tabpfn": TabPFNModel,
}


def build_model(model_name: str, task: str, seed: int, device: str, **hyperparams):
    """Factory: crea la instancia del modelo segun el nombre."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_name}' no implementado todavia. "
            f"Disponibles: {list(MODEL_REGISTRY.keys())}"
        )

    ModelClass = MODEL_REGISTRY[model_name]

    if model_name in {"xgboost", "mlp", "tabnet", "ft_transformer", "tabpfn"}:
        return ModelClass(task=task, seed=seed, device=device, **hyperparams)
    else:
        return ModelClass(task=task, seed=seed, **hyperparams)


def train_tabular(config: dict, run_dir: Path, logger) -> dict:
    """Pipeline completo de entrenamiento tabular."""
    logger.info("=" * 60)
    logger.info("Iniciando pipeline TABULAR")
    logger.info("=" * 60)

    # 1. Cargar y dividir datos
    logger.info("[1/5] Cargando dataset...")
    data = load_and_prepare(
        path=config["dataset"],
        target=config["target"],
        seed=config.get("seed", 42),
    )
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    task_detected = data["task_detected"]
    feature_names = data["feature_names"]

    if config["task"] != task_detected:
        logger.warning(
            f"Tarea declarada ('{config['task']}') no coincide con la detectada "
            f"('{task_detected}'). Se usara la declarada."
        )

    # 2. Preprocesar
    logger.info("[2/5] Preprocesando datos...")
    prep = TabularPreprocessor(scale_features=True)
    prep.fit(X_train, y_train, task=config["task"])
    X_train_prep = prep.transform(X_train)
    X_test_prep = prep.transform(X_test)
    y_train_prep = prep.transform_target(y_train)
    y_test_prep = prep.transform_target(y_test)

    # 3. Crear modelo
    logger.info(f"[3/5] Creando modelo: {config['model']}")
    hyperparams = {
        k: v for k, v in config.items()
        if k in {
            "n_estimators", "max_depth", "learning_rate", "subsample", "min_samples_split",
            "hidden_dims", "dropout", "patience", "batch", "epochs",
        }
        and v is not None
    }

    if config["model"] in {"mlp", "tabnet", "ft_transformer", "tabpfn"}:
        hyperparams["input_dim"] = X_train_prep.shape[1]
        if config["task"] == "classification":
            hyperparams["num_classes"] = len(set(y_train_prep))
        else:
            hyperparams["num_classes"] = 1

    model = build_model(
        model_name=config["model"],
        task=config["task"],
        seed=config.get("seed", 42),
        device=config.get("device", "cpu"),
        **hyperparams,
    )

    # 4. Entrenar
    logger.info("[4/5] Entrenando modelo...")
    t_start = time.time()
    model.train(X_train_prep, y_train_prep)
    train_time = time.time() - t_start
    logger.info(f"Tiempo de entrenamiento: {train_time:.2f} segundos")

    # 5. Evaluar
    logger.info("[5/5] Evaluando en conjunto de test...")
    y_pred = model.predict(X_test_prep)

    y_proba = None
    if config["task"] == "classification":
        try:
            y_proba = model.predict_proba(X_test_prep)
        except Exception as e:
            logger.warning(f"No se pudieron obtener probabilidades: {e}")

    class_names = None
    if prep.target_encoder is not None:
        class_names = list(prep.target_encoder.classes_)

    metrics = evaluate_and_save(
        y_true=y_test_prep,
        y_pred=y_pred,
        y_proba=y_proba,
        task=config["task"],
        run_dir=run_dir,
        class_names=class_names,
    )
    metrics["train_time_seconds"] = round(train_time, 2)

    model.save(run_dir / "model.pkl")

    try:
        importances = model.get_feature_importances(feature_names)
        # Convertir numpy floats a Python floats para que sean JSON-serializables
        importances = {k: float(v) for k, v in importances.items()}
        with open(run_dir / "feature_importances.json", "w") as f:
            json.dump(importances, f, indent=2)
        logger.info("Feature importances guardadas")
    except Exception as e:
        logger.warning(f"No se pudieron obtener feature importances: {e}")

    logger.info("=" * 60)
    logger.info("METRICAS FINALES:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    return metrics