"""
YOLO para deteccion de objetos.
Usa la libreria ultralytics (versiones YOLOv8/v11/v12).
"""
import shutil
import yaml
from pathlib import Path

from ultralytics import YOLO


class YOLOModel:
    """Wrapper de YOLO con interfaz consistente al resto del proyecto."""

    def __init__(self, weights: str = "yolov8n.pt", device: str = "cpu", seed: int = 42, **kwargs):
        """
        Args:
            weights: pesos preentrenados (yolov8n.pt, yolov8s.pt, ...) o ruta a .pt propio
            device: 'cpu' o numero de GPU (ej: '0')
            seed: semilla para reproducibilidad
        """
        self.weights = weights
        self.device = device
        self.seed = seed

        # Cargar modelo (descarga pesos automaticamente la primera vez)
        self.model = YOLO(weights)
        print(f"[yolo] Modelo cargado: {weights} | device={device}")

    def train(self, data_yaml: str, epochs: int = 50, batch: int = 16,
              imgsz: int = 640, project_dir: Path = None, run_name: str = "yolo_run"):
        """
        Entrena YOLO con el dataset definido en data.yaml.

        Args:
            data_yaml: ruta al data.yaml del dataset
            epochs: numero de epocas
            batch: tamano de batch
            imgsz: tamano de imagen (640 estandar)
            project_dir: carpeta donde guardar resultados (ej: runs/<name>/yolo)
            run_name: subcarpeta dentro de project_dir
        """
        print(f"[yolo] Entrenando: epochs={epochs}, batch={batch}, imgsz={imgsz}")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            seed=self.seed,
            project=str(project_dir) if project_dir else "runs/yolo",
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        print(f"[yolo] Entrenamiento completado")
        return results

    def validate(self, data_yaml: str = None):
        """Evalua el modelo en el conjunto de validacion."""
        print(f"[yolo] Validando modelo...")
        metrics = self.model.val(data=data_yaml, device=self.device)

        # Extraer metricas principales en dict legible
        result = {
            "mAP50": float(metrics.box.map50),       # mean Average Precision @ IoU 0.5
            "mAP50-95": float(metrics.box.map),      # mAP promedio de IoU 0.5 a 0.95
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        print(f"[yolo] Metricas: {result}")
        return result

    def predict(self, source, save: bool = True, output_dir: Path = None):
        """
        Predice sobre una imagen, carpeta o video.

        Args:
            source: ruta de archivo, carpeta o glob
            save: si True, guarda imagenes con cajas dibujadas
            output_dir: donde guardar predicciones
        """
        kwargs = {"source": source, "device": self.device, "save": save}
        if output_dir is not None:
            kwargs["project"] = str(output_dir.parent)
            kwargs["name"] = output_dir.name
            kwargs["exist_ok"] = True

        results = self.model.predict(**kwargs)
        return results

    def save(self, path: str):
        """Guarda el modelo entrenado (.pt)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")

        # ultralytics ya guarda automaticamente best.pt durante train()
        # Aqui copiamos best.pt al destino final si existe
        if hasattr(self.model, "trainer") and self.model.trainer is not None:
            best = Path(self.model.trainer.best)
            if best.exists():
                shutil.copy2(best, path)
                print(f"[yolo] Modelo guardado en: {path}")
                return
        # Fallback: guardar el modelo actual
        self.model.save(str(path))
        print(f"[yolo] Modelo guardado en: {path}")

    def load(self, path: str):
        """Carga un modelo previamente guardado."""
        self.model = YOLO(path)
        print(f"[yolo] Modelo cargado desde: {path}")
        return self