"""
SSD (Single Shot MultiBox Detector) para deteccion de objetos.
Usa torchvision con backbone VGG16 preentrenado en COCO.
Estructura muy similar a Faster R-CNN (mismo dataset adapter, mismo loop).
"""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

from models.detection.yolo_to_torch_dataset import YoloTorchvisionDataset, collate_fn


class SSDModel:
    """Wrapper de SSD con interfaz consistente al proyecto."""

    def __init__(self, weights: str = "ssd300_vgg16", device: str = "cpu",
                 seed: int = 42, num_classes: int = None, **kwargs):
        self.weights_arg = weights
        self.device_str = device
        self.seed = seed
        self.num_classes = num_classes
        self.model = None
        self.class_names = None

        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        print(f"[ssd] Configurado | device={self.device}")

    def _build_model(self, num_classes: int):
        """Construye SSD300-VGG16 preentrenado y ajusta la cabeza clasificadora."""
        # Cargar pesos preentrenados en COCO
        model = ssd300_vgg16(weights="DEFAULT")

        # SSD tiene multiples cabezas de clasificacion (una por feature map)
        # Necesitamos reemplazarlas todas para que salgan (num_classes + 1) clases
        # (+1 por el fondo)
        in_channels = [module.in_channels for module in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()

        model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes + 1,
        )

        return model.to(self.device)

    def _load_data_yaml(self, data_yaml_path: str):
        """Lee el data.yaml y devuelve rutas + info de clases."""
        import yaml
        with open(data_yaml_path) as f:
            data_cfg = yaml.safe_load(f)

        base = Path(data_cfg.get("path", Path(data_yaml_path).parent))
        train_images = base / data_cfg["train"]
        val_images = base / data_cfg["val"]
        train_labels = Path(str(train_images).replace("images", "labels"))
        val_labels = Path(str(val_images).replace("images", "labels"))

        return {
            "train_images": train_images,
            "train_labels": train_labels,
            "val_images": val_images,
            "val_labels": val_labels,
            "num_classes": data_cfg["nc"],
            "class_names": data_cfg["names"],
        }

    def train(self, data_yaml: str, epochs: int = 20, batch: int = 8,
              imgsz: int = 300, project_dir: Path = None, run_name: str = "ssd_run"):
        """
        Entrena SSD. Nota: SSD300 usa imagenes de 300x300 (redimensiona internamente).
        """
        cfg = self._load_data_yaml(data_yaml)
        self.num_classes = cfg["num_classes"]
        self.class_names = cfg["class_names"]

        self.model = self._build_model(self.num_classes)

        train_ds = YoloTorchvisionDataset(cfg["train_images"], cfg["train_labels"])
        val_ds = YoloTorchvisionDataset(cfg["val_images"], cfg["val_labels"])

        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=True,
            collate_fn=collate_fn, num_workers=2,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch, shuffle=False,
            collate_fn=collate_fn, num_workers=2,
        )

        # SGD es el estandar para SSD (paper original)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        print(f"[ssd] Entrenando: epochs={epochs}, batch={batch}, "
              f"train={len(train_ds)}, val={len(val_ds)}")

        history = {"epoch": [], "train_loss": []}
        best_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for images, targets in train_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["epoch"].append(epoch)
            history["train_loss"].append(avg_loss)

            lr_scheduler.step()
            print(f"[ssd] Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        print(f"[ssd] Entrenamiento completado | best_loss={best_loss:.4f}")
        return {"history": history, "best_loss": best_loss, "val_loader": val_loader}

    def validate(self, val_loader=None, data_yaml: str = None, iou_threshold: float = 0.5):
        """Evalua el modelo en el conjunto de validacion (mismo metodo que Faster R-CNN)."""
        if val_loader is None:
            if data_yaml is None:
                raise ValueError("Se requiere val_loader o data_yaml para validar")
            cfg = self._load_data_yaml(data_yaml)
            val_ds = YoloTorchvisionDataset(cfg["val_images"], cfg["val_labels"])
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                    collate_fn=collate_fn, num_workers=0)

        self.model.eval()

        all_tp = 0
        all_fp = 0
        all_fn = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)

                for out, target in zip(outputs, targets):
                    pred_boxes = out["boxes"].cpu()
                    pred_scores = out["scores"].cpu()
                    gt_boxes = target["boxes"].cpu()

                    mask = pred_scores >= 0.5
                    pred_boxes = pred_boxes[mask]

                    tp = 0
                    matched_gt = set()
                    for pb in pred_boxes:
                        best_iou = 0
                        best_idx = -1
                        for i, gb in enumerate(gt_boxes):
                            if i in matched_gt:
                                continue
                            iou = self._iou(pb, gb)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = i
                        if best_iou >= iou_threshold:
                            tp += 1
                            matched_gt.add(best_idx)

                    fp = len(pred_boxes) - tp
                    fn = len(gt_boxes) - len(matched_gt)
                    all_tp += tp
                    all_fp += fp
                    all_fn += fn

        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "mAP50": round(precision * recall, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        print(f"[ssd] Metricas: {metrics}")
        return metrics

    @staticmethod
    def _iou(box1, box2):
        """Calcula IoU entre dos cajas [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return float(inter / union) if union > 0 else 0.0

    def save(self, path: str):
        """Guarda el modelo entrenado."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        torch.save({
            "state_dict": self.model.state_dict(),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }, path)
        print(f"[ssd] Modelo guardado en: {path}")

    def load(self, path: str):
        """Carga un modelo previamente guardado."""
        ckpt = torch.load(path, map_location=self.device)
        self.num_classes = ckpt["num_classes"]
        self.class_names = ckpt.get("class_names", None)
        self.model = self._build_model(self.num_classes)
        self.model.load_state_dict(ckpt["state_dict"])
        print(f"[ssd] Modelo cargado desde: {path}")
        return self