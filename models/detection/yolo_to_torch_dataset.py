"""
Adaptador que lee un dataset en formato YOLO y lo entrega en formato
compatible con los modelos de deteccion de torchvision (Faster R-CNN, SSD).

YOLO usa: clase x_centro y_centro ancho alto (todos normalizados 0-1)
torchvision usa: boxes en formato PASCAL VOC (x1, y1, x2, y2) en pixeles absolutos
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class YoloTorchvisionDataset(Dataset):
    """
    Dataset de PyTorch que lee un split de un dataset YOLO
    y lo entrega en el formato que espera torchvision.

    Cada item devuelve: (imagen_tensor, target_dict)
    donde target_dict tiene:
        - boxes: tensor (N, 4) con [x1, y1, x2, y2] en pixeles
        - labels: tensor (N,) con clases (1-indexed; 0 es fondo en torchvision)
    """

    def __init__(self, images_dir: str, labels_dir: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        # Solo tomar imagenes que tienen su label
        image_exts = {".jpg", ".jpeg", ".png"}
        self.image_files = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in image_exts
            and (self.labels_dir / f"{p.stem}.txt").exists()
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No se encontraron pares imagen/label en {images_dir}")

        print(f"[dataset] Cargado: {len(self.image_files)} imagenes desde {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Cargar imagen y convertir a tensor [C, H, W] con valores 0-1
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        img_tensor = torch.from_numpy(
            __import__("numpy").array(img)
        ).permute(2, 0, 1).float() / 255.0

        # Leer etiquetas YOLO y convertir a formato torchvision
        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = parts
                cls = int(cls)
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)

                # Convertir YOLO (normalizado, centro+dim) -> VOC (pixeles, esquinas)
                x1 = (xc - w / 2) * W
                y1 = (yc - h / 2) * H
                x2 = (xc + w / 2) * W
                y2 = (yc + h / 2) * H

                boxes.append([x1, y1, x2, y2])
                # torchvision reserva la clase 0 para fondo, sumamos 1
                labels.append(cls + 1)

        # Si no hubo cajas, dejamos tensores vacios con las shapes correctas
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    """
    Collate function especial para deteccion.
    Torchvision espera listas (no tensores apilados) porque cada imagen
    puede tener un numero distinto de cajas.
    """
    return tuple(zip(*batch))