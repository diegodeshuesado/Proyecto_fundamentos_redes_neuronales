"""
Script auxiliar para preparar datasets en formato YOLO.
Toma una carpeta con images/ y labels/ planos, y los divide en train/val.
Genera el data.yaml requerido por ultralytics.
Se ejecuta UNA SOLA VEZ antes de entrenar.
"""
import argparse
import random
import shutil
from pathlib import Path
import yaml


def prepare_yolo_dataset(source_dir: str, output_dir: str,
                          val_split: float = 0.2, seed: int = 42):
    """
    Args:
        source_dir: carpeta con images/, labels/, classes.txt
        output_dir: carpeta destino con la estructura YOLO completa
        val_split: porcentaje para validación (default 20%)
        seed: para reproducibilidad del split
    """
    source = Path(source_dir)
    output = Path(output_dir)

    # Validar estructura fuente
    images_src = source / "images"
    labels_src = source / "labels"
    classes_file = source / "classes.txt"

    if not images_src.exists() or not labels_src.exists():
        raise FileNotFoundError(f"Faltan carpetas 'images/' o 'labels/' en {source}")
    if not classes_file.exists():
        raise FileNotFoundError(f"Falta 'classes.txt' en {source}")

    # Leer clases
    with open(classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"[prepare] Clases encontradas: {class_names}")

    # Listar imágenes válidas (que tengan su label correspondiente)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in images_src.iterdir() if p.suffix.lower() in image_extensions]
    valid_pairs = []
    for img in images:
        label = labels_src / f"{img.stem}.txt"
        if label.exists():
            valid_pairs.append((img, label))
        else:
            print(f"[prepare] AVISO: imagen sin label, se omite: {img.name}")

    print(f"[prepare] Imágenes válidas: {len(valid_pairs)}")

    # Shuffle + split
    random.seed(seed)
    random.shuffle(valid_pairs)
    n_val = int(len(valid_pairs) * val_split)
    val_pairs = valid_pairs[:n_val]
    train_pairs = valid_pairs[n_val:]
    print(f"[prepare] Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Crear estructura destino
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copiar archivos
    def copy_pairs(pairs, split):
        for img, lbl in pairs:
            shutil.copy2(img, output / "images" / split / img.name)
            shutil.copy2(lbl, output / "labels" / split / lbl.name)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    print(f"[prepare] Archivos copiados a {output}")

    # Generar data.yaml
    data_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = output / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, default_flow_style=False)
    print(f"[prepare] data.yaml generado: {yaml_path}")
    print(f"[prepare] ✓ Dataset listo para YOLO")

    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte dataset plano a estructura YOLO train/val")
    parser.add_argument("--source", type=str, required=True,
                        help="Carpeta con images/, labels/, classes.txt")
    parser.add_argument("--output", type=str, required=True,
                        help="Carpeta destino con estructura YOLO")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Porcentaje para validación (default 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_yolo_dataset(args.source, args.output, args.val_split, args.seed)