#!/bin/bash
# Download and prepare COCO 2017 for object detection.
# Run on EC2: bash prepare_coco.sh
#
# Structure after setup:
#   /home/ubuntu/data/coco/
#   ├── images/
#   │   ├── train2017/  (118K images, ~19GB)
#   │   └── val2017/    (5K images, ~1GB)
#   ├── labels/
#   │   ├── train2017/
#   │   └── val2017/
#   └── coco.yaml       (ultralytics format)

set -eu
DATA_DIR="/home/ubuntu/data/coco"
mkdir -p "$DATA_DIR/images" "$DATA_DIR/labels"
cd "$DATA_DIR"

echo "=== Downloading COCO 2017 ==="

# Images
if [ ! -d images/train2017 ]; then
    echo "Downloading train2017 images (~19GB)..."
    wget -q --show-progress "http://images.cocodataset.org/zips/train2017.zip" -O train2017.zip
    unzip -q train2017.zip -d images/
    rm train2017.zip
fi

if [ ! -d images/val2017 ]; then
    echo "Downloading val2017 images (~1GB)..."
    wget -q --show-progress "http://images.cocodataset.org/zips/val2017.zip" -O val2017.zip
    unzip -q val2017.zip -d images/
    rm val2017.zip
fi

# Annotations
if [ ! -f annotations/instances_train2017.json ]; then
    echo "Downloading annotations..."
    wget -q --show-progress "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -O annotations.zip
    unzip -q annotations.zip
    rm annotations.zip
fi

# Convert COCO annotations to YOLO format
echo "=== Converting to YOLO format ==="
python3 << 'PY'
import json
from pathlib import Path

def coco_to_yolo(coco_json, output_dir, images_dir):
    """Convert COCO JSON annotations to YOLO txt format."""
    with open(coco_json) as f:
        data = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build image id -> (w, h, filename) map
    img_info = {}
    for img in data["images"]:
        img_info[img["id"]] = (img["width"], img["height"], img["file_name"])

    # Group annotations by image
    img_anns = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    # COCO category id -> 0-indexed class id
    cat_ids = sorted(set(ann["category_id"] for ann in data["annotations"]))
    cat_map = {cid: idx for idx, cid in enumerate(cat_ids)}

    count = 0
    for img_id, (w, h, fname) in img_info.items():
        txt_name = Path(fname).stem + ".txt"
        txt_path = output_dir / txt_name

        lines = []
        for ann in img_anns.get(img_id, []):
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]  # COCO: x, y, w, h (top-left)
            # Convert to YOLO: cx, cy, w, h (center, normalized)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cls = cat_map[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        txt_path.write_text("\n".join(lines))
        count += 1

    print(f"  Converted {count} images, {len(cat_map)} classes")
    return len(cat_map)

data_dir = Path("/home/ubuntu/data/coco")

n_classes = coco_to_yolo(
    data_dir / "annotations/instances_train2017.json",
    data_dir / "labels/train2017",
    data_dir / "images/train2017",
)
coco_to_yolo(
    data_dir / "annotations/instances_val2017.json",
    data_dir / "labels/val2017",
    data_dir / "images/val2017",
)

# Create YOLO dataset yaml
yaml_content = f"""# COCO 2017 Object Detection
path: /home/ubuntu/data/coco
train: images/train2017
val: images/val2017

nc: {n_classes}
names: {list(range(n_classes))}
"""
(data_dir / "coco.yaml").write_text(yaml_content)
print(f"  Created coco.yaml ({n_classes} classes)")
PY

echo "=== Installing ultralytics ==="
pip install -q ultralytics

echo "=== Done! ==="
echo "  Images: $(ls images/train2017/ | wc -l) train, $(ls images/val2017/ | wc -l) val"
echo "  Labels: $(ls labels/train2017/ | wc -l) train, $(ls labels/val2017/ | wc -l) val"
