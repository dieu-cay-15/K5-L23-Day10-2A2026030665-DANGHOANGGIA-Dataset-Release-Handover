import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_DIR = ROOT / "validator"
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(VALIDATOR_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import detection_rules  # noqa: E402
import metrics  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _create_synthetic_delivery(
    base_dir: Path,
    items_spec: list[dict],
    num_classes: int = 80,
) -> Path:
    """Tạo một thư mục delivery giả lập để kiểm thử đơn vị 5 luật Detection."""
    delivery = base_dir / "delivery"
    (delivery / "images").mkdir(parents=True, exist_ok=True)
    (delivery / "labels").mkdir(parents=True, exist_ok=True)

    classes_text = "".join(f"{i} class_{i}\n" for i in range(num_classes))
    (delivery / "classes.txt").write_text(classes_text, encoding="utf-8")

    manifest_items = {}
    panoptic_images = []

    for idx, spec in enumerate(items_spec, start=1):
        name = spec["name"]
        width = spec.get("width", 640)
        height = spec.get("height", 480)
        img_bytes = spec.get("img_bytes", f"fake_jpeg_bytes_{name}".encode("utf-8"))
        boxes = spec.get("boxes", ["0 0.500000 0.500000 0.200000 0.200000"])
        missing = spec.get("missing", False)
        override_hash = spec.get("override_hash", None)

        jpg_rel = f"images/{name}.jpg"
        lbl_rel = f"labels/{name}.txt"

        if not missing:
            (delivery / jpg_rel).write_bytes(img_bytes)

        lbl_bytes = ("\n".join(boxes) + ("\n" if boxes else "")).encode("utf-8")
        (delivery / lbl_rel).write_bytes(lbl_bytes)

        manifest_items[name] = {
            "image": jpg_rel,
            "label": lbl_rel,
            "mask": f"masks/{name}.png",
            "image_sha256": override_hash if override_hash is not None else _sha256(img_bytes),
            "label_sha256": _sha256(lbl_bytes),
            "mask_sha256": _sha256(b"dummy_mask"),
        }
        panoptic_images.append({
            "id": idx,
            "file_name": f"{name}.jpg",
            "width": width,
            "height": height,
        })

    (delivery / "manifest.json").write_text(
        json.dumps({"dataset": "lab250", "num_classes": num_classes, "items": manifest_items}, indent=2),
        encoding="utf-8",
    )
    (delivery / "panoptic.json").write_text(
        json.dumps({"categories": [], "images": panoptic_images, "annotations": []}),
        encoding="utf-8",
    )
    return delivery


class DetectionRulesTest(unittest.TestCase):
    """Kiểm thử 5 luật Detection/File."""

    def test_clean_delivery_including_edge_rounding_and_natural_overlap(self):
        """Lô sạch (kể cả box chạm mép bị làm tròn .6f và cặp box tự nhiên chồng nhau) không báo lỗi."""
        with tempfile.TemporaryDirectory() as tmp:
            # Box chạm mép phải có sai số làm tròn .6f: cx + w/2 = 0.938272 + 0.123457/2 = 1.0000005
            # Cặp box cùng lớp chồng nhau tự nhiên (IoU ~ 0.65) nhưng khác kích thước và lệch tâm > 10px
            delivery = _create_synthetic_delivery(
                Path(tmp),
                [
                    {
                        "name": "img_clean_01",
                        "width": 640,
                        "height": 480,
                        "boxes": [
                            "0 0.938272 0.500000 0.123457 0.200000",
                            "79 0.061728 0.500000 0.123457 0.200000",
                        ],
                    },
                    {
                        "name": "img_clean_02",
                        "width": 640,
                        "height": 480,
                        "boxes": [
                            "2 0.500000 0.500000 0.300000 0.300000",
                            "2 0.520000 0.520000 0.260000 0.250000",
                        ],
                    },
                ],
            )
            errors = detection_rules.validate_detection_rules(delivery)
            self.assertEqual(errors, [])

    def test_missing_image_detected_without_false_wrong_hash(self):
        """Kiểm thử missing_image: thiếu file ảnh trong images/, không báo trùng wrong_hash."""
        with tempfile.TemporaryDirectory() as tmp:
            delivery = _create_synthetic_delivery(
                Path(tmp),
                [
                    {"name": "img_ok", "missing": False},
                    {"name": "img_missing", "missing": True},
                ],
            )
            errors = detection_rules.validate_detection_rules(delivery)
            self.assertEqual(errors, [{"type": "missing_image", "target": "img_missing"}])

    def test_wrong_hash_detected_against_valid_format_fake_hash(self):
        """Kiểm thử wrong_hash: mã sha256 đúng định dạng 64 ký tự hex nhưng là của ảnh khác."""
        with tempfile.TemporaryDirectory() as tmp:
            fake_hash = _sha256(b"other_image_content")
            delivery = _create_synthetic_delivery(
                Path(tmp),
                [
                    {"name": "img_ok"},
                    {"name": "img_bad_hash", "override_hash": fake_hash},
                ],
            )
            errors = detection_rules.validate_detection_rules(delivery)
            self.assertEqual(errors, [{"type": "wrong_hash", "target": "img_bad_hash"}])

    def test_invalid_class_detected(self):
        """Kiểm thử invalid_class: class_id = 80, class_id âm (-1), hoặc không phải số nguyên."""
        with tempfile.TemporaryDirectory() as tmp:
            delivery = _create_synthetic_delivery(
                Path(tmp),
                [
                    {"name": "img_valid", "boxes": ["0 0.5 0.5 0.2 0.2", "79 0.3 0.3 0.1 0.1"]},
                    {"name": "img_cls_80", "boxes": ["80 0.5 0.5 0.2 0.2"]},
                    {"name": "img_cls_neg", "boxes": ["-1 0.5 0.5 0.2 0.2"]},
                    {"name": "img_cls_nan", "boxes": ["car 0.5 0.5 0.2 0.2"]},
                ],
            )
            errors = detection_rules.validate_detection_rules(delivery)
            self.assertEqual(
                metrics.pairs(errors),
                {
                    ("invalid_class", "img_cls_80"),
                    ("invalid_class", "img_cls_neg"),
                    ("invalid_class", "img_cls_nan"),
                },
            )

    def test_bbox_out_of_bounds_detected(self):
        """Kiểm thử bbox_out_of_bounds: tràn nhẹ mép phải/trái/trên/dưới hoặc kích thước <= 0."""
        with tempfile.TemporaryDirectory() as tmp:
            delivery = _create_synthetic_delivery(
                Path(tmp),
                [
                    {"name": "img_in_bounds", "boxes": ["1 0.500000 0.500000 0.400000 0.400000"]},
                    {"name": "img_over_right", "boxes": ["1 0.900000 0.500000 0.260000 0.200000"]},
                    {"name": "img_over_left", "boxes": ["1 0.050000 0.500000 0.150000 0.200000"]},
                    {"name": "img_over_bottom", "boxes": ["1 0.500000 0.950000 0.200000 0.140000"]},
                    {"name": "img_zero_w", "boxes": ["1 0.500000 0.500000 0.000000 0.200000"]},
                ],
            )
            errors = detection_rules.validate_detection_rules(delivery)
            self.assertEqual(
                metrics.pairs(errors),
                {
                    ("bbox_out_of_bounds", "img_over_right"),
                    ("bbox_out_of_bounds", "img_over_left"),
                    ("bbox_out_of_bounds", "img_over_bottom"),
                    ("bbox_out_of_bounds", "img_zero_w"),
                },
            )

    def test_duplicate_box_detected_including_small_boxes(self):
        """Kiểm thử duplicate_box: cùng lớp, cùng kích thước, lệch 1-2 pixel (kể cả box nhỏ)."""
        with tempfile.TemporaryDirectory() as tmp:
            W, H = 640, 480
            x1, y1, w1, h1 = 0.500000, 0.500000, 0.200000, 0.200000
            x1_dup = x1 + 2.0 / W
            y1_dup = y1 - 1.0 / H

            x2, y2, w2, h2 = 0.300000, 0.300000, 6.0 / W, 6.0 / H
            x2_dup = x2 + 2.0 / W
            y2_dup = y2 + 2.0 / H

            delivery = _create_synthetic_delivery(
                Path(tmp),
                [
                    {
                        "name": "img_dup_normal",
                        "width": W,
                        "height": H,
                        "boxes": [
                            f"3 {x1:.6f} {y1:.6f} {w1:.6f} {h1:.6f}",
                            f"3 {x1_dup:.6f} {y1_dup:.6f} {w1:.6f} {h1:.6f}",
                        ],
                    },
                    {
                        "name": "img_dup_small",
                        "width": W,
                        "height": H,
                        "boxes": [
                            f"5 {x2:.6f} {y2:.6f} {w2:.6f} {h2:.6f}",
                            f"5 {x2_dup:.6f} {y2_dup:.6f} {w2:.6f} {h2:.6f}",
                        ],
                    },
                    {
                        "name": "img_diff_class_same_pos",
                        "width": W,
                        "height": H,
                        "boxes": [
                            f"3 {x1:.6f} {y1:.6f} {w1:.6f} {h1:.6f}",
                            f"4 {x1_dup:.6f} {y1_dup:.6f} {w1:.6f} {h1:.6f}",
                        ],
                    },
                ],
            )
            errors = detection_rules.validate_detection_rules(delivery)
            self.assertEqual(
                metrics.pairs(errors),
                {
                    ("duplicate_box", "img_dup_normal"),
                    ("duplicate_box", "img_dup_small"),
                },
            )

    def test_detection_rules_cli(self):
        """Kiểm thử giao diện dòng lệnh validator/detection_rules.py <delivery> --out <errors.json>."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            delivery = _create_synthetic_delivery(
                tmp_path,
                [
                    {"name": "img_001", "missing": True},
                    {"name": "img_002", "override_hash": "0" * 64},
                    {"name": "img_003", "boxes": ["80 0.5 0.5 0.2 0.2"]},
                    {"name": "img_004", "boxes": ["0 0.95 0.5 0.20 0.20"]},
                    {
                        "name": "img_005",
                        "width": 500,
                        "height": 500,
                        "boxes": [
                            "1 0.500000 0.500000 0.100000 0.100000",
                            "1 0.502000 0.498000 0.100000 0.100000",
                        ],
                    },
                ],
            )
            out_file = tmp_path / "errors.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_DIR / "detection_rules.py"),
                    str(delivery),
                    "--out",
                    str(out_file),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(out_file.is_file())
            doc = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(
                metrics.pairs(doc["errors"]),
                {
                    ("missing_image", "img_001"),
                    ("wrong_hash", "img_002"),
                    ("invalid_class", "img_003"),
                    ("bbox_out_of_bounds", "img_004"),
                    ("duplicate_box", "img_005"),
                },
            )

    def test_multi_seed_injected_detection_faults_f1_is_1(self):
        """Mô phỏng gài lỗi ngẫu nhiên qua 10 seed theo đúng thuật toán lab và kiểm chứng F1 = 1.0."""
        kinds = ["missing_image", "wrong_hash", "invalid_class", "bbox_out_of_bounds", "duplicate_box"]
        for seed in range(100, 110):
            rng = random.Random(seed)
            specs = []
            truth = set()
            for i in range(30):
                W = rng.choice([320, 480, 640])
                H = rng.choice([240, 360, 480])
                cx = rng.uniform(0.25, 0.85)
                cy = rng.uniform(0.25, 0.75)
                bw = rng.uniform(0.05, 0.25)
                bh = rng.uniform(0.05, 0.25)
                cls_id = rng.randrange(0, 80)
                specs.append({
                    "name": f"img_{i:04d}",
                    "width": W,
                    "height": H,
                    "boxes": [f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"],
                })

            chosen = rng.sample(specs, 10)
            for idx, sp in enumerate(chosen):
                kind = kinds[idx % len(kinds)]
                truth.add((kind, sp["name"]))
                if kind == "missing_image":
                    sp["missing"] = True
                elif kind == "wrong_hash":
                    sp["override_hash"] = _sha256(f"fake_other_{seed}_{idx}".encode("utf-8"))
                elif kind == "invalid_class":
                    parts = sp["boxes"][0].split()
                    parts[0] = "80"
                    sp["boxes"][0] = " ".join(parts)
                elif kind == "bbox_out_of_bounds":
                    parts = sp["boxes"][0].split()
                    x, w = float(parts[1]), float(parts[3])
                    over = rng.uniform(0.02, 0.08)
                    right = 1.0 + over
                    left = x - w / 2
                    parts[1], parts[3] = f"{(left + right) / 2:.6f}", f"{right - left:.6f}"
                    sp["boxes"][0] = " ".join(parts)
                elif kind == "duplicate_box":
                    parts = sp["boxes"][0].split()
                    x, y, w, h = map(float, parts[1:5])
                    x = min(1 - w / 2, max(w / 2, x + rng.choice([-2, -1, 1, 2]) / sp["width"]))
                    y = min(1 - h / 2, max(h / 2, y + rng.choice([-2, -1, 1, 2]) / sp["height"]))
                    sp["boxes"].append(f"{parts[0]} {x:.6f} {y:.6f} {parts[3]} {parts[4]}")

            with tempfile.TemporaryDirectory() as tmp:
                delivery = _create_synthetic_delivery(Path(tmp), specs)
                pred = metrics.pairs(detection_rules.validate_detection_rules(delivery))
                res = metrics.score(truth, pred)
                self.assertEqual(res["f1"], 1.0, f"Seed {seed} failed: {res}")


if __name__ == "__main__":
    unittest.main()
