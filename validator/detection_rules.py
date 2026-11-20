#!/usr/bin/env python3
"""Rule Validator — Detection & File.

Kiểm tra 5 loại lỗi luật phía File & Detection chỉ dùng thư viện chuẩn của Python:
1. missing_image      : manifest có ảnh nhưng thiếu file trên đĩa.
2. wrong_hash         : sha256 thực tế của file khác với sha256 ghi trong manifest.json.
3. invalid_class      : class_id trong nhãn YOLO nằm ngoài tập lớp hợp lệ (0..79).
4. bbox_out_of_bounds : bounding box YOLO tràn ra ngoài biên ảnh [0, 1] (hoặc w, h <= 0).
5. duplicate_box      : hai box cùng lớp, cùng kích thước, tâm chỉ lệch 1–2 pixel.

Cách tích hợp trong `validator/validate.py`:
    from validator.detection_rules import validate_detection_rules
    errors.extend(validate_detection_rules(delivery_dir))

Chạy độc lập (CLI):
    python validator/detection_rules.py <thư_mục_delivery> --out <errors.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

# Dung sai cho sai số làm tròn .6f khi xuất nhãn YOLO ở sát mép ảnh (~5e-7).
# Lỗi tràn biên thật nhỏ nhất cũng cỡ >= 1 pixel (~1.5e-3 trên ảnh 640px).
BBOX_BOUNDS_EPS = 1e-4

# Ngưỡng phát hiện duplicate_box theo kích thước + độ lệch tâm (tính bằng pixel):
# Trong lab, duplicate_box giữ nguyên kích thước và dịch tâm 1–2 pixel theo mỗi trục.
DUP_MAX_SIZE_DIFF_PX = 1.5
DUP_MAX_CENTER_SHIFT_PX = 3.5
DUP_MAX_REL_SIZE_DIFF = 0.05


def sha256_file(path: Path) -> str:
    """Tính mã băm SHA-256 (hex thường) trên bytes thực tế của file."""
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def read_jpeg_size(path: Path) -> tuple[int, int] | None:
    """Đọc kích thước (width, height) từ JPEG SOF header bằng thư viện chuẩn."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    pos = 2
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    }
    while pos < len(data) - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
            continue
        if pos + 2 > len(data):
            break
        seg_len = struct.unpack(">H", data[pos : pos + 2])[0]
        if seg_len < 2 or pos + seg_len > len(data):
            break
        if marker in sof_markers and seg_len >= 7:
            height, width = struct.unpack(">HH", data[pos + 3 : pos + 7])
            if width > 0 and height > 0:
                return width, height
        pos += seg_len
    return None


def load_manifest_items(delivery_dir: Path) -> tuple[dict[str, dict], int]:
    """Đọc danh sách items và num_classes từ manifest.json."""
    manifest_path = delivery_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {manifest_path}")
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = doc.get("items", {})
    num_classes = int(doc.get("num_classes", 80))
    return items, num_classes


def load_valid_classes(delivery_dir: Path, default_num_classes: int = 80) -> set[int]:
    """Đọc tập class_id hợp lệ từ classes.txt (mặc định 0..default_num_classes-1)."""
    classes_path = delivery_dir / "classes.txt"
    if classes_path.is_file():
        valid: set[int] = set()
        for line in classes_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    valid.add(int(parts[0]))
                except ValueError:
                    continue
        if valid:
            return valid
    return set(range(default_num_classes))


def load_image_dimensions(delivery_dir: Path, manifest_items: dict[str, dict]) -> dict[str, tuple[int, int]]:
    """Lấy kích thước (width, height) của từng ảnh từ panoptic.json hoặc header JPEG."""
    dims: dict[str, tuple[int, int]] = {}
    panoptic_path = delivery_dir / "panoptic.json"
    if panoptic_path.is_file():
        try:
            doc = json.loads(panoptic_path.read_text(encoding="utf-8"))
            for im in doc.get("images", []):
                stem = Path(im.get("file_name", "")).stem
                w = int(im.get("width", 0))
                h = int(im.get("height", 0))
                if stem and w > 0 and h > 0:
                    dims[stem] = (w, h)
        except (OSError, ValueError, KeyError):
            pass

    for name, info in manifest_items.items():
        if name not in dims:
            img_rel = info.get("image", f"images/{name}.jpg")
            size = read_jpeg_size(delivery_dir / img_rel)
            if size is not None:
                dims[name] = size
            else:
                dims[name] = (640, 640)
    return dims


def parse_yolo_file(label_path: Path) -> list[list[str]]:
    """Đọc các dòng khác rỗng của file nhãn YOLO."""
    if not label_path.is_file():
        return []
    rows: list[list[str]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(stripped.split())
    return rows


def check_missing_image(delivery_dir: Path, manifest_items: dict[str, dict]) -> list[dict]:
    """1. missing_image: manifest có ảnh nhưng thiếu file trên đĩa."""
    errors: list[dict] = []
    for name, info in manifest_items.items():
        img_rel = info.get("image", f"images/{name}.jpg")
        img_path = delivery_dir / img_rel
        if not img_path.is_file():
            errors.append({"type": "missing_image", "target": name})
    return errors


def check_wrong_hash(
    delivery_dir: Path,
    manifest_items: dict[str, dict],
    missing_images: set[str] | None = None,
) -> list[dict]:
    """2. wrong_hash: sha256 thực tế của file khác với sha256 ghi trong manifest.json."""
    skip = missing_images or set()
    errors: list[dict] = []
    for name, info in manifest_items.items():
        if name in skip:
            continue
        img_rel = info.get("image", f"images/{name}.jpg")
        img_path = delivery_dir / img_rel
        if not img_path.is_file():
            continue
        expected_hash = str(info.get("image_sha256", "")).strip().lower()
        actual_hash = sha256_file(img_path)
        if not expected_hash or actual_hash != expected_hash:
            errors.append({"type": "wrong_hash", "target": name})
    return errors


def check_invalid_class(
    delivery_dir: Path,
    manifest_items: dict[str, dict],
    valid_classes: set[int],
) -> list[dict]:
    """3. invalid_class: class_id không phải số nguyên trong tập lớp hợp lệ (0..79)."""
    errors: list[dict] = []
    for name, info in manifest_items.items():
        label_rel = info.get("label", f"labels/{name}.txt")
        rows = parse_yolo_file(delivery_dir / label_rel)
        has_invalid = False
        for row in rows:
            if not row:
                continue
            try:
                cls_id = int(row[0])
            except ValueError:
                has_invalid = True
                break
            if cls_id not in valid_classes:
                has_invalid = True
                break
        if has_invalid:
            errors.append({"type": "invalid_class", "target": name})
    return errors


def is_box_out_of_bounds(x: float, y: float, w: float, h: float, eps: float = BBOX_BOUNDS_EPS) -> bool:
    """Kiểm tra một box YOLO chuẩn hoá (x_center, y_center, width, height) có tràn biên [0, 1] không."""
    if any(math.isnan(v) or math.isinf(v) for v in (x, y, w, h)):
        return True
    if w <= 0.0 or h <= 0.0:
        return True
    left = x - w / 2.0
    right = x + w / 2.0
    top = y - h / 2.0
    bottom = y + h / 2.0
    return (
        left < -eps
        or top < -eps
        or right > 1.0 + eps
        or bottom > 1.0 + eps
    )


def check_bbox_out_of_bounds(delivery_dir: Path, manifest_items: dict[str, dict]) -> list[dict]:
    """4. bbox_out_of_bounds: box vượt ra ngoài biên ảnh [0, 1]."""
    errors: list[dict] = []
    for name, info in manifest_items.items():
        label_rel = info.get("label", f"labels/{name}.txt")
        rows = parse_yolo_file(delivery_dir / label_rel)
        out_of_bounds = False
        for row in rows:
            if len(row) < 5:
                out_of_bounds = True
                break
            try:
                x, y, w, h = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            except ValueError:
                out_of_bounds = True
                break
            if is_box_out_of_bounds(x, y, w, h):
                out_of_bounds = True
                break
        if out_of_bounds:
            errors.append({"type": "bbox_out_of_bounds", "target": name})
    return errors


def is_duplicate_pair(
    box_a: tuple[str, float, float, float, float],
    box_b: tuple[str, float, float, float, float],
    img_width: int,
    img_height: int,
) -> bool:
    """Kiểm tra hai box có phải là lỗi duplicate_box (cùng lớp, cùng kích thước, lệch 1–2 px)."""
    cls_a, xa, ya, wa, ha = box_a
    cls_b, xb, yb, wb, hb = box_b
    if cls_a != cls_b:
        return False

    dw_px = abs(wa - wb) * img_width
    dh_px = abs(ha - hb) * img_height
    if dw_px > DUP_MAX_SIZE_DIFF_PX or dh_px > DUP_MAX_SIZE_DIFF_PX:
        return False

    rel_w = abs(wa - wb) / max(wa, wb, 1e-9)
    rel_h = abs(ha - hb) / max(ha, hb, 1e-9)
    if rel_w > DUP_MAX_REL_SIZE_DIFF or rel_h > DUP_MAX_REL_SIZE_DIFF:
        return False

    dx_px = abs(xa - xb) * img_width
    dy_px = abs(ya - yb) * img_height
    return dx_px <= DUP_MAX_CENTER_SHIFT_PX and dy_px <= DUP_MAX_CENTER_SHIFT_PX


def check_duplicate_box(
    delivery_dir: Path,
    manifest_items: dict[str, dict],
    image_dims: dict[str, tuple[int, int]],
) -> list[dict]:
    """5. duplicate_box: một box bị lặp lại cùng lớp, cùng kích thước, chỉ lệch 1–2 pixel."""
    errors: list[dict] = []
    for name, info in manifest_items.items():
        label_rel = info.get("label", f"labels/{name}.txt")
        rows = parse_yolo_file(delivery_dir / label_rel)
        if len(rows) < 2:
            continue
        width, height = image_dims.get(name, (640, 640))
        parsed: list[tuple[str, float, float, float, float]] = []
        for row in rows:
            if len(row) < 5:
                continue
            try:
                parsed.append((row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])))
            except ValueError:
                continue

        found_dup = False
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                if is_duplicate_pair(parsed[i], parsed[j], width, height):
                    found_dup = True
                    break
            if found_dup:
                break
        if found_dup:
            errors.append({"type": "duplicate_box", "target": name})
    return errors


def validate_detection_rules(delivery_dir: Path) -> list[dict]:
    """Chạy toàn bộ 5 luật Detection/File trên thư mục delivery."""
    manifest_items, num_classes = load_manifest_items(delivery_dir)
    valid_classes = load_valid_classes(delivery_dir, num_classes)
    image_dims = load_image_dimensions(delivery_dir, manifest_items)

    missing_errors = check_missing_image(delivery_dir, manifest_items)
    missing_targets = {e["target"] for e in missing_errors}

    hash_errors = check_wrong_hash(delivery_dir, manifest_items, missing_targets)
    class_errors = check_invalid_class(delivery_dir, manifest_items, valid_classes)
    bounds_errors = check_bbox_out_of_bounds(delivery_dir, manifest_items)
    dup_errors = check_duplicate_box(delivery_dir, manifest_items, image_dims)

    return missing_errors + hash_errors + class_errors + bounds_errors + dup_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Rule Validator — Detection & File")
    parser.add_argument("delivery", help="Đường dẫn tới thư mục delivery")
    parser.add_argument("--out", required=True, help="Đường dẫn file errors.json đầu ra")
    args = parser.parse_args()

    delivery_dir = Path(args.delivery)
    if not delivery_dir.is_dir():
        raise SystemExit(f"Thư mục delivery không tồn tại: {delivery_dir}")

    errors = validate_detection_rules(delivery_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"errors": errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Detection validator finished: {len(errors)} errors -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
