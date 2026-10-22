"""
Data Preprocessing Module for PCB Defect Detection & Synthesis
==============================================================
Tiền xử lý dữ liệu theo protocol của bài báo, nhưng chỉ dùng dữ liệu
thực sự có trong repository và không để rò rỉ split.

Các nguyên tắc bắt buộc
-----------------------
1. ID / NPI split:
   - Chỉ dùng official ``train`` của các bo mạch SEEN cho train/val ID.
     Không lấy ảnh official ``val``/``test`` vào bất kỳ tập train nào.
   - ID val là 15% official train của seen boards (held-out theo ảnh,
     stratified theo board).
   - Low-data là 10% của ID train full, lấy từ chính official train.
   - NPI train dùng đúng low-data seen ở trên.
   - NPI val dùng official ``val`` + ``test`` của unseen boards 10/11/12.
   - Official train của unseen boards KHÔNG được dùng để sinh dữ liệu ID.

2. Ảnh normal:
   - Chỉ lấy từ official train của seen boards.
   - Ưu tiên crop vùng sạch (không giao bbox lỗi), đủ foreground/texture.
   - Chỉ dùng Telea khi không tìm được crop sạch; sau đó kiểm tra halo/
     blur/texture và loại bỏ ảnh không đạt.
   - Ghi manifest để lần chạy sau không tái sử dụng ảnh normal cũ sai nguồn.

3. Thống kê bbox:
   - Tính trên ID train full (official train, seen boards) để cung cấp
     Q1/median/Q3 cho MaskSizePrior và DualInpaintLoss.

4. Xuất file:
   - Danh sách split và YAML dùng đường dẫn tuyệt đối được tính tại runtime,
     không hard-code /kaggle/working hay ổ đĩa cụ thể.
"""

import json
import logging
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PROTOCOL = "official_train_only_v2"
NORMAL_PROTOCOL = "clean_crop_or_verified_telea_v2"

PCB_CLASSES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(PCB_CLASSES)}
CLASS_ID_TO_NAME = {idx: name for idx, name in enumerate(PCB_CLASSES)}


def _as_posix(path: Path) -> str:
    return path.resolve().as_posix()


class PCBDataModule:
    """Quản lý split, thống kê bbox và sinh ảnh normal sạch cho pipeline."""

    def __init__(
        self,
        raw_dataset_dir: str | Path = "data/pcb-defect-dataset",
        output_dir: str | Path = "data",
        image_size: int = 600,
        random_seed: int = 42,
    ):
        self.raw_dir = Path(raw_dataset_dir)
        self.output_dir = Path(output_dir)
        self.image_size = int(image_size)
        self.seed = int(random_seed)
        self._rng = random.Random(self.seed)

    # ─────────────────────────────────────────────────────────
    # 1. Scan raw dataset
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def extract_board_id(filename: str) -> str:
        """Trích xuất mã bo mạch từ tên file (ví dụ 'light_01' -> '01')."""
        import re

        match = re.search(r"light_(\d+)", filename)
        return match.group(1) if match else "unknown"

    def scan_dataset(self) -> List[Dict]:
        """Quét toàn bộ ảnh và nhãn YOLO trong raw dataset."""
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dataset: '{self.raw_dir.resolve()}'. "
                "Hãy giải nén DeepPCB vào 'data/pcb-defect-dataset/'."
            )

        actual_raw_dir = self.raw_dir
        if not any((actual_raw_dir / s / "images").exists() for s in ["train", "val", "test"]):
            for sub in actual_raw_dir.iterdir():
                if sub.is_dir() and any(
                    (sub / s / "images").exists() for s in ["train", "val", "test"]
                ):
                    actual_raw_dir = sub
                    logger.info(f"Phát hiện dữ liệu trong thư mục con: {actual_raw_dir}")
                    break

        records: List[Dict] = []
        for split in ["train", "val", "test"]:
            img_dir = actual_raw_dir / split / "images"
            lbl_dir = actual_raw_dir / split / "labels"
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                stem = img_path.stem
                lbl_path = lbl_dir / f"{stem}.txt"
                if not lbl_path.exists():
                    alt_path = lbl_dir / f"{stem.replace('_600', '_256')}.txt"
                    if alt_path.exists():
                        lbl_path = alt_path

                board_id = self.extract_board_id(img_path.name)
                boxes: List[Dict] = []
                if lbl_path.exists():
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            cls_id = int(parts[0])
                            cx, cy, w, h = [float(v) for v in parts[1:5]]
                            boxes.append(
                                {
                                    "class_id": cls_id,
                                    "class_name": CLASS_ID_TO_NAME.get(cls_id, f"class_{cls_id}"),
                                    "cx": cx,
                                    "cy": cy,
                                    "w": w,
                                    "h": h,
                                    "px_w": w * self.image_size,
                                    "px_h": h * self.image_size,
                                    "x1": max(0, int((cx - w / 2) * self.image_size)),
                                    "y1": max(0, int((cy - h / 2) * self.image_size)),
                                    "x2": min(self.image_size, int((cx + w / 2) * self.image_size)),
                                    "y2": min(self.image_size, int((cy + h / 2) * self.image_size)),
                                }
                            )

                records.append(
                    {
                        "original_split": split,
                        "filename": img_path.name,
                        "image_path": str(img_path.resolve()),
                        "label_path": str(lbl_path.resolve()) if lbl_path.exists() else None,
                        "board_id": board_id,
                        "boxes": boxes,
                        "num_boxes": len(boxes),
                    }
                )

        if not records:
            raise FileNotFoundError(
                f"Không tìm thấy ảnh trong '{actual_raw_dir.resolve()}'. "
                "Kiểm tra cấu trúc train/images, val/images, test/images."
            )

        split_counts = Counter(r["original_split"] for r in records)
        board_counts = Counter(r["board_id"] for r in records)
        logger.info(f"Đã quét {len(records)} ảnh. Split gốc: {dict(split_counts)}")
        logger.info(f"Bo mạch: {dict(sorted(board_counts.items()))}")
        return records

    # ─────────────────────────────────────────────────────────
    # 2. Bbox statistics (official train, seen boards only)
    # ─────────────────────────────────────────────────────────
    def compute_bbox_statistics(self, records: List[Dict], source: str = "id_train_full") -> Dict:
        """Tính Q1/median/Q3 kích thước bbox trên tập được chỉ định."""
        if not records:
            raise ValueError("Không có record nào để tính thống kê bbox.")

        ws_px, hs_px = [], []
        ws_norm, hs_norm = [], []
        class_counts = Counter()

        for rec in records:
            for box in rec.get("boxes", []):
                ws_px.append(float(box["px_w"]))
                hs_px.append(float(box["px_h"]))
                ws_norm.append(float(box["w"]))
                hs_norm.append(float(box["h"]))
                class_counts[box.get("class_name", "unknown")] += 1

        if not ws_px:
            raise ValueError("Tập thống kê không có bbox lỗi nào.")

        stats = {
            "total_instances": len(ws_px),
            "source_split": source,
            "reference_image_size": self.image_size,
            "split_protocol": SPLIT_PROTOCOL,
            "class_distribution": dict(class_counts),
            "pixel_size": {
                "width": {
                    "mean": float(np.mean(ws_px)),
                    "min": float(np.min(ws_px)),
                    "q1": float(np.percentile(ws_px, 25)),
                    "median": float(np.median(ws_px)),
                    "q3": float(np.percentile(ws_px, 75)),
                    "max": float(np.max(ws_px)),
                },
                "height": {
                    "mean": float(np.mean(hs_px)),
                    "min": float(np.min(hs_px)),
                    "q1": float(np.percentile(hs_px, 25)),
                    "median": float(np.median(hs_px)),
                    "q3": float(np.percentile(hs_px, 75)),
                    "max": float(np.max(hs_px)),
                },
            },
            "normalized_size": {
                "width": {
                    "q1": float(np.percentile(ws_norm, 25)),
                    "median": float(np.median(ws_norm)),
                    "q3": float(np.percentile(ws_norm, 75)),
                },
                "height": {
                    "q1": float(np.percentile(hs_norm, 25)),
                    "median": float(np.median(hs_norm)),
                    "q3": float(np.percentile(hs_norm, 75)),
                },
            },
        }

        stats_path = self.output_dir / "dataset_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Đã lưu bbox stats ({source}) vào {stats_path}: "
            f"W Q1/median/Q3 = {stats['pixel_size']['width']['q1']:.2f}/"
            f"{stats['pixel_size']['width']['median']:.2f}/"
            f"{stats['pixel_size']['width']['q3']:.2f}px"
        )
        return stats

    # ─────────────────────────────────────────────────────────
    # 3. Split ID / NPI without leakage
    # ─────────────────────────────────────────────────────────
    def split_id_and_npi(
        self,
        records: List[Dict],
        unseen_boards: Optional[List[str]] = None,
        low_data_ratio: float = 0.1,
    ) -> Dict[str, Dict]:
        """
        Tạo split ID/NPI chỉ từ official train của seen boards.

        - ID train full: 85% official train seen.
        - ID val: 15% official train seen (held-out).
        - ID train low: 10% của ID train full.
        - NPI train: ID train low.
        - NPI val: official val + test của unseen boards.
        """
        all_boards = sorted({str(r["board_id"]) for r in records})
        logger.info(f"Các bo mạch tìm thấy: {all_boards}")

        if unseen_boards is None:
            unseen_boards = ["10", "11", "12"]
        unseen_boards = [str(b) for b in unseen_boards]
        seen_boards = [b for b in all_boards if b not in unseen_boards]

        missing_unseen = [b for b in unseen_boards if b not in all_boards]
        if missing_unseen:
            logger.warning(f"Không thấy dữ liệu cho unseen boards: {missing_unseen}")
        if not seen_boards:
            raise ValueError("Không có seen board nào sau khi loại unseen boards.")

        official_train_seen = [
            r
            for r in records
            if str(r["board_id"]) in seen_boards and r["original_split"] == "train"
        ]
        unseen_val_pool = [
            r
            for r in records
            if str(r["board_id"]) in unseen_boards
            and r["original_split"] in ("val", "test")
        ]

        if not official_train_seen:
            raise ValueError("Không có ảnh official train nào của seen boards.")
        if not unseen_val_pool:
            raise ValueError("Không có ảnh official val/test nào của unseen boards.")

        # Stratified 85/15 theo từng seen board, chỉ trên official train.
        by_board: Dict[str, List[Dict]] = defaultdict(list)
        for rec in official_train_seen:
            by_board[str(rec["board_id"])].append(rec)

        id_train_full: List[Dict] = []
        id_val: List[Dict] = []
        per_board_counts: Dict[str, Dict[str, int]] = {}

        for board in sorted(by_board.keys()):
            recs = sorted(by_board[board], key=lambda r: r["image_path"])
            self._rng.shuffle(recs)
            if len(recs) <= 1:
                n_val = 0
            else:
                n_val = max(1, int(round(len(recs) * 0.15)))
                n_val = min(n_val, len(recs) - 1)
            val_part = recs[:n_val]
            train_part = recs[n_val:]
            id_val.extend(val_part)
            id_train_full.extend(train_part)
            per_board_counts[board] = {
                "official_train": len(recs),
                "id_train_full": len(train_part),
                "id_val": len(val_part),
            }

        id_train_full.sort(key=lambda r: r["image_path"])
        id_val.sort(key=lambda r: r["image_path"])

        n_low = max(1, int(round(len(id_train_full) * float(low_data_ratio))))
        shuffled_train = list(id_train_full)
        self._rng.shuffle(shuffled_train)
        id_train_low = shuffled_train[:n_low]

        # Bảo đảm 6 lớp đều xuất hiện trong low-data nếu có thể. Nếu phải bổ sung
        # thêm ảnh, ghi rõ trong summary; đây là sai số nhỏ so với 10% mục tiêu.
        present_classes = {
            int(box["class_id"])
            for rec in id_train_low
            for box in rec.get("boxes", [])
        }
        if len(present_classes) < len(PCB_CLASSES):
            for rec in shuffled_train[n_low:]:
                rec_classes = {int(b["class_id"]) for b in rec.get("boxes", [])}
                missing = rec_classes - present_classes
                if missing:
                    id_train_low.append(rec)
                    present_classes |= rec_classes
                if len(present_classes) == len(PCB_CLASSES):
                    break

        id_train_low.sort(key=lambda r: r["image_path"])
        npi_val_unseen = sorted(unseen_val_pool, key=lambda r: r["image_path"])

        splits = {
            "id": {
                "train_full": id_train_full,
                "train_low": id_train_low,
                "val": id_val,
                "seen_boards": seen_boards,
                "unseen_boards": unseen_boards,
            },
            "npi": {
                "train_low": id_train_low,
                "train_full": id_train_full,
                "val_unseen": npi_val_unseen,
                "unseen_boards": unseen_boards,
                "seen_boards": seen_boards,
            },
            "protocol": {
                "name": SPLIT_PROTOCOL,
                "low_data_ratio": float(low_data_ratio),
                "id_train_source": "official_train_seen_boards_only",
                "id_val_source": "held_out_15pct_official_train_seen_boards",
                "npi_train_source": "same_as_id_train_low",
                "npi_val_source": "official_val_test_unseen_boards_only",
                "official_val_test_seen_used_for_training": False,
                "unseen_boards_used_for_id_synthesis": False,
            },
        }

        logger.info(
            f"ID: train_full={len(id_train_full)}, low(10%)={len(id_train_low)}, "
            f"val={len(id_val)} | NPI val unseen={len(npi_val_unseen)}"
        )
        logger.info(f"ID per-board: {per_board_counts}")
        return splits

    # ─────────────────────────────────────────────────────────
    # 4. Normal image generation: clean crop first, verified Telea fallback
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        p2 = float(np.percentile(blur, 2))
        p50 = float(np.percentile(blur, 50))
        threshold = max(10.0, min(0.45 * p50, p2 + 8.0))
        fg = (blur > threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        if fg.mean() < 0.05:
            fg = (blur > 12).astype(np.uint8)
        return fg.astype(bool)

    @staticmethod
    def _defect_mask_for_record(rec: Dict, height: int, width: int, expand: int = 4) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for box in rec.get("boxes", []):
            x1 = max(0, int(box["x1"]) - expand)
            y1 = max(0, int(box["y1"]) - expand)
            x2 = min(width, int(box["x2"]) + expand)
            y2 = min(height, int(box["y2"]) + expand)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
        return mask

    def _find_clean_crop(
        self,
        img_bgr: np.ndarray,
        defect_mask: np.ndarray,
        min_size: int = 512,
    ) -> Optional[Tuple[np.ndarray, List[int], Dict]]:
        """
        Tìm crop sạch lớn nhất, không giao defect, đủ foreground/texture.

        min_size=512 để tránh upscale quá mạnh (ví dụ 344 -> 600) làm mờ
        micro-trace. Nếu không có crop >=512, chuyển sang Telea + kiểm định.
        """
        h, w = img_bgr.shape[:2]
        fg = self._foreground_mask(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(gx, gy)

        max_size = min(self.image_size, h, w)
        sizes = []
        size = max_size
        while size >= min_size:
            sizes.append(size)
            size -= 64
        if min_size not in sizes and min_size <= max_size:
            sizes.append(min_size)

        for size in sizes:
            step = max(24, size // 8)
            best = None
            for y in range(0, h - size + 1, step):
                for x in range(0, w - size + 1, step):
                    if defect_mask[y : y + size, x : x + size].any():
                        continue
                    fg_frac = float(fg[y : y + size, x : x + size].mean())
                    if fg_frac < 0.55:
                        continue
                    edge_mean = float(edge[y : y + size, x : x + size].mean())
                    if edge_mean < 3.0:
                        continue
                    score = edge_mean + 8.0 * fg_frac
                    if best is None or score > best[0]:
                        best = (score, x, y)

            if best is not None:
                _, x, y = best
                crop = img_bgr[y : y + size, x : x + size]
                if size != self.image_size:
                    crop = cv2.resize(
                        crop,
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                meta = {
                    "crop_box": [int(x), int(y), int(x + size), int(y + size)],
                    "crop_size": int(size),
                    "resized_to": [self.image_size, self.image_size],
                }
                return crop, [int(x), int(y), int(x + size), int(y + size)], meta
        return None

    @staticmethod
    def _verify_telea_result(
        original_bgr: np.ndarray,
        normal_bgr: np.ndarray,
        defect_mask: np.ndarray,
    ) -> Tuple[bool, Dict]:
        """
        Kiểm tra halo/blur sau Telea.

        Tiêu chí:
          - màu vùng inpaint không lệch quá nhiều so với vành xung quanh;
          - texture (Laplacian variance) không bị mất hoàn toàn;
          - gradient tại biên mask không tăng đột biến (halo sắc cạnh).
        """
        mask = (defect_mask > 127).astype(np.uint8)
        if int(mask.sum()) < 20:
            return False, {"reason": "mask_quá_nhỏ_để_kiểm_định"}

        mask3 = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        ring3 = (cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1) > 0) & (mask3 == 0)
        if int(ring3.sum()) < 60:
            return False, {"reason": "ring_quá_nhỏ"}

        orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        norm_gray = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2GRAY)
        in_mask = mask3 > 0

        mean_in = normal_bgr[in_mask].mean(axis=0)
        mean_ring = normal_bgr[ring3].mean(axis=0)
        color_diff = float(np.abs(mean_in - mean_ring).max())

        std_in = float(norm_gray[in_mask].std())
        std_ring = float(norm_gray[ring3].std())
        std_ratio = std_in / max(std_ring, 1e-3)

        lap = cv2.Laplacian(norm_gray, cv2.CV_32F, ksize=3)
        lap_in = float(lap[in_mask].var())
        lap_ring = float(lap[ring3].var())
        lap_ratio = lap_in / max(lap_ring, 1e-3)

        gx = cv2.Sobel(norm_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(norm_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        boundary = (cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1) > 0) & (mask == 0)
        boundary_grad = float(grad[boundary].mean()) if boundary.any() else 0.0
        ring_grad = float(grad[ring3].mean())

        metrics = {
            "color_diff": round(color_diff, 3),
            "std_ratio": round(std_ratio, 3),
            "laplacian_ratio": round(lap_ratio, 4),
            "boundary_grad": round(boundary_grad, 3),
            "ring_grad": round(ring_grad, 3),
        }

        if color_diff > 32.0:
            return False, {**metrics, "reason": "màu_lệch_quá_32"}
        if std_ratio < 0.20 or std_ratio > 2.8:
            return False, {**metrics, "reason": "std_ratio_ngoài_[0.2,2.8]"}
        if lap_ratio < 0.12:
            return False, {**metrics, "reason": "texture_bị_mờ_(laplacian_ratio<0.12)"}
        if boundary_grad > max(2.6 * ring_grad + 10.0, 45.0):
            return False, {**metrics, "reason": "halo_biên_gradient_cao"}
        return True, metrics

    def _make_normal_image(self, rec: Dict) -> Optional[Dict]:
        """Trả về dict chứa ảnh normal + method/metadata, hoặc None nếu bị loại."""
        img = cv2.imread(rec["image_path"])
        if img is None:
            return None

        h, w = img.shape[:2]
        defect_mask = self._defect_mask_for_record(rec, h, w, expand=4)

        # 1) Ưu tiên crop vùng sạch.
        crop_result = self._find_clean_crop(img, defect_mask)
        if crop_result is not None:
            normal_img, crop_box, meta = crop_result
            return {
                "image": normal_img,
                "method": "clean_crop",
                "meta": {"crop_box": crop_box, **meta},
            }

        # 2) Telea fallback, chỉ chấp nhận nếu vượt kiểm định halo.
        if int((defect_mask > 0).sum()) == 0:
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = cv2.resize(
                    img, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4
                )
            return {"image": img, "method": "whole_image", "meta": {"crop_box": [0, 0, w, h]}}

        normal_img = cv2.inpaint(img, defect_mask, 3, cv2.INPAINT_TELEA)
        ok, metrics = self._verify_telea_result(img, normal_img, defect_mask)
        if not ok:
            logger.debug(f"Loại Telea {Path(rec['image_path']).name}: {metrics}")
            return None

        if normal_img.shape[0] != self.image_size or normal_img.shape[1] != self.image_size:
            normal_img = cv2.resize(
                normal_img,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return {
            "image": normal_img,
            "method": "telea_verified",
            "meta": {"inpaint_metrics": metrics, "crop_box": [0, 0, w, h]},
        }

    def _clean_stale_normals(self, output_dir: Path, keep_paths: set, stale_dir: Path):
        """Di chuyển các ảnh normal cũ không thuộc manifest hiện tại ra khỏi glob."""
        stale_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for pattern in ("normal_*.jpg", "normal_*.jpeg", "normal_*.png"):
            for p in output_dir.glob(pattern):
                if p.resolve() in keep_paths:
                    continue
                dst = stale_dir / p.name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(p), str(dst))
                moved += 1
        if moved:
            logger.info(f"Đã chuyển {moved} ảnh normal cũ/không hợp lệ vào {stale_dir}")

    def generate_normal_images(
        self,
        records: List[Dict],
        target_count: int = 799,
        output_dir: Optional[Path] = None,
        allowed_boards: Optional[List[str]] = None,
        allowed_splits: Tuple[str, ...] = ("train",),
    ) -> List[Path]:
        """
        Sinh ảnh normal sạch. Chỉ dùng bản ghi thuộc allowed_boards và
        allowed_splits (mặc định official train). Không lấy ảnh từ official
        val/test hoặc unseen boards vào pool sinh ảnh ID.
        """
        if output_dir is None:
            output_dir = self.output_dir / "pcb_normal"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        allowed_boards = [str(b) for b in (allowed_boards or [])]
        if not allowed_boards:
            raise ValueError("generate_normal_images cần allowed_boards (seen boards).")

        manifest_path = output_dir / "normal_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                paths = [Path(item["path"]) for item in old.get("generated", [])]
                valid = (
                    old.get("protocol") == NORMAL_PROTOCOL
                    and old.get("allowed_boards") == allowed_boards
                    and old.get("allowed_splits") == list(allowed_splits)
                    and len(paths) >= target_count
                    and all(p.exists() for p in paths[:target_count])
                )
                if valid:
                    logger.info(
                        f"Manifest normal hợp lệ: dùng {target_count} ảnh đã sinh "
                        f"từ boards {allowed_boards}, splits {list(allowed_splits)}."
                    )
                    return paths[:target_count]
                logger.info("Manifest normal cũ không hợp lệ hoặc thiếu ảnh; sinh lại.")
            except Exception as e:
                logger.warning(f"Không đọc được manifest normal cũ: {e}. Sinh lại.")

        candidates = [
            r
            for r in records
            if str(r["board_id"]) in allowed_boards and r["original_split"] in allowed_splits
        ]
        if not candidates:
            raise ValueError(
                f"Không có ảnh nguồn normal nào thuộc boards={allowed_boards}, "
                f"splits={list(allowed_splits)}."
            )

        # Round-robin theo board để tăng đa dạng sản phẩm.
        by_board: Dict[str, List[Dict]] = defaultdict(list)
        for r in candidates:
            by_board[str(r["board_id"])].append(r)
        board_order = sorted(by_board.keys())
        for b in board_order:
            by_board[b].sort(key=lambda r: r["image_path"])
            self._rng.shuffle(by_board[b])
        cursor = {b: 0 for b in board_order}

        generated: List[Dict] = []
        generated_paths: List[Path] = []
        method_counts = Counter()
        rejected_telea = 0
        rejected_other = 0

        pbar = tqdm(total=target_count, desc="Sinh ảnh phôi sạch (crop sạch / Telea verified)", unit="ảnh")
        stop = False
        while len(generated) < target_count and not stop:
            progressed = False
            for board in board_order:
                if len(generated) >= target_count:
                    break
                idx = cursor[board]
                if idx >= len(by_board[board]):
                    continue
                progressed = True
                rec = by_board[board][idx]
                cursor[board] = idx + 1

                result = self._make_normal_image(rec)
                if result is None:
                    rejected_telea += 1
                    continue

                out_name = f"normal_{board}_{len(generated):04d}.jpg"
                out_path = output_dir / out_name
                if not cv2.imwrite(str(out_path), result["image"], [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    rejected_other += 1
                    continue

                method_counts[result["method"]] += 1
                generated_paths.append(out_path)
                generated.append(
                    {
                        "path": str(out_path.resolve()),
                        "source_image": str(Path(rec["image_path"]).resolve()),
                        "board_id": board,
                        "original_split": rec["original_split"],
                        "method": result["method"],
                        "meta": result["meta"],
                    }
                )
                pbar.update(1)

            if not progressed:
                stop = True

        pbar.close()

        if len(generated) < target_count:
            logger.warning(
                f"Chỉ sinh được {len(generated)}/{target_count} ảnh normal đạt kiểm định. "
                "Không tạo ảnh giả để đủ số lượng."
            )

        keep = {p.resolve() for p in generated_paths}
        self._clean_stale_normals(output_dir, keep, output_dir / "_stale")

        manifest = {
            "protocol": NORMAL_PROTOCOL,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "allowed_boards": allowed_boards,
            "allowed_splits": list(allowed_splits),
            "target_count": int(target_count),
            "generated_count": len(generated),
            "method_counts": dict(method_counts),
            "rejected_telea_or_validation": int(rejected_telea),
            "rejected_write_or_other": int(rejected_other),
            "generated": generated,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Đã tạo {len(generated_paths)} ảnh normal | methods={dict(method_counts)} | "
            f"manifest={manifest_path}"
        )
        return generated_paths

    # ─────────────────────────────────────────────────────────
    # 5. Export splits, YAML, metadata
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _write_list(path: Path, records: List[Dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(f"{Path(rec['image_path']).resolve().as_posix()}\n")

    def export_splits_to_disk(self, splits: Dict[str, Dict]) -> Dict[str, str]:
        """
        Ghi danh sách split, YAML và metadata.

        YAML dùng đường dẫn tuyệt đối được tính tại runtime (không hard-code
        /kaggle/working), để chạy được trên local, Kaggle hoặc Colab.
        """
        splits_dir = self.output_dir / "prepared_splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        id_train_txt = splits_dir / "id_train_low.txt"
        id_train_full_txt = splits_dir / "id_train_full.txt"
        id_val_txt = splits_dir / "id_val.txt"
        npi_train_txt = splits_dir / "npi_train_low.txt"
        npi_val_txt = splits_dir / "npi_val_unseen.txt"

        self._write_list(id_train_txt, splits["id"]["train_low"])
        self._write_list(id_train_full_txt, splits["id"]["train_full"])
        self._write_list(id_val_txt, splits["id"]["val"])
        self._write_list(npi_train_txt, splits["npi"]["train_low"])
        self._write_list(npi_val_txt, splits["npi"]["val_unseen"])

        dataset_root = self.output_dir.resolve().as_posix()
        id_yaml_path = self.output_dir / "id_setup.yaml"
        id_yaml_path.write_text(
            "# In-Distribution (ID) Dataset Config\n"
            f"# split_protocol: {SPLIT_PROTOCOL}\n"
            "# Train/val chỉ lấy từ official train của seen boards.\n"
            f"path: {dataset_root}\n"
            f"train: {_as_posix(id_train_txt)}\n"
            f"val: {_as_posix(id_val_txt)}\n\n"
            f"nc: {len(PCB_CLASSES)}\n"
            f"names: {PCB_CLASSES}\n",
            encoding="utf-8",
        )

        npi_yaml_path = self.output_dir / "npi_setup.yaml"
        npi_yaml_path.write_text(
            "# New Product Introduction (NPI) Dataset Config\n"
            f"# split_protocol: {SPLIT_PROTOCOL}\n"
            "# Train trên seen boards; val trên official val+test của unseen boards.\n"
            f"path: {dataset_root}\n"
            f"train: {_as_posix(npi_train_txt)}\n"
            f"val: {_as_posix(npi_val_txt)}\n\n"
            f"nc: {len(PCB_CLASSES)}\n"
            f"names: {PCB_CLASSES}\n",
            encoding="utf-8",
        )

        camg_train_meta = []
        for rec in splits["id"]["train_low"]:
            camg_train_meta.append(
                {
                    "image_path": str(Path(rec["image_path"]).resolve()),
                    "label_path": rec.get("label_path"),
                    "board_id": rec["board_id"],
                    "original_split": rec["original_split"],
                    "split_protocol": SPLIT_PROTOCOL,
                    "boxes": rec["boxes"],
                    "num_boxes": len(rec["boxes"]),
                }
            )

        meta_path = self.output_dir / "train_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(camg_train_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Đã xuất ID YAML: {id_yaml_path}")
        logger.info(f"Đã xuất NPI YAML: {npi_yaml_path}")
        logger.info(f"Đã xuất CAMG/LoRA metadata: {meta_path}")
        return {
            "id_yaml": str(id_yaml_path),
            "npi_yaml": str(npi_yaml_path),
            "train_metadata": str(meta_path),
            "id_train_low": str(id_train_txt),
            "id_train_full": str(id_train_full_txt),
            "id_val": str(id_val_txt),
            "npi_train_low": str(npi_train_txt),
            "npi_val_unseen": str(npi_val_txt),
        }

    # ─────────────────────────────────────────────────────────
    # 6. Run all
    # ─────────────────────────────────────────────────────────
    def run_pipeline(self, target_normal_count: int = 799, low_data_ratio: float = 0.1) -> Dict:
        """Thực thi toàn bộ quy trình tiền xử lý."""
        logger.info("=" * 66)
        logger.info("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU (OFFICIAL-TRAIN-ONLY PROTOCOL)")
        logger.info("=" * 66)

        records = self.scan_dataset()
        splits = self.split_id_and_npi(records, low_data_ratio=low_data_ratio)

        # Thống kê trên ID train full, không dùng val/test.
        stats = self.compute_bbox_statistics(splits["id"]["train_full"], source="id_train_full")

        # Ảnh normal chỉ từ official train của seen boards.
        normal_paths = self.generate_normal_images(
            records,
            target_count=target_normal_count,
            allowed_boards=splits["id"]["seen_boards"],
            allowed_splits=("train",),
        )

        yaml_paths = self.export_splits_to_disk(splits)

        def counts_by_board(recs: List[Dict]) -> Dict[str, int]:
            return dict(sorted(Counter(str(r["board_id"]) for r in recs).items()))

        def counts_by_class(recs: List[Dict]) -> Dict[str, int]:
            c = Counter()
            for r in recs:
                for b in r.get("boxes", []):
                    c[b.get("class_name", "unknown")] += 1
            return dict(sorted(c.items()))

        normal_manifest_path = self.output_dir / "pcb_normal" / "normal_manifest.json"
        normal_manifest = {}
        if normal_manifest_path.exists():
            try:
                with open(normal_manifest_path, "r", encoding="utf-8") as f:
                    normal_manifest = json.load(f)
            except Exception:
                normal_manifest = {}

        summary = {
            "split_protocol": SPLIT_PROTOCOL,
            "normal_protocol": NORMAL_PROTOCOL,
            "total_raw_images": len(records),
            "raw_split_counts": dict(Counter(r["original_split"] for r in records)),
            "seen_boards": splits["id"]["seen_boards"],
            "unseen_boards": splits["npi"]["unseen_boards"],
            "id_train_full_count": len(splits["id"]["train_full"]),
            "id_train_low_count": len(splits["id"]["train_low"]),
            "id_val_count": len(splits["id"]["val"]),
            "npi_train_count": len(splits["npi"]["train_low"]),
            "npi_unseen_val_count": len(splits["npi"]["val_unseen"]),
            "id_train_low_by_board": counts_by_board(splits["id"]["train_low"]),
            "id_train_low_by_class": counts_by_class(splits["id"]["train_low"]),
            "id_val_by_board": counts_by_board(splits["id"]["val"]),
            "npi_val_by_board": counts_by_board(splits["npi"]["val_unseen"]),
            "normal_images_count": len(normal_paths),
            "normal_method_counts": normal_manifest.get("method_counts", {}),
            "normal_source_boards": normal_manifest.get("allowed_boards", []),
            "normal_source_splits": normal_manifest.get("allowed_splits", []),
            "normal_manifest": str(normal_manifest_path.resolve()) if normal_manifest_path.exists() else None,
            "bbox_q1_width": stats["pixel_size"]["width"]["q1"],
            "bbox_median_width": stats["pixel_size"]["width"]["median"],
            "bbox_q3_width": stats["pixel_size"]["width"]["q3"],
            "bbox_q1_height": stats["pixel_size"]["height"]["q1"],
            "bbox_median_height": stats["pixel_size"]["height"]["median"],
            "bbox_q3_height": stats["pixel_size"]["height"]["q3"],
            "exported_files": yaml_paths,
            "limitations": (
                "Repo chỉ có dữ liệu các board 01,04-09 (7 seen boards); "
                "paper mô tả 9 seen boards. Các board 02/03 không có trong raw dataset. "
                "Do đó số ảnh low-data và thành phần board không thể trùng khớp tuyệt đối paper."
            ),
        }

        summary_file = self.output_dir / "data_prep_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Hoàn tất tiền xử lý! Báo cáo: {summary_file}")
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dm = PCBDataModule()
    dm.run_pipeline()
