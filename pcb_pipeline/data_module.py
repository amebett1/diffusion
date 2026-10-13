"""
Data Preprocessing Module for PCB Defect Detection & Synthesis
==============================================================
Triển khai khâu tiền xử lý dữ liệu theo chuẩn bài báo:
"A Synthetic Defect Image Generation Method for Enhanced Industrial
Defect Detection Based on Inpainting Diffusion and Context-aware Mask Generation"

Các chức năng chính:
1. Board Grouping: Tách ID sản phẩm (10 loại bo mạch PCB).
2. Product-wise Split (ID vs NPI):
   - ID setup: 85% train, 15% val trên các sản phẩm đã biết (seen products).
   - NPI setup: Train trên seen products, Val trên unseen products (3 loại bo mạch mới).
3. Low-data Subsampling (10%): Tái lập kịch bản dữ liệu khan hiếm theo bài báo.
4. Normal Images Generation (799 ảnh): Tạo 799 ảnh phôi sạch 600x600 đưa vào data/pcb_normal/.
5. BBox Statistics (Q1, Q3): Phân tích phân vị kích thước bounding box phục vụ Step 2.
6. YAML & Metadata Export: Xuất YOLO dataset config và JSON metadata cho CAMG và Diffusion.
"""

import os
import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 6 lớp lỗi PCB chuẩn theo paper (DeepPCB)
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


class PCBDataModule:
    """
    Module quản lý và tiền xử lý toàn bộ dữ liệu cho pipeline:
    - Parser dữ liệu DeepPCB
    - Phân chia ID và NPI
    - Trích xuất 10% low-data
    - Sinh 799 ảnh normal sạch
    - Tính phân phối Q1/Q3 bbox
    """

    def __init__(
        self,
        raw_dataset_dir: str | Path = "data/pcb-defect-dataset",
        output_dir: str | Path = "data",
        image_size: int = 600,
        random_seed: int = 42,
    ):
        self.raw_dir = Path(raw_dataset_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.seed = random_seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def extract_board_id(filename: str) -> str:
        """Trích xuất mã bo mạch từ tên file (ví dụ 'light_01' -> '01')."""
        match = re.search(r"light_(\d+)", filename)
        if match:
            return match.group(1)
        return "unknown"

    def scan_dataset(self) -> List[Dict]:
        """
        Quét toàn bộ ảnh và nhãn trong raw dataset, đọc thông tin bounding box.
        Trả về danh sách các sample record.
        """
        records = []
        for split in ["train", "val", "test"]:
            img_dir = self.raw_dir / split / "images"
            lbl_dir = self.raw_dir / split / "labels"
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                stem = img_path.stem
                lbl_path = lbl_dir / f"{stem}.txt"
                # Hỗ trợ fallback nếu còn sót đuôi _256
                if not lbl_path.exists():
                    alt_path = lbl_dir / f"{stem.replace('_600', '_256')}.txt"
                    if alt_path.exists():
                        lbl_path = alt_path

                board_id = self.extract_board_id(img_path.name)
                boxes = []
                if lbl_path.exists():
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                cx, cy, w, h = [float(x) for x in parts[1:5]]
                                boxes.append({
                                    "class_id": cls_id,
                                    "class_name": CLASS_ID_TO_NAME.get(cls_id, f"class_{cls_id}"),
                                    "cx": cx,
                                    "cy": cy,
                                    "w": w,
                                    "h": h,
                                    # Tọa độ pixel (x1, y1, x2, y2)
                                    "px_w": w * self.image_size,
                                    "px_h": h * self.image_size,
                                    "x1": max(0, int((cx - w / 2) * self.image_size)),
                                    "y1": max(0, int((cy - h / 2) * self.image_size)),
                                    "x2": min(self.image_size, int((cx + w / 2) * self.image_size)),
                                    "y2": min(self.image_size, int((cy + h / 2) * self.image_size)),
                                })

                records.append({
                    "original_split": split,
                    "filename": img_path.name,
                    "image_path": str(img_path.resolve()),
                    "label_path": str(lbl_path.resolve()) if lbl_path.exists() else None,
                    "board_id": board_id,
                    "boxes": boxes,
                    "num_boxes": len(boxes),
                })

        logger.info(f"Đã quét tổng cộng {len(records)} ảnh trong raw dataset.")
        return records

    def compute_bbox_statistics(self, records: List[Dict]) -> Dict:
        """
        Tính toán phân phối kích thước bbox (Q1, median, Q3, mean, min, max)
        theo chuẩn công thức bài báo để cung cấp cho Step 2 (Loss L_rec).
        """
        ws_px, hs_px = [], []
        ws_norm, hs_norm = [], []
        class_counts = Counter()

        for rec in records:
            for box in rec["boxes"]:
                ws_px.append(box["px_w"])
                hs_px.append(box["px_h"])
                ws_norm.append(box["w"])
                hs_norm.append(box["h"])
                class_counts[box["class_name"]] += 1

        stats = {
            "total_instances": len(ws_px),
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
                    "q3": float(np.percentile(ws_norm, 75)),
                },
                "height": {
                    "q1": float(np.percentile(hs_norm, 25)),
                    "q3": float(np.percentile(hs_norm, 75)),
                },
            },
        }

        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Đã lưu thống kê kích thước BBox vào {stats_path}")
        return stats

    def split_id_and_npi(
        self,
        records: List[Dict],
        unseen_boards: Optional[List[str]] = None,
        low_data_ratio: float = 0.1,
    ) -> Dict[str, Dict]:
        """
        Phân tách dữ liệu theo 2 kịch bản của bài báo:
        1. In-Distribution (ID):
           - Bỏ các unseen products.
           - Lấy 85% train, 15% validation từ các retained images.
        2. New Product Introduction (NPI):
           - Train set: Tương đương ID (chỉ dùng seen products).
           - Val set: 100% ảnh thuộc unseen products.
        """
        all_boards = sorted(list({r["board_id"] for r in records}))
        logger.info(f"Các loại bo mạch tìm thấy: {all_boards}")

        if unseen_boards is None:
            # Chọn 3 bo mạch làm unseen products (chiếm ~30% như trong paper 9 seen / 3 unseen)
            unseen_boards = ["10", "11", "12"]

        seen_boards = [b for b in all_boards if b not in unseen_boards]
        logger.info(f"Seen boards ({len(seen_boards)}): {seen_boards}")
        logger.info(f"Unseen boards ({len(unseen_boards)}): {unseen_boards}")

        # Tách records theo seen vs unseen
        seen_records = [r for r in records if r["board_id"] in seen_boards]
        unseen_records = [r for r in records if r["board_id"] in unseen_boards]

        # Xáo trộn có kiểm soát
        rng = random.Random(self.seed)
        shuffled_seen = seen_records.copy()
        rng.shuffle(shuffled_seen)

        # ID split: 85% train, 15% val
        n_id_train = int(len(shuffled_seen) * 0.85)
        id_train_full = shuffled_seen[:n_id_train]
        id_val = shuffled_seen[n_id_train:]

        # Lấy 10% Low-data train theo bài báo ("used only 10% of training images")
        n_low_data = max(1, int(len(id_train_full) * low_data_ratio))
        id_train_low = id_train_full[:n_low_data]

        # NPI split: Train = id_train_low (hoặc id_train_full), Val = unseen_records
        splits = {
            "id": {
                "train_full": id_train_full,
                "train_low": id_train_low,
                "val": id_val,
                "seen_boards": seen_boards,
            },
            "npi": {
                "train_low": id_train_low,
                "train_full": id_train_full,
                "val_unseen": unseen_records,
                "unseen_boards": unseen_boards,
                "seen_boards": seen_boards,
            },
        }

        logger.info(
            f"ID Split: Train Full={len(id_train_full)}, Train Low (10%)={len(id_train_low)}, Val={len(id_val)}"
        )
        logger.info(
            f"NPI Split: Train={len(id_train_low)}, Unseen Test Val={len(unseen_records)}"
        )
        return splits

    def generate_normal_images(
        self,
        records: List[Dict],
        target_count: int = 799,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Tạo đúng 799 ảnh không lỗi (non-defective normal images) 600x600:
        Theo bài báo: "we added 799 non-defective images to use for the generation process
        by cropping non-defective areas in the product."

        Thuật toán:
        - Chọn 799 ảnh đại diện từ các bo mạch (ưu tiên unseen boards cho NPI và seen boards cho ID).
        - Sử dụng thuật toán Inpainting cục bộ (Telea) để xóa sạch các bounding box lỗi nhỏ (~30px, <0.5% diện tích),
          tái tạo nền bo mạch hoàn toàn sạch, giữ nguyên độ phân giải và đặc trưng bề mặt vật liệu 600x600.
        """
        if output_dir is None:
            output_dir = self.output_dir / "pcb_normal"
        output_dir.mkdir(parents=True, exist_ok=True)

        existing_normals = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        if len(existing_normals) >= target_count:
            logger.info(f"Đã có sẵn {len(existing_normals)} ảnh normal trong {output_dir}. Bỏ qua bước sinh.")
            return existing_normals[:target_count]

        logger.info(f"Bắt đầu trích xuất/sinh {target_count} ảnh normal vào {output_dir}...")

        # Chọn mẫu đồng đều từ các bo mạch
        boards_map = defaultdict(list)
        for r in records:
            boards_map[r["board_id"]].append(r)

        selected_records = []
        keys = list(boards_map.keys())
        idx = 0
        while len(selected_records) < target_count:
            b_key = keys[idx % len(keys)]
            if boards_map[b_key]:
                selected_records.append(boards_map[b_key].pop(0))
            idx += 1

        generated_paths = []
        for i, rec in enumerate(selected_records):
            img = cv2.imread(rec["image_path"])
            if img is None:
                continue

            H, W, _ = img.shape
            mask = np.zeros((H, W), dtype=np.uint8)

            # Đánh dấu các vùng lỗi cần xóa
            for box in rec["boxes"]:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                # Mở rộng 2 pixel biên để xóa triệt để vết lỗi
                x1 = max(0, x1 - 2)
                y1 = max(0, y1 - 2)
                x2 = min(W, x2 + 2)
                y2 = min(H, y2 + 2)
                mask[y1:y2, x1:x2] = 255

            if np.sum(mask > 0) > 0:
                # Xóa lỗi, phục hồi bề mặt bo mạch sạch
                normal_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            else:
                normal_img = img

            out_filename = f"normal_{rec['board_id']}_{i:04d}.jpg"
            out_path = output_dir / out_filename
            cv2.imwrite(str(out_path), normal_img)
            generated_paths.append(out_path)

            if len(generated_paths) >= target_count:
                break

        logger.info(f"✅ Đã tạo thành công {len(generated_paths)} ảnh normal vào {output_dir}")
        return generated_paths

    def export_splits_to_disk(self, splits: Dict[str, Dict]) -> Dict[str, str]:
        """
        Ghi danh sách ảnh và cấu hình YOLO YAML cho cả 2 kịch bản ID và NPI
        để phục vụ huấn luyện downstream detector (YOLOv8, YOLOv9, YOLO11, RT-DETR).
        """
        splits_dir = self.output_dir / "prepared_splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        yaml_paths = {}

        # 1. Kịch bản In-Distribution (ID)
        id_train_txt = splits_dir / "id_train_low.txt"
        id_val_txt = splits_dir / "id_val.txt"

        with open(id_train_txt, "w", encoding="utf-8") as f:
            for r in splits["id"]["train_low"]:
                f.write(f"{r['image_path']}\n")

        with open(id_val_txt, "w", encoding="utf-8") as f:
            for r in splits["id"]["val"]:
                f.write(f"{r['image_path']}\n")

        id_yaml_content = f"""# In-Distribution (ID) Dataset Config (10% Low-Data)
train: {str(id_train_txt.resolve()).replace('\\', '/')}
val: {str(id_val_txt.resolve()).replace('\\', '/')}

nc: {len(PCB_CLASSES)}
names: {PCB_CLASSES}
"""
        id_yaml_path = self.output_dir / "id_setup.yaml"
        with open(id_yaml_path, "w", encoding="utf-8") as f:
            f.write(id_yaml_content)
        yaml_paths["id_yaml"] = str(id_yaml_path)

        # 2. Kịch bản New Product Introduction (NPI)
        npi_train_txt = splits_dir / "npi_train_low.txt"
        npi_val_unseen_txt = splits_dir / "npi_val_unseen.txt"

        with open(npi_train_txt, "w", encoding="utf-8") as f:
            for r in splits["npi"]["train_low"]:
                f.write(f"{r['image_path']}\n")

        with open(npi_val_unseen_txt, "w", encoding="utf-8") as f:
            for r in splits["npi"]["val_unseen"]:
                f.write(f"{r['image_path']}\n")

        npi_yaml_content = f"""# New Product Introduction (NPI) Dataset Config
# Train trên Seen Products, Val trên 100% Unseen Products
train: {str(npi_train_txt.resolve()).replace('\\', '/')}
val: {str(npi_val_unseen_txt.resolve()).replace('\\', '/')}

nc: {len(PCB_CLASSES)}
names: {PCB_CLASSES}
"""
        npi_yaml_path = self.output_dir / "npi_setup.yaml"
        with open(npi_yaml_path, "w", encoding="utf-8") as f:
            f.write(npi_yaml_content)
        yaml_paths["npi_yaml"] = str(npi_yaml_path)

        # 3. Ghi file metadata tổng hợp (train_metadata.json cho CAMG & LoRA)
        camg_train_meta = []
        for r in splits["id"]["train_low"]:
            camg_train_meta.append({
                "image_path": r["image_path"],
                "label_path": r["label_path"],
                "board_id": r["board_id"],
                "boxes": r["boxes"],
            })

        meta_path = self.output_dir / "train_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(camg_train_meta, f, indent=2, ensure_ascii=False)
        yaml_paths["train_metadata"] = str(meta_path)

        logger.info(f"✅ Đã xuất cấu hình ID YAML -> {id_yaml_path}")
        logger.info(f"✅ Đã xuất cấu hình NPI YAML -> {npi_yaml_path}")
        logger.info(f"✅ Đã xuất CAMG train metadata -> {meta_path}")

        return yaml_paths

    def run_pipeline(self, target_normal_count: int = 799, low_data_ratio: float = 0.1) -> Dict:
        """Thực thi trọn vẹn toàn bộ quy trình tiền xử lý."""
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHUẨN BÀI BÁO")
        logger.info("=" * 60)

        records = self.scan_dataset()
        stats = self.compute_bbox_statistics(records)
        splits = self.split_id_and_npi(records, low_data_ratio=low_data_ratio)
        normal_paths = self.generate_normal_images(records, target_count=target_normal_count)
        yaml_paths = self.export_splits_to_disk(splits)

        summary = {
            "total_raw_images": len(records),
            "normal_images_count": len(normal_paths),
            "id_train_low_count": len(splits["id"]["train_low"]),
            "id_val_count": len(splits["id"]["val"]),
            "npi_train_count": len(splits["npi"]["train_low"]),
            "npi_unseen_val_count": len(splits["npi"]["val_unseen"]),
            "seen_boards": splits["id"]["seen_boards"],
            "unseen_boards": splits["npi"]["unseen_boards"],
            "bbox_q1_width": stats["pixel_size"]["width"]["q1"],
            "bbox_q3_width": stats["pixel_size"]["width"]["q3"],
            "bbox_q1_height": stats["pixel_size"]["height"]["q1"],
            "bbox_q3_height": stats["pixel_size"]["height"]["q3"],
            "exported_files": yaml_paths,
        }

        summary_file = self.output_dir / "data_prep_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"🎉 Hoàn tất tiền xử lý! Báo cáo chi tiết: {summary_file}")
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dm = PCBDataModule()
    dm.run_pipeline()
