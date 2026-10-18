"""
Step 3: Quality Filtering based on Defect Detection Confidence (YOLO11)
========================================================================
Triển khai chuẩn theo Mục 3.4 của bài báo:
"A Synthetic Defect Image Generation Method for Enhanced Industrial
Defect Detection Based on Inpainting Diffusion and Context-aware Mask Generation"
(Journal of Intelligent Manufacturing, 2026)

Mục tiêu & Cơ chế kỹ thuật:
1. Huấn luyện mô hình YOLO11 (Judge Detector) trên tập ảnh lỗi thực tế (10% low-data real split).
2. Đánh giá chất lượng từng ảnh tổng hợp bằng độ tin cậy phát hiện (Detection Confidence):
   - Chấm điểm có đối chiếu mục tiêu (Target-aware IoU Matching): Kiểm tra xem YOLO11 có phát hiện
     đúng lớp khuyết tật mục tiêu (class_id) tại đúng tọa độ bounding box tổng hợp (IoU >= 0.1).
3. Quy tắc loại bỏ (Rejection Rule):
   - Loại bỏ k = 25% số lượng ảnh có confidence thấp nhất (hoặc không phát hiện đúng loại lỗi).
   - Giữ lại 75% ảnh đạt chuẩn chất lượng cao nhất (D_syn, accepted).
4. Chuẩn hóa & Xuất Dataset cho Downstream Benchmark:
   - Tự động xuất file nhãn bounding box chuẩn YOLO (.txt: class_id x_c y_c w h).
   - Tự động kết hợp ảnh lỗi thực (10% low-data) và ảnh tổng hợp accepted thành
     tập dữ liệu mở rộng D_train, aug (augmented_dataset.yaml).
5. Cơ chế Dry-run / Heuristic Scorer (No-GPU Fallback):
   - Khi chạy kiểm thử trên máy cá nhân (chưa có GPU / chưa cài ultralytics), tự động dùng
     Heuristic Scorer dựa trên năng lượng biên cạnh Sobel và độ tương phản cục bộ của vùng lỗi.
   - Khi đưa lên Google Colab, tự động kích hoạt toàn bộ sức mạnh GPU và ultralytics.
"""

import os
import json
import math
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Danh sách 6 lớp khuyết tật PCB chuẩn DeepPCB
PCB_DEFECT_CLASSES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]


def compute_iou(box1: Union[List[float], Tuple[float, ...]], box2: Union[List[float], Tuple[float, ...]]) -> float:
    """
    Tính chỉ số Intersection over Union (IoU) giữa 2 bounding boxes dạng [x1, y1, x2, y2].
    """
    if len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


# ─────────────────────────────────────────────────────────
# 1. YOLO11 Trainer (Huấn luyện Judge Model trên ảnh thực)
# ─────────────────────────────────────────────────────────

class YOLO11DefectTrainer:
    """
    Huấn luyện mô hình phát hiện YOLO11 trên tập ảnh lỗi PCB thực tế (10% low-data split)
    để làm bộ giám định chất lượng (Confidence Judge).
    """

    def __init__(
        self,
        model_size: str = "s",   # n / s / m / l / x
        device: str = "cuda",
        project: str = "outputs/yolo_training",
    ):
        self.model_size = model_size
        self.device = device
        self.project = Path(project)
        self.project.mkdir(parents=True, exist_ok=True)
        self.model_name = f"yolo11{model_size}.pt"

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            self.has_ultralytics = True
            logger.info(f"[YOLO11 Judge] Đã nạp kiến trúc {self.model_name} qua thư viện ultralytics.")
        except ImportError:
            self.has_ultralytics = False
            self.model = None
            logger.warning(
                "[YOLO11 Judge] Chưa cài đặt 'ultralytics' (hoặc môi trường không có GPU). "
                "Chế độ Dry-run / Setup được kích hoạt. Khi chạy trên Google Colab, mô hình sẽ tự động train."
            )

    def train(
        self,
        data_yaml: Union[str, Path] = "data/id_setup.yaml",
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        name: str = "pcb_defect_judge",
    ) -> str:
        """
        Huấn luyện YOLO11 trên tập lỗi thực tế (100 epochs với AdamW theo chuẩn bài báo).

        Args:
            data_yaml: File cấu hình dataset YOLO (chuẩn ID split 10% từ Step 0)
            epochs:    Số epochs huấn luyện (chuẩn bài báo là 100 epochs)
            batch:     Batch size
            imgsz:     Kích thước ảnh input (640)
            name:      Tên thí nghiệm

        Returns:
            Đường dẫn tới file trọng số tốt nhất (best.pt)
        """
        import torch

        data_yaml_path = Path(data_yaml)
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Không tìm thấy dataset YAML: {data_yaml_path}")

        best_weights_dir = self.project / name / "weights"
        best_weights_dir.mkdir(parents=True, exist_ok=True)
        best_weights = best_weights_dir / "best.pt"

        # Huấn luyện mô hình YOLO11 trên GPU với ultralytics
        should_train = (
            self.has_ultralytics
            and self.model is not None
            and (self.device == "cuda" or (isinstance(self.device, str) and "cuda" in self.device and torch.cuda.is_available()))
        )

        if should_train:
            logger.info("=" * 60)
            logger.info(f"[YOLO11 Judge] Bắt đầu huấn luyện GPU {self.model_name} trên {data_yaml_path}:")
            logger.info(f"  Epochs: {epochs}, Batch size: {batch}, Image size: {imgsz}, Optimizer: AdamW, Device: {self.device}")
            logger.info("=" * 60)

            self.model.train(
                data=str(data_yaml_path.resolve()).replace("\\", "/"),
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                optimizer="AdamW",
                lr0=1e-3,
                cos_lr=True,
                device=self.device,
                project=str(self.project),
                name=name,
                exist_ok=True,
                verbose=True,
            )
            logger.info(f"[YOLO11 Judge] Huấn luyện hoàn tất! Trọng số lưu tại: {best_weights}")
            return str(best_weights.resolve())

        # Fallback khi môi trường không có GPU
        logger.info(f"[YOLO11 Judge] Không có GPU hoặc ultralytics, checkpoint giả lập sẵn sàng tại: {best_weights}")
        if not best_weights.exists():
            best_weights.write_text("yolo11_stub_weights")
        return str(best_weights.resolve())


# ─────────────────────────────────────────────────────────
# 2. YOLO11 Confidence Scorer (Đánh giá chất lượng ảnh)
# ─────────────────────────────────────────────────────────

class YOLO11ConfidenceScorer:
    """
    Module chấm điểm độ tin cậy phát hiện lỗi cho từng ảnh tổng hợp:
    - Khi có mô hình YOLO11 (GPU / Colab): Chấm điểm Target-aware IoU matching.
    - Khi không có GPU / mô hình (Local testing): Heuristic Scorer dựa trên
      Sobel Gradient Magnitude và Độ tương phản vùng lỗi để tránh crash pipeline.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        imgsz: int = 640,
    ):
        self.device = device
        self.imgsz = imgsz
        self.weights_path = Path(weights_path) if weights_path else None
        self.model = None
        self.use_heuristic = True

        if self.weights_path and self.weights_path.exists():
            try:
                # Kiểm tra xem file có phải stub không
                if self.weights_path.stat().st_size > 1000:
                    from ultralytics import YOLO
                    self.model = YOLO(str(self.weights_path))
                    self.use_heuristic = False
                    logger.info(f"[Scorer] Đã nạp YOLO11 Judge Weights <- {self.weights_path}")
            except Exception as e:
                logger.warning(f"[Scorer] Không thể nạp trọng số YOLO từ {weights_path}: {e}")

        if self.use_heuristic:
            logger.info("[Scorer] Kích hoạt Heuristic Quality Scorer (Sobel Edge Energy + Local Contrast).")

    def score_record(self, record: Dict) -> float:
        """
        Chấm điểm chất lượng cho một record ảnh tổng hợp từ Step 2.

        Đặc tả Target-aware IoU Matching:
        - Đối chiếu target_class_id và target_bbox của khuyết tật được tạo.
        - Nếu YOLO11 phát hiện đúng lớp lỗi tại vùng tương ứng (IoU >= 0.1) -> Lấy confidence phát hiện.
        - Nếu không phát hiện đúng lớp lỗi tại vùng này -> Chấm 0.0 điểm (ảnh hỏng/kém chất lượng).
        """
        img_path = record.get("synthetic_image_path") or record.get("generated_path")
        if not img_path or not Path(img_path).exists():
            return 0.0

        target_cid = int(record.get("class_id", 0))
        target_bbox = record.get("bbox_pixel", [])
        defect_type = record.get("defect_type", "unknown")

        # Tự động trích xuất target_bbox pixel nếu record chỉ có bbox_norm
        if (not target_bbox or len(target_bbox) != 4) and record.get("bbox_norm"):
            try:
                bn = record["bbox_norm"]
                with Image.open(img_path) as im:
                    orig_w, orig_h = im.size
                # bbox_norm từ Step 1 là [xc, yc, bw, bh] chuẩn hóa
                xc, yc, bw, bh = bn[0] * orig_w, bn[1] * orig_h, bn[2] * orig_w, bn[3] * orig_h
                target_bbox = [max(0.0, xc - bw / 2.0), max(0.0, yc - bh / 2.0), min(float(orig_w), xc + bw / 2.0), min(float(orig_h), yc + bh / 2.0)]
            except Exception:
                pass

        # ── 1. SỬ DỤNG MÔ HÌNH YOLO11 THỰC TẾ (TRÊN GPU / COLAB) ─────────
        if not self.use_heuristic and self.model is not None:
            try:
                results = self.model.predict(
                    str(img_path),
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
                if not results or len(results[0].boxes) == 0:
                    return 0.0

                boxes = results[0].boxes
                pred_cids = boxes.cls.cpu().numpy().astype(int)
                pred_confs = boxes.conf.cpu().numpy().astype(float)
                pred_xyxy = boxes.xyxy.cpu().numpy().astype(float)

                matching_confs = []
                same_class_confs = []

                for cid, conf, xyxy in zip(pred_cids, pred_confs, pred_xyxy):
                    if cid == target_cid:
                        same_class_confs.append(conf)
                        if target_bbox and len(target_bbox) == 4:
                            iou = compute_iou(xyxy, target_bbox)
                            if iou >= 0.1:
                                matching_confs.append(conf)

                if matching_confs:
                    # Phát hiện đúng lớp và đúng vùng tọa độ
                    return float(max(matching_confs))
                elif same_class_confs and not target_bbox:
                    # Không có target_bbox, lấy max confidence của đúng lớp
                    return float(max(same_class_confs))
                elif same_class_confs:
                    # Đúng lớp nhưng vị trí lệch hơn 90% -> Giảm 50% điểm
                    return float(max(same_class_confs) * 0.5)

                # YOLO11 không phát hiện ra lớp khuyết tật mục tiêu
                return 0.0

            except Exception as e:
                logger.warning(f"[Scorer] Lỗi khi inference YOLO trên {img_path}: {e}")

        # ── 2. HEURISTIC QUALITY SCORER (LOCAL DRY-RUN / NO-GPU) ────────
        return self._heuristic_score(img_path, target_bbox, defect_type)

    def _heuristic_score(
        self,
        img_path: Union[str, Path],
        target_bbox: List[float],
        defect_type: str,
    ) -> float:
        """
        Bộ chấm điểm mô phỏng dựa trên thị giác máy tính cổ điển:
        - Tính năng lượng cạnh Sobel (Edge Energy) trong vùng lỗi: phát hiện độ sắc nét.
        - Tính độ tương phản cục bộ (Local Contrast) so với vùng đệm xung quanh:
          loại bỏ các khuyết tật bị mờ nhạt hoặc bết màu.
        - Trả về điểm confidence chuẩn hóa trong khoảng [0.35..0.95].
        """
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return 0.0

            H, W, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Lấy tọa độ vùng bbox lỗi
            if target_bbox and len(target_bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in target_bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
            else:
                # Nếu không có bbox, lấy vùng trung tâm ảnh
                x1, y1, x2, y2 = W // 4, H // 4, 3 * W // 4, 3 * H // 4

            if x2 <= x1 or y2 <= y1:
                return 0.0

            crop = gray[y1:y2, x1:x2]
            if crop.size == 0:
                return 0.0

            # 1. Năng lượng biên cạnh Sobel
            sobel_x = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            edge_score = float(np.mean(edge_mag) / 255.0)

            # 2. Độ tương phản so với viền bao xung quanh (Local Context Margin)
            margin = 8
            my1, my2 = max(0, y1 - margin), min(H, y2 + margin)
            mx1, mx2 = max(0, x1 - margin), min(W, x2 + margin)
            surrounding = gray[my1:my2, mx1:mx2]

            mean_crop = float(np.mean(crop))
            mean_surr = float(np.mean(surrounding))
            contrast_score = abs(mean_crop - mean_surr) / 255.0

            # 3. Kết hợp chỉ số chất lượng
            combined_metric = 0.5 * min(1.0, edge_score * 3.5) + 0.5 * min(1.0, contrast_score * 4.0)

            # Thêm nhiễu ngẫu nhiên xác định (deterministic seed) theo tên file
            # để đảm bảo xếp hạng phân vị tự nhiên và ổn định
            seed_val = sum(ord(c) for c in Path(img_path).stem) % 1000
            np.random.seed(seed_val)
            perturbation = np.random.uniform(-0.05, 0.05)

            final_conf = np.clip(0.40 + combined_metric * 0.50 + perturbation, 0.10, 0.98)
            return float(round(final_conf, 4))

        except Exception as e:
            logger.warning(f"[Scorer Heuristic] Không thể tính điểm cho {img_path}: {e}")
            return 0.50

    def score_batch(self, records: List[Dict]) -> List[float]:
        """Chấm điểm danh sách records ảnh."""
        return [self.score_record(rec) for rec in records]


# ─────────────────────────────────────────────────────────
# 3. Quality Filter (Loại bỏ 25% thấp nhất & Xuất Dataset)
# ─────────────────────────────────────────────────────────

class QualityFilter:
    """
    Bộ lọc chất lượng ảnh tổng hợp:
    - Loại bỏ k = 25% số lượng ảnh có confidence score thấp nhất.
    - Giữ lại 75% ảnh đạt chuẩn chất lượng cao nhất.
    - Tự động xuất nhãn bounding box chuẩn YOLO (.txt).
    - Tạo file dataset kết hợp (augmented_dataset.yaml) cho bước đánh giá hạ nguồn.
    """

    def __init__(
        self,
        scorer: YOLO11ConfidenceScorer,
        reject_ratio: float = 0.25,
    ):
        self.scorer = scorer
        self.reject_ratio = reject_ratio

    def filter(
        self,
        generated_metadata: List[Dict],
        output_dir: Union[str, Path] = "outputs/filtered",
        copy_accepted: bool = True,
        data_yaml: Optional[Union[str, Path]] = "data/id_setup.yaml",
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Lọc toàn cục danh sách ảnh tổng hợp theo độ tin cậy.

        Args:
            generated_metadata: List metadata từ Step 2 (generated_metadata.json)
            output_dir:         Thư mục lưu ảnh và dataset sau lọc
            copy_accepted:      Copy các ảnh đạt chuẩn sang thư mục accepted
            data_yaml:          File cấu hình dataset lỗi thực tế gốc

        Returns:
            (accepted_records, rejected_records)
        """
        out_dir = Path(output_dir)
        accepted_dir = out_dir / "accepted"
        accepted_imgs_dir = accepted_dir / "images"
        accepted_lbls_dir = accepted_dir / "labels"
        rejected_dir = out_dir / "rejected"

        accepted_imgs_dir.mkdir(parents=True, exist_ok=True)
        accepted_lbls_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)

        if not generated_metadata:
            logger.warning("[Filter] Danh sách ảnh rỗng!")
            return [], []

        logger.info(f"[Filter] Đang chấm điểm {len(generated_metadata)} ảnh tổng hợp...")
        scores = self.scorer.score_batch(generated_metadata)

        # Gắn confidence score vào từng record
        scored_records = []
        for rec, score in zip(generated_metadata, scores):
            rec_copy = dict(rec)
            rec_copy["confidence"] = score
            scored_records.append(rec_copy)

        # Sắp xếp theo confidence giảm dần (chất lượng cao nhất lên đầu)
        scored_records.sort(key=lambda r: float(r["confidence"]), reverse=True)

        # Phân tách theo tỷ lệ loại bỏ (25% thấp nhất bị reject)
        n_total = len(scored_records)
        n_keep = max(1, int(round(n_total * (1.0 - self.reject_ratio))))
        accepted = scored_records[:n_keep]
        rejected = scored_records[n_keep:]

        logger.info("=" * 60)
        logger.info(f"[Quality Filter] Kết quả lọc chất lượng (Tỷ lệ loại bỏ {self.reject_ratio*100:.0f}%):")
        logger.info(f"  Tổng ảnh đầu vào:   {n_total}")
        logger.info(f"  Chấp nhận (Keep):   {len(accepted)} ({len(accepted)/n_total*100:.1f}%)")
        logger.info(f"  Loại bỏ (Reject):   {len(rejected)} ({len(rejected)/n_total*100:.1f}%)")

        acc_scores = [r["confidence"] for r in accepted]
        rej_scores = [r["confidence"] for r in rejected]
        logger.info(f"  Accepted Confidence: Min={min(acc_scores):.4f}, Mean={np.mean(acc_scores):.4f}, Max={max(acc_scores):.4f}")
        if rej_scores:
            logger.info(f"  Rejected Confidence: Min={min(rej_scores):.4f}, Mean={np.mean(rej_scores):.4f}, Max={max(rej_scores):.4f}")
        logger.info("=" * 60)

        # ── XUẤT ẢNH & FILE NHÃN YOLO CHO CÁC ẢNH ACCEPTED ────────────────
        accepted_img_paths = []
        for rec in accepted:
            src_path_str = rec.get("synthetic_image_path") or rec.get("generated_path")
            src_path = Path(src_path_str)

            if copy_accepted and src_path.exists():
                dst_img = accepted_imgs_dir / src_path.name
                shutil.copy2(src_path, dst_img)
                rec["filtered_image_path"] = str(dst_img.resolve()).replace("\\", "/")
                accepted_img_paths.append(rec["filtered_image_path"])
            else:
                accepted_img_paths.append(str(src_path.resolve()).replace("\\", "/"))

            # Tạo file nhãn YOLO (.txt): class_id x_c y_c w h
            stem = src_path.stem
            lbl_file = accepted_lbls_dir / f"{stem}.txt"

            cid = int(rec.get("class_id", 0))
            bbox_norm = rec.get("bbox_norm", [])
            bbox_pixel = rec.get("bbox_pixel", [])

            if bbox_norm and len(bbox_norm) == 4:
                # bbox_norm từ Step 1 đã ở định dạng chuẩn YOLO: [xc, yc, w, h] chuẩn hóa [0, 1]
                xc, yc, w, h = [float(v) for v in bbox_norm]
            elif bbox_pixel and len(bbox_pixel) == 4:
                # Nếu chỉ có bbox_pixel [x1, y1, x2, y2], lấy kích thước ảnh thực để tính
                try:
                    with Image.open(src_path) as im:
                        orig_w, orig_h = im.size
                except Exception:
                    orig_w, orig_h = 600, 600

                px1, py1, px2, py2 = [float(v) for v in bbox_pixel]
                xc = (px1 + px2) / (2.0 * orig_w)
                yc = (py1 + py2) / (2.0 * orig_h)
                w = (px2 - px1) / float(orig_w)
                h = (py2 - py1) / float(orig_h)
            else:
                xc, yc, w, h = 0.5, 0.5, 0.05, 0.05

            xc = float(np.clip(xc, 0.001, 0.999))
            yc = float(np.clip(yc, 0.001, 0.999))
            w = float(np.clip(w, 0.001, 0.999))
            h = float(np.clip(h, 0.001, 0.999))

            lbl_file.write_text(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
            rec["yolo_label_path"] = str(lbl_file.resolve()).replace("\\", "/")

        # Lưu metadata
        with open(out_dir / "accepted_metadata.json", "w", encoding="utf-8") as f:
            json.dump(accepted, f, indent=2, ensure_ascii=False)

        with open(out_dir / "rejected_metadata.json", "w", encoding="utf-8") as f:
            json.dump(rejected, f, indent=2, ensure_ascii=False)

        # ── TẠO FILE AUGMENTED DATASET YAML CHO DOWNSTREAM BENCHMARK ───────
        self._export_augmented_dataset(accepted_img_paths, out_dir, data_yaml)

        logger.info(f"[Filter] Hoàn tất lọc! Kết quả và nhãn YOLO đã lưu tại: {out_dir}")
        return accepted, rejected

    def filter_by_class(
        self,
        generated_metadata: List[Dict],
        output_dir: Union[str, Path] = "outputs/filtered",
        data_yaml: Optional[Union[str, Path]] = "data/id_setup.yaml",
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Lọc độc lập theo từng loại lỗi (defect_type) để bảo đảm tỷ lệ cân bằng giữa các lớp.
        """
        by_class: Dict[str, List[Dict]] = {}
        for rec in generated_metadata:
            cls_name = rec.get("defect_type", "unknown")
            by_class.setdefault(cls_name, []).append(rec)

        all_accepted, all_rejected = [], []
        logger.info(f"[Filter By Class] Tiến hành lọc độc lập cho {len(by_class)} lớp khuyết tật...")

        for cls_name, records in by_class.items():
            cls_out = Path(output_dir) / "classes" / cls_name
            acc, rej = self.filter(records, cls_out, copy_accepted=False, data_yaml=None)
            all_accepted.extend(acc)
            all_rejected.extend(rej)
            logger.info(f"  [{cls_name:16s}] Keep: {len(acc):3d} | Reject: {len(rej):3d}")

        # Xuất dataset chung sau khi gom tất cả các lớp
        out_dir = Path(output_dir)
        acc_paths = [r.get("filtered_image_path") or r.get("synthetic_image_path") for r in all_accepted]
        self._export_augmented_dataset(acc_paths, out_dir, data_yaml)

        with open(out_dir / "accepted_metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_accepted, f, indent=2, ensure_ascii=False)

        with open(out_dir / "rejected_metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_rejected, f, indent=2, ensure_ascii=False)

        return all_accepted, all_rejected

    def _export_augmented_dataset(
        self,
        synthetic_img_paths: List[str],
        output_dir: Path,
        base_yaml_path: Optional[Union[str, Path]],
    ):
        """
        Gộp danh sách ảnh lỗi thực tế (10% low-data) và ảnh tổng hợp accepted
        thành file augmented_train.txt và file augmented_dataset.yaml.
        """
        real_train_imgs = []
        val_img_source = ""

        # Đọc cấu hình dataset gốc nếu có
        if base_yaml_path and Path(base_yaml_path).exists():
            try:
                import yaml
                with open(base_yaml_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)

                train_ref = cfg.get("train", "")
                val_img_source = cfg.get("val", "")

                if train_ref and Path(train_ref).exists():
                    with open(train_ref, "r", encoding="utf-8") as f:
                        real_train_imgs = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.warning(f"Không thể đọc base dataset YAML {base_yaml_path}: {e}")

        # Gộp danh sách ảnh thực và ảnh sinh
        merged_train_imgs = list(real_train_imgs) + [p.replace("\\", "/") for p in synthetic_img_paths if p]

        aug_train_txt = output_dir / "augmented_train.txt"
        with open(aug_train_txt, "w", encoding="utf-8") as f:
            for p in merged_train_imgs:
                f.write(f"{p}\n")

        # Tạo file YAML mở rộng
        aug_yaml_path = output_dir / "augmented_dataset.yaml"
        aug_yaml_content = (
            f"# Augmented PCB Defect Dataset (Real 10% + Top 75% Accepted Synthetic)\n"
            f"path: .\n"
            f"train: {aug_train_txt.resolve().as_posix()}\n"
            f"val: {val_img_source}\n\n"
            f"nc: 6\n"
            f"names: {PCB_DEFECT_CLASSES}\n"
        )
        aug_yaml_path.write_text(aug_yaml_content, encoding="utf-8")
        logger.info(f"[Dataset Exporter] Đã tạo Augmented Dataset YAML: {aug_yaml_path}")
        logger.info(f"  Tổng ảnh train: {len(merged_train_imgs)} (Thực: {len(real_train_imgs)} + Tổng hợp: {len(synthetic_img_paths)})")


# ─────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Step 3: Quality Filtering using YOLO11")
    subparsers = parser.add_subparsers(dest="cmd")

    # 1. Huấn luyện YOLO judge
    tr = subparsers.add_parser("train_judge", help="Huấn luyện YOLO11 judge trên ảnh lỗi thực")
    tr.add_argument("--data_yaml",   default="data/id_setup.yaml", help="File cấu hình dataset lỗi thực (ID 10%)")
    tr.add_argument("--epochs",      type=int, default=100, help="Số epochs huấn luyện (chuẩn bài báo là 100)")
    tr.add_argument("--batch",       type=int, default=16, help="Batch size")
    tr.add_argument("--model_size",  default="s", choices=["n", "s", "m", "l", "x"], help="Kích thước mô hình YOLO11")
    tr.add_argument("--device",      default="cuda", help="Thiết bị tính toán (cuda / cpu)")
    tr.add_argument("--project",     default="outputs/yolo_training", help="Thư mục lưu weights")

    # 2. Lọc ảnh
    fl = subparsers.add_parser("filter", help="Lọc 25% ảnh confidence thấp nhất")
    fl.add_argument("--generated_meta", required=True, help="File metadata từ Step 2 (generated_metadata.json)")
    fl.add_argument("--yolo_weights",   default=None, help="File trọng số best.pt (nếu có)")
    fl.add_argument("--data_yaml",      default="data/id_setup.yaml", help="Dataset YAML gốc để ghép dataset")
    fl.add_argument("--output_dir",     default="outputs/filtered", help="Thư mục xuất kết quả")
    fl.add_argument("--reject_ratio",   type=float, default=0.25, help="Tỷ lệ loại bỏ (0.25 = 25%)")
    fl.add_argument("--by_class",       action="store_true", help="Lọc độc lập theo từng loại lỗi")
    fl.add_argument("--device",         default="cuda", help="Thiết bị tính toán")

    args = parser.parse_args()

    if args.cmd == "train_judge":
        trainer = YOLO11DefectTrainer(
            model_size=args.model_size,
            device=args.device,
            project=args.project,
        )
        trainer.train(
            data_yaml=args.data_yaml,
            epochs=args.epochs,
            batch=args.batch,
        )

    elif args.cmd == "filter":
        with open(args.generated_meta, "r", encoding="utf-8") as f:
            gen_meta = json.load(f)

        scorer = YOLO11ConfidenceScorer(weights_path=args.yolo_weights, device=args.device)
        flt = QualityFilter(scorer, reject_ratio=args.reject_ratio)

        if args.by_class:
            flt.filter_by_class(gen_meta, output_dir=args.output_dir, data_yaml=args.data_yaml)
        else:
            flt.filter(gen_meta, output_dir=args.output_dir, data_yaml=args.data_yaml)

    else:
        parser.print_help()
