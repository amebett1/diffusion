"""
Step 3: Quality Filtering using YOLO11
----------------------------------------
- Train YOLO11 trên ảnh lỗi PCB thực để làm "judge"
- Đánh giá confidence score cho từng ảnh tổng hợp
- Loại bỏ 25% ảnh có confidence thấp nhất
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# YOLO11 Trainer (train trên ảnh lỗi thực)
# ─────────────────────────────────────────────────────────

class YOLO11DefectTrainer:
    """
    Huấn luyện YOLO11 trên tập ảnh lỗi PCB thực tế để
    tạo một bộ đánh giá chất lượng (confidence scorer).

    Yêu cầu: ultralytics >= 8.3 (YOLO11 được hỗ trợ từ v8.3+)
    """

    def __init__(
        self,
        model_size: str = "n",   # n / s / m / l / x
        device: str = "cuda",
        project: str = "outputs/yolo_training",
    ):
        from ultralytics import YOLO
        self.model_size = model_size
        self.device = device
        self.project = project
        model_name = f"yolo11{model_size}.pt"
        logger.info(f"[YOLO11] Khởi tạo model: {model_name}")
        self.model = YOLO(model_name)

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        batch: int = 16,
        imgsz: int = 640,
        name: str = "pcb_defect_judge",
    ) -> str:
        """
        Huấn luyện YOLO11 trên tập lỗi thực.

        Args:
            data_yaml: Đường dẫn tới YOLO dataset YAML
                       (chứa train/val paths + class names)
            epochs:    Số epoch (50 đủ cho judge)
            batch:     Batch size
            imgsz:     Kích thước ảnh input
            name:      Tên run

        Returns:
            Path tới best.pt weights
        """
        logger.info(f"[YOLO11] Bắt đầu train: {epochs} epochs, batch={batch}")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            project=self.project,
            name=name,
            exist_ok=True,
            verbose=True,
        )
        best_weights = Path(self.project) / name / "weights" / "best.pt"
        logger.info(f"[YOLO11] ✅ Train xong → {best_weights}")
        return str(best_weights)

    @staticmethod
    def build_dataset_yaml(
        real_defect_dir: str | Path,
        output_yaml: str | Path,
        defect_classes: Optional[list[str]] = None,
        val_split: float = 0.2,
    ):
        """
        Tạo YOLO dataset YAML từ thư mục ảnh lỗi thực.
        Cấu trúc thư mục kỳ vọng:
            real_defect_dir/
              <class_name>/
                img1.jpg
                img2.jpg
                ...

        Args:
            real_defect_dir: Thư mục gốc chứa ảnh lỗi thực
            output_yaml:     Đường dẫn ghi file YAML
            defect_classes:  Danh sách lớp (tự động detect nếu None)
            val_split:       Tỷ lệ validation
        """
        import yaml
        import random
        from sklearn.model_selection import train_test_split

        real_dir = Path(real_defect_dir)
        if defect_classes is None:
            defect_classes = sorted(
                d.name for d in real_dir.iterdir() if d.is_dir()
            )

        train_imgs, val_imgs = [], []
        train_labels, val_labels = [], []

        for cls_idx, cls_name in enumerate(defect_classes):
            cls_dir = real_dir / cls_name
            imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
            if len(imgs) < 2:
                continue
            tr, vl = train_test_split(imgs, test_size=val_split, random_state=42)
            train_imgs.extend(tr)
            val_imgs.extend(vl)

        # Tạo label files (full-image bounding box 0 0.5 0.5 1.0 1.0)
        label_dir = real_dir / "labels"
        label_dir.mkdir(exist_ok=True)

        def _write_labels(img_list):
            for img_path in img_list:
                cls_name = img_path.parent.name
                cls_idx = defect_classes.index(cls_name)
                lbl_path = label_dir / (img_path.stem + ".txt")
                lbl_path.write_text(f"{cls_idx} 0.5 0.5 1.0 1.0\n")

        _write_labels(train_imgs)
        _write_labels(val_imgs)

        yaml_data = {
            "path": str(real_dir.resolve()),
            "train": [str(p.relative_to(real_dir)) for p in train_imgs],
            "val":   [str(p.relative_to(real_dir)) for p in val_imgs],
            "nc":    len(defect_classes),
            "names": defect_classes,
        }
        with open(output_yaml, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        logger.info(f"[YOLO11] Dataset YAML → {output_yaml}")
        logger.info(f"  Classes: {defect_classes}")
        logger.info(f"  Train: {len(train_imgs)} | Val: {len(val_imgs)}")
        return str(output_yaml)


# ─────────────────────────────────────────────────────────
# YOLO11 Confidence Scorer
# ─────────────────────────────────────────────────────────

class YOLO11ConfidenceScorer:
    """
    Dùng YOLO11 đã train để score ảnh tổng hợp.
    Confidence = max detection confidence trong ảnh.
    Nếu không detect được gì → confidence = 0.
    """

    def __init__(self, weights_path: str, device: str = "cuda", imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.device = device
        self.imgsz = imgsz
        logger.info(f"[Scorer] YOLO11 judge loaded ← {weights_path}")

    def score_image(self, image_path: str | Path) -> float:
        """Trả về confidence score [0, 1] cho 1 ảnh."""
        results = self.model.predict(
            str(image_path),
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if results and len(results[0].boxes) > 0:
            conf = float(results[0].boxes.conf.max().cpu())
        else:
            conf = 0.0
        return conf

    def score_batch(self, image_paths: list[str | Path]) -> list[float]:
        """Score nhiều ảnh, trả về list confidence scores."""
        scores = []
        for p in image_paths:
            try:
                s = self.score_image(p)
            except Exception as e:
                logger.warning(f"[Scorer] Lỗi {p}: {e}")
                s = 0.0
            scores.append(s)
        return scores


# ─────────────────────────────────────────────────────────
# Quality Filter
# ─────────────────────────────────────────────────────────

class QualityFilter:
    """
    Lọc ảnh tổng hợp: loại bỏ 25% có confidence thấp nhất.
    Theo thiết lập trong ảnh: "Loại 25% ảnh có độ tin cậy thấp nhất"
    """

    def __init__(self, scorer: YOLO11ConfidenceScorer, reject_ratio: float = 0.25):
        self.scorer = scorer
        self.reject_ratio = reject_ratio

    def filter(
        self,
        generated_metadata: list[dict],
        output_dir: str | Path,
        copy_accepted: bool = True,
    ) -> tuple[list[dict], list[dict]]:
        """
        Args:
            generated_metadata: List records từ Step 2 (có key 'generated_path')
            output_dir:         Thư mục lưu ảnh đã lọc
            copy_accepted:      Nếu True → copy ảnh accepted sang output_dir

        Returns:
            (accepted_records, rejected_records)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [rec["generated_path"] for rec in generated_metadata]
        logger.info(f"[Filter] Scoring {len(image_paths)} ảnh...")
        scores = self.scorer.score_batch(image_paths)

        # Gắn score vào records
        scored = [
            {**rec, "confidence": score}
            for rec, score in zip(generated_metadata, scores)
        ]

        # Sắp xếp theo confidence giảm dần
        scored.sort(key=lambda r: -r["confidence"])

        n_keep = int(len(scored) * (1 - self.reject_ratio))
        accepted = scored[:n_keep]
        rejected = scored[n_keep:]

        logger.info(
            f"[Filter] Tổng={len(scored)} | "
            f"Chấp nhận={len(accepted)} | "
            f"Loại bỏ={len(rejected)} "
            f"({self.reject_ratio*100:.0f}% thấp nhất)"
        )

        # Thống kê
        all_scores = [r["confidence"] for r in scored]
        accepted_scores = [r["confidence"] for r in accepted]
        rejected_scores = [r["confidence"] for r in rejected]
        logger.info(
            f"[Filter] Confidence thống kê:\n"
            f"  Tất cả:    min={min(all_scores):.3f} max={max(all_scores):.3f} "
            f"mean={np.mean(all_scores):.3f}\n"
            f"  Accepted:  min={min(accepted_scores):.3f} mean={np.mean(accepted_scores):.3f}\n"
            f"  Rejected:  max={max(rejected_scores):.3f} mean={np.mean(rejected_scores):.3f}"
        )

        # Copy ảnh accepted
        if copy_accepted:
            accepted_dir = output_dir / "accepted"
            accepted_dir.mkdir(exist_ok=True)
            for rec in accepted:
                src = Path(rec["generated_path"])
                dst = accepted_dir / src.name
                shutil.copy2(src, dst)
                rec["filtered_path"] = str(dst)

        # Lưu metadata
        with open(output_dir / "accepted_metadata.json", "w", encoding="utf-8") as f:
            json.dump(accepted, f, indent=2, ensure_ascii=False)
        with open(output_dir / "rejected_metadata.json", "w", encoding="utf-8") as f:
            json.dump(rejected, f, indent=2, ensure_ascii=False)

        logger.info(f"[Filter] 💾 Kết quả lọc → {output_dir}")
        return accepted, rejected

    def filter_by_class(
        self,
        generated_metadata: list[dict],
        output_dir: str | Path,
    ) -> dict[str, list[dict]]:
        """
        Lọc độc lập theo từng loại lỗi (defect_type).
        Đảm bảo phân phối đều giữa các lớp sau khi lọc.
        """
        # Nhóm theo defect_type
        by_class: dict[str, list] = {}
        for rec in generated_metadata:
            cls = rec.get("defect_type", "unknown")
            by_class.setdefault(cls, []).append(rec)

        all_accepted = {}
        for cls_name, records in by_class.items():
            cls_out = Path(output_dir) / cls_name
            accepted, rejected = self.filter(records, cls_out)
            all_accepted[cls_name] = accepted
            logger.info(f"  [{cls_name}] {len(accepted)} ảnh chấp nhận")

        return all_accepted


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    # Train YOLO judge
    tr = subparsers.add_parser("train_judge", help="Train YOLO11 judge trên ảnh lỗi thực")
    tr.add_argument("--real_defect_dir", required=True)
    tr.add_argument("--output_yaml",     default="outputs/pcb_dataset.yaml")
    tr.add_argument("--epochs",          type=int, default=50)
    tr.add_argument("--model_size",      default="n", choices=["n","s","m","l","x"])
    tr.add_argument("--device",          default="cuda")

    # Filter
    fl = subparsers.add_parser("filter", help="Lọc 25% ảnh confidence thấp nhất")
    fl.add_argument("--generated_meta",  required=True, help="generated_metadata.json từ Step 2")
    fl.add_argument("--yolo_weights",    required=True, help="best.pt từ train_judge")
    fl.add_argument("--output_dir",      default="outputs/filtered")
    fl.add_argument("--reject_ratio",    type=float, default=0.25)
    fl.add_argument("--by_class",        action="store_true",
                    help="Lọc độc lập theo từng defect_type")
    fl.add_argument("--device",          default="cuda")

    args = parser.parse_args()

    if args.cmd == "train_judge":
        trainer = YOLO11DefectTrainer(model_size=args.model_size, device=args.device)
        yaml_path = YOLO11DefectTrainer.build_dataset_yaml(
            args.real_defect_dir, args.output_yaml
        )
        trainer.train(yaml_path, epochs=args.epochs)

    elif args.cmd == "filter":
        with open(args.generated_meta, encoding="utf-8") as f:
            gen_meta = json.load(f)
        scorer = YOLO11ConfidenceScorer(args.yolo_weights, device=args.device)
        flt = QualityFilter(scorer, reject_ratio=args.reject_ratio)
        if args.by_class:
            flt.filter_by_class(gen_meta, args.output_dir)
        else:
            flt.filter(gen_meta, args.output_dir)

    else:
        parser.print_help()
