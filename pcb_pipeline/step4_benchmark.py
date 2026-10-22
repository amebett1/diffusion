"""
Step 4: Downstream Benchmark (Resource-limited YOLO11s demo)
============================================================
Mục tiêu của file này
---------------------
Đánh giá tác động của dữ liệu tổng hợp lên detector hạ nguồn trong một
thiết lập TÀI NGUYÊN HẠN CHẾ:

    Baseline  : 10% official train (real low-data) của seen boards
    Augmented : đúng 10% real low-data ở trên + ảnh synthetic đã accepted

Phạm vi trung thực
------------------
- Chỉ chạy YOLO11s. KHÔNG chạy YOLOv8, YOLOv9 hay RT-DETR trong demo này.
- Không tuyên bố tái lập đầy đủ Table 3 / Table 6 của paper.
- Sau khi train, LUÔN load lại ``best.pt`` từ đĩa và đánh giá checkpoint đó,
  không gọi ``.val()`` trên model đang nằm trong bộ nhớ sau train.
- ``iou`` truyền vào Ultralytics là NMS IoU, dùng 0.7 (khoảng chuẩn 0.5-0.7).
  Không truyền 0.25 như IoU matching của paper.
- Precision/Recall được lấy tại confidence 0.25. Ultralytics dùng match IoU
  mặc định 0.5 cho P/R; paper dùng 0.25. Vì API hiện tại không cho đặt trực
  tiếp match IoU, báo cáo ghi rõ khác biệt này thay vì gắn nhãn sai.

Output
------
- benchmark_summary.json
- benchmark_comparison.csv
- benchmark_report.md
"""

import sys
import os
import json
import csv
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import yaml
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PCB_DEFECT_CLASSES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]

# Chỉ cho phép YOLO11s trong demo tài nguyên hạn chế.
ALLOWED_MODELS = {
    "yolo11": "yolo11s.pt",
    "yolo11s": "yolo11s.pt",
}

SKIPPED_MODEL_HINTS = {
    "yolov8": "YOLOv8 bị bỏ qua trong demo này",
    "yolov8s": "YOLOv8 bị bỏ qua trong demo này",
    "yolov8n": "YOLOv8 bị bỏ qua trong demo này",
    "yolov9": "YOLOv9 bị bỏ qua trong demo này",
    "yolov9s": "YOLOv9 bị bỏ qua trong demo này",
    "yolov9c": "YOLOv9 bị bỏ qua trong demo này",
    "rtdetr": "RT-DETR bị bỏ qua trong demo này",
    "rtdetr-l": "RT-DETR bị bỏ qua trong demo này",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DownstreamBenchmarker:
    """Huấn luyện/đánh giá YOLO11s trên baseline và augmented dataset."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/benchmark",
        device: str = "cuda",
        epochs: int = 100,
        batch_size: int = 8,
        imgsz: int = 600,
        lr0: float = 1e-4,
        lrf: float = 0.01,
        optimizer: str = "AdamW",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.7,
        seed: int = 42,
        workers: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir = self.output_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.imgsz = int(imgsz)
        self.lr0 = float(lr0)
        self.lrf = float(lrf)
        self.optimizer = optimizer
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.seed = int(seed)
        self.workers = int(workers) if workers is not None else (0 if os.name == "nt" else 4)

    # ─────────────────────────────────────────────────────────
    # 1. Path/YAML normalization
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _candidate_roots(primary_yaml: Path) -> List[Path]:
        roots = [primary_yaml.parent, PROJECT_ROOT, Path.cwd()]
        return [p.resolve() for p in roots if p.exists()]

    @staticmethod
    def _rebase_known_path(raw: str) -> Optional[Path]:
        """Cứu các path kiểu /kaggle/working/diffusion/... bằng suffix data/ hoặc outputs/."""
        s = str(raw).strip().replace("\\", "/")
        for marker in ("outputs/", "data/"):
            idx = s.find(marker)
            if idx >= 0:
                cand = (PROJECT_ROOT / s[idx:]).resolve()
                if cand.exists():
                    return cand
        return None

    def _resolve_existing_path(self, raw: Union[str, Path], base_dirs: List[Path]) -> Optional[Path]:
        s = str(raw).strip()
        if not s:
            return None
        p = Path(s)
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            for b in base_dirs:
                candidates.append(b / p)
            candidates.append(PROJECT_ROOT / p)
            candidates.append(Path.cwd() / p)

        for c in candidates:
            try:
                if c.exists():
                    return c.resolve()
            except OSError:
                continue

        rebased = self._rebase_known_path(s)
        return rebased

    def _read_image_entries(self, value, base_dirs: List[Path]) -> List[str]:
        """Đọc ảnh từ directory, file .txt, list hoặc một file ảnh."""
        entries: List[str] = []
        if value is None:
            return entries

        if isinstance(value, (list, tuple)):
            for item in value:
                entries.extend(self._read_image_entries(item, base_dirs))
            return entries

        raw = str(value).strip()
        if not raw:
            return entries

        resolved = self._resolve_existing_path(raw, base_dirs)
        if resolved is None:
            logger.warning(f"[Benchmark] Không tìm thấy dataset entry: {raw}")
            return entries

        if resolved.is_dir():
            for p in sorted(resolved.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                    entries.append(str(p.resolve()))
            return entries

        if resolved.suffix.lower() in {".txt", ".csv"}:
            with open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Danh sách của Ultralytics thường chỉ có đường dẫn ảnh.
                    # Không split theo space để tránh làm hỏng path có dấu cách.
                    item = line.split(",")[0].strip() if resolved.suffix.lower() == ".csv" else line
                    p = self._resolve_existing_path(item, [resolved.parent] + base_dirs)
                    if p is not None and p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                        entries.append(str(p))
            return entries

        if resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES:
            entries.append(str(resolved))
        return entries

    @staticmethod
    def _dedupe(paths: List[str]) -> List[str]:
        seen = set()
        out = []
        for p in paths:
            key = str(Path(p).resolve()).lower() if os.name == "nt" else str(Path(p).resolve())
            if key not in seen:
                seen.add(key)
                out.append(str(Path(p).resolve()))
        return out

    def _write_yaml(
        self,
        out_yaml: Path,
        train_images: List[str],
        val_images: List[str],
        names: List[str],
        comment: str,
    ) -> Dict:
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        train_list = out_yaml.parent / f"{out_yaml.stem}_train.txt"
        val_list = out_yaml.parent / f"{out_yaml.stem}_val.txt"

        train_list.write_text(
            "\n".join(str(Path(p).resolve().as_posix()) for p in train_images) + "\n",
            encoding="utf-8",
        )
        val_list.write_text(
            "\n".join(str(Path(p).resolve().as_posix()) for p in val_images) + "\n",
            encoding="utf-8",
        )

        content = (
            f"# {comment}\n"
            "# Auto-generated by step4_benchmark.py; paths are runtime-adaptive.\n"
            f"path: {out_yaml.parent.resolve().as_posix()}\n"
            f"train: {train_list.name}\n"
            f"val: {val_list.name}\n\n"
            f"nc: {len(names)}\n"
            f"names: {names}\n"
        )
        out_yaml.write_text(content, encoding="utf-8")
        return {
            "yaml": str(out_yaml.resolve()),
            "train_list": str(train_list.resolve()),
            "val_list": str(val_list.resolve()),
            "train_count": len(train_images),
            "val_count": len(val_images),
        }

    def _prepare_dataset_yaml(self, src_yaml: Union[str, Path], out_yaml: Path, split_name: str) -> Dict:
        """Chuẩn hóa một YAML bất kỳ thành YAML tự chứa, path tuyệt đối runtime."""
        src = Path(src_yaml)
        if not src.exists():
            raise FileNotFoundError(f"Không tìm thấy dataset YAML: {src}")

        with open(src, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        base_dirs = self._candidate_roots(src)
        declared_path = data.get("path")
        if declared_path:
            p = self._resolve_existing_path(declared_path, base_dirs)
            if p is not None:
                base_dirs.insert(0, p)

        train_images = self._dedupe(self._read_image_entries(data.get("train"), base_dirs))
        val_images = self._dedupe(self._read_image_entries(data.get("val"), base_dirs))

        if not train_images:
            raise RuntimeError(f"Không đọc được ảnh train từ {src}")
        if not val_images:
            raise RuntimeError(f"Không đọc được ảnh val từ {src}")

        names = data.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        if not isinstance(names, list) or len(names) == 0:
            names = list(PCB_DEFECT_CLASSES)

        info = self._write_yaml(
            out_yaml=out_yaml,
            train_images=train_images,
            val_images=val_images,
            names=names,
            comment=f"Normalized {split_name} dataset from {src.resolve().as_posix()}",
        )
        info["source_yaml"] = str(src.resolve())
        logger.info(
            f"[Benchmark] Dataset YAML chuẩn hóa: {out_yaml} "
            f"(train={len(train_images)}, val={len(val_images)})"
        )
        return info

    def _prepare_augmented_yaml(
        self,
        aug_yaml: Union[str, Path],
        base_info: Dict,
        base_names: List[str],
        split_name: str,
    ) -> Dict:
        """
        Chuẩn hóa augmented YAML. Nếu YAML cũ trỏ tới /kaggle/... hoặc list
        synthetic bị thiếu, tái tạo từ accepted_metadata.json.
        """
        base_train = self._read_image_entries(base_info["train_list"], [self.datasets_dir])
        base_val = self._read_image_entries(base_info["val_list"], [self.datasets_dir])
        base_train = self._dedupe(base_train)

        aug_path = Path(aug_yaml)
        can_rebuild = False
        try:
            if aug_path.exists():
                with open(aug_path, "r", encoding="utf-8") as f:
                    aug_data = yaml.safe_load(f) or {}
                aug_base_dirs = self._candidate_roots(aug_path)
                declared = aug_data.get("path")
                if declared:
                    p = self._resolve_existing_path(declared, aug_base_dirs)
                    if p is not None:
                        aug_base_dirs.insert(0, p)

                aug_train = self._dedupe(
                    self._read_image_entries(aug_data.get("train"), aug_base_dirs)
                )
                if len(aug_train) > len(base_train):
                    info = self._write_yaml(
                        out_yaml=self.datasets_dir / f"augmented_{split_name}.yaml",
                        train_images=aug_train,
                        val_images=base_val,
                        names=base_names,
                        comment=(
                            f"Augmented {split_name} dataset: real low-data + accepted synthetic. "
                            f"Source: {aug_path.resolve().as_posix()}"
                        ),
                    )
                    info["source_yaml"] = str(aug_path.resolve())
                    info["rebuilt_from_metadata"] = False
                    info["synthetic_count"] = len(aug_train) - len(base_train)
                    logger.info(
                        f"[Benchmark] Augmented YAML: {info['yaml']} "
                        f"(train={info['train_count']}, synthetic≈{info['synthetic_count']})"
                    )
                    return info
                logger.warning(
                    "[Benchmark] Augmented YAML không có thêm ảnh synthetic so với baseline; "
                    "sẽ thử tái tạo từ accepted_metadata.json."
                )
            else:
                logger.warning(f"[Benchmark] Không thấy augmented YAML: {aug_path}")
            can_rebuild = True
        except Exception as e:
            logger.warning(f"[Benchmark] Không đọc được augmented YAML ({e}); thử tái tạo từ metadata.")
            can_rebuild = True

        if can_rebuild:
            candidates = [
                aug_path.parent / "accepted_metadata.json" if aug_path.parent.exists() else None,
                self.output_dir.parent / "filtered" / "accepted_metadata.json",
                PROJECT_ROOT / "outputs" / "pcb_synthesis" / "filtered" / "accepted_metadata.json",
            ]
            meta_path = next((p for p in candidates if p is not None and p.exists()), None)
            if meta_path is None:
                raise FileNotFoundError(
                    "Không thể tạo augmented dataset: không có augmented_dataset.yaml hợp lệ "
                    "và cũng không tìm thấy accepted_metadata.json. Hãy chạy Step 3 trước."
                )

            with open(meta_path, "r", encoding="utf-8") as f:
                accepted = json.load(f)

            synthetic_paths = []
            for rec in accepted:
                p = rec.get("filtered_image_path") or rec.get("synthetic_image_path")
                if p:
                    pp = Path(p)
                    if pp.exists():
                        synthetic_paths.append(str(pp.resolve()))
            synthetic_paths = self._dedupe(synthetic_paths)

            if not synthetic_paths:
                raise RuntimeError(
                    f"accepted_metadata.json tại {meta_path} không có ảnh synthetic tồn tại."
                )

            aug_train = self._dedupe(base_train + synthetic_paths)
            info = self._write_yaml(
                out_yaml=self.datasets_dir / f"augmented_{split_name}.yaml",
                train_images=aug_train,
                val_images=base_val,
                names=base_names,
                comment=(
                    f"Augmented {split_name} dataset rebuilt from accepted_metadata.json "
                    f"({meta_path.resolve().as_posix()})"
                ),
            )
            info["source_yaml"] = str(meta_path.resolve())
            info["rebuilt_from_metadata"] = True
            info["synthetic_count"] = len(synthetic_paths)
            logger.info(
                f"[Benchmark] Đã tái tạo augmented YAML: {info['yaml']} "
                f"(real={len(base_train)}, synthetic={len(synthetic_paths)})"
            )
            return info

        raise RuntimeError("Không thể chuẩn bị augmented dataset YAML.")

    # ─────────────────────────────────────────────────────────
    # 2. Model / metrics
    # ─────────────────────────────────────────────────────────
    def _resolve_models(self, requested: List[str]) -> Tuple[List[str], List[str]]:
        evaluated: List[str] = []
        skipped: List[str] = []
        for m in requested:
            key = str(m).lower().strip()
            if key in ALLOWED_MODELS:
                canonical = "yolo11s"
                if canonical not in evaluated:
                    evaluated.append(canonical)
            else:
                skipped.append(str(m))
                hint = SKIPPED_MODEL_HINTS.get(key, "không nằm trong phạm vi demo")
                logger.warning(f"[Benchmark] Bỏ qua '{m}': {hint}.")
        if not evaluated:
            raise ValueError(
                "Không có model hợp lệ. Demo này chỉ chạy YOLO11/YOLO11s. "
                f"Models yêu cầu: {requested}"
            )
        return evaluated, skipped

    def _get_model_instance(self, model_name: str):
        from ultralytics import YOLO

        key = model_name.lower().strip()
        weights_name = ALLOWED_MODELS.get(key)
        if weights_name is None:
            raise ValueError(f"Model '{model_name}' không được phép trong demo YOLO11s.")

        local_weights = PROJECT_ROOT / weights_name
        weights_arg = str(local_weights) if local_weights.exists() else weights_name
        logger.info(f"[Benchmark] Khởi tạo YOLO11s với weights: {weights_arg}")
        return YOLO(weights_arg), weights_name

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            v = float(value)
            return v if v == v and abs(v) != float("inf") else default
        except Exception:
            return default

    def _extract_metrics(self, val_results, model_name: str) -> Dict:
        metrics = {
            "model": model_name,
            "map50": 0.0,
            "map50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class_ap50": {},
            "per_class_ap50_95": {},
        }

        box = getattr(val_results, "box", None)
        if box is None:
            logger.warning("[Benchmark] val_results không có box metrics.")
            return metrics

        metrics["map50"] = self._safe_float(getattr(box, "map50", 0.0))
        metrics["map50_95"] = self._safe_float(getattr(box, "map", 0.0))
        metrics["precision"] = self._safe_float(getattr(box, "mp", 0.0))
        metrics["recall"] = self._safe_float(getattr(box, "mr", 0.0))

        maps = getattr(box, "maps", None)
        all_ap = getattr(box, "all_ap", None)
        names = getattr(val_results, "names", {})
        if isinstance(names, list):
            name_to_idx = {name: i for i, name in enumerate(names)}
        elif isinstance(names, dict):
            name_to_idx = {str(v): int(k) for k, v in names.items()}
        else:
            name_to_idx = {name: i for i, name in enumerate(PCB_DEFECT_CLASSES)}

        for cls_name in PCB_DEFECT_CLASSES:
            idx = name_to_idx.get(cls_name)
            if idx is None:
                metrics["per_class_ap50"][cls_name] = 0.0
                metrics["per_class_ap50_95"][cls_name] = 0.0
                continue

            if all_ap is not None and len(all_ap) > idx:
                try:
                    metrics["per_class_ap50"][cls_name] = round(float(all_ap[idx][0]), 4)
                except Exception:
                    metrics["per_class_ap50"][cls_name] = 0.0
            else:
                metrics["per_class_ap50"][cls_name] = 0.0

            if maps is not None and len(maps) > idx:
                try:
                    metrics["per_class_ap50_95"][cls_name] = round(float(maps[idx]), 4)
                except Exception:
                    metrics["per_class_ap50_95"][cls_name] = 0.0
            else:
                metrics["per_class_ap50_95"][cls_name] = 0.0

        return metrics

    # ─────────────────────────────────────────────────────────
    # 3. Train + evaluate best.pt
    # ─────────────────────────────────────────────────────────
    def train_and_eval_single(
        self,
        model_name: str,
        data_yaml: str,
        exp_name: str,
        is_augmented: bool = False,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
        n_epochs = int(epochs or self.epochs)
        batch = int(batch_size or self.batch_size)
        project_dir = self.output_dir / "runs"

        logger.info("=" * 72)
        tag = "AUGMENTED (Real low-data + Synthetic)" if is_augmented else "BASELINE (Real low-data)"
        logger.info(f"HUẤN LUYỆN DOWNSTREAM: YOLO11s | {tag}")
        logger.info(f"   Dataset:  {data_yaml}")
        logger.info(
            f"   Epochs: {n_epochs} | Batch: {batch} | Imgsz: {self.imgsz} | "
            f"Optimizer: {self.optimizer}"
        )
        logger.info(f"   Sau train sẽ load best.pt và val với conf={self.conf_thresh}, NMS IoU={self.iou_thresh}")
        logger.info("=" * 72)

        model, weights_name = self._get_model_instance(model_name)
        t0 = time.time()
        model.train(
            data=data_yaml,
            epochs=n_epochs,
            batch=batch,
            imgsz=self.imgsz,
            device=self.device,
            optimizer=self.optimizer,
            lr0=self.lr0,
            lrf=self.lrf,
            cos_lr=True,
            degrees=10.0,
            fliplr=0.5,
            flipud=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            project=str(project_dir),
            name=exp_name,
            exist_ok=True,
            seed=self.seed,
            workers=self.workers,
            verbose=True,
            plots=False,
            save=True,
        )
        train_time_min = round((time.time() - t0) / 60.0, 2)

        trainer = getattr(model, "trainer", None)
        save_dir = Path(getattr(trainer, "save_dir", project_dir / exp_name))
        best_ckpt = save_dir / "weights" / "best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(
                f"Không tìm thấy best.pt sau khi train tại {best_ckpt}. "
                "Không đánh giá model in-memory sau train."
            )

        logger.info(f"[Benchmark] Load best.pt để đánh giá: {best_ckpt}")
        from ultralytics import YOLO

        eval_model = YOLO(str(best_ckpt))
        val_results = eval_model.val(
            data=data_yaml,
            split="val",
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            plots=False,
            verbose=False,
        )

        metrics = self._extract_metrics(val_results, model_name)
        metrics.update(
            {
                "train_time_minutes": train_time_min,
                "exp_name": exp_name,
                "is_augmented": bool(is_augmented),
                "weights": weights_name,
                "best_ckpt": best_ckpt.resolve().as_posix(),
                "data_yaml": str(Path(data_yaml).resolve()),
                "conf_thresh": self.conf_thresh,
                "nms_iou": self.iou_thresh,
                "eval_checkpoint": "best.pt",
                "precision_recall_match_iou": "Ultralytics default (0.5); paper match IoU 0.25 not configured",
            }
        )

        logger.info(
            f"Hoàn tất {exp_name}: mAP@50={metrics['map50']*100:.2f}% | "
            f"mAP@50-95={metrics['map50_95']*100:.2f}% | "
            f"P={metrics['precision']*100:.2f}% | R={metrics['recall']*100:.2f}%"
        )
        return metrics

    # ─────────────────────────────────────────────────────────
    # 4. Run benchmark
    # ─────────────────────────────────────────────────────────
    def run_benchmark(
        self,
        models: List[str] = ["yolo11s"],
        base_yaml: str = "data/id_setup.yaml",
        aug_yaml: str = "outputs/pcb_synthesis/filtered/augmented_dataset.yaml",
        split_name: str = "id",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
        n_epochs = int(epochs or self.epochs)
        batch = int(batch_size or self.batch_size)
        evaluated_models, skipped_models = self._resolve_models(models)

        logger.info("#" * 72)
        logger.info(f"BẮT ĐẦU DOWNSTREAM BENCHMARK [{split_name.upper()}] — YOLO11s DEMO")
        logger.info(f"   Models đánh giá: {evaluated_models}")
        logger.info(f"   Models bỏ qua:   {skipped_models if skipped_models else 'không có'}")
        logger.info(f"   Baseline YAML:   {base_yaml}")
        logger.info(f"   Augmented YAML:  {aug_yaml}")
        logger.info(f"   Epochs: {n_epochs} | Batch: {batch} | Imgsz: {self.imgsz}")
        logger.info("#" * 72)

        base_info = self._prepare_dataset_yaml(
            src_yaml=base_yaml,
            out_yaml=self.datasets_dir / f"base_{split_name}.yaml",
            split_name=f"base_{split_name}",
        )

        with open(base_info["yaml"], "r", encoding="utf-8") as f:
            base_yaml_data = yaml.safe_load(f) or {}
        base_names = base_yaml_data.get("names", PCB_DEFECT_CLASSES)
        if isinstance(base_names, dict):
            base_names = [base_names[k] for k in sorted(base_names.keys(), key=lambda x: int(x))]

        aug_info = self._prepare_augmented_yaml(
            aug_yaml=aug_yaml,
            base_info=base_info,
            base_names=base_names,
            split_name=split_name,
        )

        if aug_info["train_count"] <= base_info["train_count"]:
            raise RuntimeError(
                "Augmented dataset không có thêm ảnh synthetic nào so với baseline. "
                "Không chạy benchmark giả tạo với hai dataset giống nhau."
            )

        results = {
            "split_name": split_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": n_epochs,
            "batch_size": batch,
            "imgsz": self.imgsz,
            "models_requested": list(models),
            "models_evaluated": evaluated_models,
            "models_skipped": skipped_models,
            "protocol": {
                "scope": "resource_limited_yolo11s_demo",
                "baseline": "10% official train seen boards (real)",
                "augmented": "same real low-data + accepted synthetic",
                "evaluation_split": "val",
                "eval_checkpoint": "best.pt",
                "eval_checkpoint_note": "reloaded from disk before validation",
                "conf_thresh": self.conf_thresh,
                "nms_iou": self.iou_thresh,
                "precision_recall_match_iou": (
                    "Ultralytics default 0.5; paper uses 0.25 for P/R matching. "
                    "This demo does not mislabel NMS IoU as match IoU."
                ),
                "paper_reproduction": False,
            },
            "datasets": {
                "baseline": base_info,
                "augmented": aug_info,
            },
            "comparisons": {},
        }

        for model_name in tqdm(evaluated_models, desc="Đánh giá Downstream Benchmark", unit="model"):
            exp_base = f"{split_name}_{model_name}_baseline"
            exp_aug = f"{split_name}_{model_name}_augmented"

            logger.info(f"\n>>> [1/2] Baseline YOLO11s...")
            base_metrics = self.train_and_eval_single(
                model_name=model_name,
                data_yaml=base_info["yaml"],
                exp_name=exp_base,
                is_augmented=False,
                epochs=n_epochs,
                batch_size=batch,
            )

            logger.info(f"\n>>> [2/2] Augmented YOLO11s...")
            aug_metrics = self.train_and_eval_single(
                model_name=model_name,
                data_yaml=aug_info["yaml"],
                exp_name=exp_aug,
                is_augmented=True,
                epochs=n_epochs,
                batch_size=batch,
            )

            p_base = base_metrics["precision"] * 100.0
            p_aug = aug_metrics["precision"] * 100.0
            delta_p = round(p_aug - p_base, 2)

            r_base = base_metrics["recall"] * 100.0
            r_aug = aug_metrics["recall"] * 100.0
            delta_r = round(r_aug - r_base, 2)

            map50_base = base_metrics["map50"] * 100.0
            map50_aug = aug_metrics["map50"] * 100.0
            delta_map50 = round(map50_aug - map50_base, 2)
            percent_gain = round(delta_map50 / max(map50_base, 1e-6) * 100.0, 2)

            map_base = base_metrics["map50_95"] * 100.0
            map_aug = aug_metrics["map50_95"] * 100.0
            delta_map = round(map_aug - map_base, 2)

            per_class_diff = {}
            improved, regressed = [], []
            for cls_name in PCB_DEFECT_CLASSES:
                c_base = base_metrics["per_class_ap50"].get(cls_name, 0.0) * 100.0
                c_aug = aug_metrics["per_class_ap50"].get(cls_name, 0.0) * 100.0
                d = round(c_aug - c_base, 2)
                per_class_diff[cls_name] = {
                    "baseline": round(c_base, 2),
                    "augmented": round(c_aug, 2),
                    "delta": d,
                }
                if d > 0.05:
                    improved.append(cls_name)
                elif d < -0.05:
                    regressed.append(cls_name)

            results["comparisons"][model_name] = {
                "baseline": base_metrics,
                "augmented": aug_metrics,
                "summary": {
                    "precision_baseline": round(p_base, 2),
                    "precision_augmented": round(p_aug, 2),
                    "delta_precision": delta_p,
                    "recall_baseline": round(r_base, 2),
                    "recall_augmented": round(r_aug, 2),
                    "delta_recall": delta_r,
                    "map50_baseline": round(map50_base, 2),
                    "map50_augmented": round(map50_aug, 2),
                    "delta_map50": delta_map50,
                    "percent_gain": percent_gain,
                    "map50_95_baseline": round(map_base, 2),
                    "map50_95_augmented": round(map_aug, 2),
                    "delta_map50_95": delta_map,
                    "per_class_comparison": per_class_diff,
                    "classes_improved": improved,
                    "classes_regressed": regressed,
                },
            }

            logger.info(
                f"\n[KẾT QUẢ YOLO11s]:\n"
                f"   Precision: {p_base:.2f}% -> {p_aug:.2f}% (Δ {delta_p:+.2f} pp)\n"
                f"   Recall:    {r_base:.2f}% -> {r_aug:.2f}% (Δ {delta_r:+.2f} pp)\n"
                f"   mAP@50:    {map50_base:.2f}% -> {map50_aug:.2f}% "
                f"(Δ {delta_map50:+.2f} pp, relative {percent_gain:+.2f}%)"
            )

        results["conclusion"] = self._build_conclusion(results)
        self._save_json_report(results)
        self._save_csv_report(results)
        self._save_markdown_report(results)
        self._print_console_summary(results)
        return results

    # ─────────────────────────────────────────────────────────
    # 5. Dynamic, honest conclusion & reports
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _trend_word(delta: float, positive: str = "tăng", negative: str = "giảm") -> str:
        if delta > 0.05:
            return positive
        if delta < -0.05:
            return negative
        return "gần như không đổi"

    def _build_conclusion(self, results: Dict) -> List[str]:
        lines: List[str] = []
        for model_name, comp in results.get("comparisons", {}).items():
            s = comp["summary"]
            lines.append(
                f"{model_name.upper()}: mAP@50 {self._trend_word(s['delta_map50'])} "
                f"{s['delta_map50']:+.2f} pp ({s['map50_baseline']:.2f}% -> "
                f"{s['map50_augmented']:.2f}%); Precision "
                f"{self._trend_word(s['delta_precision'])} {s['delta_precision']:+.2f} pp; "
                f"Recall {self._trend_word(s['delta_recall'])} {s['delta_recall']:+.2f} pp."
            )
            lines.append(
                f"  Per-class AP@50: {len(s['classes_improved'])} lớp tăng, "
                f"{len(s['classes_regressed'])} lớp giảm. "
                f"Tăng: {s['classes_improved'] or 'không có'}; "
                f"Giảm: {s['classes_regressed'] or 'không có'}."
            )

        comparisons = list(results.get("comparisons", {}).values())
        if comparisons:
            s = comparisons[0]["summary"]
            if s["delta_map50"] > 0.05:
                lines.append(
                    "Kết luận tổng quát: trong lần chạy này, dữ liệu tổng hợp có cải thiện "
                    "mAP@50 của YOLO11s. Đây là quan sát thực nghiệm, không phải bằng chứng "
                    "tái lập đầy đủ paper."
                )
            elif s["delta_map50"] < -0.05:
                lines.append(
                    "Kết luận tổng quát: trong lần chạy này, dữ liệu tổng hợp làm GIẢM "
                    "mAP@50 của YOLO11s. Cần báo cáo trung thực kết quả âm này và kiểm tra "
                    "chất lượng mask/inpainting/filter trước khi kết luận."
                )
            else:
                lines.append(
                    "Kết luận tổng quát: trong lần chạy này, dữ liệu tổng hợp không tạo ra "
                    "thay đổi đáng kể về mAP@50 của YOLO11s."
                )

        lines.append(
            "Phạm vi: chỉ YOLO11s, một seed, một thiết lập low-data; không chạy YOLOv8, "
            "YOLOv9 hay RT-DETR. Kết quả không đại diện cho toàn bộ bảng so sánh của paper."
        )
        return lines

    def _save_json_report(self, results: Dict):
        json_path = self.output_dir / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"[Reporter] Đã lưu JSON report: {json_path}")

    def _save_csv_report(self, results: Dict):
        csv_path = self.output_dir / "benchmark_comparison.csv"
        headers = [
            "Model",
            "Setup",
            "Precision Base (%)",
            "Precision Aug (%)",
            "Delta Precision (pp)",
            "Recall Base (%)",
            "Recall Aug (%)",
            "Delta Recall (pp)",
            "mAP@50 Base (%)",
            "mAP@50 Aug (%)",
            "Delta mAP@50 (pp)",
            "% Gain",
            "mAP@50-95 Base (%)",
            "mAP@50-95 Aug (%)",
            "Delta mAP@50-95 (pp)",
            "Train Time Base (min)",
            "Train Time Aug (min)",
            "Conf",
            "NMS IoU",
        ]
        for c in PCB_DEFECT_CLASSES:
            headers.extend([f"{c}_Base", f"{c}_Aug", f"{c}_Delta"])

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for model_name, comp in results.get("comparisons", {}).items():
                s = comp["summary"]
                base = comp["baseline"]
                aug = comp["augmented"]
                row = [
                    model_name.upper(),
                    results.get("split_name", "id").upper(),
                    s["precision_baseline"],
                    s["precision_augmented"],
                    s["delta_precision"],
                    s["recall_baseline"],
                    s["recall_augmented"],
                    s["delta_recall"],
                    s["map50_baseline"],
                    s["map50_augmented"],
                    s["delta_map50"],
                    s["percent_gain"],
                    s["map50_95_baseline"],
                    s["map50_95_augmented"],
                    s["delta_map50_95"],
                    base.get("train_time_minutes", 0),
                    aug.get("train_time_minutes", 0),
                    results.get("protocol", {}).get("conf_thresh", 0.25),
                    results.get("protocol", {}).get("nms_iou", 0.7),
                ]
                per_class = s.get("per_class_comparison", {})
                for c in PCB_DEFECT_CLASSES:
                    cdiff = per_class.get(c, {})
                    row.extend(
                        [
                            cdiff.get("baseline", 0.0),
                            cdiff.get("augmented", 0.0),
                            cdiff.get("delta", 0.0),
                        ]
                    )
                writer.writerow(row)

        logger.info(f"[Reporter] Đã lưu CSV report: {csv_path}")

    def _save_markdown_report(self, results: Dict):
        md_path = self.output_dir / "benchmark_report.md"
        split_name = results.get("split_name", "id").upper()
        protocol = results.get("protocol", {})
        datasets = results.get("datasets", {})
        base_info = datasets.get("baseline", {})
        aug_info = datasets.get("augmented", {})

        md = [
            f"# Downstream Benchmark Report — {split_name} (YOLO11s demo)",
            "",
            f"> **Thời gian:** {results.get('timestamp')}",
            f"> **Phạm vi:** {protocol.get('scope', 'resource_limited_yolo11s_demo')}",
            f"> **Models đánh giá:** {', '.join(results.get('models_evaluated', []))}",
            f"> **Models bỏ qua:** {', '.join(results.get('models_skipped', [])) or 'không có'}",
            f"> **Epochs:** {results.get('epochs')} | **Batch:** {results.get('batch_size')} | "
            f"**Imgsz:** {results.get('imgsz')} | **Optimizer:** AdamW",
            f"> **Đánh giá:** load lại `best.pt`; conf = {protocol.get('conf_thresh')}; "
            f"NMS IoU = {protocol.get('nms_iou')}",
            f"> **Baseline train:** {base_info.get('train_count', 0)} ảnh | "
            f"**Augmented train:** {aug_info.get('train_count', 0)} ảnh "
            f"(synthetic: {aug_info.get('synthetic_count', 'n/a')})",
            f"> **Val:** {base_info.get('val_count', 0)} ảnh (dùng chung cho cả hai setup)",
            "",
            "## 1. Bảng hiệu năng tổng thể (chỉ YOLO11s)",
            "",
            "| Model | Precision Base (%) | Precision Aug (%) | Δ Precision (pp) | "
            "Recall Base (%) | Recall Aug (%) | Δ Recall (pp) | mAP@50 Base (%) | "
            "mAP@50 Aug (%) | Δ mAP@50 (pp) | % Gain |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for model_name, comp in results.get("comparisons", {}).items():
            s = comp["summary"]
            md.append(
                f"| **{model_name.upper()}** | "
                f"{s['precision_baseline']:.2f}% | {s['precision_augmented']:.2f}% | "
                f"{s['delta_precision']:+.2f} | "
                f"{s['recall_baseline']:.2f}% | {s['recall_augmented']:.2f}% | "
                f"{s['delta_recall']:+.2f} | "
                f"{s['map50_baseline']:.2f}% | {s['map50_augmented']:.2f}% | "
                f"{s['delta_map50']:+.2f} | {s['percent_gain']:+.2f}% |"
            )

        md.extend(
            [
                "",
                "## 2. Per-class AP@50",
                "",
                "| Model | Setup | " + " | ".join(f"`{c}`" for c in PCB_DEFECT_CLASSES)
                + " | Mean AP@50 |",
                "|---|:---:|" + "|".join([":---:" for _ in PCB_DEFECT_CLASSES]) + "|:---:|",
            ]
        )

        for model_name, comp in results.get("comparisons", {}).items():
            s = comp["summary"]
            per_class = s.get("per_class_comparison", {})

            base_cells = [f"**{model_name.upper()}**", "Baseline"]
            base_cells.extend(
                f"{per_class.get(c, {}).get('baseline', 0.0):.2f}%" for c in PCB_DEFECT_CLASSES
            )
            base_cells.append(f"{s['map50_baseline']:.2f}%")
            md.append("| " + " | ".join(base_cells) + " |")

            aug_cells = [f"**{model_name.upper()}**", "Augmented"]
            aug_cells.extend(
                f"{per_class.get(c, {}).get('augmented', 0.0):.2f}%" for c in PCB_DEFECT_CLASSES
            )
            aug_cells.append(f"{s['map50_augmented']:.2f}%")
            md.append("| " + " | ".join(aug_cells) + " |")

            diff_cells = [f"**{model_name.upper()}**", "*Δ*"]
            diff_cells.extend(
                f"{per_class.get(c, {}).get('delta', 0.0):+.2f} pp" for c in PCB_DEFECT_CLASSES
            )
            diff_cells.append(f"{s['delta_map50']:+.2f} pp")
            md.append("| " + " | ".join(diff_cells) + " |")

        md.extend(
            [
                "",
                "## 3. Kết luận tự động (không hard-code)",
                "",
            ]
        )
        for line in results.get("conclusion", []):
            md.append(f"- {line}")

        md.extend(
            [
                "",
                "## 4. Giới hạn & lưu ý trung thực",
                "",
                "- Chỉ chạy **YOLO11s**. YOLOv8, YOLOv9 và RT-DETR bị bỏ qua để tiết kiệm GPU/VRAM.",
                "- Đây **không phải** tái lập đầy đủ Table 3/6 của paper; chỉ là benchmark demo có kiểm soát.",
                "- Precision/Recall lấy tại confidence 0.25. Ultralytics dùng match IoU mặc định 0.5 cho P/R; "
                "paper dùng 0.25. Không truyền 0.25 vào tham số NMS IoU để tránh gắn nhãn sai.",
                "- Nếu Δ mAP@50 âm hoặc gần 0, báo cáo vẫn giữ nguyên; không có kết luận tô hồng.",
                "- Chất lượng ảnh synthetic phụ thuộc Step 1/2/3 và số mask hợp lệ thực tế.",
                "- Seed cố định và split official-train-only; kết quả vẫn có thể dao động do phần cứng/thư viện.",
            ]
        )

        md_path.write_text("\n".join(md), encoding="utf-8")
        logger.info(f"[Reporter] Đã lưu Markdown report: {md_path}")

    def _print_console_summary(self, results: Dict):
        split_name = results.get("split_name", "id").upper()
        print("\n" + "=" * 110)
        print(f"KẾT QUẢ DOWNSTREAM BENCHMARK [{split_name}] — YOLO11s DEMO (không phải full paper reproduction)")
        print("=" * 110)
        print(
            f"{'Model':<10} | {'Precision (Base -> Aug | Δ)':<30} | "
            f"{'Recall (Base -> Aug | Δ)':<30} | {'mAP@50 (Base -> Aug | Δ | Gain)':<32}"
        )
        print("-" * 110)
        for model_name, comp in results.get("comparisons", {}).items():
            s = comp["summary"]
            p_str = (
                f"{s['precision_baseline']:.1f}% -> {s['precision_augmented']:.1f}% "
                f"({s['delta_precision']:+.1f}pp)"
            )
            r_str = (
                f"{s['recall_baseline']:.1f}% -> {s['recall_augmented']:.1f}% "
                f"({s['delta_recall']:+.1f}pp)"
            )
            map_str = (
                f"{s['map50_baseline']:.1f}% -> {s['map50_augmented']:.1f}% "
                f"({s['delta_map50']:+.1f}pp | {s['percent_gain']:+.1f}%)"
            )
            print(f"{model_name.upper():<10} | {p_str:<30} | {r_str:<30} | {map_str:<32}")
        print("=" * 110)
        for line in results.get("conclusion", []):
            print(f"- {line}")
        print(f"Báo cáo chi tiết: {self.output_dir / 'benchmark_report.md'}")
        print("=" * 110 + "\n")


# ─────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Step 4: Downstream Benchmark (resource-limited YOLO11s demo)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11s"],
        help="Chỉ yolo11/yolo11s được chạy; model khác bị bỏ qua.",
    )
    parser.add_argument(
        "--split",
        choices=["id", "npi"],
        default="id",
        help="Kịch bản đánh giá: 'id' hoặc 'npi'",
    )
    parser.add_argument(
        "--base_yaml",
        default=None,
        help="Dataset YAML baseline (mặc định: data/id_setup.yaml hoặc data/npi_setup.yaml)",
    )
    parser.add_argument(
        "--aug_yaml",
        default="outputs/pcb_synthesis/filtered/augmented_dataset.yaml",
        help="Dataset YAML augmented từ Step 3",
    )
    parser.add_argument("--output_dir", default="outputs/benchmark")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU (0.5-0.7). Không dùng 0.25 như match IoU của paper.",
    )
    parser.add_argument("--imgsz", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()
    if not args.base_yaml:
        args.base_yaml = "data/id_setup.yaml" if args.split == "id" else "data/npi_setup.yaml"

    benchmarker = DownstreamBenchmarker(
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        lr0=args.lr0,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        workers=args.workers,
    )
    benchmarker.run_benchmark(
        models=args.models,
        base_yaml=args.base_yaml,
        aug_yaml=args.aug_yaml,
        split_name=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
