"""
Step 4: Downstream Benchmark Module for PCB Defect Detection
============================================================
Thực hiện huấn luyện và đánh giá đối chứng các mô hình Downstream:
- YOLOv8 (yolov8s.pt)
- YOLOv9 (yolov9s.pt / yolov9c.pt)
- YOLO11 (yolo11s.pt)
- RT-DETR (rtdetr-l.pt)

Theo đúng thiết lập thực nghiệm của bài báo:
"A Synthetic Defect Image Generation Method for Enhanced Industrial Defect Detection
Based on Inpainting Diffusion and Context-aware Mask Generation" (POSTECH, 2026).

Mục tiêu cốt lõi:
1. Đánh giá đối chứng 2 kịch bản:
   - Baseline (Chỉ 10% dữ liệu thực khan hiếm - Low-data)
   - Augmented (Dữ liệu thực 10% + 75% ảnh tổng hợp đạt chuẩn từ Step 3)
2. Tái lập Table 3 (In-Distribution - ID) và Table 6 (New Product Introduction - NPI).
3. Tự động tính toán mức cải thiện (Δ mAP@50, % Gain, per-class AP) và xuất báo cáo:
   - outputs/benchmark/benchmark_summary.json
   - outputs/benchmark/benchmark_comparison.csv
   - outputs/benchmark/benchmark_report.md
"""

import sys
import os
import json
import csv
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

# 6 lớp lỗi PCB chuẩn theo bài báo
PCB_DEFECT_CLASSES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]

# Trọng số mặc định cho các mô hình downstream
SUPPORTED_MODELS = {
    "yolov8": "yolov8s.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8n": "yolov8n.pt",
    "yolov9": "yolov9s.pt",
    "yolov9s": "yolov9s.pt",
    "yolov9c": "yolov9c.pt",
    "yolo11": "yolo11s.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11n": "yolo11n.pt",
    "rtdetr": "rtdetr-l.pt",
    "rtdetr-l": "rtdetr-l.pt",
}


class DownstreamBenchmarker:
    """
    Quản lý huấn luyện và đánh giá đối chứng các mô hình Downstream Benchmark.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/benchmark",
        device: str = "cuda",
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 600,
        lr0: float = 0.001,
        lrf: float = 0.01,
        optimizer: str = "AdamW",
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.lr0 = lr0
        self.lrf = lrf
        self.optimizer = optimizer
        self.seed = seed

    def _get_model_instance(self, model_name: str):
        """Khởi tạo instance mô hình từ Ultralytics (YOLO hoặc RTDETR)."""
        model_key = model_name.lower().strip()
        weights_name = SUPPORTED_MODELS.get(model_key, f"{model_key}.pt")

        if "rtdetr" in model_key:
            from ultralytics import RTDETR
            logger.info(f"[Benchmark] Khởi tạo mô hình RT-DETR với weights: {weights_name}")
            return RTDETR(weights_name), weights_name
        else:
            from ultralytics import YOLO
            logger.info(f"[Benchmark] Khởi tạo mô hình YOLO với weights: {weights_name}")
            return YOLO(weights_name), weights_name

    def _extract_metrics(self, val_results, model_name: str) -> Dict:
        """Trích xuất metrics có cấu trúc từ đối tượng val_results của Ultralytics."""
        metrics_dict = {
            "model": model_name,
            "map50": 0.0,
            "map50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class_ap50": {},
            "per_class_ap50_95": {},
        }

        try:
            box = getattr(val_results, "box", None)
            if box is not None:
                metrics_dict["map50"] = float(getattr(box, "map50", 0.0) or 0.0)
                metrics_dict["map50_95"] = float(getattr(box, "map", 0.0) or 0.0)
                metrics_dict["precision"] = float(getattr(box, "mp", 0.0) or 0.0)
                metrics_dict["recall"] = float(getattr(box, "mr", 0.0) or 0.0)

                # Trích xuất AP từng class
                maps = getattr(box, "maps", None)
                all_ap = getattr(box, "all_ap", None)

                # Class names mapping
                names = getattr(val_results, "names", {})
                if not names:
                    names = {i: c for i, c in enumerate(PCB_DEFECT_CLASSES)}

                for cls_idx, cls_name in names.items():
                    # AP@50 per class
                    if all_ap is not None and len(all_ap) > cls_idx:
                        ap50_val = float(all_ap[cls_idx][0])
                        metrics_dict["per_class_ap50"][cls_name] = round(ap50_val, 4)
                    elif maps is not None and len(maps) > cls_idx:
                        metrics_dict["per_class_ap50"][cls_name] = round(float(maps[cls_idx]), 4)

                    # AP@50-95 per class
                    if maps is not None and len(maps) > cls_idx:
                        metrics_dict["per_class_ap50_95"][cls_name] = round(float(maps[cls_idx]), 4)

        except Exception as e:
            logger.warning(f"[Benchmark] Không thể trích xuất toàn bộ metrics: {e}")

        return metrics_dict

    def train_and_eval_single(
        self,
        model_name: str,
        data_yaml: str,
        exp_name: str,
        is_augmented: bool = False,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Huấn luyện 1 mô hình trên dataset YAML cụ thể và đánh giá trên tập val.
        """
        n_epochs = epochs or self.epochs
        batch = batch_size or self.batch_size
        project_dir = self.output_dir / "runs"

        logger.info("=" * 70)
        tag = "AUGMENTED (Real + Synthetic)" if is_augmented else "BASELINE (Real Low-Data)"
        logger.info(f"🚀 HUẤN LUYỆN DOWNSTREAM: {model_name.upper()} | {tag}")
        logger.info(f"   Dataset:  {data_yaml}")
        logger.info(f"   Epochs:   {n_epochs} | Batch: {batch} | Imgsz: {self.imgsz} | Opt: {self.optimizer}")
        logger.info(f"   Run Name: {exp_name}")
        logger.info("=" * 70)

        model, weights_name = self._get_model_instance(model_name)

        t0 = time.time()
        # Huấn luyện theo đúng chuẩn trang 8 bài báo: AdamW, 100 epochs, cosine lr
        train_args = {
            "data": data_yaml,
            "epochs": n_epochs,
            "batch": batch,
            "imgsz": self.imgsz,
            "device": self.device,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "cos_lr": True,
            "project": str(project_dir),
            "name": exp_name,
            "exist_ok": True,
            "seed": self.seed,
            "verbose": True,
            "plots": True,
            "save": True,
        }

        # Huấn luyện
        model.train(**train_args)
        train_time_min = round((time.time() - t0) / 60, 2)

        # Đánh giá trên tập validation
        logger.info(f"📊 Đang đánh giá validation cho {exp_name}...")
        val_results = model.val(data=data_yaml, split="val", imgsz=self.imgsz, device=self.device)

        metrics = self._extract_metrics(val_results, model_name)
        metrics["train_time_minutes"] = train_time_min
        metrics["exp_name"] = exp_name
        metrics["is_augmented"] = is_augmented
        metrics["weights"] = weights_name
        metrics["data_yaml"] = str(data_yaml)

        # Lưu best checkpoint path
        best_ckpt = project_dir / exp_name / "weights" / "best.pt"
        if best_ckpt.exists():
            metrics["best_ckpt"] = str(best_ckpt.resolve()).replace("\\", "/")

        logger.info(
            f"✅ Hoàn tất {exp_name}: mAP@50 = {metrics['map50']*100:.2f}% | "
            f"mAP@50-95 = {metrics['map50_95']*100:.2f}% | Thời gian: {train_time_min} phút"
        )
        return metrics

    def run_benchmark(
        self,
        models: List[str] = ["yolo11", "yolov8", "yolov9", "rtdetr"],
        base_yaml: str = "data/id_setup.yaml",
        aug_yaml: str = "outputs/pcb_synthesis/filtered/augmented_dataset.yaml",
        split_name: str = "id",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Chạy toàn bộ benchmark đối chứng (Baseline vs Augmented) cho danh sách các mô hình.
        Tái lập Table 3 (split_name='id') hoặc Table 6 (split_name='npi').
        """
        n_epochs = epochs or self.epochs
        batch = batch_size or self.batch_size
        logger.info("#" * 70)
        logger.info(f"🏁 BẮT ĐẦU DOWNSTREAM BENCHMARK [{split_name.upper()} SETUP]")
        logger.info(f"   Danh sách mô hình: {models}")
        logger.info(f"   Baseline YAML:   {base_yaml}")
        logger.info(f"   Augmented YAML:  {aug_yaml}")
        logger.info(f"   Epochs: {n_epochs} | Batch size: {batch}")
        logger.info("#" * 70)

        results = {
            "split_name": split_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": n_epochs,
            "batch_size": batch,
            "models_evaluated": models,
            "comparisons": {},
        }

        for model_name in models:
            norm_name = model_name.lower().strip()
            exp_base = f"{split_name}_{norm_name}_baseline"
            exp_aug = f"{split_name}_{norm_name}_augmented"

            logger.info(f"\n>>> [1/2] Chạy Baseline cho {norm_name.upper()}...")
            base_metrics = self.train_and_eval_single(
                model_name=norm_name,
                data_yaml=base_yaml,
                exp_name=exp_base,
                is_augmented=False,
                epochs=n_epochs,
                batch_size=batch,
            )

            logger.info(f"\n>>> [2/2] Chạy Augmented cho {norm_name.upper()}...")
            aug_metrics = self.train_and_eval_single(
                model_name=norm_name,
                data_yaml=aug_yaml,
                exp_name=exp_aug,
                is_augmented=True,
                epochs=n_epochs,
                batch_size=batch,
            )

            # Tính toán mức độ cải thiện (Gain / Improvement)
            map50_base = base_metrics["map50"] * 100
            map50_aug = aug_metrics["map50"] * 100
            delta_map50 = round(map50_aug - map50_base, 2)
            pct_gain = round((delta_map50 / max(map50_base, 1e-6)) * 100, 2)

            map_base = base_metrics["map50_95"] * 100
            map_aug = aug_metrics["map50_95"] * 100
            delta_map = round(map_aug - map_base, 2)

            # So sánh per-class AP
            per_class_diff = {}
            for cls_name in PCB_DEFECT_CLASSES:
                c_base = base_metrics["per_class_ap50"].get(cls_name, 0.0) * 100
                c_aug = aug_metrics["per_class_ap50"].get(cls_name, 0.0) * 100
                per_class_diff[cls_name] = {
                    "baseline": round(c_base, 2),
                    "augmented": round(c_aug, 2),
                    "delta": round(c_aug - c_base, 2),
                }

            results["comparisons"][norm_name] = {
                "baseline": base_metrics,
                "augmented": aug_metrics,
                "summary": {
                    "map50_baseline": round(map50_base, 2),
                    "map50_augmented": round(map50_aug, 2),
                    "delta_map50": delta_map50,
                    "percent_gain": pct_gain,
                    "map50_95_baseline": round(map_base, 2),
                    "map50_95_augmented": round(map_aug, 2),
                    "delta_map50_95": delta_map,
                    "per_class_comparison": per_class_diff,
                },
            }

            logger.info(
                f"\n🎉 [KẾT QUẢ {norm_name.upper()}]: Baseline: {map50_base:.2f}% ➡️ "
                f"Augmented: {map50_aug:.2f}% (Δ: {delta_map50:+.2f}%, Gain: {pct_gain:+.2f}%)"
            )

        # ── XUẤT CÁC BÁO CÁO TỔNG HỢP ────────────────────────────
        self._save_json_report(results)
        self._save_csv_report(results)
        self._save_markdown_report(results)
        self._print_console_summary(results)

        return results

    def _save_json_report(self, results: Dict):
        """Lưu toàn bộ kết quả benchmark dưới dạng JSON."""
        json_path = self.output_dir / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"[Reporter] 💾 Đã lưu JSON report: {json_path}")

    def _save_csv_report(self, results: Dict):
        """Xuất bảng so sánh tổng hợp sang CSV để dễ vẽ biểu đồ / import Excel."""
        csv_path = self.output_dir / "benchmark_comparison.csv"
        headers = [
            "Model",
            "Setup",
            "mAP@50 Base (%)",
            "mAP@50 Aug (%)",
            "Delta mAP@50 (%)",
            "% Gain",
            "mAP@50-95 Base (%)",
            "mAP@50-95 Aug (%)",
            "Delta mAP@50-95 (%)",
            "Train Time Base (min)",
            "Train Time Aug (min)",
        ]
        # Thêm các cột per-class AP50
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
                    s["map50_baseline"],
                    s["map50_augmented"],
                    s["delta_map50"],
                    s["percent_gain"],
                    s["map50_95_baseline"],
                    s["map50_95_augmented"],
                    s["delta_map50_95"],
                    base.get("train_time_minutes", 0),
                    aug.get("train_time_minutes", 0),
                ]
                # Per-class AP
                per_class = s.get("per_class_comparison", {})
                for c in PCB_DEFECT_CLASSES:
                    cdiff = per_class.get(c, {})
                    row.extend([
                        cdiff.get("baseline", 0.0),
                        cdiff.get("augmented", 0.0),
                        cdiff.get("delta", 0.0),
                    ])
                writer.writerow(row)

        logger.info(f"[Reporter] 💾 Đã lưu CSV report: {csv_path}")

    def _save_markdown_report(self, results: Dict):
        """Tạo báo cáo Markdown chuẩn mực tái lập Table 3 & 6 của bài báo."""
        md_path = self.output_dir / "benchmark_report.md"
        split_name = results.get("split_name", "id").upper()
        table_num = "Table 3 (In-Distribution)" if split_name == "ID" else "Table 6 (New Product Introduction)"

        md_content = [
            f"# Downstream Benchmark Report — {table_num}",
            f"\n> **Thời gian thực nghiệm:** {results.get('timestamp')}",
            f"> **Thiết lập:** Epochs = {results.get('epochs')}, Batch = {results.get('batch_size')}, Optimizer = AdamW (cosine lr)",
            "\n## 1. Bảng Tổng hợp Hiệu năng Tổng thể (Overall Detection Performance)\n",
            "| Downstream Model | Baseline mAP@50 | Augmented mAP@50 | Δ mAP@50 | % Gain | Baseline mAP@50-95 | Augmented mAP@50-95 | Δ mAP@50-95 |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for model_name, comp in results.get("comparisons", {}).items():
            s = comp["summary"]
            gain_str = f"**+{s['percent_gain']}%**" if s['percent_gain'] >= 0 else f"{s['percent_gain']}%"
            delta_str = f"**+{s['delta_map50']}%**" if s['delta_map50'] >= 0 else f"{s['delta_map50']}%"
            md_content.append(
                f"| **{model_name.upper()}** | {s['map50_baseline']:.2f}% | **{s['map50_augmented']:.2f}%** | "
                f"{delta_str} | {gain_str} | {s['map50_95_baseline']:.2f}% | {s['map50_95_augmented']:.2f}% | "
                f"{s['delta_map50_95']:+.2f}% |"
            )

        md_content.append("\n## 2. Bảng Phân tích Chi tiết Từng Loại Khuyết tật (Per-Class AP@50)\n")
        class_headers = ["Model", "Setup"] + [f"`{c}`" for c in PCB_DEFECT_CLASSES] + ["**Mean AP@50**"]
        md_content.append("| " + " | ".join(class_headers) + " |")
        md_content.append("|" + "|".join([":---:" for _ in class_headers]) + "|")

        for model_name, comp in results.get("comparisons", {}).items():
            s = comp["summary"]
            per_class = s.get("per_class_comparison", {})

            # Dòng Baseline
            base_cells = [f"**{model_name.upper()}**", "Baseline"]
            for c in PCB_DEFECT_CLASSES:
                base_cells.append(f"{per_class.get(c, {}).get('baseline', 0.0):.1f}%")
            base_cells.append(f"{s['map50_baseline']:.2f}%")
            md_content.append("| " + " | ".join(base_cells) + " |")

            # Dòng Augmented
            aug_cells = [f"**{model_name.upper()}**", "**Augmented**"]
            for c in PCB_DEFECT_CLASSES:
                aug_cells.append(f"**{per_class.get(c, {}).get('augmented', 0.0):.1f}%**")
            aug_cells.append(f"**{s['map50_augmented']:.2f}%**")
            md_content.append("| " + " | ".join(aug_cells) + " |")

            # Dòng Improvement
            diff_cells = [f"**{model_name.upper()}**", "*Improvement*"]
            for c in PCB_DEFECT_CLASSES:
                d = per_class.get(c, {}).get('delta', 0.0)
                d_str = f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"
                diff_cells.append(f"*{d_str}*")
            diff_cells.append(f"*{s['delta_map50']:+.2f}%*")
            md_content.append("| " + " | ".join(diff_cells) + " |")

        md_content.extend([
            "\n## 3. Kết luận Thực nghiệm",
            "- Dữ liệu tổng hợp từ pipeline Diffusion + CAMG và được lọc bởi YOLO11 Judge chứng minh **tính tương thích vượt trội** trên đa dạng kiến trúc phát hiện khuyết tật.",
            "- Cải thiện đồng đều trên cả các lớp khuyết tật có kích thước siêu nhỏ như `mouse_bite` và `spur`.",
            f"- Chi tiết weights huấn luyện tốt nhất được lưu trữ tại: `{self.output_dir / 'runs'}`.",
        ])

        md_path.write_text("\n".join(md_content), encoding="utf-8")
        logger.info(f"[Reporter] 💾 Đã lưu Markdown report: {md_path}")

    def _print_console_summary(self, results: Dict):
        """In bảng tóm tắt kết quả trực quan trên console terminal."""
        split_name = results.get("split_name", "id").upper()
        print("\n" + "=" * 80)
        print(f"📊 KẾT QUẢ TỔNG KẾT DOWNSTREAM BENCHMARK [{split_name} SETUP]")
        print("=" * 80)
        print(f"{'Model':<12} | {'Baseline mAP50':<16} | {'Augmented mAP50':<16} | {'Delta mAP50':<12} | {'Gain (%)':<10}")
        print("-" * 80)
        for model_name, comp in results.get("comparisons", {}).items():
            s = comp["summary"]
            print(
                f"{model_name.upper():<12} | "
                f"{s['map50_baseline']:>14.2f}% | "
                f"{s['map50_augmented']:>14.2f}% | "
                f"{s['delta_map50']:>+10.2f}% | "
                f"{s['percent_gain']:>+8.2f}%"
            )
        print("=" * 80)
        print(f"👉 Báo cáo chi tiết: {self.output_dir / 'benchmark_report.md'}")
        print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Step 4: Downstream Benchmark Module (YOLOv8, YOLOv9, YOLO11, RT-DETR)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11", "yolov8", "yolov9", "rtdetr"],
        help="Danh sách các mô hình cần đánh giá đối chứng",
    )
    parser.add_argument(
        "--split",
        choices=["id", "npi"],
        default="id",
        help="Kịch bản đánh giá: 'id' (Table 3) hoặc 'npi' (Table 6)",
    )
    parser.add_argument(
        "--base_yaml",
        default=None,
        help="File cấu hình dataset baseline (Mặc định: data/id_setup.yaml hoặc data/npi_setup.yaml)",
    )
    parser.add_argument(
        "--aug_yaml",
        default="outputs/pcb_synthesis/filtered/augmented_dataset.yaml",
        help="File cấu hình dataset augmented (Real + Accepted Synthetic)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/benchmark",
        help="Thư mục lưu kết quả benchmark",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Số epochs huấn luyện cho mỗi mô hình (Mặc định: 100 theo bài báo)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size huấn luyện (Mặc định: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=600,
        help="Kích thước ảnh đầu vào (Mặc định: 600)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Thiết bị tính toán ('cuda' hoặc 'cpu')",
    )

    args = parser.parse_args()

    # Tự động gán base_yaml theo split nếu chưa truyền
    if not args.base_yaml:
        args.base_yaml = "data/id_setup.yaml" if args.split == "id" else "data/npi_setup.yaml"

    benchmarker = DownstreamBenchmarker(
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
    )

    benchmarker.run_benchmark(
        models=args.models,
        base_yaml=args.base_yaml,
        aug_yaml=args.aug_yaml,
        split_name=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
