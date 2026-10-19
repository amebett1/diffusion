"""
PCB Defect Synthesis Pipeline — Main Orchestrator
===================================================
Kết hợp 3 bước thành một pipeline hoàn chỉnh:

  Step 1: Mask Generation    (DINOv2 + BBox)
  Step 2: Image Synthesis    (SD Inpainting + LoRA + TI)
  Step 3: Quality Filtering  (YOLO11 confidence scorer)

Cách dùng nhanh:
    python run_pipeline.py --help

Chạy toàn bộ pipeline:
    python run_pipeline.py full \\
        --image_dir data/pcb_normal \\
        --real_defect_dir data/pcb_real_defects \\
        --output_dir outputs/pcb_synthesis \\
        --lora_ckpt outputs/lora_checkpoints/epoch_010

Chạy từng bước riêng:
    python run_pipeline.py step1 --image_dir data/pcb_normal --output_dir outputs
    python run_pipeline.py step2 --mask_meta outputs/masks/mask_metadata.json ...
    python run_pipeline.py step3 --gen_meta outputs/generated/generated_metadata.json ...
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pcb_pipeline.step1_mask_generator import PCBMaskGenerator, CAMGGenerator, CAMGTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Shared config / defaults
# ─────────────────────────────────────────────────────────
PCB_DEFECT_TYPES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]

DEFAULT_CONFIG = {
    # Step 1
    "dino_model":        "dinov2_vitb14",
    "mask_strategy":     "saliency",       # saliency | random | grid
    "n_proposals":       5,
    "images_per_defect": 1,
    # Step 2 — LoRA & TI
    "lora_rank":         8,
    "lora_alpha":        16.0,
    "lora_epochs":       10,
    "lora_batch":        32,
    "lr_unet":           1e-4,
    "lr_text_encoder":   5e-5,
    "lr_ti":             5e-4,
    # Step 2 — Inference
    "inpaint_steps":     50,               # ~4 giây/ảnh
    "guidance_scale":    7.5,
    "negative_prompt":   "blurry, low quality, deformed, ugly, watermark, signature, text, logo",
    # Step 3
    "reject_ratio":      0.25,             # Loại 25%
    "yolo_size":         "s",              # YOLO11s chuẩn theo bài báo
    "yolo_epochs":       100,              # 100 epochs chuẩn theo trang 8 bài báo
    "device":            "cuda",
}


# ─────────────────────────────────────────────────────────
# Helper: load/merge config
# ─────────────────────────────────────────────────────────
def load_config(config_path: str | None, overrides: dict) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def save_config(cfg: dict, output_dir: Path):
    (output_dir / "pipeline_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────
# STEP 1: Mask Generation (CAMG)
# ─────────────────────────────────────────────────────────
def run_step1_train(args, cfg: dict) -> str:
    logger.info("=" * 60)
    logger.info("STEP 1a — Huấn luyện CAMG Module (Prediction Heads)")
    logger.info("=" * 60)
    meta_path = getattr(args, "metadata_path", "data/train_metadata.json")
    epochs = getattr(args, "epochs", 10)
    batch_size = getattr(args, "batch_size", 32)
    lr = getattr(args, "lr", 1e-4)

    trainer = CAMGTrainer(device=cfg["device"], lr=lr)
    ckpt = trainer.train(metadata_path=meta_path, epochs=epochs, batch_size=batch_size)
    logger.info(f"Huấn luyện CAMG hoàn tất -> {ckpt}")
    return ckpt


def run_step1(args, cfg: dict) -> list[dict]:
    logger.info("=" * 60)
    logger.info("STEP 1b — Sinh Mask lỗi bằng CAMG (Top-k Sampling)")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir) / "masks"
    weights_path = getattr(args, "camg_weights", None)
    k = getattr(args, "k", 5)

    meta_json = output_dir / "mask_metadata.json"
    if meta_json.exists():
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                cached_records = json.load(f)
            if cached_records and len(cached_records) > 0:
                logger.info(f"[CAMG] Tái sử dụng {len(cached_records)} masks đã sinh trước đó từ: {meta_json}")
                return cached_records
        except Exception as e:
            logger.warning(f"[CAMG] Không thể đọc mask metadata cũ: {e}. Đang sinh lại...")

    img_dir = Path(args.image_dir)
    if not img_dir.exists() or len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) == 0:
        raw_dir = Path("data/pcb-defect-dataset")
        if raw_dir.exists():
            logger.info(f"Không tìm thấy ảnh sạch trong '{img_dir}'. Tự động tạo 799 ảnh normal từ {raw_dir}...")
            from pcb_pipeline.data_module import PCBDataModule
            dm = PCBDataModule(raw_dataset_dir=raw_dir, output_dir="data")
            dm.run_pipeline()
        else:
            raise FileNotFoundError(
                f"Không tìm thấy thư mục ảnh sạch '{img_dir}' và cũng không tìm thấy '{raw_dir}'.\n"
                f"Hãy tải dataset và giải nén vào 'data/pcb-defect-dataset/', sau đó chạy 'python run_pipeline.py prepare_data'!"
            )

    gen = CAMGGenerator(
        weights_path=weights_path,
        device=cfg["device"],
        k=k,
    )
    records = gen.generate_batch(
        image_dir=args.image_dir,
        output_dir=output_dir,
        defect_types=args.defect_types or PCB_DEFECT_TYPES,
        images_per_defect=getattr(args, "images_per_defect", 1),
    )
    logger.info(f"Step 1 xong: {len(records)} mask records")
    return records


# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# STEP 2a: LoRA Training (nếu chưa có checkpoint)
# ─────────────────────────────────────────────────────────
def run_step2_train(args, cfg: dict):
    # Xác định đường dẫn train_metadata.json
    metadata_path = getattr(args, "metadata_path", None)
    if not metadata_path or not Path(metadata_path).exists():
        if getattr(args, "real_defect_dir", None):
            alt = Path(args.real_defect_dir) / "train_metadata.json"
            if alt.exists():
                metadata_path = str(alt)
        if not metadata_path or not Path(metadata_path).exists():
            default_meta = Path("data/train_metadata.json")
            if default_meta.exists():
                metadata_path = str(default_meta)

    if not metadata_path or not Path(metadata_path).exists():
        logger.warning("[Step 2 Train] Bỏ qua: không tìm thấy file train_metadata.json!")
        return None

    logger.info("=" * 60)
    logger.info("STEP 2a — Fine-tune LoRA (U-Net & Text Encoder) + Textual Inversion")
    logger.info(f"  rank={cfg['lora_rank']}  alpha={cfg['lora_alpha']}")
    logger.info(f"  epochs={cfg['lora_epochs']}  batch={cfg['lora_batch']}")
    logger.info(f"  lr_unet={cfg['lr_unet']}  lr_text_encoder={cfg['lr_text_encoder']}  lr_ti={cfg['lr_ti']}")
    logger.info(f"  metadata={metadata_path}")
    logger.info("=" * 60)

    lora_out = Path(args.output_dir) / "lora_checkpoints"
    from pcb_pipeline.step2_lora_inpainting import LoRAInpaintTrainer
    trainer = LoRAInpaintTrainer(
        model_id=args.model_id,
        rank=cfg["lora_rank"],
        alpha=cfg["lora_alpha"],
        lr_unet=cfg["lr_unet"],
        lr_text_encoder=cfg["lr_text_encoder"],
        lr_ti=cfg["lr_ti"],
        device=cfg["device"],
        output_dir=str(lora_out),
    )

    ckpt_path = trainer.train(
        data_json=metadata_path,
        epochs=cfg["lora_epochs"],
        batch_size=cfg["lora_batch"],
    )
    return ckpt_path


# ─────────────────────────────────────────────────────────
# STEP 2b: Image Synthesis (Inpainting)
# ─────────────────────────────────────────────────────────
def run_step2_generate(mask_records: list[dict], args, cfg: dict) -> list[dict]:
    from pcb_pipeline.step2_lora_inpainting import PCBDefectInpainter

    logger.info("=" * 60)
    logger.info("STEP 2b — Sinh ảnh lỗi PCB (SD Inpainting + LoRA)")
    logger.info(f"  steps={cfg['inpaint_steps']}  guidance={cfg['guidance_scale']}")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir) / "generated"
    lora_ckpt = getattr(args, "lora_ckpt", None)

    # Tìm file ti_ckpt từ args hoặc trong thư mục lora_ckpt
    ti_ckpt = getattr(args, "ti_ckpt", None)
    if not ti_ckpt and lora_ckpt:
        for name in ["ti_embeddings.pt", "ti_embedding.pt"]:
            p = Path(lora_ckpt) / name
            if p.exists():
                ti_ckpt = str(p)
                break

    inpainter = PCBDefectInpainter(
        model_id=args.model_id,
        lora_ckpt=lora_ckpt,
        ti_ckpt=ti_ckpt,
        device=cfg["device"],
        num_inference_steps=cfg["inpaint_steps"],
        guidance_scale=cfg["guidance_scale"],
        negative_prompt=cfg.get("negative_prompt", "blurry, low quality, deformed, ugly, watermark, signature, text, logo"),
    )

    t0 = time.time()
    results = inpainter.generate_batch(mask_records, output_dir)
    elapsed = time.time() - t0
    per_img = elapsed / max(len(results), 1)
    logger.info(f"Step 2b xong: {len(results)} ảnh | {per_img:.1f}s/ảnh")
    return results


# ─────────────────────────────────────────────────────────
# STEP 3: Quality Filtering
# ─────────────────────────────────────────────────────────
def run_step3(generated_records: list[dict], args, cfg: dict) -> list[dict]:
    from pcb_pipeline.step3_quality_filter import (
        YOLO11DefectTrainer,
        YOLO11ConfidenceScorer,
        QualityFilter,
    )

    logger.info("=" * 60)
    logger.info("STEP 3 — Lọc ảnh bằng YOLO11 (loại 25% thấp nhất)")
    logger.info("=" * 60)

    out_base = Path(args.output_dir)
    output_dir = out_base if out_base.name == "filtered" else out_base / "filtered"
    yolo_weights = getattr(args, "yolo_weights", None)
    data_yaml = getattr(args, "data_yaml", None) or "data/id_setup.yaml"

    # Train YOLO judge nếu chưa có weights
    if not yolo_weights and Path(data_yaml).exists():
        trainer = YOLO11DefectTrainer(
            model_size=cfg.get("yolo_size", "s"),
            device=cfg["device"],
            project=str(Path(args.output_dir) / "yolo_training"),
        )
        yolo_weights = trainer.train(
            data_yaml,
            epochs=cfg.get("yolo_epochs", 100),
        )

    scorer = YOLO11ConfidenceScorer(weights_path=yolo_weights, device=cfg["device"])
    flt = QualityFilter(scorer, reject_ratio=cfg.get("reject_ratio", 0.25))

    if getattr(args, "by_class", False):
        accepted, rejected = flt.filter_by_class(generated_records, output_dir, data_yaml=data_yaml)
    else:
        accepted, rejected = flt.filter(generated_records, output_dir, data_yaml=data_yaml)

    logger.info(f"Step 3 xong: {len(accepted)} ảnh accepted / {len(rejected)} rejected")
    aug_yaml = output_dir / "augmented_dataset.yaml"
    if aug_yaml.exists():
        logger.info(f"   Augmented Dataset YAML: {aug_yaml.resolve()}")
    return accepted


# ─────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────
def run_full_pipeline(args, cfg: dict):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out)

    logger.info("Bắt đầu PCB Defect Synthesis Pipeline")
    logger.info(f"   Output: {out.resolve()}")
    t_start = time.time()

    # Tự động kiểm tra ảnh sạch pcb_normal
    img_dir = Path(args.image_dir)
    if not img_dir.exists() or len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) == 0:
        raw_dir = Path("data/pcb-defect-dataset")
        if raw_dir.exists():
            logger.info(f"Chưa tìm thấy ảnh sạch trong '{img_dir}'. Đang tự động tạo 799 ảnh normal theo chuẩn bài báo...")
            from pcb_pipeline.data_module import PCBDataModule
            dm = PCBDataModule(raw_dataset_dir=raw_dir, output_dir="data")
            dm.run_pipeline()
        else:
            raise FileNotFoundError(
                f"Không tìm thấy thư mục '{img_dir}' và cũng không tìm thấy '{raw_dir}'.\n"
                f"Hãy tải dataset và giải nén vào 'data/pcb-defect-dataset/', sau đó chạy 'python run_pipeline.py prepare_data'!"
            )

    # Step 1: Mask Generation
    mask_records = run_step1(args, cfg)

    # Step 2a: LoRA Training (nếu cần)
    if not getattr(args, "lora_ckpt", None) and getattr(args, "real_defect_dir", None):
        args.lora_ckpt = run_step2_train(args, cfg)

    # Step 2b: Image Synthesis
    gen_records = run_step2_generate(mask_records, args, cfg)

    # Step 3: Quality Filtering
    final_records = run_step3(gen_records, args, cfg)

    # Step 4: Downstream Benchmark (nếu yêu cầu qua cờ --run_benchmark)
    benchmark_results = None
    aug_yaml_path = out / "filtered" / "augmented_dataset.yaml"
    if getattr(args, "run_benchmark", False):
        setattr(args, "models", getattr(args, "benchmark_models", ["yolo11"]))
        setattr(args, "epochs", getattr(args, "benchmark_epochs", 100))
        setattr(args, "aug_yaml", str(aug_yaml_path))
        benchmark_results = run_step4(args, cfg)

    # Tổng kết
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("PIPELINE HOÀN TẤT")
    logger.info(f"   Tổng thời gian: {elapsed/60:.1f} phút")
    logger.info(f"   Ảnh sau lọc:   {len(final_records)}")
    if aug_yaml_path.exists():
        logger.info(f"   Augmented YAML: {aug_yaml_path.resolve()}")
    if benchmark_results:
        logger.info(f"   Benchmark Dir:  {(out / 'benchmark').resolve()}")
    logger.info(f"   Output:        {out.resolve()}")
    logger.info("=" * 60)

    # Lưu report
    report = {
        "total_masks":        len(mask_records),
        "total_generated":    len(gen_records),
        "total_accepted":     len(final_records),
        "rejected_ratio":     cfg["reject_ratio"],
        "augmented_yaml":     str(aug_yaml_path.resolve()) if aug_yaml_path.exists() else None,
        "benchmark_executed": bool(benchmark_results),
        "elapsed_minutes":    round(elapsed / 60, 1),
        "config":             cfg,
    }
    (out / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return final_records


# ─────────────────────────────────────────────────────────
# STEP 4: Downstream Benchmark
# ─────────────────────────────────────────────────────────
def run_step4(args, cfg: dict) -> dict:
    from pcb_pipeline.step4_benchmark import DownstreamBenchmarker

    logger.info("=" * 60)
    logger.info("STEP 4 — Downstream Benchmark (YOLO11)")
    logger.info("=" * 60)

    out_base = Path(args.output_dir)
    bench_out = out_base if out_base.name == "benchmark" else out_base / "benchmark"
    split_name = getattr(args, "split", "id")
    base_yaml = getattr(args, "base_yaml", None)
    if not base_yaml:
        base_yaml = "data/id_setup.yaml" if split_name == "id" else "data/npi_setup.yaml"

    aug_yaml = getattr(args, "aug_yaml", None)
    if not aug_yaml:
        candidate = out_base / "filtered" / "augmented_dataset.yaml"
        aug_yaml = str(candidate) if candidate.exists() else "outputs/pcb_synthesis/filtered/augmented_dataset.yaml"

    models = getattr(args, "models", None) or ["yolo11"]
    epochs = getattr(args, "epochs", None) or cfg.get("benchmark_epochs", 100)
    batch_size = getattr(args, "batch_size", None) or cfg.get("benchmark_batch", 8)
    lr0 = getattr(args, "lr0", None) or cfg.get("benchmark_lr0", 1e-4)
    conf = getattr(args, "conf", None) or cfg.get("benchmark_conf", 0.25)
    iou = getattr(args, "iou", None) or cfg.get("benchmark_iou", 0.25)
    imgsz = getattr(args, "imgsz", 600)

    benchmarker = DownstreamBenchmarker(
        output_dir=bench_out,
        device=cfg["device"],
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        conf_thresh=conf,
        iou_thresh=iou,
    )
    results = benchmarker.run_benchmark(
        models=models,
        base_yaml=base_yaml,
        aug_yaml=aug_yaml,
        split_name=split_name,
        epochs=epochs,
        batch_size=batch_size,
    )
    return results


def run_prepare_data(args, cfg: dict):
    from pcb_pipeline.data_module import PCBDataModule
    from setup_env_paths import adapt_paths

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dataset gốc: '{raw_dir}'.\n"
            f"Vui lòng tải dataset và giải nén vào đúng thư mục '{raw_dir}' rồi thử lại!"
        )
    logger.info("=" * 60)
    logger.info("PREPARE DATA — Phân chia ID/NPI splits & Tạo 799 ảnh phôi sạch pcb_normal")
    logger.info("=" * 60)
    dm = PCBDataModule(raw_dataset_dir=raw_dir, output_dir=out_dir)
    dm.run_pipeline()
    adapt_paths(mode="relative")
    logger.info("Dữ liệu đã sẵn sàng để chạy pipeline!")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PCB Defect Synthesis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Shared args helper ───────────────────────────────
    def add_common(p):
        p.add_argument("--output_dir",       default="outputs/pcb_synthesis")
        p.add_argument("--model_id",         default="runwayml/stable-diffusion-inpainting")
        p.add_argument("--device",           default="cuda")
        p.add_argument("--config",           default=None, help="JSON config file")
        p.add_argument("--defect_types",     nargs="+", default=None)

    # ── prepare_data ─────────────────────────────────────
    p_prep = subparsers.add_parser("prepare_data", help="Tiền xử lý dataset: tạo ID/NPI splits và sinh 799 ảnh phôi sạch pcb_normal")
    p_prep.add_argument("--raw_dir",    default="data/pcb-defect-dataset", help="Thư mục dataset gốc đã tải về")
    p_prep.add_argument("--output_dir", default="data", help="Thư mục lưu cấu hình và ảnh normal")

    # ── full ─────────────────────────────────────────────
    p_full = subparsers.add_parser("full", help="Chạy toàn bộ pipeline")
    add_common(p_full)
    p_full.add_argument("--image_dir",       required=True,  help="Thư mục ảnh PCB không lỗi")
    p_full.add_argument("--real_defect_dir", default=None,   help="Thư mục ảnh lỗi thực (để train LoRA+YOLO)")
    p_full.add_argument("--metadata_path",   default="data/train_metadata.json", help="Đường dẫn train_metadata.json")
    p_full.add_argument("--camg_weights",    default=None,   help="Đường dẫn trọng số camg_best.pth")
    p_full.add_argument("--k",               type=int, default=5, help="Top-k sampling cho CAMG")
    p_full.add_argument("--images_per_defect", type=int, default=1)
    p_full.add_argument("--mask_strategy",   default="saliency", choices=["saliency","random","grid"])
    p_full.add_argument("--lora_ckpt",       default=None,   help="Đường dẫn LoRA checkpoint (nếu đã train)")
    p_full.add_argument("--ti_ckpt",         default=None,   help="Đường dẫn ti_embeddings.pt (nếu đã train)")
    p_full.add_argument("--lora_rank",       type=int,   default=None)
    p_full.add_argument("--lora_alpha",      type=float, default=None)
    p_full.add_argument("--lora_epochs",     type=int,   default=None)
    p_full.add_argument("--lora_batch",      type=int,   default=None)
    p_full.add_argument("--lr_unet",         type=float, default=None)
    p_full.add_argument("--lr_text_encoder", type=float, default=None)
    p_full.add_argument("--lr_ti",           type=float, default=None, help="Learning rate cho Textual Inversion")
    p_full.add_argument("--inpaint_steps",   type=int,   default=None)
    p_full.add_argument("--guidance",        type=float, default=None)
    p_full.add_argument("--negative_prompt", default=None)
    p_full.add_argument("--yolo_weights",    default=None,   help="best.pt của YOLO judge (nếu đã train)")
    p_full.add_argument("--data_yaml",       default="data/id_setup.yaml", help="File cấu hình dataset lỗi thực (ID 10%% hoặc NPI)")
    p_full.add_argument("--yolo_size",       default=None, choices=["n", "s", "m", "l", "x"], help="Kích thước mô hình YOLO11 (mặc định 's' theo bài báo)")
    p_full.add_argument("--yolo_epochs",     type=int, default=None, help="Số epochs huấn luyện YOLO judge (100 theo bài báo)")
    p_full.add_argument("--reject_ratio",    type=float, default=None)
    p_full.add_argument("--by_class",        action="store_true", help="Lọc độc lập theo từng loại lỗi để cân bằng class")
    p_full.add_argument("--run_benchmark",    action="store_true", help="Tự động chạy Step 4 Benchmark sau khi hoàn tất Step 3")
    p_full.add_argument("--benchmark_models", nargs="+", default=["yolo11"], help="Danh sách mô hình benchmark khi chạy full (mặc định: yolo11)")
    p_full.add_argument("--benchmark_epochs", type=int, default=100, help="Số epochs benchmark cho mỗi model (mặc định 100 theo bài báo)")

    # ── step1_train ──────────────────────────────────────
    p1t = subparsers.add_parser("step1_train", help="Huấn luyện prediction heads của CAMG")
    add_common(p1t)
    p1t.add_argument("--metadata_path",      default="data/train_metadata.json", help="train_metadata.json từ Data Module")
    p1t.add_argument("--epochs",             type=int,   default=10)
    p1t.add_argument("--batch_size",         type=int,   default=32)
    p1t.add_argument("--lr",                 type=float, default=1e-4)

    # ── step1 ────────────────────────────────────────────
    p1 = subparsers.add_parser("step1", help="Chỉ chạy Step 1: Tạo mask bằng CAMG")
    add_common(p1)
    p1.add_argument("--image_dir",           required=True)
    p1.add_argument("--camg_weights",        default=None, help="Đường dẫn trọng số camg_best.pth")
    p1.add_argument("--k",                   type=int, default=5, help="Top-k sampling (mặc định k=5)")
    p1.add_argument("--images_per_defect",   type=int, default=1)

    # ── step2_train ──────────────────────────────────────
    p2t = subparsers.add_parser("step2_train", help="Chỉ chạy Step 2a: Fine-tune LoRA")
    add_common(p2t)
    p2t.add_argument("--real_defect_dir",    default=None, help="Thư mục ảnh lỗi thực")
    p2t.add_argument("--metadata_path",      default="data/train_metadata.json", help="train_metadata.json từ Data Module")
    p2t.add_argument("--lora_rank",          type=int,   default=None)
    p2t.add_argument("--lora_alpha",         type=float, default=None)
    p2t.add_argument("--lora_epochs",        type=int,   default=None)
    p2t.add_argument("--lora_batch",         type=int,   default=None)
    p2t.add_argument("--lr_unet",            type=float, default=None)
    p2t.add_argument("--lr_text_encoder",    type=float, default=None)
    p2t.add_argument("--lr_ti",              type=float, default=None, help="Learning rate cho Textual Inversion")

    # ── step2_generate ───────────────────────────────────
    p2g = subparsers.add_parser("step2_generate", help="Chỉ chạy Step 2b: Sinh ảnh lỗi")
    add_common(p2g)
    p2g.add_argument("--mask_meta",          required=True, help="mask_metadata.json từ Step 1")
    p2g.add_argument("--lora_ckpt",          default=None, help="Thư mục chứa LoRA checkpoint")
    p2g.add_argument("--ti_ckpt",            default=None, help="File ti_embeddings.pt")
    p2g.add_argument("--inpaint_steps",      type=int,   default=None)
    p2g.add_argument("--guidance",           type=float, default=None)
    p2g.add_argument("--negative_prompt",    default=None)

    # ── step3 ────────────────────────────────────────────
    p3 = subparsers.add_parser("step3", help="Chỉ chạy Step 3: Lọc ảnh YOLO11")
    add_common(p3)
    p3.add_argument("--gen_meta",            required=True, help="generated_metadata.json từ Step 2")
    p3.add_argument("--yolo_weights",        default=None, help="File best.pt nếu đã train trước")
    p3.add_argument("--data_yaml",           default="data/id_setup.yaml", help="Dataset YAML gốc (ID 10%%)")
    p3.add_argument("--real_defect_dir",     default=None)
    p3.add_argument("--yolo_size",           default=None, choices=["n", "s", "m", "l", "x"], help="Kích thước mô hình YOLO11 (mặc định 's' theo bài báo)")
    p3.add_argument("--yolo_epochs",         type=int, default=None, help="Số epochs huấn luyện YOLO judge (mặc định 100)")
    p3.add_argument("--reject_ratio",        type=float, default=None)
    p3.add_argument("--by_class",            action="store_true")

    # ── step4 ────────────────────────────────────────────
    p4 = subparsers.add_parser("step4", help="Chỉ chạy Step 4: Downstream Benchmark (mặc định: yolo11)")
    add_common(p4)
    p4.add_argument("--models",              nargs="+", default=["yolo11"], help="Danh sách mô hình benchmark (mặc định: yolo11)")
    p4.add_argument("--split",               choices=["id", "npi"], default="id", help="Kịch bản đánh giá: 'id' (Table 3) hoặc 'npi' (Table 6)")
    p4.add_argument("--base_yaml",           default=None, help="Dataset YAML baseline (data/id_setup.yaml hoặc data/npi_setup.yaml)")
    p4.add_argument("--aug_yaml",            default=None, help="Dataset YAML augmented từ Step 3")
    p4.add_argument("--epochs",              type=int, default=100, help="Số epochs huấn luyện (mặc định 100 theo bài báo)")
    p4.add_argument("--batch_size",          type=int, default=8, help="Batch size huấn luyện (mặc định 8 theo bài báo)")
    p4.add_argument("--lr0",                 type=float, default=1e-4, help="Learning rate AdamW (mặc định 1e-4 theo bài báo)")
    p4.add_argument("--conf",                type=float, default=0.25, help="Confidence threshold đánh giá Precision/Recall (mặc định 0.25 theo bài báo)")
    p4.add_argument("--iou",                 type=float, default=0.25, help="IoU threshold đánh giá Precision/Recall (mặc định 0.25 theo bài báo)")
    p4.add_argument("--imgsz",               type=int, default=600, help="Kích thước ảnh đầu vào (mặc định 600)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(
        getattr(args, "config", None),
        {
            "device":            getattr(args, "device", "cuda"),
            "mask_strategy":     getattr(args, "mask_strategy", None),
            "images_per_defect": getattr(args, "images_per_defect", None),
            "lora_rank":         getattr(args, "lora_rank", None),
            "lora_alpha":        getattr(args, "lora_alpha", None),
            "lora_epochs":       getattr(args, "lora_epochs", None),
            "lora_batch":        getattr(args, "lora_batch", None),
            "lr_unet":           getattr(args, "lr_unet", None),
            "lr_text_encoder":   getattr(args, "lr_text_encoder", None),
            "lr_ti":             getattr(args, "lr_ti", None),
            "inpaint_steps":     getattr(args, "inpaint_steps", None),
            "guidance_scale":    getattr(args, "guidance", None),
            "negative_prompt":   getattr(args, "negative_prompt", None),
            "reject_ratio":      getattr(args, "reject_ratio", None),
            "yolo_size":         getattr(args, "yolo_size", None),
            "yolo_epochs":       getattr(args, "yolo_epochs", None),
            "benchmark_epochs":  getattr(args, "benchmark_epochs", None),
        },
    )

    if args.command == "full":
        run_full_pipeline(args, cfg)

    elif args.command == "step1_train":
        run_step1_train(args, cfg)

    elif args.command == "step1":
        run_step1(args, cfg)

    elif args.command == "step2_train":
        run_step2_train(args, cfg)

    elif args.command == "step2_generate":
        with open(args.mask_meta, encoding="utf-8") as f:
            mask_records = json.load(f)
        run_step2_generate(mask_records, args, cfg)

    elif args.command == "step3":
        with open(args.gen_meta, encoding="utf-8") as f:
            gen_records = json.load(f)
        run_step3(gen_records, args, cfg)

    elif args.command == "step4":
        run_step4(args, cfg)

    elif args.command == "prepare_data":
        run_prepare_data(args, cfg)


if __name__ == "__main__":
    main()
