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

import argparse
import json
import logging
import time
from pathlib import Path

from pcb_pipeline.step1_mask_generator import PCBMaskGenerator
from pcb_pipeline.step2_lora_inpainting import PCBDefectInpainter, LoRAInpaintTrainer
from pcb_pipeline.step3_quality_filter import (
    YOLO11DefectTrainer,
    YOLO11ConfidenceScorer,
    QualityFilter,
)

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
    "scratch",
    "missing_hole",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

DEFAULT_CONFIG = {
    # Step 1
    "dino_model":        "dinov2_vitb14",
    "mask_strategy":     "saliency",       # saliency | random | grid
    "n_proposals":       5,
    "images_per_defect": 1,
    # Step 2 — LoRA
    "lora_rank":         8,
    "lora_alpha":        16.0,
    "lora_epochs":       10,
    "lora_batch":        32,
    # Step 2 — Inference
    "inpaint_steps":     50,               # ~4 giây/ảnh
    "guidance_scale":    7.5,
    # Step 3
    "reject_ratio":      0.25,             # Loại 25%
    "yolo_size":         "n",
    "yolo_epochs":       50,
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
# STEP 1: Mask Generation
# ─────────────────────────────────────────────────────────
def run_step1(args, cfg: dict) -> list[dict]:
    logger.info("=" * 60)
    logger.info("STEP 1 — Tạo mask lỗi bằng DINOv2")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir) / "masks"
    gen = PCBMaskGenerator(
        dino_model=cfg["dino_model"],
        device=cfg["device"],
        strategy=cfg["mask_strategy"],
        n_proposals=cfg["n_proposals"],
    )
    records = gen.generate_batch(
        image_dir=args.image_dir,
        output_dir=output_dir,
        defect_types=args.defect_types or PCB_DEFECT_TYPES,
        images_per_defect=cfg["images_per_defect"],
    )
    logger.info(f"✅ Step 1 xong: {len(records)} mask records")
    return records


# ─────────────────────────────────────────────────────────
# STEP 2a: LoRA Training (nếu chưa có checkpoint)
# ─────────────────────────────────────────────────────────
def run_step2_train(args, cfg: dict):
    if not args.real_defect_dir:
        logger.warning("[Step 2 Train] Bỏ qua: không có --real_defect_dir")
        return

    logger.info("=" * 60)
    logger.info("STEP 2a — Fine-tune LoRA + Textual Inversion")
    logger.info(f"  rank={cfg['lora_rank']}  alpha={cfg['lora_alpha']}")
    logger.info(f"  epochs={cfg['lora_epochs']}  batch={cfg['lora_batch']}")
    logger.info("=" * 60)

    lora_out = Path(args.output_dir) / "lora_checkpoints"
    trainer = LoRAInpaintTrainer(
        model_id=args.model_id,
        rank=cfg["lora_rank"],
        alpha=cfg["lora_alpha"],
        device=cfg["device"],
        output_dir=str(lora_out),
    )

    # Dùng real defect dir làm data train (nếu có metadata sẵn thì dùng)
    real_meta = Path(args.real_defect_dir) / "train_metadata.json"
    if not real_meta.exists():
        logger.error(f"Không tìm thấy {real_meta}. Tạo file JSON với cấu trúc:")
        logger.error('  [{"image_path": "...", "mask_path": "...", "defect_type": "..."}]')
        return

    trainer.train(
        data_json=str(real_meta),
        epochs=cfg["lora_epochs"],
        batch_size=cfg["lora_batch"],
    )
    # Trả về đường dẫn checkpoint mới nhất
    return str(lora_out / f"epoch_{cfg['lora_epochs']:03d}")


# ─────────────────────────────────────────────────────────
# STEP 2b: Image Synthesis (Inpainting)
# ─────────────────────────────────────────────────────────
def run_step2_generate(mask_records: list[dict], args, cfg: dict) -> list[dict]:
    logger.info("=" * 60)
    logger.info("STEP 2b — Sinh ảnh lỗi PCB (SD Inpainting + LoRA)")
    logger.info(f"  steps={cfg['inpaint_steps']}  guidance={cfg['guidance_scale']}")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir) / "generated"
    lora_ckpt = getattr(args, "lora_ckpt", None)
    ti_ckpt = (Path(lora_ckpt) / "ti_embedding.pt") if lora_ckpt else None

    inpainter = PCBDefectInpainter(
        model_id=args.model_id,
        lora_ckpt=lora_ckpt,
        ti_ckpt=ti_ckpt,
        device=cfg["device"],
        num_inference_steps=cfg["inpaint_steps"],
        guidance_scale=cfg["guidance_scale"],
    )

    t0 = time.time()
    results = inpainter.generate_batch(mask_records, output_dir)
    elapsed = time.time() - t0
    per_img = elapsed / max(len(results), 1)
    logger.info(f"✅ Step 2b xong: {len(results)} ảnh | {per_img:.1f}s/ảnh")
    return results


# ─────────────────────────────────────────────────────────
# STEP 3: Quality Filtering
# ─────────────────────────────────────────────────────────
def run_step3(generated_records: list[dict], args, cfg: dict) -> list[dict]:
    logger.info("=" * 60)
    logger.info("STEP 3 — Lọc ảnh bằng YOLO11 (loại 25% thấp nhất)")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir) / "filtered"
    yolo_weights = getattr(args, "yolo_weights", None)

    # Train YOLO judge nếu chưa có weights
    if not yolo_weights and args.real_defect_dir:
        trainer = YOLO11DefectTrainer(
            model_size=cfg["yolo_size"],
            device=cfg["device"],
            project=str(Path(args.output_dir) / "yolo_training"),
        )
        yaml_path = YOLO11DefectTrainer.build_dataset_yaml(
            args.real_defect_dir,
            Path(args.output_dir) / "pcb_yolo_dataset.yaml",
        )
        yolo_weights = trainer.train(yaml_path, epochs=cfg["yolo_epochs"])

    if not yolo_weights:
        logger.error("[Step 3] Không có YOLO weights! Bỏ qua bước lọc.")
        return generated_records

    scorer = YOLO11ConfidenceScorer(yolo_weights, device=cfg["device"])
    flt = QualityFilter(scorer, reject_ratio=cfg["reject_ratio"])

    accepted, rejected = flt.filter(generated_records, output_dir)
    logger.info(f"✅ Step 3 xong: {len(accepted)} ảnh accepted / {len(rejected)} rejected")
    return accepted


# ─────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────
def run_full_pipeline(args, cfg: dict):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out)

    logger.info("🚀 Bắt đầu PCB Defect Synthesis Pipeline")
    logger.info(f"   Output: {out.resolve()}")
    t_start = time.time()

    # Step 1: Mask Generation
    mask_records = run_step1(args, cfg)

    # Step 2a: LoRA Training (nếu cần)
    if not getattr(args, "lora_ckpt", None) and getattr(args, "real_defect_dir", None):
        args.lora_ckpt = run_step2_train(args, cfg)

    # Step 2b: Image Synthesis
    gen_records = run_step2_generate(mask_records, args, cfg)

    # Step 3: Quality Filtering
    final_records = run_step3(gen_records, args, cfg)

    # Tổng kết
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("🎉 PIPELINE HOÀN TẤT")
    logger.info(f"   Tổng thời gian: {elapsed/60:.1f} phút")
    logger.info(f"   Ảnh sau lọc:   {len(final_records)}")
    logger.info(f"   Output:        {out.resolve()}")
    logger.info("=" * 60)

    # Lưu report
    report = {
        "total_masks":      len(mask_records),
        "total_generated":  len(gen_records),
        "total_accepted":   len(final_records),
        "rejected_ratio":   cfg["reject_ratio"],
        "elapsed_minutes":  round(elapsed / 60, 1),
        "config":           cfg,
    }
    (out / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return final_records


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

    # ── full ─────────────────────────────────────────────
    p_full = subparsers.add_parser("full", help="Chạy toàn bộ 3 bước pipeline")
    add_common(p_full)
    p_full.add_argument("--image_dir",       required=True,  help="Thư mục ảnh PCB không lỗi")
    p_full.add_argument("--real_defect_dir", default=None,   help="Thư mục ảnh lỗi thực (để train LoRA+YOLO)")
    p_full.add_argument("--lora_ckpt",       default=None,   help="Đường dẫn LoRA checkpoint (nếu đã train)")
    p_full.add_argument("--yolo_weights",    default=None,   help="best.pt của YOLO judge (nếu đã train)")
    p_full.add_argument("--images_per_defect", type=int, default=1)
    p_full.add_argument("--mask_strategy",   default="saliency", choices=["saliency","random","grid"])
    p_full.add_argument("--lora_rank",       type=int,   default=None)
    p_full.add_argument("--lora_alpha",      type=float, default=None)
    p_full.add_argument("--lora_epochs",     type=int,   default=None)
    p_full.add_argument("--lora_batch",      type=int,   default=None)
    p_full.add_argument("--inpaint_steps",   type=int,   default=None)
    p_full.add_argument("--reject_ratio",    type=float, default=None)

    # ── step1 ────────────────────────────────────────────
    p1 = subparsers.add_parser("step1", help="Chỉ chạy Step 1: Tạo mask")
    add_common(p1)
    p1.add_argument("--image_dir",           required=True)
    p1.add_argument("--images_per_defect",   type=int, default=1)
    p1.add_argument("--mask_strategy",       default="saliency", choices=["saliency","random","grid"])

    # ── step2_train ──────────────────────────────────────
    p2t = subparsers.add_parser("step2_train", help="Chỉ chạy Step 2a: Fine-tune LoRA")
    add_common(p2t)
    p2t.add_argument("--real_defect_dir",    required=True)
    p2t.add_argument("--lora_rank",          type=int,   default=None)
    p2t.add_argument("--lora_alpha",         type=float, default=None)
    p2t.add_argument("--lora_epochs",        type=int,   default=None)
    p2t.add_argument("--lora_batch",         type=int,   default=None)

    # ── step2_generate ───────────────────────────────────
    p2g = subparsers.add_parser("step2_generate", help="Chỉ chạy Step 2b: Sinh ảnh lỗi")
    add_common(p2g)
    p2g.add_argument("--mask_meta",          required=True, help="mask_metadata.json từ Step 1")
    p2g.add_argument("--lora_ckpt",          default=None)
    p2g.add_argument("--inpaint_steps",      type=int,   default=None)
    p2g.add_argument("--guidance",           type=float, default=None)

    # ── step3 ────────────────────────────────────────────
    p3 = subparsers.add_parser("step3", help="Chỉ chạy Step 3: Lọc ảnh YOLO11")
    add_common(p3)
    p3.add_argument("--gen_meta",            required=True, help="generated_metadata.json từ Step 2")
    p3.add_argument("--yolo_weights",        default=None)
    p3.add_argument("--real_defect_dir",     default=None)
    p3.add_argument("--reject_ratio",        type=float, default=None)
    p3.add_argument("--by_class",            action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(
        getattr(args, "config", None),
        {
            "device":          args.device,
            "mask_strategy":   getattr(args, "mask_strategy", None),
            "images_per_defect": getattr(args, "images_per_defect", None),
            "lora_rank":       getattr(args, "lora_rank", None),
            "lora_alpha":      getattr(args, "lora_alpha", None),
            "lora_epochs":     getattr(args, "lora_epochs", None),
            "lora_batch":      getattr(args, "lora_batch", None),
            "inpaint_steps":   getattr(args, "inpaint_steps", None),
            "guidance_scale":  getattr(args, "guidance", None),
            "reject_ratio":    getattr(args, "reject_ratio", None),
        },
    )

    if args.command == "full":
        run_full_pipeline(args, cfg)

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

        yolo_weights = getattr(args, "yolo_weights", None)
        if not yolo_weights and args.real_defect_dir:
            trainer = YOLO11DefectTrainer(device=cfg["device"])
            yaml_path = YOLO11DefectTrainer.build_dataset_yaml(
                args.real_defect_dir,
                Path(args.output_dir) / "pcb_yolo_dataset.yaml",
            )
            yolo_weights = trainer.train(yaml_path)

        scorer = YOLO11ConfidenceScorer(yolo_weights, device=cfg["device"])
        flt = QualityFilter(scorer, reject_ratio=cfg["reject_ratio"])

        if getattr(args, "by_class", False):
            flt.filter_by_class(gen_records, args.output_dir)
        else:
            flt.filter(gen_records, args.output_dir)


if __name__ == "__main__":
    main()
