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
    "num_samples":       829,              # Chuẩn bài báo: 829 ảnh lỗi tổng hợp cho PCB (Trang 6 Col 2)
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
# GPU Detection & Status Logging
# ─────────────────────────────────────────────────────────
def log_gpu_info():
    """Kiểm tra và ghi log cấu hình GPU khả dụng (hỗ trợ Dual GPU Kaggle 2x T4)."""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            logger.info("=" * 70)
            logger.info(f"PHÁT HIỆN HỆ THỐNG GPU CUDA: {count} GPU khả dụng")
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                logger.info(f"   [GPU {i}] {props.name} | VRAM: {vram_gb:.1f} GB")
            if count >= 2:
                logger.info("   -> Phát hiện Dual GPU (Kaggle 2x T4): Sẵn sàng tăng tốc huấn luyện & sinh ảnh!")
            logger.info("=" * 70)
        else:
            logger.warning("Không tìm thấy GPU CUDA. Pipeline sẽ chạy trên CPU.")
    except Exception as e:
        logger.debug(f"Không thể kiểm tra GPU: {e}")


# ─────────────────────────────────────────────────────────
# Pipeline Checkpoint & Modular State Manager
# ─────────────────────────────────────────────────────────
class PipelineCheckpointManager:
    """Quản lý tiến trình, lưu checkpoint trạng thái và hỗ trợ khôi phục (--resume)."""

    STAGE_DEFS = [
        ("0",  "Data Prep & 799 Normals", "data/id_setup.yaml"),
        ("1",  "CAMG Mask Generation",    "outputs/pcb_synthesis/masks/mask_metadata.json"),
        ("2a", "LoRA Fine-tuning",        "outputs/pcb_synthesis/lora_checkpoints"),
        ("2b", "Diffusion Inpainting",    "outputs/pcb_synthesis/generated/generated_metadata.json"),
        ("3",  "Quality Filtering",       "outputs/pcb_synthesis/filtered/augmented_dataset.yaml"),
        ("4",  "Downstream Benchmark",    "outputs/pcb_synthesis/benchmark/benchmark_report.md"),
    ]

    STAGE_KEYS = [s[0] for s in STAGE_DEFS]

    ALIASES = {
        "0": "0", "prep": "0", "data": "0", "prepare_data": "0",
        "1": "1", "mask": "1", "masks": "1", "camg": "1", "step1": "1",
        "2a": "2a", "lora": "2a", "lora_train": "2a", "step2a": "2a",
        "2b": "2b", "inpaint": "2b", "generate": "2b", "step2b": "2b",
        "3": "3", "filter": "3", "quality_filter": "3", "step3": "3",
        "4": "4", "benchmark": "4", "bench": "4", "step4": "4",
    }

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.status_file = self.output_dir / "pipeline_status.json"
        self.status_data = self._load()

    def _load(self) -> dict:
        if self.status_file.exists():
            try:
                with open(self.status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"updated_at": None, "stages": {}}

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(self.status_data, f, indent=2, ensure_ascii=False)

    def is_completed(self, stage_id: str, artifact_path: str | Path | None = None) -> bool:
        stages = self.status_data.get("stages", {})
        info = stages.get(stage_id, {})
        chk = artifact_path or info.get("target_artifact")
        if chk:
            p = Path(chk)
            if p.exists():
                if p.is_dir():
                    if any(p.iterdir()):
                        return True
                elif p.stat().st_size > 0:
                    return True
        return info.get("status") == "COMPLETED"

    def mark_stage(self, stage_id: str, status: str, elapsed: float = 0.0, artifact: str | None = None, extra: dict | None = None):
        if "stages" not in self.status_data:
            self.status_data["stages"] = {}
        stage_entry = self.status_data["stages"].get(stage_id, {})
        stage_entry.update({
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_artifact": str(artifact) if artifact else stage_entry.get("target_artifact"),
        })
        if extra:
            stage_entry.update(extra)
        self.status_data["stages"][stage_id] = stage_entry
        self.save()

    def should_execute(self, stage_id: str, from_stage: str | None, to_stage: str | None) -> bool:
        sid = self.ALIASES.get(str(stage_id).lower(), stage_id)
        if sid not in self.STAGE_KEYS:
            return True
        idx = self.STAGE_KEYS.index(sid)

        if from_stage:
            f_id = self.ALIASES.get(str(from_stage).lower(), from_stage)
            if f_id in self.STAGE_KEYS:
                if idx < self.STAGE_KEYS.index(f_id):
                    return False

        if to_stage:
            t_id = self.ALIASES.get(str(to_stage).lower(), to_stage)
            if t_id in self.STAGE_KEYS:
                if idx > self.STAGE_KEYS.index(t_id):
                    return False

        return True

    def render_dashboard(self) -> str:
        lines = []
        bar = "=" * 82
        sep = "-" * 82
        lines.append(bar)
        lines.append(f"{'PCB DEFECT SYNTHESIS PIPELINE EXECUTION DASHBOARD':^82}")
        lines.append(bar)
        header = f"{'Stage':<6} {'Tên công đoạn':<25} {'Trạng thái':<14} {'Thời gian':<12} {'Output / Artifact':<22}"
        lines.append(header)
        lines.append(sep)

        for sid, name, default_art in self.STAGE_DEFS:
            stage_entry = self.status_data.get("stages", {}).get(sid, {})
            status = stage_entry.get("status", "PENDING")
            elapsed_sec = stage_entry.get("elapsed_seconds", None)
            art = stage_entry.get("target_artifact", default_art)
            art_name = Path(art).name if art else "-"

            if status == "COMPLETED":
                status_str = "[HOÀN TẤT]"
            elif status == "RUNNING":
                status_str = "[ĐANG CHẠY...]"
            elif status == "SKIPPED":
                status_str = "[BỎ QUA]"
            else:
                status_str = "[CHƯA CHẠY]"

            if elapsed_sec is not None and elapsed_sec > 0:
                if elapsed_sec >= 3600:
                    time_str = f"{int(elapsed_sec//3600)}h {int((elapsed_sec%3600)//60)}m"
                elif elapsed_sec >= 60:
                    time_str = f"{int(elapsed_sec//60)}m {int(elapsed_sec%60)}s"
                else:
                    time_str = f"{elapsed_sec:.1f}s"
            else:
                time_str = "--"

            lines.append(f"{sid:<6} {name:<25} {status_str:<14} {time_str:<12} {art_name:<22}")

        lines.append(bar)
        return "\n".join(lines)


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
    import random
    from pcb_pipeline.step2_lora_inpainting import PCBDefectInpainter

    num_samples = getattr(args, "num_samples", None)
    if num_samples is None:
        num_samples = cfg.get("num_samples", 829)

    # Lấy mẫu cân bằng theo từng loại lỗi chuẩn theo bài báo (Trang 6 Cột 2: 829 ảnh)
    if num_samples and len(mask_records) > num_samples:
        logger.info(f"[Step 2b] Lấy mẫu cân bằng {num_samples} ảnh từ {len(mask_records)} masks (chuẩn thực nghiệm bài báo)...")
        by_defect = {}
        for r in mask_records:
            by_defect.setdefault(r["defect_type"], []).append(r)

        selected = []
        n_types = len(by_defect)
        base_per_type = num_samples // n_types
        remainder = num_samples % n_types

        rng = random.Random(42)  # Cố định seed 42 để đảm bảo tính tái lập (reproducibility)
        for idx, (dtype, records) in enumerate(sorted(by_defect.items())):
            quota = base_per_type + (1 if idx < remainder else 0)
            shuffled = list(records)
            rng.shuffle(shuffled)
            selected.extend(shuffled[:quota])

        logger.info(f"   -> Đã chọn chính xác {len(selected)} masks (cân bằng ~{base_per_type} ảnh/class cho 6 loại lỗi).")
        mask_records = selected

    logger.info("=" * 60)
    logger.info(f"STEP 2b — Sinh {len(mask_records)} ảnh lỗi PCB (SD Inpainting + LoRA)")
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

    manager = PipelineCheckpointManager(out)
    log_gpu_info()

    logger.info("Bắt đầu PCB Defect Synthesis Pipeline")
    logger.info(f"   Thư mục Output: {out.resolve()}")
    logger.info("\n" + manager.render_dashboard())
    t_start = time.time()

    resume = getattr(args, "resume", False)
    from_st = getattr(args, "from_stage", None)
    to_st = getattr(args, "to_stage", None)

    # ─────────────────────────────────────────────────────
    # STAGE 0: Data Prep (ID/NPI splits + 799 normal images)
    # ─────────────────────────────────────────────────────
    img_dir = Path(args.image_dir)
    has_normals = img_dir.exists() and len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) > 0
    id_yaml = Path("data/id_setup.yaml")

    if manager.should_execute("0", from_st, to_st):
        if resume and has_normals and id_yaml.exists():
            logger.info("[RESUME] Stage 0 (Data Prep) đã có 799 ảnh phôi sạch & ID setup. Bỏ qua.")
            manager.mark_stage("0", "COMPLETED", artifact=str(id_yaml))
        elif not has_normals:
            raw_dir = Path("data/pcb-defect-dataset")
            if raw_dir.exists():
                logger.info(f"Chưa tìm thấy ảnh sạch trong '{img_dir}'. Đang tự động tạo 799 ảnh normal theo chuẩn bài báo...")
                manager.mark_stage("0", "RUNNING")
                t0 = time.time()
                from pcb_pipeline.data_module import PCBDataModule
                dm = PCBDataModule(raw_dataset_dir=raw_dir, output_dir="data")
                dm.run_pipeline()
                manager.mark_stage("0", "COMPLETED", elapsed=time.time() - t0, artifact=str(id_yaml))
            else:
                raise FileNotFoundError(
                    f"Không tìm thấy thư mục '{img_dir}' và cũng không tìm thấy '{raw_dir}'.\n"
                    f"Hãy tải dataset và giải nén vào 'data/pcb-defect-dataset/', sau đó chạy 'python run_pipeline.py prepare_data'!"
                )
        else:
            manager.mark_stage("0", "COMPLETED", artifact=str(id_yaml))
    else:
        logger.info("[SKIP] Bỏ qua Stage 0 theo cấu hình stage.")
        manager.mark_stage("0", "SKIPPED")

    # ─────────────────────────────────────────────────────
    # STAGE 1: CAMG Mask Generation
    # ─────────────────────────────────────────────────────
    mask_records = []
    mask_meta_path = out / "masks" / "mask_metadata.json"

    if manager.should_execute("1", from_st, to_st):
        if resume and mask_meta_path.exists() and mask_meta_path.stat().st_size > 0:
            try:
                with open(mask_meta_path, "r", encoding="utf-8") as f:
                    mask_records = json.load(f)
                if mask_records and len(mask_records) > 0:
                    logger.info(f"[RESUME] Stage 1 (CAMG Masks) đã có sẵn {len(mask_records)} masks tại: {mask_meta_path}. Tái sử dụng!")
                    manager.mark_stage("1", "COMPLETED", artifact=str(mask_meta_path), extra={"count": len(mask_records)})
            except Exception as e:
                logger.warning(f"Lỗi đọc mask metadata cũ: {e}. Đang sinh lại...")

        if not mask_records:
            manager.mark_stage("1", "RUNNING")
            t0 = time.time()
            mask_records = run_step1(args, cfg)
            manager.mark_stage("1", "COMPLETED", elapsed=time.time() - t0, artifact=str(mask_meta_path), extra={"count": len(mask_records)})
    else:
        logger.info("[SKIP] Bỏ qua Stage 1 theo cấu hình stage.")
        manager.mark_stage("1", "SKIPPED")
        if mask_meta_path.exists():
            try:
                with open(mask_meta_path, "r", encoding="utf-8") as f:
                    mask_records = json.load(f)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────
    # STAGE 2a: LoRA & Textual Inversion Training
    # ─────────────────────────────────────────────────────
    lora_ckpt_target = getattr(args, "lora_ckpt", None)
    if not lora_ckpt_target:
        candidate_lora = out / "lora_checkpoints" / "epoch_010"
        if candidate_lora.exists():
            lora_ckpt_target = str(candidate_lora)
        else:
            cand_root = out / "lora_checkpoints"
            if cand_root.exists() and any(cand_root.iterdir()):
                lora_ckpt_target = str(cand_root)

    if manager.should_execute("2a", from_st, to_st):
        if resume and lora_ckpt_target and Path(lora_ckpt_target).exists():
            logger.info(f"[RESUME] Stage 2a (LoRA Checkpoint) đã có sẵn tại: {lora_ckpt_target}. Bỏ qua huấn luyện.")
            args.lora_ckpt = lora_ckpt_target
            manager.mark_stage("2a", "COMPLETED", artifact=str(lora_ckpt_target))
        elif not getattr(args, "lora_ckpt", None) and getattr(args, "real_defect_dir", None):
            manager.mark_stage("2a", "RUNNING")
            t0 = time.time()
            args.lora_ckpt = run_step2_train(args, cfg)
            manager.mark_stage("2a", "COMPLETED", elapsed=time.time() - t0, artifact=str(args.lora_ckpt or out / "lora_checkpoints"))
        else:
            manager.mark_stage("2a", "SKIPPED", artifact=str(args.lora_ckpt or "Không train"))
    else:
        logger.info("[SKIP] Bỏ qua Stage 2a theo cấu hình stage.")
        manager.mark_stage("2a", "SKIPPED")
        if lora_ckpt_target:
            args.lora_ckpt = lora_ckpt_target

    # ─────────────────────────────────────────────────────
    # STAGE 2b: Diffusion Inpainting Synthesis
    # ─────────────────────────────────────────────────────
    gen_records = []
    gen_meta_path = out / "generated" / "generated_metadata.json"

    if manager.should_execute("2b", from_st, to_st):
        if resume and gen_meta_path.exists() and gen_meta_path.stat().st_size > 0:
            try:
                with open(gen_meta_path, "r", encoding="utf-8") as f:
                    gen_records = json.load(f)
                if gen_records and len(gen_records) > 0:
                    logger.info(f"[RESUME] Stage 2b (Diffusion Inpaint) đã có sẵn {len(gen_records)} ảnh tại: {gen_meta_path}. Tái sử dụng!")
                    manager.mark_stage("2b", "COMPLETED", artifact=str(gen_meta_path), extra={"count": len(gen_records)})
            except Exception as e:
                logger.warning(f"Lỗi đọc generated metadata cũ: {e}. Đang sinh lại...")

        if not gen_records:
            if not mask_records:
                raise RuntimeError("Không có mask_records từ Stage 1 để sinh ảnh!")
            manager.mark_stage("2b", "RUNNING")
            t0 = time.time()
            gen_records = run_step2_generate(mask_records, args, cfg)
            manager.mark_stage("2b", "COMPLETED", elapsed=time.time() - t0, artifact=str(gen_meta_path), extra={"count": len(gen_records)})
    else:
        logger.info("[SKIP] Bỏ qua Stage 2b theo cấu hình stage.")
        manager.mark_stage("2b", "SKIPPED")
        if gen_meta_path.exists():
            try:
                with open(gen_meta_path, "r", encoding="utf-8") as f:
                    gen_records = json.load(f)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────
    # STAGE 3: Quality Filtering (YOLO11s)
    # ─────────────────────────────────────────────────────
    final_records = []
    aug_yaml_path = out / "filtered" / "augmented_dataset.yaml"
    acc_meta_path = out / "filtered" / "accepted_metadata.json"

    if manager.should_execute("3", from_st, to_st):
        if resume and aug_yaml_path.exists() and acc_meta_path.exists():
            try:
                with open(acc_meta_path, "r", encoding="utf-8") as f:
                    final_records = json.load(f)
                if final_records and len(final_records) > 0:
                    logger.info(f"[RESUME] Stage 3 (Quality Filter) đã có sẵn {len(final_records)} ảnh accepted tại: {aug_yaml_path}. Tái sử dụng!")
                    manager.mark_stage("3", "COMPLETED", artifact=str(aug_yaml_path), extra={"count": len(final_records)})
            except Exception as e:
                logger.warning(f"Lỗi đọc filter metadata cũ: {e}. Đang lọc lại...")

        if not final_records:
            if not gen_records:
                raise RuntimeError("Không có gen_records từ Stage 2b để lọc chất lượng!")
            manager.mark_stage("3", "RUNNING")
            t0 = time.time()
            final_records = run_step3(gen_records, args, cfg)
            manager.mark_stage("3", "COMPLETED", elapsed=time.time() - t0, artifact=str(aug_yaml_path), extra={"count": len(final_records)})
    else:
        logger.info("[SKIP] Bỏ qua Stage 3 theo cấu hình stage.")
        manager.mark_stage("3", "SKIPPED")
        if acc_meta_path.exists():
            try:
                with open(acc_meta_path, "r", encoding="utf-8") as f:
                    final_records = json.load(f)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────
    # STAGE 4: Downstream Benchmark
    # ─────────────────────────────────────────────────────
    benchmark_results = None
    bench_report_path = out / "benchmark" / "benchmark_report.md"

    if getattr(args, "run_benchmark", False) and manager.should_execute("4", from_st, to_st):
        if resume and bench_report_path.exists() and bench_report_path.stat().st_size > 0:
            logger.info(f"[RESUME] Stage 4 (Downstream Benchmark) đã hoàn tất tại: {bench_report_path}. Bỏ qua.")
            manager.mark_stage("4", "COMPLETED", artifact=str(bench_report_path))
            benchmark_results = {"report": str(bench_report_path)}
        else:
            manager.mark_stage("4", "RUNNING")
            t0 = time.time()
            setattr(args, "models", getattr(args, "benchmark_models", ["yolo11"]))
            setattr(args, "epochs", getattr(args, "benchmark_epochs", 100))
            setattr(args, "aug_yaml", str(aug_yaml_path))
            benchmark_results = run_step4(args, cfg)
            manager.mark_stage("4", "COMPLETED", elapsed=time.time() - t0, artifact=str(bench_report_path))
    else:
        if not getattr(args, "run_benchmark", False):
            manager.mark_stage("4", "SKIPPED", artifact="Không kích hoạt (--run_benchmark)")
        else:
            manager.mark_stage("4", "SKIPPED")

    # ─────────────────────────────────────────────────────
    # Tổng kết & Dashboard cuối cùng
    # ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("\n" + manager.render_dashboard())
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION HOÀN TẤT")
    logger.info(f"   Tổng thời gian phiên:    {elapsed/60:.1f} phút")
    logger.info(f"   Ảnh sau lọc (Accepted):  {len(final_records)}")
    if aug_yaml_path.exists():
        logger.info(f"   Augmented YAML:          {aug_yaml_path.resolve()}")
    if benchmark_results:
        logger.info(f"   Benchmark Report:        {bench_report_path.resolve()}")
    logger.info(f"   Output Directory:        {out.resolve()}")
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
        "stages":             manager.status_data.get("stages", {}),
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
    p_full.add_argument("--num_samples",     type=int, default=829, help="Số lượng ảnh lỗi tổng hợp cần sinh (chuẩn bài báo: 829 ảnh)")
    p_full.add_argument("--yolo_weights",    default=None,   help="best.pt của YOLO judge (nếu đã train)")
    p_full.add_argument("--data_yaml",       default="data/id_setup.yaml", help="File cấu hình dataset lỗi thực (ID 10%% hoặc NPI)")
    p_full.add_argument("--yolo_size",       default=None, choices=["n", "s", "m", "l", "x"], help="Kích thước mô hình YOLO11 (mặc định 's' theo bài báo)")
    p_full.add_argument("--yolo_epochs",     type=int, default=None, help="Số epochs huấn luyện YOLO judge (100 theo bài báo)")
    p_full.add_argument("--reject_ratio",    type=float, default=None)
    p_full.add_argument("--by_class",        action="store_true", help="Lọc độc lập theo từng loại lỗi để cân bằng class")
    p_full.add_argument("--run_benchmark",    action="store_true", help="Tự động chạy Step 4 Benchmark sau khi hoàn tất Step 3")
    p_full.add_argument("--benchmark_models", nargs="+", default=["yolo11"], help="Danh sách mô hình benchmark khi chạy full (mặc định: yolo11)")
    p_full.add_argument("--benchmark_epochs", type=int, default=100, help="Số epochs benchmark cho mỗi model (mặc định 100 theo bài báo)")
    p_full.add_argument("--resume",           action="store_true", help="Tự động khôi phục và bỏ qua các giai đoạn đã hoàn thành trước đó")
    p_full.add_argument("--from_stage",       default=None, help="Bắt đầu từ giai đoạn cụ thể: 0 (prep), 1 (mask), 2a (lora), 2b (inpaint), 3 (filter), 4 (benchmark)")
    p_full.add_argument("--to_stage",         default=None, help="Dừng lại sau giai đoạn cụ thể: 0, 1, 2a, 2b, 3, 4")

    # ── status ───────────────────────────────────────────
    p_stat = subparsers.add_parser("status", help="Xem bảng trạng thái dashboard tiến trình của pipeline")
    p_stat.add_argument("--output_dir", default="outputs/pcb_synthesis", help="Thư mục output chứa pipeline_status.json")

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
    p2g.add_argument("--num_samples",        type=int, default=829, help="Số lượng ảnh lỗi tổng hợp cần sinh (chuẩn bài báo: 829 ảnh)")

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

    elif args.command == "status":
        manager = PipelineCheckpointManager(args.output_dir)
        print("\n" + manager.render_dashboard())


if __name__ == "__main__":
    main()
