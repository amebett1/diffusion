"""
PCB Defect Synthesis Pipeline — Main Orchestrator
==================================================
Ghép các bước theo protocol của paper, nhưng chỉ trong phạm vi tài nguyên
thực tế của repo:

  Step 0: Data Prep          (ID/NPI split + ảnh normal sạch)
  Step 1: Mask Generation    (CAMG + DINOv2 + size prior)
  Step 2a: LoRA/TI Training  (SD Inpainting)
  Step 2b: Image Synthesis   (inpainting diffusion)
  Step 3: Quality Filtering  (YOLO11 judge hoặc heuristic cảnh báo rõ)
  Step 4: Benchmark          (CHỈ YOLO11s, nếu bật --run_benchmark)

Nguyên tắc an toàn
------------------
- Không bao giờ chạy CAMG với prediction heads random. Nếu thiếu
  ``camg_best.pth``, pipeline tự train CAMG trước khi sinh mask.
- Không tái sử dụng mask cache cũ nếu sai version, sai checkpoint hoặc
  kích thước mask bất thường (ví dụ mask 120x120 của bản cũ).
- Không train LoRA nếu đã có checkpoint hợp lệ; nếu chưa có thì tự train.
- Benchmark mặc định chỉ YOLO11s; NMS IoU = 0.7, conf = 0.25.
- Không gọi ``adapt_paths()`` sau data prep vì có thể ghi đè path tuyệt đối
  runtime-adaptive bằng path tương đối dễ hỏng.

Ví dụ
-----
Chuẩn bị dữ liệu:
    python run_pipeline.py prepare_data

Chạy toàn bộ pipeline (tự train CAMG/LoRA nếu thiếu):
    python run_pipeline.py full --run_benchmark

Chạy riêng từng bước:
    python run_pipeline.py step1_train
    python run_pipeline.py step1
    python run_pipeline.py step2_train
    python run_pipeline.py step2_generate --mask_meta outputs/pcb_synthesis/masks/mask_metadata.json
    python run_pipeline.py step3 --gen_meta outputs/pcb_synthesis/generated/generated_metadata.json
    python run_pipeline.py step4
"""

import sys
import argparse
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RAW_DIR = DATA_DIR / "pcb-defect-dataset"
DEFAULT_NORMAL_DIR = DATA_DIR / "pcb_normal"
DEFAULT_TRAIN_META = DATA_DIR / "train_metadata.json"
DEFAULT_STATS_PATH = DATA_DIR / "dataset_stats.json"

from pcb_pipeline.data_module import (  # noqa: E402
    NORMAL_PROTOCOL,
    SPLIT_PROTOCOL,
    PCBDataModule,
)
from pcb_pipeline.step1_mask_generator import (  # noqa: E402
    CAMGGenerator,
    CAMGTrainer,
    MASK_GENERATION_VERSION,
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
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]

DEFAULT_CONFIG = {
    # Step 0
    "normal_target": 799,
    "low_data_ratio": 0.1,
    # Step 1
    "camg_epochs": 10,
    "camg_batch": 32,
    "camg_lr": 1e-4,
    "camg_workers": 0,
    "dino_model": "dinov2_vitb14",
    "mask_strategy": "saliency",
    "n_proposals": 5,
    "images_per_defect": 1,
    # Step 2 — LoRA & TI
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lora_epochs": 10,
    "lora_batch": 32,
    "lr_unet": 1e-4,
    "lr_text_encoder": 5e-5,
    "lr_ti": 5e-4,
    # Step 2 — Inference
    "num_samples": 829,
    "inpaint_steps": 50,
    "guidance_scale": 7.5,
    "negative_prompt": "blurry, low quality, deformed, ugly, watermark, signature, text, logo",
    # Step 3
    "reject_ratio": 0.25,
    "yolo_size": "s",
    "yolo_epochs": 100,
    # Step 4
    "benchmark_models": ["yolo11s"],
    "benchmark_epochs": 100,
    "benchmark_batch": 8,
    "benchmark_lr0": 1e-4,
    "benchmark_conf": 0.25,
    "benchmark_iou": 0.7,
    "benchmark_imgsz": 600,
    "benchmark_workers": None,
    "device": "cuda",
}


def _int_or(value, default: int) -> int:
    return int(value) if value is not None and str(value) != "" else int(default)


def _float_or(value, default: float) -> float:
    return float(value) if value is not None and str(value) != "" else float(default)


def _resolve_path(value, default=None) -> Path:
    raw = value if value is not None else default
    if raw is None:
        raise ValueError("Không có đường dẫn để resolve.")
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _file_ok(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _resolve_runtime_device(requested: str) -> str:
    if str(requested).startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "Yêu cầu CUDA nhưng torch.cuda.is_available()=False. "
                    "Chuyển sang CPU; pipeline sẽ rất chậm và không đại diện thiết lập GPU."
                )
                return "cpu"
        except Exception:
            return "cpu"
    return str(requested)


def load_config(config_path: Optional[str], overrides: dict) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def save_config(cfg: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pipeline_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────
def log_gpu_info():
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
            logger.info("=" * 70)
        else:
            logger.warning("Không tìm thấy GPU CUDA. Pipeline sẽ chạy trên CPU.")
    except Exception as e:
        logger.debug(f"Không thể kiểm tra GPU: {e}")


# ─────────────────────────────────────────────────────────
# Pipeline checkpoint manager
# ─────────────────────────────────────────────────────────
class PipelineCheckpointManager:
    STAGE_DEFS = [
        ("0", "Data Prep & Normal Images", "data/data_prep_summary.json"),
        ("1", "CAMG Mask Generation", "outputs/pcb_synthesis/masks/mask_metadata.json"),
        ("2a", "LoRA / TI Fine-tuning", "outputs/pcb_synthesis/lora_checkpoints"),
        ("2b", "Diffusion Inpainting", "outputs/pcb_synthesis/generated/generated_metadata.json"),
        ("3", "Quality Filtering", "outputs/pcb_synthesis/filtered/augmented_dataset.yaml"),
        ("4", "Downstream Benchmark", "outputs/pcb_synthesis/benchmark/benchmark_report.md"),
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

    def mark_stage(
        self,
        stage_id: str,
        status: str,
        elapsed: float = 0.0,
        artifact: str | None = None,
        extra: dict | None = None,
    ):
        if "stages" not in self.status_data:
            self.status_data["stages"] = {}
        entry = self.status_data["stages"].get(stage_id, {})
        entry.update({
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_artifact": str(artifact) if artifact else entry.get("target_artifact"),
        })
        if extra:
            entry.update(extra)
        self.status_data["stages"][stage_id] = entry
        self.save()

    def should_execute(self, stage_id: str, from_stage: str | None, to_stage: str | None) -> bool:
        sid = self.ALIASES.get(str(stage_id).lower(), stage_id)
        if sid not in self.STAGE_KEYS:
            return True
        idx = self.STAGE_KEYS.index(sid)

        if from_stage:
            f_id = self.ALIASES.get(str(from_stage).lower(), from_stage)
            if f_id in self.STAGE_KEYS and idx < self.STAGE_KEYS.index(f_id):
                return False

        if to_stage:
            t_id = self.ALIASES.get(str(to_stage).lower(), to_stage)
            if t_id in self.STAGE_KEYS and idx > self.STAGE_KEYS.index(t_id):
                return False
        return True

    def render_dashboard(self) -> str:
        lines = []
        bar = "=" * 82
        sep = "-" * 82
        lines.append(bar)
        lines.append(f"{'PCB DEFECT SYNTHESIS PIPELINE EXECUTION DASHBOARD':^82}")
        lines.append(bar)
        lines.append(f"{'Stage':<6} {'Tên công đoạn':<25} {'Trạng thái':<14} {'Thời gian':<12} {'Output / Artifact':<22}")
        lines.append(sep)

        for sid, name, default_art in self.STAGE_DEFS:
            entry = self.status_data.get("stages", {}).get(sid, {})
            status = entry.get("status", "PENDING")
            elapsed_sec = entry.get("elapsed_seconds", None)
            art = entry.get("target_artifact", default_art)
            art_name = Path(art).name if art else "-"

            # `prepare_data` có thể chạy trực tiếp mà không qua manager.
            # Suy ra trạng thái Stage 0 từ artifact đã được validate.
            if status == "PENDING" and sid == "0":
                try:
                    if _validate_data_prep(DATA_DIR, 799)[0]:
                        status = "COMPLETED"
                except Exception:
                    pass

            status_str = {
                "COMPLETED": "[HOÀN TẤT]",
                "RUNNING": "[ĐANG CHẠY...]",
                "SKIPPED": "[BỎ QUA]",
            }.get(status, "[CHƯA CHẠY]")

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
# Data prep validation / auto-run
# ─────────────────────────────────────────────────────────
def _resolve_list_line(line: str, data_dir: Path) -> Optional[Path]:
    p = Path(line.strip())
    candidates = [p] if p.is_absolute() else [PROJECT_ROOT / p, data_dir.parent / p, Path.cwd() / p]
    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except OSError:
            continue
    return None


def _path_has_split(path: Path, split_name: str) -> bool:
    return f"/{split_name}/" in path.as_posix().lower()


def _board_from_name(path: Path) -> Optional[str]:
    m = re.search(r"light_(\d+)", path.name)
    return m.group(1) if m else None


def _validate_split_lists(
    data_dir: Path,
    summary: dict,
) -> Tuple[bool, str]:
    splits_dir = data_dir / "prepared_splits"
    files = {
        "id_train_low": splits_dir / "id_train_low.txt",
        "id_train_full": splits_dir / "id_train_full.txt",
        "id_val": splits_dir / "id_val.txt",
        "npi_train_low": splits_dir / "npi_train_low.txt",
        "npi_val_unseen": splits_dir / "npi_val_unseen.txt",
    }
    for name, p in files.items():
        if not _file_ok(p):
            return False, f"thiếu {name}: {p}"

    seen_boards = set(str(b) for b in summary.get("seen_boards", []))
    unseen_boards = set(str(b) for b in summary.get("unseen_boards", []))
    if not seen_boards or not unseen_boards:
        return False, "summary thiếu seen_boards/unseen_boards"

    def load(name: str) -> List[Path]:
        out = []
        with open(files[name], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = _resolve_list_line(line, data_dir)
                if p is None:
                    raise FileNotFoundError(f"{name} chứa path không tồn tại: {line}")
                out.append(p)
        return out

    try:
        id_low = load("id_train_low")
        id_full = load("id_train_full")
        id_val = load("id_val")
        npi_val = load("npi_val_unseen")
    except FileNotFoundError as e:
        return False, str(e)

    if not id_low or not id_full or not id_val or not npi_val:
        return False, "một split rỗng"

    # Không lấy official val/test của seen boards vào train/val ID.
    for name, recs in [("id_train_low", id_low), ("id_val", id_val)]:
        bad = [p for p in recs if not _path_has_split(p, "train")]
        if bad:
            return False, f"{name} chứa ảnh không thuộc official train: {bad[0]}"

    low_set = {str(p).lower() for p in id_low}
    full_set = {str(p).lower() for p in id_full}
    if not low_set.issubset(full_set):
        return False, "id_train_low không phải tập con của id_train_full"
    if low_set & {str(p).lower() for p in id_val}:
        return False, "id_train_low giao với id_val"

    for p in id_low + id_val:
        board = _board_from_name(p)
        if board is None or board not in seen_boards:
            return False, f"ID split chứa board không thuộc seen: {p}"

    for p in npi_val:
        board = _board_from_name(p)
        if board is None or board not in unseen_boards:
            return False, f"NPI val chứa board không thuộc unseen: {p}"
        if not (_path_has_split(p, "val") or _path_has_split(p, "test")):
            return False, f"NPI val chứa ảnh không thuộc official val/test: {p}"

    return True, "ok"


def _validate_data_prep(data_dir: Path, target_normals: int) -> Tuple[bool, str]:
    summary_path = data_dir / "data_prep_summary.json"
    if not _file_ok(summary_path):
        return False, "thiếu data_prep_summary.json"
    try:
        summary = _load_json(summary_path)
    except Exception as e:
        return False, f"không đọc được summary: {e}"

    if summary.get("split_protocol") != SPLIT_PROTOCOL:
        return False, "split_protocol cũ/không hợp lệ"
    if summary.get("normal_protocol") != NORMAL_PROTOCOL:
        return False, "normal_protocol cũ/không hợp lệ"

    for key in ("id_yaml", "npi_yaml", "train_metadata", "normal_manifest"):
        p = summary.get("exported_files", {}).get(key) if isinstance(summary.get("exported_files"), dict) else summary.get(key)
        if key == "id_yaml":
            p = data_dir / "id_setup.yaml"
        elif key == "npi_yaml":
            p = data_dir / "npi_setup.yaml"
        elif key == "train_metadata":
            p = data_dir / "train_metadata.json"
        elif key == "normal_manifest":
            p = data_dir / "pcb_normal" / "normal_manifest.json"
        if p is None or not _file_ok(Path(p)):
            return False, f"thiếu artifact {key}"

    for p in [
        data_dir / "id_setup.yaml",
        data_dir / "npi_setup.yaml",
        data_dir / "train_metadata.json",
        DEFAULT_STATS_PATH,
        data_dir / "pcb_normal" / "normal_manifest.json",
    ]:
        if not _file_ok(p):
            return False, f"thiếu artifact: {p}"

    try:
        manifest = _load_json(data_dir / "pcb_normal" / "normal_manifest.json")
    except Exception as e:
        return False, f"không đọc được normal manifest: {e}"

    if manifest.get("protocol") != NORMAL_PROTOCOL:
        return False, "normal manifest sai protocol"
    if list(manifest.get("allowed_splits", [])) != ["train"]:
        return False, "normal manifest không chỉ dùng official train"
    if list(manifest.get("allowed_boards", [])) != list(summary.get("seen_boards", [])):
        return False, "normal manifest dùng board khác seen_boards"
    generated = manifest.get("generated", [])
    if len(generated) < int(target_normals):
        return False, f"chỉ có {len(generated)}/{target_normals} ảnh normal"
    for item in generated:
        p = Path(item.get("path", ""))
        if not _file_ok(p, min_bytes=256):
            return False, f"ảnh normal thiếu/hỏng: {p}"
        src = Path(item.get("source_image", ""))
        if not _file_ok(src):
            return False, f"nguồn normal thiếu: {src}"
        if not _path_has_split(src, "train"):
            return False, f"normal lấy từ split không phải official train: {src}"

    # Metadata CAMG/LoRA phải chỉ chứa official train của seen boards.
    try:
        meta = _load_json(data_dir / "train_metadata.json")
    except Exception as e:
        return False, f"không đọc được train_metadata: {e}"
    if not isinstance(meta, list) or not meta:
        return False, "train_metadata rỗng"
    seen_boards = set(str(b) for b in summary.get("seen_boards", []))
    for rec in meta:
        p = Path(rec.get("image_path", ""))
        if not _file_ok(p):
            return False, f"train_metadata thiếu ảnh: {p}"
        if rec.get("original_split") != "train":
            return False, f"train_metadata chứa split {rec.get('original_split')}: {p}"
        if str(rec.get("board_id")) not in seen_boards:
            return False, f"train_metadata chứa unseen board: {p}"

    try:
        stats = _load_json(DEFAULT_STATS_PATH)
        median_w = float(stats["pixel_size"]["width"]["median"])
        if not (10.0 <= median_w <= 60.0):
            return False, f"bbox median width bất thường: {median_w}"
    except Exception as e:
        return False, f"dataset_stats lỗi: {e}"

    ok_lists, reason = _validate_split_lists(data_dir, summary)
    if not ok_lists:
        return False, reason
    return True, "ok"


def _ensure_data_prep(args, cfg: dict, force: bool = False) -> dict:
    data_dir = DATA_DIR
    target = int(cfg.get("normal_target", 799))
    if not force:
        ok, reason = _validate_data_prep(data_dir, target)
        if ok:
            logger.info("[Data Prep] Artifact hiện có hợp lệ; bỏ qua tạo lại.")
            try:
                return _load_json(data_dir / "data_prep_summary.json")
            except Exception:
                pass
        else:
            logger.warning(f"[Data Prep] Artifact cũ không hợp lệ ({reason}); tạo lại từ raw dataset.")

    raw_dir = _resolve_path(getattr(args, "raw_dir", None), DEFAULT_RAW_DIR)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy raw dataset: '{raw_dir}'.\n"
            "Hãy giải nén DeepPCB vào 'data/pcb-defect-dataset/' rồi chạy lại."
        )

    dm = PCBDataModule(
        raw_dataset_dir=str(raw_dir),
        output_dir=str(data_dir),
        image_size=600,
        random_seed=42,
    )
    summary = dm.run_pipeline(
        target_normal_count=target,
        low_data_ratio=float(cfg.get("low_data_ratio", 0.1)),
    )

    ok, reason = _validate_data_prep(data_dir, target)
    if not ok:
        logger.error(f"[Data Prep] Sau khi chạy vẫn chưa hợp lệ: {reason}")
    else:
        logger.info("[Data Prep] Đã tạo artifact hợp lệ.")
    return summary


# ─────────────────────────────────────────────────────────
# Checkpoint / cache helpers
# ─────────────────────────────────────────────────────────
def _find_camg_checkpoint(out_dir: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = _resolve_path(explicit)
        if _file_ok(p, min_bytes=1024):
            return p
        raise FileNotFoundError(
            f"Không tìm thấy CAMG checkpoint hợp lệ tại '{p}'. "
            "Không chạy CAMG với heads random."
        )
    for candidate in [
        out_dir / "camg_checkpoints" / "camg_best.pth",
        out_dir / "camg_best.pth",
        PROJECT_ROOT / "outputs" / "camg_checkpoints" / "camg_best.pth",
    ]:
        if _file_ok(candidate, min_bytes=1024):
            return candidate.resolve()
    return None


def _lora_checkpoint_valid(path: Path) -> bool:
    try:
        p = Path(path)
        if p.is_dir():
            return _file_ok(p / "pytorch_lora_weights.pt", 1024) and _file_ok(
                p / "ti_embeddings.pt", 1024
            )
    except OSError:
        pass
    return False


def _find_lora_checkpoint(root: Path) -> Optional[Path]:
    root = Path(root)
    if _lora_checkpoint_valid(root):
        return root.resolve()
    if root.exists() and root.is_dir():
        candidates = sorted(
            [p for p in root.glob("epoch_*") if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        for p in candidates:
            if _lora_checkpoint_valid(p):
                return p.resolve()
    return None


def _find_ti_checkpoint(lora_dir: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = _resolve_path(explicit)
        if _file_ok(p, 1024):
            return p
        raise FileNotFoundError(f"Không tìm thấy TI checkpoint: {p}")
    p = Path(lora_dir) / "ti_embeddings.pt"
    return p.resolve() if _file_ok(p, 1024) else None


def _archive_files(target_dir: Path, patterns: List[str], label: str):
    target_dir = Path(target_dir)
    if not target_dir.exists():
        return
    stale_dir = target_dir / "_stale" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    moved = 0
    for pattern in patterns:
        for p in target_dir.glob(pattern):
            if not p.exists() or p.name == "_stale":
                continue
            try:
                stale_dir.mkdir(parents=True, exist_ok=True)
                dst = stale_dir / p.name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(p), str(dst))
                moved += 1
            except OSError as e:
                logger.warning(f"Không archive được {p}: {e}")
    if moved:
        logger.info(f"Đã chuyển {moved} file cũ vào {stale_dir}")


def _mask_stats(records: List[dict]) -> Dict:
    widths, heights = [], []
    for rec in records:
        try:
            x1, y1, x2, y2 = [float(v) for v in rec.get("bbox_pixel", [])[:4]]
            if x2 > x1 and y2 > y1:
                widths.append(x2 - x1)
                heights.append(y2 - y1)
        except Exception:
            continue
    if not widths:
        return {}
    import numpy as np

    return {
        "count": len(widths),
        "width_min": float(np.min(widths)),
        "width_q1": float(np.percentile(widths, 25)),
        "width_median": float(np.median(widths)),
        "width_q3": float(np.percentile(widths, 75)),
        "width_max": float(np.max(widths)),
        "height_median": float(np.median(heights)),
        "height_q3": float(np.percentile(heights, 75)),
    }


def _validate_mask_cache(
    mask_dir: Path,
    weights_path: Path,
    expected_image_count: int,
) -> Tuple[bool, str, List[dict]]:
    meta_path = mask_dir / "mask_metadata.json"
    info_path = mask_dir / "mask_generation_info.json"
    if not _file_ok(meta_path):
        return False, "thiếu mask_metadata.json", []
    if not _file_ok(info_path):
        return False, "thiếu mask_generation_info.json", []
    try:
        records = _load_json(meta_path)
        info = _load_json(info_path)
    except Exception as e:
        return False, f"không đọc được mask cache: {e}", []

    if not isinstance(records, list) or not records:
        return False, "mask metadata rỗng", []
    if info.get("mask_version") != MASK_GENERATION_VERSION:
        return False, f"mask_version cũ ({info.get('mask_version')})", records
    if info.get("allow_random_weights"):
        return False, "mask cache được sinh bằng random heads", records
    try:
        cached_weights = Path(str(info.get("weights_path", ""))).resolve()
    except Exception:
        cached_weights = None
    if cached_weights is None or cached_weights != Path(weights_path).resolve():
        return False, "mask cache dùng checkpoint khác", records

    normal_paths = set()
    for rec in records:
        npath = Path(rec.get("normal_image_path", ""))
        mpath = Path(rec.get("mask_path", ""))
        if not _file_ok(npath, 256):
            return False, f"normal image thiếu: {npath}", records
        if not _file_ok(mpath, 64):
            return False, f"mask thiếu: {mpath}", records
        normal_paths.add(str(npath.resolve()).lower())

    stats = _mask_stats(records)
    if not stats:
        return False, "mask metadata không có bbox hợp lệ", records
    if stats["width_median"] > 60.0 or stats["width_q3"] > 100.0:
        return False, (
            "kích thước mask bất thường "
            f"(median={stats['width_median']:.1f}px, q3={stats['width_q3']:.1f}px); "
            "có thể là mask 120x120 của bản cũ"
        ), records
    min_expected_images = min(int(expected_image_count), 100)
    if len(normal_paths) < min_expected_images:
        return False, (
            f"cache chỉ phủ {len(normal_paths)} ảnh normal, cần ít nhất {min_expected_images}"
        ), records
    if int(info.get("total_saved", -1)) != len(records):
        return False, "total_saved trong info không khớp metadata", records

    return True, "ok", records


def _prune_stale_masks(mask_dir: Path, records: List[dict]):
    keep = {str(Path(r["mask_path"]).resolve()).lower() for r in records if r.get("mask_path")}
    stale_dir = Path(mask_dir) / "_stale" / f"masks_{time.strftime('%Y%m%d_%H%M%S')}"
    moved = 0
    for p in Path(mask_dir).glob("*_mask.png"):
        if str(p.resolve()).lower() in keep:
            continue
        try:
            stale_dir.mkdir(parents=True, exist_ok=True)
            dst = stale_dir / p.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(p), str(dst))
            moved += 1
        except OSError:
            pass
    if moved:
        logger.info(f"[CAMG] Đã chuyển {moved} mask cũ/không dùng vào {stale_dir}")


def _validate_generated_cache(gen_meta: Path) -> Tuple[bool, str, List[dict]]:
    if not _file_ok(gen_meta):
        return False, "thiếu generated_metadata.json", []
    try:
        records = _load_json(gen_meta)
    except Exception as e:
        return False, f"không đọc được generated metadata: {e}", []
    if not isinstance(records, list) or not records:
        return False, "generated metadata rỗng", []

    for rec in records:
        p = Path(rec.get("synthetic_image_path", ""))
        if not _file_ok(p, 1024):
            return False, f"ảnh synthetic thiếu/hỏng: {p}", records
        if rec.get("generation_mode") != "diffusion":
            return False, f"ảnh không phải diffusion thật: {p}", records
        if rec.get("generation_version") != "ddim_consistent_v2":
            return False, f"cache dùng generation version cũ: {p}", records
        if not rec.get("using_lora"):
            return False, f"ảnh sinh không dùng LoRA: {p}", records
        if not rec.get("ti_loaded"):
            return False, f"ảnh sinh không dùng Textual Inversion: {p}", records
        if rec.get("latent_overwrite") != "consistent_single_noise_ddim":
            return False, f"cache dùng latent overwrite cũ: {p}", records
    return True, "ok", records


def _validate_filtered_cache(out_dir: Path) -> Tuple[bool, str, List[dict]]:
    filtered = Path(out_dir) / "filtered"
    meta = filtered / "accepted_metadata.json"
    aug = filtered / "augmented_dataset.yaml"
    if not _file_ok(meta) or not _file_ok(aug):
        return False, "thiếu accepted metadata hoặc augmented YAML", []
    try:
        records = _load_json(meta)
    except Exception as e:
        return False, f"không đọc được accepted metadata: {e}", []
    if not isinstance(records, list) or not records:
        return False, "accepted metadata rỗng", []
    for rec in records:
        p = rec.get("filtered_image_path") or rec.get("synthetic_image_path")
        lp = rec.get("yolo_label_path")
        if not p or not _file_ok(Path(p), 512):
            return False, f"ảnh accepted thiếu: {p}", records
        if not lp or not _file_ok(Path(lp), 4):
            return False, f"nhãn YOLO accepted thiếu: {lp}", records
    return True, "ok", records


def _validate_benchmark_summary(out_dir: Path) -> bool:
    p = Path(out_dir) / "benchmark" / "benchmark_summary.json"
    if not _file_ok(p):
        return False
    try:
        data = _load_json(p)
        protocol = data.get("protocol", {})
        if protocol.get("scope") != "resource_limited_yolo11s_demo":
            return False
        if data.get("models_evaluated") != ["yolo11s"]:
            return False
        if protocol.get("eval_checkpoint") != "best.pt":
            return False
        if abs(float(protocol.get("nms_iou", 0.0)) - 0.7) > 1e-6:
            return False
        aug = data.get("datasets", {}).get("augmented", {})
        if int(aug.get("synthetic_count", 0)) <= 0:
            return False
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────
# STEP 1: CAMG training + mask generation
# ─────────────────────────────────────────────────────────
def run_step1_train(args, cfg: dict, output_dir: Optional[Path] = None) -> str:
    logger.info("=" * 66)
    logger.info("STEP 1a — Huấn luyện CAMG prediction heads")
    logger.info("=" * 66)

    metadata_path = _resolve_path(getattr(args, "metadata_path", None), DEFAULT_TRAIN_META)
    if not _file_ok(metadata_path):
        raise FileNotFoundError(
            f"Không tìm thấy train metadata: {metadata_path}\n"
            "Chạy 'python run_pipeline.py prepare_data' trước."
        )

    out = Path(args.output_dir) if getattr(args, "output_dir", None) else Path("outputs/pcb_synthesis")
    ckpt_dir = Path(output_dir) if output_dir else _resolve_path(
        getattr(args, "camg_out", None), out / "camg_checkpoints"
    )

    trainer = CAMGTrainer(
        device=cfg["device"],
        lr=_float_or(getattr(args, "camg_lr", None), cfg.get("camg_lr", 1e-4)),
        output_dir=str(ckpt_dir),
        num_workers=_int_or(getattr(args, "camg_workers", None), cfg.get("camg_workers", 0)),
    )
    ckpt = trainer.train(
        metadata_path=str(metadata_path),
        epochs=_int_or(getattr(args, "camg_epochs", None), cfg.get("camg_epochs", 10)),
        batch_size=_int_or(getattr(args, "camg_batch", None), cfg.get("camg_batch", 32)),
    )
    if not _file_ok(Path(ckpt), 1024):
        raise RuntimeError(f"CAMG training không tạo checkpoint hợp lệ: {ckpt}")
    logger.info(f"Huấn luyện CAMG hoàn tất -> {ckpt}")
    return str(Path(ckpt).resolve())


def _normal_image_count(img_dir: Path) -> int:
    if not img_dir.exists():
        return 0
    return len(
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.png"))
    )


def run_step1(args, cfg: dict) -> List[dict]:
    logger.info("=" * 66)
    logger.info("STEP 1b — Sinh mask lỗi bằng CAMG (Top-k + size prior)")
    logger.info("=" * 66)

    out = Path(args.output_dir)
    output_dir = out / "masks"
    img_dir = _resolve_path(getattr(args, "image_dir", None), DEFAULT_NORMAL_DIR)

    if _normal_image_count(img_dir) == 0:
        logger.warning(f"Không thấy ảnh normal trong {img_dir}; chạy data prep trước.")
        _ensure_data_prep(args, cfg)
        if _normal_image_count(img_dir) == 0:
            raise FileNotFoundError(
                f"Không có ảnh normal trong '{img_dir}' sau data prep. "
                "Kiểm tra 'data/pcb_normal/' và raw dataset."
            )

    weights = _find_camg_checkpoint(out, getattr(args, "camg_weights", None))
    if weights is None:
        logger.warning(
            "[CAMG] Chưa có camg_best.pth. Tự động huấn luyện CAMG trước khi sinh mask; "
            "TUYỆT ĐỐI không chạy với prediction heads random."
        )
        weights = Path(run_step1_train(args, cfg, output_dir=out / "camg_checkpoints"))

    if not getattr(args, "force_masks", False):
        ok, reason, cached = _validate_mask_cache(
            output_dir, weights, expected_image_count=_normal_image_count(img_dir)
        )
        if ok:
            logger.info(f"[CAMG] Tái sử dụng {len(cached)} mask cache hợp lệ: {output_dir}")
            return cached
        logger.warning(f"[CAMG] Không dùng mask cache cũ: {reason}. Sinh lại.")

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = CAMGGenerator(
        weights_path=str(weights),
        device=cfg["device"],
        k=_int_or(getattr(args, "k", None), 5),
        allow_random_weights=False,
        stats_path=DEFAULT_STATS_PATH,
    )
    records = generator.generate_batch(
        image_dir=str(img_dir),
        output_dir=str(output_dir),
        defect_types=getattr(args, "defect_types", None) or PCB_DEFECT_TYPES,
        images_per_defect=_int_or(getattr(args, "images_per_defect", None), 1),
    )
    if not records:
        raise RuntimeError("CAMG không sinh được mask nào. Kiểm tra ảnh normal/foreground.")

    _prune_stale_masks(output_dir, records)
    stats = _mask_stats(records)
    logger.info(
        f"Step 1 xong: {len(records)} mask | "
        f"W median={stats.get('width_median', 0):.1f}px, "
        f"Q3={stats.get('width_q3', 0):.1f}px"
    )
    if stats.get("width_median", 0) > 60:
        raise RuntimeError(
            "Mask vừa sinh có kích thước bất thường (>60px median). "
            "Kiểm tra dataset_stats.json và MaskSizePrior."
        )
    return records


# ─────────────────────────────────────────────────────────
# STEP 2a: LoRA + TI training
# ─────────────────────────────────────────────────────────
def run_step2_train(args, cfg: dict) -> str:
    logger.info("=" * 66)
    logger.info("STEP 2a — Fine-tune LoRA (U-Net/Text Encoder) + Textual Inversion")
    logger.info("=" * 66)

    metadata_path = _resolve_path(getattr(args, "metadata_path", None), DEFAULT_TRAIN_META)
    if not _file_ok(metadata_path):
        raise FileNotFoundError(
            f"Không tìm thấy train metadata: {metadata_path}\n"
            "Chạy 'python run_pipeline.py prepare_data' trước."
        )

    out = Path(args.output_dir)
    lora_out = out / "lora_checkpoints"
    target_epochs = _int_or(
        getattr(args, "lora_epochs", None), cfg.get("lora_epochs", 10)
    )
    force_lora = bool(getattr(args, "force_lora", False))
    if force_lora:
        _archive_files(lora_out, ["epoch_*", "training_state.json"], "lora")
        existing = None
    else:
        existing = _find_lora_checkpoint(lora_out)

    start_epoch = 0
    if existing is not None and not force_lora:
        state_path = lora_out / "training_state.json"
        state = _load_json(state_path) if state_path.exists() else None
        if state is None:
            # Checkpoint cũ không có state: giữ hành vi cũ là dùng luôn.
            logger.info(f"[LoRA] Checkpoint hợp lệ đã có: {existing}; bỏ qua train.")
            return str(existing)
        completed = int(state.get("completed_epochs", 0))
        if completed >= target_epochs:
            logger.info(
                f"[LoRA] Checkpoint đã đủ {completed}/{target_epochs} epoch; bỏ qua train."
            )
            return str(existing)
        start_epoch = completed
        logger.warning(
            f"[LoRA] Checkpoint mới có {completed}/{target_epochs} epoch; "
            f"resume từ epoch {completed}."
        )

    from pcb_pipeline.step2_lora_inpainting import LoRAInpaintTrainer

    trainer = LoRAInpaintTrainer(
        model_id=getattr(args, "model_id", "runwayml/stable-diffusion-inpainting"),
        rank=_int_or(getattr(args, "lora_rank", None), cfg.get("lora_rank", 8)),
        alpha=_float_or(getattr(args, "lora_alpha", None), cfg.get("lora_alpha", 16.0)),
        lr_unet=_float_or(getattr(args, "lr_unet", None), cfg.get("lr_unet", 1e-4)),
        lr_text_encoder=_float_or(
            getattr(args, "lr_text_encoder", None), cfg.get("lr_text_encoder", 5e-5)
        ),
        lr_ti=_float_or(getattr(args, "lr_ti", None), cfg.get("lr_ti", 5e-4)),
        device=cfg["device"],
        output_dir=str(lora_out),
    )
    ckpt = trainer.train(
        data_json=str(metadata_path),
        epochs=target_epochs,
        batch_size=_int_or(getattr(args, "lora_batch", None), cfg.get("lora_batch", 32)),
        start_epoch=start_epoch,
    )
    ckpt_path = Path(ckpt).resolve()
    if not _lora_checkpoint_valid(ckpt_path):
        raise RuntimeError(
            f"LoRA training xong nhưng checkpoint thiếu pytorch_lora_weights.pt/"
            f"ti_embeddings.pt: {ckpt_path}"
        )
    return str(ckpt_path)


# ─────────────────────────────────────────────────────────
# STEP 2b: Image synthesis
# ─────────────────────────────────────────────────────────
def _balance_mask_records(mask_records: List[dict], num_samples: int, seed: int = 42) -> List[dict]:
    if not num_samples or len(mask_records) <= num_samples:
        return mask_records
    import random

    by_defect: Dict[str, List[dict]] = {}
    for r in mask_records:
        by_defect.setdefault(r.get("defect_type", "unknown"), []).append(r)

    selected: List[dict] = []
    n_types = len(by_defect)
    base_per_type = num_samples // n_types
    remainder = num_samples % n_types
    rng = random.Random(seed)
    for idx, (dtype, records) in enumerate(sorted(by_defect.items())):
        quota = base_per_type + (1 if idx < remainder else 0)
        shuffled = list(records)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:quota])
    return selected


def run_step2_generate(mask_records: List[dict], args, cfg: dict) -> List[dict]:
    from pcb_pipeline.step2_lora_inpainting import PCBDefectInpainter

    out = Path(args.output_dir)
    output_dir = out / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "resume", False):
        ok, reason, cached = _validate_generated_cache(output_dir / "generated_metadata.json")
        if ok:
            logger.info(f"[Step 2b] Dùng {len(cached)} ảnh generated cache hợp lệ.")
            return cached
        logger.warning(f"[Step 2b] Không dùng generated cache: {reason}")

    # ── Resolve LoRA/TI checkpoint, tự train nếu chưa có ──
    explicit_lora = getattr(args, "lora_ckpt", None)
    if explicit_lora:
        p = _resolve_path(explicit_lora)
        if not _lora_checkpoint_valid(p):
            raise FileNotFoundError(
                f"LoRA checkpoint không hợp lệ/không đủ file: {p}. "
                "Cần pytorch_lora_weights.pt và ti_embeddings.pt."
            )
        lora_dir = p
    else:
        lora_dir = _find_lora_checkpoint(out / "lora_checkpoints")
        if lora_dir is None:
            logger.warning("[LoRA] Chưa có checkpoint hợp lệ; tự động huấn luyện LoRA.")
            lora_dir = Path(run_step2_train(args, cfg))

    ti_ckpt = _find_ti_checkpoint(lora_dir, getattr(args, "ti_ckpt", None))
    if ti_ckpt is None:
        raise FileNotFoundError(
            f"Thiếu ti_embeddings.pt trong {lora_dir}. "
            "Không sinh ảnh khi chưa có Textual Inversion của paper."
        )
    args.lora_ckpt = str(lora_dir)

    num_samples = _int_or(getattr(args, "num_samples", None), cfg.get("num_samples", 829))
    if num_samples and len(mask_records) > num_samples:
        logger.info(
            f"[Step 2b] Chọn cân bằng {num_samples} mask từ {len(mask_records)} mask."
        )
        mask_records = _balance_mask_records(mask_records, num_samples, seed=42)

    if not getattr(args, "resume", False):
        _archive_files(output_dir, ["syn_*.png", "generated_metadata.json"], "generated")

    logger.info("=" * 66)
    logger.info(f"STEP 2b — Sinh {len(mask_records)} ảnh lỗi PCB")
    logger.info(
        f"  steps={cfg['inpaint_steps']} | guidance={cfg['guidance_scale']} | "
        f"LoRA={lora_dir}"
    )
    logger.info("=" * 66)

    inpainter = PCBDefectInpainter(
        model_id=getattr(args, "model_id", "runwayml/stable-diffusion-inpainting"),
        lora_ckpt=str(lora_dir),
        ti_ckpt=str(ti_ckpt),
        lora_rank=_int_or(getattr(args, "lora_rank", None), cfg.get("lora_rank", 8)),
        lora_alpha=_float_or(getattr(args, "lora_alpha", None), cfg.get("lora_alpha", 16.0)),
        device=cfg["device"],
        num_inference_steps=_int_or(
            getattr(args, "inpaint_steps", None), cfg.get("inpaint_steps", 50)
        ),
        guidance_scale=_float_or(
            getattr(args, "guidance", None), cfg.get("guidance_scale", 7.5)
        ),
        negative_prompt=getattr(args, "negative_prompt", None)
        or cfg.get("negative_prompt", DEFAULT_CONFIG["negative_prompt"]),
        allow_fallback=False,
    )

    t0 = time.time()
    results = inpainter.generate_batch(mask_records, output_dir)
    elapsed = time.time() - t0
    per_img = elapsed / max(len(results), 1)
    logger.info(f"Step 2b xong: {len(results)} ảnh | {per_img:.1f}s/ảnh")
    if not results:
        raise RuntimeError(
            "Step 2b không sinh được ảnh nào. Kiểm tra mask đầu vào, VRAM và checkpoint LoRA/TI."
        )

    bad = [
        r for r in results
        if r.get("generation_mode") != "diffusion"
        or not r.get("using_lora")
        or not r.get("ti_loaded")
    ]
    if bad:
        raise RuntimeError(
            f"Có {len(bad)} ảnh không đúng cấu hình diffusion+LoRA+TI. "
            "Không đưa các ảnh fallback vào benchmark."
        )
    return results


# ─────────────────────────────────────────────────────────
# STEP 3: Quality filtering
# ─────────────────────────────────────────────────────────
def _find_yolo_judge(out: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = _resolve_path(explicit)
        if _file_ok(p, 1024):
            return p
        raise FileNotFoundError(f"Không tìm thấy YOLO judge weights: {p}")
    candidates = [
        out / "yolo_training" / "pcb_defect_judge" / "weights" / "best.pt",
        out / "yolo_training" / "weights" / "best.pt",
    ]
    for p in candidates:
        if _file_ok(p, 1024):
            return p.resolve()
    return None


def run_step3(generated_records: List[dict], args, cfg: dict) -> List[dict]:
    from pcb_pipeline.step3_quality_filter import (
        YOLO11DefectTrainer,
        YOLO11ConfidenceScorer,
        QualityFilter,
    )

    logger.info("=" * 66)
    logger.info("STEP 3 — Lọc ảnh bằng YOLO11 (loại 25% confidence thấp nhất)")
    logger.info("=" * 66)

    out = Path(args.output_dir)
    output_dir = out / "filtered"
    data_yaml = _resolve_path(getattr(args, "data_yaml", None), DATA_DIR / "id_setup.yaml")
    if not _file_ok(data_yaml):
        raise FileNotFoundError(f"Không tìm thấy dataset YAML: {data_yaml}")

    yolo_weights = _find_yolo_judge(out, getattr(args, "yolo_weights", None))
    if yolo_weights is None:
        logger.info("[YOLO Judge] Chưa có weights; huấn luyện YOLO11 judge trên real low-data.")
        trainer = YOLO11DefectTrainer(
            model_size=cfg.get("yolo_size", "s"),
            device=cfg["device"],
            project=str(out / "yolo_training"),
        )
        yolo_weights = Path(
            trainer.train(
                data_yaml,
                epochs=_int_or(getattr(args, "yolo_epochs", None), cfg.get("yolo_epochs", 100)),
                batch=8,
                imgsz=600,
            )
        )

    scorer = YOLO11ConfidenceScorer(weights_path=yolo_weights, device=cfg["device"])
    setattr(args, "step3_scorer_mode", "heuristic_sobel" if scorer.use_heuristic else "yolo11")
    if scorer.use_heuristic:
        logger.warning(
            "[Step 3] YOLO11 judge không khả dụng (stub/CPU/thiếu ultralytics). "
            "Đang dùng heuristic Sobel/contrast. Kết quả lọc KHÔNG phải YOLO11 judge "
            "và phải ghi rõ trong báo cáo."
        )

    if not getattr(args, "resume", False):
        _archive_files(
            output_dir,
            [
                "accepted", "rejected", "classes", "augmented_train.txt",
                "augmented_val.txt", "augmented_dataset.yaml",
                "accepted_metadata.json", "rejected_metadata.json",
            ],
            "filtered",
        )

    flt = QualityFilter(scorer, reject_ratio=cfg.get("reject_ratio", 0.25))
    if getattr(args, "by_class", False):
        accepted, rejected = flt.filter_by_class(
            generated_records, output_dir, data_yaml=data_yaml
        )
    else:
        accepted, rejected = flt.filter(
            generated_records, output_dir, data_yaml=data_yaml
        )

    if not accepted:
        raise RuntimeError("Step 3 không giữ được ảnh synthetic nào.")
    aug_yaml = output_dir / "augmented_dataset.yaml"
    if not _file_ok(aug_yaml):
        raise RuntimeError(f"Step 3 không tạo được augmented YAML: {aug_yaml}")

    logger.info(f"Step 3 xong: {len(accepted)} accepted / {len(rejected)} rejected")
    logger.info(f"   Augmented Dataset YAML: {aug_yaml.resolve()}")
    return accepted


# ─────────────────────────────────────────────────────────
# STEP 4: Downstream benchmark
# ─────────────────────────────────────────────────────────
def run_step4(args, cfg: dict) -> dict:
    from pcb_pipeline.step4_benchmark import DownstreamBenchmarker

    logger.info("=" * 66)
    logger.info("STEP 4 — Downstream Benchmark (CHỈ YOLO11s)")
    logger.info("=" * 66)

    out_base = Path(args.output_dir)
    bench_out = out_base if out_base.name == "benchmark" else out_base / "benchmark"
    split_name = getattr(args, "split", "id")
    base_yaml = _resolve_path(
        getattr(args, "base_yaml", None),
        DATA_DIR / ("id_setup.yaml" if split_name == "id" else "npi_setup.yaml"),
    )

    aug_yaml = getattr(args, "aug_yaml", None)
    if aug_yaml:
        aug_yaml = _resolve_path(aug_yaml)
    else:
        candidate = out_base / "filtered" / "augmented_dataset.yaml"
        if not _file_ok(candidate):
            raise FileNotFoundError(
                f"Không tìm thấy augmented epoch YAML: {candidate}. Chạy Step 3 trước."
            )
        aug_yaml = candidate.resolve()

    benchmarker = DownstreamBenchmarker(
        output_dir=bench_out,
        device=cfg["device"],
        epochs=_int_or(
            getattr(args, "epochs", None), cfg.get("benchmark_epochs", 100)
        ),
        batch_size=_int_or(
            getattr(args, "batch_size", None), cfg.get("benchmark_batch", 8)
        ),
        imgsz=_int_or(
            getattr(args, "imgsz", None), cfg.get("benchmark_imgsz", 600)
        ),
        lr0=_float_or(
            getattr(args, "lr0", None), cfg.get("benchmark_lr0", 1e-4)
        ),
        conf_thresh=_float_or(
            getattr(args, "conf", None), cfg.get("benchmark_conf", 0.25)
        ),
        iou_thresh=_float_or(
            getattr(args, "iou", None), cfg.get("benchmark_iou", 0.7)
        ),
        workers=getattr(args, "workers", None)
        if getattr(args, "workers", None) is not None
        else cfg.get("benchmark_workers"),
    )
    return benchmarker.run_benchmark(
        models=getattr(args, "models", None)
        or cfg.get("benchmark_models", ["yolo11s"]),
        base_yaml=str(base_yaml),
        aug_yaml=str(aug_yaml),
        split_name=split_name,
        epochs=_int_or(getattr(args, "epochs", None), cfg.get("benchmark_epochs", 100)),
        batch_size=_int_or(
            getattr(args, "batch_size", None), cfg.get("benchmark_batch", 8)
        ),
    )


# ─────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────
def run_full_pipeline(args, cfg: dict):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out)

    manager = PipelineCheckpointManager(out)
    log_gpu_info()

    logger.info("Bắt đầu PCB Defect Synthesis Pipeline")
    logger.info(f"   Output: {out}")
    logger.info("\n" + manager.render_dashboard())
    t_start = time.time()

    resume = bool(getattr(args, "resume", False))
    from_st = getattr(args, "from_stage", None)
    to_st = getattr(args, "to_stage", None)

    # ── STAGE 0: Data prep ──
    data_summary = {}
    if manager.should_execute("0", from_st, to_st):
        manager.mark_stage("0", "RUNNING")
        t0 = time.time()
        data_summary = _ensure_data_prep(
            args, cfg, force=bool(getattr(args, "force_data_prep", False))
        )
        manager.mark_stage(
            "0", "COMPLETED", elapsed=time.time() - t0,
            artifact=str(DATA_DIR / "data_prep_summary.json"),
            extra={
                "id_train_low": data_summary.get("id_train_low_count"),
                "normals": data_summary.get("normal_images_count"),
            },
        )
    else:
        logger.info("[SKIP] Bỏ qua Stage 0 theo cấu hình stage.")
        manager.mark_stage("0", "SKIPPED")
        ok, reason = _validate_data_prep(DATA_DIR, int(cfg.get("normal_target", 799)))
        if not ok:
            raise RuntimeError(
                f"Stage 0 bị bỏ qua nhưng data prep không hợp lệ: {reason}. "
                "Chạy 'python run_pipeline.py prepare_data' trước."
            )
        data_summary = _load_json(DATA_DIR / "data_prep_summary.json")

    # ── STAGE 1: CAMG masks ──
    mask_records: List[dict] = []
    mask_meta_path = out / "masks" / "mask_metadata.json"
    if manager.should_execute("1", from_st, to_st):
        manager.mark_stage("1", "RUNNING")
        t0 = time.time()
        mask_records = run_step1(args, cfg)
        manager.mark_stage(
            "1", "COMPLETED", elapsed=time.time() - t0,
            artifact=str(mask_meta_path), extra={"count": len(mask_records)},
        )
    else:
        logger.info("[SKIP] Bỏ qua Stage 1 theo cấu hình stage.")
        manager.mark_stage("1", "SKIPPED")
        if _file_ok(mask_meta_path):
            mask_records = _load_json(mask_meta_path)

    # ── STAGE 2a: LoRA checkpoint ──
    # Không tự bỏ qua chỉ vì đã thấy epoch_*: checkpoint có thể mới train một
    # phần. run_step2_train() đọc training_state.json để skip hoặc resume đúng.
    lora_dir: Optional[Path] = None
    explicit_lora = getattr(args, "lora_ckpt", None)
    if explicit_lora:
        explicit_path = _resolve_path(explicit_lora)
        if not _lora_checkpoint_valid(explicit_path):
            raise FileNotFoundError(
                f"--lora_ckpt không hợp lệ/không đủ file: {explicit_path}. "
                "Cần pytorch_lora_weights.pt và ti_embeddings.pt."
            )
        lora_dir = explicit_path
    if manager.should_execute("2a", from_st, to_st):
        if explicit_lora and lora_dir is not None and not getattr(args, "force_lora", False):
            logger.info(f"[LoRA] Dùng checkpoint chỉ định: {lora_dir}")
            manager.mark_stage("2a", "COMPLETED", artifact=str(lora_dir))
        else:
            manager.mark_stage("2a", "RUNNING")
            t0 = time.time()
            lora_dir = Path(run_step2_train(args, cfg))
            manager.mark_stage(
                "2a", "COMPLETED", elapsed=time.time() - t0, artifact=str(lora_dir)
            )
    else:
        logger.info("[SKIP] Bỏ qua Stage 2a theo cấu hình stage.")
        manager.mark_stage("2a", "SKIPPED")
        if lora_dir is None:
            lora_dir = _find_lora_checkpoint(out / "lora_checkpoints")
    if lora_dir is not None:
        args.lora_ckpt = str(lora_dir)
    elif manager.should_execute("2b", from_st, to_st) and not getattr(args, "lora_ckpt", None):
        raise RuntimeError("Không có LoRA checkpoint hợp lệ để chạy Step 2b.")

    # ── STAGE 2b: Diffusion generation ──
    gen_records: List[dict] = []
    gen_meta_path = out / "generated" / "generated_metadata.json"
    if manager.should_execute("2b", from_st, to_st):
        reused = False
        if resume:
            ok, reason, cached = _validate_generated_cache(gen_meta_path)
            if ok:
                logger.info(f"[RESUME] Dùng {len(cached)} ảnh generated hợp lệ: {gen_meta_path}")
                gen_records = cached
                manager.mark_stage(
                    "2b", "COMPLETED", artifact=str(gen_meta_path),
                    extra={"count": len(cached)},
                )
                reused = True
            else:
                logger.warning(f"[RESUME] Không dùng generated cache: {reason}")
        if not reused:
            if not mask_records:
                raise RuntimeError("Không có mask_records từ Stage 1 để sinh ảnh.")
            manager.mark_stage("2b", "RUNNING")
            t0 = time.time()
            gen_records = run_step2_generate(mask_records, args, cfg)
            manager.mark_stage(
                "2b", "COMPLETED", elapsed=time.time() - t0,
                artifact=str(gen_meta_path), extra={"count": len(gen_records)},
            )
    else:
        logger.info("[SKIP] Bỏ qua Stage 2b theo cấu hình stage.")
        manager.mark_stage("2b", "SKIPPED")
        if _file_ok(gen_meta_path):
            gen_records = _load_json(gen_meta_path)

    # ── STAGE 3: Quality filter ──
    final_records: List[dict] = []
    filtered_dir = out / "filtered"
    aug_yaml_path = filtered_dir / "augmented_dataset.yaml"
    if manager.should_execute("3", from_st, to_st):
        reused = False
        if resume:
            ok, reason, cached = _validate_filtered_cache(out)
            if ok:
                logger.info(f"[RESUME] Dùng {len(cached)} ảnh accepted hợp lệ: {filtered_dir}")
                final_records = cached
                manager.mark_stage(
                    "3", "COMPLETED", artifact=str(aug_yaml_path),
                    extra={"count": len(cached)},
                )
                reused = True
            else:
                logger.warning(f"[RESUME] Không dùng filtered cache: {reason}")
        if not reused:
            if not gen_records:
                raise RuntimeError("Không có gen_records từ Stage 2b để lọc chất lượng.")
            manager.mark_stage("3", "RUNNING")
            t0 = time.time()
            final_records = run_step3(gen_records, args, cfg)
            manager.mark_stage(
                "3", "COMPLETED", elapsed=time.time() - t0,
                artifact=str(aug_yaml_path), extra={"count": len(final_records)},
            )
    else:
        logger.info("[SKIP] Bỏ qua Stage 3 theo cấu hình stage.")
        manager.mark_stage("3", "SKIPPED")
        if _file_ok(filtered_dir / "accepted_metadata.json"):
            final_records = _load_json(filtered_dir / "accepted_metadata.json")

    # ── STAGE 4: Benchmark ──
    benchmark_results = None
    bench_report_path = out / "benchmark" / "benchmark_report.md"
    if getattr(args, "run_benchmark", False) and manager.should_execute("4", from_st, to_st):
        if resume and _validate_benchmark_summary(out):
            logger.info(f"[RESUME] Benchmark report hợp lệ đã có: {bench_report_path}")
            manager.mark_stage("4", "COMPLETED", artifact=str(bench_report_path))
            benchmark_results = _load_json(out / "benchmark" / "benchmark_summary.json")
        else:
            manager.mark_stage("4", "RUNNING")
            t0 = time.time()
            args.models = getattr(args, "benchmark_models", None) or ["yolo11s"]
            args.epochs = getattr(args, "benchmark_epochs", 100)
            args.batch_size = getattr(args, "benchmark_batch", 8)
            args.lr0 = getattr(args, "benchmark_lr0", 1e-4)
            args.conf = getattr(args, "benchmark_conf", 0.25)
            args.iou = getattr(args, "benchmark_iou", 0.7)
            args.imgsz = getattr(args, "benchmark_imgsz", 600)
            args.workers = getattr(args, "benchmark_workers", None)
            args.split = getattr(args, "split", "id")
            args.aug_yaml = str(aug_yaml_path)
            benchmark_results = run_step4(args, cfg)
            manager.mark_stage(
                "4", "COMPLETED", elapsed=time.time() - t0, artifact=str(bench_report_path)
            )
    else:
        if not getattr(args, "run_benchmark", False):
            manager.mark_stage("4", "SKIPPED", artifact="Không kích hoạt (--run_benchmark)")
        else:
            manager.mark_stage("4", "SKIPPED")

    # ── Summary ──
    elapsed = time.time() - t_start
    logger.info("\n" + manager.render_dashboard())
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION HOÀN TẤT")
    logger.info(f"   Tổng thời gian:          {elapsed/60:.1f} phút")
    logger.info(f"   Ảnh sau lọc (Accepted):  {len(final_records)}")
    if aug_yaml_path.exists():
        logger.info(f"   Augmented YAML:          {aug_yaml_path.resolve()}")
    if benchmark_results:
        logger.info(f"   Benchmark Report:        {bench_report_path.resolve()}")
    logger.info(f"   Output Directory:        {out}")
    logger.info("=" * 60)

    mask_stats = _mask_stats(mask_records)
    limitations = [
        "Chỉ có 7 seen boards trong repo (01,04-09), không có board 02/03 như paper.",
        "Benchmark chỉ chạy YOLO11s, không chạy YOLOv8/YOLOv9/RT-DETR.",
        "Không tuyên bố tái lập đầy đủ Table 3/6 của paper.",
        "Step 3 dùng YOLO11 judge nếu có GPU/weights; nếu không sẽ dùng heuristic và đã cảnh báo.",
    ]
    if getattr(args, "step3_scorer_mode", None) == "heuristic_sobel":
        limitations.append(
            "Lần chạy này Step 3 dùng heuristic Sobel/contrast, KHÔNG phải YOLO11 judge."
        )

    report = {
        "split_protocol": SPLIT_PROTOCOL,
        "normal_protocol": NORMAL_PROTOCOL,
        "total_masks": len(mask_records),
        "mask_size_stats": mask_stats,
        "total_generated": len(gen_records),
        "total_accepted": len(final_records),
        "rejected_ratio": cfg["reject_ratio"],
        "step3_scorer_mode": getattr(args, "step3_scorer_mode", "not_run"),
        "augmented_yaml": str(aug_yaml_path.resolve()) if aug_yaml_path.exists() else None,
        "benchmark_executed": bool(benchmark_results),
        "benchmark_models": getattr(args, "benchmark_models", ["yolo11s"]),
        "benchmark_conf": getattr(args, "benchmark_conf", 0.25),
        "benchmark_nms_iou": getattr(args, "benchmark_iou", 0.7),
        "elapsed_minutes": round(elapsed / 60, 1),
        "data_prep": data_summary,
        "config": cfg,
        "stages": manager.status_data.get("stages", {}),
        "limitations": limitations,
    }
    (out / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return final_records


def run_prepare_data(args, cfg: dict):
    logger.info("=" * 66)
    logger.info("PREPARE DATA — ID/NPI split + ảnh normal sạch")
    logger.info("=" * 66)
    raw_dir = _resolve_path(getattr(args, "raw_dir", None), DEFAULT_RAW_DIR)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy raw dataset: '{raw_dir}'.\n"
            "Hãy giải nén DeepPCB vào 'data/pcb-defect-dataset/'."
        )
    force = bool(getattr(args, "force", False))
    summary = _ensure_data_prep(args, cfg, force=force)
    logger.info("Dữ liệu đã sẵn sàng:")
    logger.info(f"  ID train low: {summary.get('id_train_low_count')}")
    logger.info(f"  ID val:       {summary.get('id_val_count')}")
    logger.info(f"  NPI val:      {summary.get('npi_unseen_val_count')}")
    logger.info(f"  Normals:      {summary.get('normal_images_count')}")
    logger.info(f"  Stats:        {DEFAULT_STATS_PATH.resolve()}")


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

    def add_common(p):
        p.add_argument("--output_dir", default="outputs/pcb_synthesis")
        p.add_argument("--model_id", default="runwayml/stable-diffusion-inpainting")
        p.add_argument("--device", default="cuda")
        p.add_argument("--config", default=None, help="JSON config file")
        p.add_argument("--defect_types", nargs="+", default=None)

    p_prep = subparsers.add_parser(
        "prepare_data", help="Tạo ID/NPI splits và ảnh normal sạch"
    )
    p_prep.add_argument("--raw_dir", default="data/pcb-defect-dataset")
    # Data prep luôn ghi vào data/ vì Step 1-4 dùng cố định data/id_setup.yaml,
    # data/npi_setup.yaml, data/dataset_stats.json và data/pcb_normal/.
    p_prep.add_argument("--normal_target", type=int, default=799)
    p_prep.add_argument("--low_data_ratio", type=float, default=0.1)
    p_prep.add_argument("--force", action="store_true", help="Tạo lại dù artifact cũ hợp lệ")

    p_full = subparsers.add_parser("full", help="Chạy toàn bộ pipeline")
    add_common(p_full)
    p_full.add_argument("--raw_dir", default="data/pcb-defect-dataset")
    p_full.add_argument("--image_dir", default="data/pcb_normal")
    p_full.add_argument("--real_defect_dir", default=None, help="Giữ để tương thích CLI cũ")
    p_full.add_argument("--metadata_path", default="data/train_metadata.json")
    p_full.add_argument("--camg_weights", default=None)
    p_full.add_argument("--camg_out", default=None)
    p_full.add_argument("--camg_epochs", type=int, default=None)
    p_full.add_argument("--camg_batch", type=int, default=None)
    p_full.add_argument("--camg_lr", type=float, default=None)
    p_full.add_argument("--camg_workers", type=int, default=None)
    p_full.add_argument("--k", type=int, default=5)
    p_full.add_argument("--images_per_defect", type=int, default=1)
    p_full.add_argument("--mask_strategy", default="saliency", choices=["saliency", "random", "grid"])
    p_full.add_argument("--lora_ckpt", default=None)
    p_full.add_argument("--ti_ckpt", default=None)
    p_full.add_argument("--lora_rank", type=int, default=None)
    p_full.add_argument("--lora_alpha", type=float, default=None)
    p_full.add_argument("--lora_epochs", type=int, default=None)
    p_full.add_argument("--lora_batch", type=int, default=None)
    p_full.add_argument("--lr_unet", type=float, default=None)
    p_full.add_argument("--lr_text_encoder", type=float, default=None)
    p_full.add_argument("--lr_ti", type=float, default=None)
    p_full.add_argument("--inpaint_steps", type=int, default=None)
    p_full.add_argument("--guidance", type=float, default=None)
    p_full.add_argument("--negative_prompt", default=None)
    p_full.add_argument("--num_samples", type=int, default=829)
    p_full.add_argument("--yolo_weights", default=None)
    p_full.add_argument("--data_yaml", default="data/id_setup.yaml")
    p_full.add_argument("--yolo_size", default=None, choices=["n", "s", "m", "l", "x"])
    p_full.add_argument("--yolo_epochs", type=int, default=None)
    p_full.add_argument("--reject_ratio", type=float, default=None)
    p_full.add_argument("--by_class", action="store_true")
    p_full.add_argument("--run_benchmark", action="store_true")
    p_full.add_argument("--benchmark_models", nargs="+", default=["yolo11s"])
    p_full.add_argument("--benchmark_epochs", type=int, default=100)
    p_full.add_argument("--benchmark_batch", type=int, default=8)
    p_full.add_argument("--benchmark_lr0", type=float, default=1e-4)
    p_full.add_argument("--benchmark_conf", type=float, default=0.25)
    p_full.add_argument("--benchmark_iou", type=float, default=0.7, help="NMS IoU (0.5-0.7)")
    p_full.add_argument("--benchmark_imgsz", type=int, default=600)
    p_full.add_argument("--benchmark_workers", type=int, default=None)
    p_full.add_argument("--split", choices=["id", "npi"], default="id")
    p_full.add_argument("--resume", action="store_true")
    p_full.add_argument("--force_data_prep", action="store_true")
    p_full.add_argument("--force_masks", action="store_true")
    p_full.add_argument("--force_lora", action="store_true")
    p_full.add_argument("--from_stage", default=None)
    p_full.add_argument("--to_stage", default=None)

    p_stat = subparsers.add_parser("status", help="Xem dashboard trạng thái pipeline")
    p_stat.add_argument("--output_dir", default="outputs/pcb_synthesis")

    p1t = subparsers.add_parser("step1_train", help="Huấn luyện CAMG prediction heads")
    add_common(p1t)
    p1t.add_argument("--metadata_path", default="data/train_metadata.json")
    p1t.add_argument("--camg_out", default=None)
    p1t.add_argument("--camg_epochs", type=int, default=None)
    p1t.add_argument("--camg_batch", type=int, default=None)
    p1t.add_argument("--camg_lr", type=float, default=None)
    p1t.add_argument("--camg_workers", type=int, default=None)

    p1 = subparsers.add_parser("step1", help="Sinh mask bằng CAMG (tự train nếu thiếu)")
    add_common(p1)
    p1.add_argument("--image_dir", default="data/pcb_normal")
    p1.add_argument("--camg_weights", default=None)
    p1.add_argument("--k", type=int, default=5)
    p1.add_argument("--images_per_defect", type=int, default=1)
    p1.add_argument("--force_masks", action="store_true")

    p2t = subparsers.add_parser("step2_train", help="Fine-tune LoRA + TI")
    add_common(p2t)
    p2t.add_argument("--metadata_path", default="data/train_metadata.json")
    p2t.add_argument("--lora_rank", type=int, default=None)
    p2t.add_argument("--lora_alpha", type=float, default=None)
    p2t.add_argument("--lora_epochs", type=int, default=None)
    p2t.add_argument("--lora_batch", type=int, default=None)
    p2t.add_argument("--lr_unet", type=float, default=None)
    p2t.add_argument("--lr_text_encoder", type=float, default=None)
    p2t.add_argument("--lr_ti", type=float, default=None)
    p2t.add_argument("--force_lora", action="store_true")

    p2g = subparsers.add_parser("step2_generate", help="Sinh ảnh lỗi (tự train LoRA nếu thiếu)")
    add_common(p2g)
    p2g.add_argument("--mask_meta", required=True)
    p2g.add_argument("--lora_ckpt", default=None)
    p2g.add_argument("--ti_ckpt", default=None)
    p2g.add_argument("--lora_rank", type=int, default=None)
    p2g.add_argument("--lora_alpha", type=float, default=None)
    p2g.add_argument("--inpaint_steps", type=int, default=None)
    p2g.add_argument("--guidance", type=float, default=None)
    p2g.add_argument("--negative_prompt", default=None)
    p2g.add_argument("--num_samples", type=int, default=829)
    p2g.add_argument("--resume", action="store_true")

    p3 = subparsers.add_parser("step3", help="Lọc ảnh bằng YOLO11 judge")
    add_common(p3)
    p3.add_argument("--gen_meta", required=True)
    p3.add_argument("--yolo_weights", default=None)
    p3.add_argument("--data_yaml", default="data/id_setup.yaml")
    p3.add_argument("--yolo_size", default=None, choices=["n", "s", "m", "l", "x"])
    p3.add_argument("--yolo_epochs", type=int, default=None)
    p3.add_argument("--reject_ratio", type=float, default=None)
    p3.add_argument("--by_class", action="store_true")
    p3.add_argument("--resume", action="store_true")

    p4 = subparsers.add_parser("step4", help="Downstream benchmark (chỉ YOLO11s)")
    add_common(p4)
    p4.add_argument("--models", nargs="+", default=["yolo11s"])
    p4.add_argument("--split", choices=["id", "npi"], default="id")
    p4.add_argument("--base_yaml", default=None)
    p4.add_argument("--aug_yaml", default=None)
    p4.add_argument("--epochs", type=int, default=100)
    p4.add_argument("--batch_size", type=int, default=8)
    p4.add_argument("--lr0", type=float, default=1e-4)
    p4.add_argument("--conf", type=float, default=0.25)
    p4.add_argument("--iou", type=float, default=0.7, help="NMS IoU (0.5-0.7), không phải match IoU 0.25")
    p4.add_argument("--imgsz", type=int, default=600)
    p4.add_argument("--workers", type=int, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(
        getattr(args, "config", None),
        {
            "device": getattr(args, "device", "cuda"),
            "normal_target": getattr(args, "normal_target", None),
            "low_data_ratio": getattr(args, "low_data_ratio", None),
            "camg_epochs": getattr(args, "camg_epochs", None),
            "camg_batch": getattr(args, "camg_batch", None),
            "camg_lr": getattr(args, "camg_lr", None),
            "camg_workers": getattr(args, "camg_workers", None),
            "mask_strategy": getattr(args, "mask_strategy", None),
            "images_per_defect": getattr(args, "images_per_defect", None),
            "lora_rank": getattr(args, "lora_rank", None),
            "lora_alpha": getattr(args, "lora_alpha", None),
            "lora_epochs": getattr(args, "lora_epochs", None),
            "lora_batch": getattr(args, "lora_batch", None),
            "lr_unet": getattr(args, "lr_unet", None),
            "lr_text_encoder": getattr(args, "lr_text_encoder", None),
            "lr_ti": getattr(args, "lr_ti", None),
            "inpaint_steps": getattr(args, "inpaint_steps", None),
            "guidance_scale": getattr(args, "guidance", None),
            "negative_prompt": getattr(args, "negative_prompt", None),
            "reject_ratio": getattr(args, "reject_ratio", None),
            "yolo_size": getattr(args, "yolo_size", None),
            "yolo_epochs": getattr(args, "yolo_epochs", None),
            "benchmark_models": getattr(args, "benchmark_models", None),
            "benchmark_epochs": getattr(args, "benchmark_epochs", None),
            "benchmark_batch": getattr(args, "benchmark_batch", None),
            "benchmark_lr0": getattr(args, "benchmark_lr0", None),
            "benchmark_conf": getattr(args, "benchmark_conf", None),
            "benchmark_iou": getattr(args, "benchmark_iou", None),
            "benchmark_imgsz": getattr(args, "benchmark_imgsz", None),
            "benchmark_workers": getattr(args, "benchmark_workers", None),
        },
    )
    cfg["device"] = _resolve_runtime_device(cfg["device"])

    if args.command == "full":
        run_full_pipeline(args, cfg)
    elif args.command == "step1_train":
        run_step1_train(args, cfg)
    elif args.command == "step1":
        run_step1(args, cfg)
    elif args.command == "step2_train":
        run_step2_train(args, cfg)
    elif args.command == "step2_generate":
        mask_records = _load_json(_resolve_path(args.mask_meta))
        run_step2_generate(mask_records, args, cfg)
    elif args.command == "step3":
        gen_records = _load_json(_resolve_path(args.gen_meta))
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
