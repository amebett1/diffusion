"""
Kiểm tra môi trường đã sẵn sàng để train/chạy pipeline hay chưa.

Script này KHÔNG train, KHÔNG sinh ảnh và KHÔNG sửa dữ liệu.
Chỉ kiểm tra:
- phiên bản Python / PyTorch / CUDA / VRAM
- các package cần thiết
- data prep, split và 799 ảnh normal
- cache Stable Diffusion Inpainting (fp16), DINOv2, YOLO11s
- output cũ có artifact không hợp lệ hay không

Cách dùng:
    python verify_setup.py
    python verify_setup.py --download-models   # tải model còn thiếu
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pcb_synthesis"
SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "ultralytics",
    "cv2",
    "PIL",
    "numpy",
    "tqdm",
    "yaml",
]

SD_REQUIRED_FILES = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/vocab.json",
    "scheduler/scheduler_config.json",
]

ERRORS: list[str] = []
WARNINGS: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    ERRORS.append(msg)
    print(f"  [FAIL] {msg}")


def check_python_and_packages() -> None:
    print("\n[1] Python và package")
    print(f"  Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        fail("Cần Python >= 3.10")
    else:
        ok("Python >= 3.10")

    missing = [m for m in REQUIRED_PACKAGES if importlib.util.find_spec(m) is None]
    if missing:
        fail("Thiếu package: " + ", ".join(missing))
        print("         Cài đặt: python -m pip install -r requirements.txt")
    else:
        ok("Đủ package bắt buộc")

    try:
        import torch
        import torchvision
        import diffusers
        import transformers
        import accelerate

        ok(
            f"torch={torch.__version__}, torchvision={torchvision.__version__}, "
            f"diffusers={diffusers.__version__}, transformers={transformers.__version__}, "
            f"accelerate={accelerate.__version__}"
        )
    except Exception as exc:
        fail(f"Không import được package chính: {exc}")
        return

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / (1024**3)
        ok(f"CUDA khả dụng: {props.name} ({vram:.1f} GB VRAM)")
        if vram < 5.5:
            warn("VRAM < 5.5 GB; Step 2 có thể không đủ, nên chạy trên Colab/Kaggle.")
        elif vram < 7.0:
            warn("VRAM 6 GB: dùng fp16 + gradient checkpointing, batch vật lý = 1; chạy được nhưng chậm.")
    else:
        warn("Không thấy CUDA; pipeline sẽ chạy CPU và rất chậm.")


def check_data() -> None:
    print("\n[2] Dữ liệu và split")
    if not (DATA_DIR / "pcb-defect-dataset").exists():
        fail("Thiếu data/pcb-defect-dataset/")
        return

    try:
        from run_pipeline import _validate_data_prep

        valid, reason = _validate_data_prep(DATA_DIR, 799)
    except Exception as exc:
        fail(f"Không chạy được kiểm tra data prep: {exc}")
        return

    if valid:
        ok(f"Data prep hợp lệ: {reason}")
    else:
        fail(f"Data prep chưa hợp lệ: {reason}")
        print("         Chạy: python run_pipeline.py prepare_data --force")
        return

    normals = list((DATA_DIR / "pcb_normal").glob("*.jpg"))
    if len(normals) == 799:
        ok("Đủ 799 ảnh normal")
    else:
        fail(f"Số ảnh normal = {len(normals)}, cần 799")

    for name, expected in [
        ("id_train_low.txt", 534),
        ("id_val.txt", 941),
        ("npi_val_unseen.txt", 550),
    ]:
        p = DATA_DIR / "prepared_splits" / name
        n = len(p.read_text(encoding="utf-8").splitlines()) if p.exists() else 0
        if n == expected:
            ok(f"{name}: {n} dòng")
        else:
            fail(f"{name}: {n} dòng, cần {expected}")


def check_models(download: bool) -> None:
    print("\n[3] Model và cache")
    from huggingface_hub import try_to_load_from_cache, snapshot_download

    missing = []
    for rel in SD_REQUIRED_FILES:
        cached = try_to_load_from_cache(SD_MODEL_ID, rel)
        if not isinstance(cached, str) or not Path(cached).exists():
            missing.append(rel)

    if missing:
        if download:
            print(f"  Đang tải {SD_MODEL_ID} (fp16, khoảng 2.7 GB)...")
            snapshot_download(
                repo_id=SD_MODEL_ID,
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "*.safetensors",
                    "*.model",
                    "*.yaml",
                    "tokenizer/*",
                    "scheduler/*",
                    "feature_extractor/*",
                ],
                ignore_patterns=["*.ckpt", "*.bin"],
            )
            missing = [
                rel
                for rel in SD_REQUIRED_FILES
                if not isinstance(try_to_load_from_cache(SD_MODEL_ID, rel), str)
            ]
        if missing:
            fail("Thiếu cache Stable Diffusion Inpainting: " + ", ".join(missing[:4]))
            print("         Cài mạng rồi chạy: python verify_setup.py --download-models")
        else:
            ok("Stable Diffusion Inpainting fp16 đã có trong cache")
    else:
        ok("Stable Diffusion Inpainting fp16 đã có trong cache")

    import torch

    dino_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    dino_weight = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vitb14_pretrain.pth"
    if dino_repo.exists() and dino_weight.exists():
        ok(f"DINOv2 ViT-B/14 đã có cache ({dino_weight.stat().st_size / 1024**2:.0f} MB)")
    else:
        warn("Chưa thấy cache DINOv2; lần train Step 1 đầu tiên sẽ tải (~330 MB).")

    yolo = PROJECT_ROOT / "yolo11s.pt"
    if yolo.exists():
        ok(f"YOLO11s weights: {yolo.name} ({yolo.stat().st_size / 1024**2:.1f} MB)")
    else:
        warn("Chưa có yolo11s.pt; Ultralytics sẽ tự tải khi chạy Step 3/4.")


def check_outputs() -> None:
    print("\n[4] Output và checkpoint")
    if not OUTPUT_DIR.exists():
        ok("outputs/pcb_synthesis chưa tồn tại (sẽ được tạo khi chạy)")
    else:
        stale = []
        mask_dir = OUTPUT_DIR / "masks"
        if mask_dir.exists():
            for p in mask_dir.glob("*.png"):
                name = p.name
                if any(f"normal_{b}_" in name for b in ("10", "11", "12")):
                    stale.append(p.name)
                    if len(stale) >= 5:
                        break
        if stale:
            fail(
                "Có mask của unseen boards 10/11/12 trong output active: "
                + ", ".join(stale)
            )
            print("         Chạy lại Step 1 để archive/regenerate mask.")
        else:
            ok("Không thấy mask unseen boards trong output active")

        camg = OUTPUT_DIR / "camg_checkpoints" / "camg_best.pth"
        lora = OUTPUT_DIR / "lora_checkpoints"
        if camg.exists():
            ok(f"CAMG checkpoint: {camg}")
        else:
            warn("Chưa có CAMG checkpoint production; pipeline sẽ tự train Step 1.")
        if lora.exists() and any(lora.glob("epoch_*/pytorch_lora_weights.pt")):
            ok(f"LoRA checkpoint: {lora}")
        else:
            warn("Chưa có LoRA checkpoint production; pipeline sẽ tự train Step 2a.")

    if (PROJECT_ROOT / "outputs" / "_stale_invalid_pcb_synthesis_20261022").exists():
        warn("Có thư mục artifact cũ đã archive; không dùng nó cho benchmark.")


def print_summary() -> int:
    print("\n" + "=" * 70)
    if ERRORS:
        print(f"KẾT QUẢ: CHƯA SẴN SÀNG ({len(ERRORS)} lỗi, {len(WARNINGS)} cảnh báo)")
        for e in ERRORS:
            print(f"  - {e}")
        return 1

    print(f"KẾT QUẢ: SẴN SÀNG ({len(WARNINGS)} cảnh báo)")
    if WARNINGS:
        for w in WARNINGS:
            print(f"  - {w}")
    print("\nLệnh chạy gợi ý:")
    print("  python run_pipeline.py full --output_dir outputs/pcb_synthesis --run_benchmark --resume")
    print("Hoặc chạy từng stage:")
    print("  python run_pipeline.py step1_train")
    print("  python run_pipeline.py step1")
    print("  python run_pipeline.py step2_train")
    print("  python run_pipeline.py step2_generate")
    print("  python run_pipeline.py step3")
    print("  python run_pipeline.py step4")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra setup pipeline")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Tự tải Stable Diffusion Inpainting fp16 nếu cache còn thiếu",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("KIỂM TRA SETUP - PCB DEFECT SYNTHESIS PIPELINE")
    print(f"Project: {PROJECT_ROOT}")
    print("=" * 70)

    check_python_and_packages()
    check_data()
    check_models(download=args.download_models)
    check_outputs()
    return print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
