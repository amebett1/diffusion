"""
Đóng gói source code hiện tại thành file nhỏ để upload lên Kaggle Dataset.

File tạo ra: diffusion_kaggle_code.zip

Cách dùng:
    python package_for_kaggle.py

Sau đó upload file zip lên Kaggle:
    Datasets -> New Dataset -> đặt tên "diffusion-code" -> upload zip.

Notebook Kaggle sẽ tự tìm:
    /kaggle/input/diffusion-code/diffusion_kaggle_code.zip
và giải nén vào /kaggle/working/diffusion.

Script KHÔNG đóng gói data/, outputs/, scratch/, checkpoint hay model weights.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "diffusion_kaggle_code.zip"

INCLUDE_FILES = [
    "run_pipeline.py",
    "requirements.txt",
    "README.md",
    "KAGGLE_SETUP.md",
    "verify_setup.py",
    "setup_env_paths.py",
    "run_on_kaggle.ipynb",
    "run_on_colab.ipynb",
    "package_for_kaggle.py",
]

INCLUDE_DIRS = [
    "pcb_pipeline",
]

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "data",
    "outputs",
    "scratch",
    "kaggle_results",
    "runs",
    "wandb",
}

EXCLUDE_SUFFIXES = {
    ".zip",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".onnx",
    ".engine",
    ".pdf",
    ".log",
    ".pyc",
    ".pyo",
}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE_FILES:
        p = ROOT / name
        if p.is_file():
            files.append(p)
        else:
            print(f"[WARN] Thiếu file: {name}")

    for dirname in INCLUDE_DIRS:
        base = ROOT / dirname
        if not base.exists():
            print(f"[WARN] Thiếu thư mục: {dirname}")
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if p.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            files.append(p)
    return sorted(set(files), key=lambda p: p.as_posix())


def main() -> int:
    required = ["run_pipeline.py", "requirements.txt", "pcb_pipeline/step2_lora_inpainting.py"]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        print("[FAIL] Thiếu file bắt buộc: " + ", ".join(missing))
        return 1

    files = iter_files()
    if OUTPUT.exists():
        OUTPUT.unlink()

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files:
            zf.write(p, p.relative_to(ROOT).as_posix())

    size_mb = OUTPUT.stat().st_size / 1024**2
    print("=" * 68)
    print("ĐÃ TẠO CODE ZIP CHO KAGGLE")
    print("=" * 68)
    print(f"File:   {OUTPUT}")
    print(f"Size:   {size_mb:.2f} MB")
    print(f"Files:  {len(files)}")
    print("=" * 68)
    print("Upload file này lên Kaggle Dataset tên 'diffusion-code',")
    print("sau đó Add Input vào notebook trước khi Save & Run All.")
    print("Xem hướng dẫn chi tiết trong KAGGLE_SETUP.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
