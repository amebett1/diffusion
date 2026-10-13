"""
Packaging Utility for Google Colab
==================================
Tự động đóng gói dự án thành 1 file ZIP duy nhất: 'diffusion_colab.zip'
để tải lên Google Drive / Google Colab nhanh chóng.

Tự động loại bỏ các file rác và file tạm:
- outputs/
- runs/
- scratch/
- __pycache__/ & *.pyc
- File PDF lớn
"""

import sys
import os
import zipfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXCLUDE_DIRS = {
    "outputs",
    "runs",
    "scratch",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pdf",
    ".zip",
    ".log",
}

STORE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".pt",
    ".pth",
}

def create_colab_package(output_zip: str = "diffusion_colab.zip"):
    root_dir = Path(__file__).resolve().parent
    zip_path = root_dir / output_zip

    print("=" * 60)
    print("📦 BẮT ĐẦU ĐÓNG GÓI DỰ ÁN CHO GOOGLE COLAB")
    print(f"📂 Thư mục nguồn: {root_dir}")
    print(f"🎯 File ZIP đích: {zip_path}")
    print("=" * 60)

    t0 = time.time()
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for foldername, subfolders, filenames in os.walk(root_dir):
            rel_dir = Path(foldername).relative_to(root_dir)
            parts = rel_dir.parts

            if any(p in EXCLUDE_DIRS for p in parts):
                continue

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in EXCLUDE_EXTENSIONS or filename == output_zip:
                    continue

                full_path = Path(foldername) / filename
                rel_path = full_path.relative_to(root_dir)

                # Sử dụng ZIP_STORED cho ảnh và weights (đã nén sẵn) để đóng gói cực nhanh
                cmode = zipfile.ZIP_STORED if ext in STORE_EXTENSIONS else zipfile.ZIP_DEFLATED

                zipf.write(full_path, arcname=str(rel_path), compress_type=cmode)
                file_count += 1
                total_uncompressed_bytes += full_path.stat().st_size

                if file_count % 1000 == 0:
                    print(f"  Đang đóng gói... ({file_count} files)")

    elapsed = time.time() - t0
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    orig_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print("=" * 60)
    print("✅ ĐÓNG GÓI THÀNH CÔNG!")
    print(f"  📁 Tổng số files:        {file_count}")
    print(f"  📊 Kích thước gốc:       {orig_size_mb:.1f} MB")
    print(f"  🗜️ Kích thước file ZIP:  {zip_size_mb:.1f} MB")
    print(f"  ⏱️ Thời gian nén:        {elapsed:.1f} giây")
    print(f"  📍 Đường dẫn file:       {zip_path.resolve()}")
    print("=" * 60)
    print("👉 Bây giờ bạn chỉ cần kéo thả file 'diffusion_colab.zip' này lên Google Drive!")

if __name__ == "__main__":
    create_colab_package()
