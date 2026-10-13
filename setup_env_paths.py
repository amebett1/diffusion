"""
Environment Path Adapter for PCB Defect Pipeline
================================================
Tự động chuẩn hóa toàn bộ đường dẫn trong:
- data/id_setup.yaml
- data/npi_setup.yaml
- data/prepared_splits/*.txt
- data/train_metadata.json

Hỗ trợ 2 chế độ:
- 'relative' (Mặc định - Khuyên dùng khi chia sẻ GitHub): Dùng đường dẫn tương đối để chạy được trên mọi máy.
- 'absolute': Chuyển thành đường dẫn tuyệt đối theo máy hiện tại (Windows / Linux / Colab).
"""

import sys
import json
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def adapt_paths(project_root: Path = None, mode: str = "relative"):
    if project_root is None:
        project_root = Path(__file__).resolve().parent

    data_dir = project_root / "data"
    splits_dir = data_dir / "prepared_splits"
    raw_dataset_dir = data_dir / "pcb-defect-dataset"

    print("=" * 60)
    print(f"🔧 Đang chuẩn hóa đường dẫn [Chế độ: {mode.upper()}] cho môi trường: {sys.platform}")
    print(f"📂 Thư mục gốc dự án: {project_root}")
    print("=" * 60)

    # 1. Chuẩn hóa các file .txt trong data/prepared_splits/
    txt_files = list(splits_dir.glob("*.txt"))
    for txt_file in txt_files:
        lines = txt_file.read_text(encoding="utf-8").splitlines()
        updated_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            normalized_line = line.replace("\\", "/")
            if "data/pcb-defect-dataset" in normalized_line:
                subpath = normalized_line.split("data/pcb-defect-dataset/")[-1]
            elif "pcb-defect-dataset" in normalized_line:
                subpath = normalized_line.split("pcb-defect-dataset/")[-1]
            else:
                subpath = Path(line).name

            if mode == "absolute":
                target_path = str((raw_dataset_dir / subpath).resolve()).replace("\\", "/")
            else:
                target_path = f"data/pcb-defect-dataset/{subpath}".replace("\\", "/")

            updated_lines.append(target_path)

        with open(txt_file, "w", encoding="utf-8") as f:
            for p in updated_lines:
                f.write(f"{p}\n")
        print(f"  ✅ Cập nhật {len(updated_lines)} đường dẫn trong: {txt_file.name}")

    # 2. Chuẩn hóa data/id_setup.yaml và data/npi_setup.yaml
    if mode == "absolute":
        id_train_ref = str((splits_dir / "id_train_low.txt").resolve()).replace("\\", "/")
        id_val_ref = str((splits_dir / "id_val.txt").resolve()).replace("\\", "/")
        npi_train_ref = str((splits_dir / "npi_train_low.txt").resolve()).replace("\\", "/")
        npi_val_ref = str((splits_dir / "npi_val_unseen.txt").resolve()).replace("\\", "/")
    else:
        id_train_ref = "data/prepared_splits/id_train_low.txt"
        id_val_ref = "data/prepared_splits/id_val.txt"
        npi_train_ref = "data/prepared_splits/npi_train_low.txt"
        npi_val_ref = "data/prepared_splits/npi_val_unseen.txt"

    id_yaml_path = data_dir / "id_setup.yaml"
    if id_yaml_path.exists():
        path_line = f"path: {str(project_root).replace('\\', '/')}\n" if mode == "absolute" else "path: .\n"
        content_id = f"""# In-Distribution (ID) Dataset Config (10% Low-Data)
{path_line}train: {id_train_ref}
val: {id_val_ref}

nc: 6
names: ['mouse_bite', 'spur', 'missing_hole', 'short', 'open_circuit', 'spurious_copper']
"""
        id_yaml_path.write_text(content_id, encoding="utf-8")
        print(f"  ✅ Cập nhật: {id_yaml_path.name}")

    npi_yaml_path = data_dir / "npi_setup.yaml"
    if npi_yaml_path.exists():
        path_line = f"path: {str(project_root).replace('\\', '/')}\n" if mode == "absolute" else "path: .\n"
        content_npi = f"""# New Product Introduction (NPI) Dataset Config
# Train trên Seen Products, Val trên 100% Unseen Products
{path_line}train: {npi_train_ref}
val: {npi_val_ref}

nc: 6
names: ['mouse_bite', 'spur', 'missing_hole', 'short', 'open_circuit', 'spurious_copper']
"""
        npi_yaml_path.write_text(content_npi, encoding="utf-8")
        print(f"  ✅ Cập nhật: {npi_yaml_path.name}")

    # 3. Chuẩn hóa data/train_metadata.json
    meta_path = data_dir / "train_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for item in metadata:
            img_p = item.get("image_path", "").replace("\\", "/")
            lbl_p = item.get("label_path", "").replace("\\", "/")

            if "data/pcb-defect-dataset" in img_p:
                sub_img = img_p.split("data/pcb-defect-dataset/")[-1]
            elif "pcb-defect-dataset" in img_p:
                sub_img = img_p.split("pcb-defect-dataset/")[-1]
            else:
                sub_img = Path(img_p).name

            if "data/pcb-defect-dataset" in lbl_p:
                sub_lbl = lbl_p.split("data/pcb-defect-dataset/")[-1]
            elif "pcb-defect-dataset" in lbl_p:
                sub_lbl = lbl_p.split("pcb-defect-dataset/")[-1]
            else:
                sub_lbl = Path(lbl_p).name

            if mode == "absolute":
                item["image_path"] = str((raw_dataset_dir / sub_img).resolve()).replace("\\", "/")
                item["label_path"] = str((raw_dataset_dir / sub_lbl).resolve()).replace("\\", "/")
            else:
                item["image_path"] = f"data/pcb-defect-dataset/{sub_img}".replace("\\", "/")
                item["label_path"] = f"data/pcb-defect-dataset/{sub_lbl}".replace("\\", "/")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Cập nhật {len(metadata)} bản ghi trong: {meta_path.name}")

    print("=" * 60)
    print("🎉 TẤT CẢ ĐƯỜNG DẪN ĐÃ ĐƯỢC ĐỒNG BỘ THÀNH CÔNG!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Environment Path Adapter")
    parser.add_argument("--mode", default="relative", choices=["relative", "absolute"], help="Chế độ đường dẫn")
    args = parser.parse_args()
    adapt_paths(mode=args.mode)
