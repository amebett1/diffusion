# PCB Defect Image Synthesis & Quality Filter Pipeline

Triển khai hoàn chỉnh theo bài báo:
> **"A Synthetic Defect Image Generation Method for Enhanced Industrial Defect Detection Based on Inpainting Diffusion and Context-aware Mask Generation"**
> *Journal of Intelligent Manufacturing (POSTECH, 2026)*

---

## 📌 Kiến trúc 3 bước cốt lõi

1. **Step 1 — Context-Aware Mask Generation (CAMG)**:
   - Backbone DINOv2 ViT-B/14 (Frozen).
   - Dự đoán phân loại patch-wise (7 classes) và kích thước bounding box (log w, log h).
   - Cơ chế lấy mẫu Top-k (k = 5) xác định vị trí khuyết tật tự nhiên nhất.
2. **Step 2 — Inpainting Diffusion + LoRA + Textual Inversion**:
   - Kiến trúc Stable Diffusion Inpainting (U-Net 9 kênh).
   - Parameter-Efficient Fine-Tuning: LoRA ($r=8, \alpha=16$) cho Attention và Output projections.
   - 6 tokens học riêng biệt qua Textual Inversion cho 6 loại lỗi PCB.
   - Hàm mất mát kép Dual-Loss ($\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{def}} + \lambda \mathcal{L}_{\text{rec}}$, $\lambda = 0.1$).
   - 50 bước DDIM Scheduler kết hợp cơ chế **Latent Overwriting** bảo toàn 100% cấu trúc bo mạch nền.
3. **Step 3 — Confidence-based Quality Filtering (YOLO11s Judge)**:
   - Huấn luyện mô hình phát hiện YOLO11s (100 epochs, AdamW optimizer, cosine lr scheduling).
   - Chấm điểm chất lượng có đối chiếu mục tiêu (Target-aware IoU matching $\ge 0.1$).
   - Quy tắc loại bỏ: Loại 25% ảnh confidence thấp nhất, giữ lại 75% ảnh đạt chuẩn.
   - Tự động xuất nhãn bounding box chuẩn YOLO (`.txt`) và sinh file `augmented_dataset.yaml` cho Downstream Benchmark.

---

## 🛠️ Hướng dẫn cài đặt & Thiết lập dữ liệu

### 1. Clone mã nguồn (Siêu nhẹ < 2 MB)
```bash
git clone <repository_url>
cd diffusion
pip install -r requirements.txt
```

### 2. Thiết lập dữ liệu (`data/`)
Do dung lượng ảnh lớn (~1.2 GB), dữ liệu được chia sẻ riêng qua Google Drive:
- Tải file **`diffusion_colab.zip`** từ Google Drive.
- Giải nén vào thư mục dự án sao cho cấu trúc như sau:

```text
data/
├── pcb-defect-dataset/        # Ảnh và nhãn DeepPCB gốc (train, val, test)
├── pcb_normal/                # 799 ảnh phôi PCB sạch không lỗi
├── prepared_splits/           # Danh sách file chia tập ID và NPI
├── id_setup.yaml              # Cấu hình YOLO kịch bản In-Distribution (10% low-data)
├── npi_setup.yaml             # Cấu hình YOLO kịch bản New Product Introduction
├── train_metadata.json        # Metadata huấn luyện CAMG và LoRA
└── dataset_stats.json         # Thống kê phân vị Q1/Q3 kích thước khuyết tật
```

> **Lưu ý về đường dẫn:** Toàn bộ cấu hình và file txt đều sử dụng **đường dẫn tương đối (`data/...`)**, cho phép dự án chạy được trên bất kỳ máy tính nào (Windows, macOS, Linux, Google Colab) mà không cần cấu hình lại!

---

## 🚀 Cách chạy Pipeline

### Cách 1: Chạy trực tiếp trên Google Colab với GPU (Khuyên dùng)
Dự án đã tích hợp sẵn notebook [run_on_colab.ipynb](file:///c:/Users/Lenovo/Downloads/diffusion/diffusion/run_on_colab.ipynb):
1. Tải `diffusion_colab.zip` lên Google Drive của bạn.
2. Mở file `run_on_colab.ipynb` trên Google Colab, bật GPU (T4 hoặc A100).
3. Bấm **Run all** để tự động giải nén, huấn luyện toàn bộ và lưu kết quả về Google Drive!

---

### Cách 2: Chạy toàn bộ Pipeline bằng Command Line (One-Click Full)
```bash
python run_pipeline.py full \
    --image_dir data/pcb_normal \
    --real_defect_dir data/pcb-defect-dataset \
    --metadata_path data/train_metadata.json \
    --data_yaml data/id_setup.yaml \
    --output_dir outputs/pcb_synthesis \
    --lora_epochs 10 \
    --lora_batch 32 \
    --inpaint_steps 50 \
    --guidance 7.5 \
    --yolo_size s \
    --yolo_epochs 100 \
    --reject_ratio 0.25 \
    --device cuda
```

---

### Cách 3: Chạy từng bước độc lập

#### Bước 1: Sinh Mask bằng CAMG
```bash
python run_pipeline.py step1 \
    --image_dir data/pcb_normal \
    --output_dir outputs \
    --k 5 \
    --images_per_defect 1
```

#### Bước 2a: Huấn luyện LoRA + Textual Inversion
```bash
python run_pipeline.py step2_train \
    --metadata_path data/train_metadata.json \
    --output_dir outputs \
    --lora_epochs 10 \
    --lora_batch 32 \
    --device cuda
```

#### Bước 2b: Sinh ảnh khuyết tật tổng hợp
```bash
python run_pipeline.py step2_generate \
    --mask_meta outputs/masks/mask_metadata.json \
    --lora_ckpt outputs/lora_checkpoints/epoch_010 \
    --output_dir outputs \
    --inpaint_steps 50 \
    --device cuda
```

#### Bước 3: Huấn luyện YOLO11 Judge & Lọc chất lượng
```bash
python run_pipeline.py step3 \
    --gen_meta outputs/generated/generated_metadata.json \
    --data_yaml data/id_setup.yaml \
    --output_dir outputs \
    --yolo_size s \
    --yolo_epochs 100 \
    --reject_ratio 0.25 \
    --device cuda
```

#### Bước 4: Downstream Benchmark (Tái lập Table 3 & 6 của bài báo)
```bash
python run_pipeline.py step4 \
    --models yolo11 yolov8 yolov9 rtdetr \
    --split id \
    --base_yaml data/id_setup.yaml \
    --aug_yaml outputs/pcb_synthesis/filtered/augmented_dataset.yaml \
    --epochs 100 \
    --batch_size 16 \
    --output_dir outputs/benchmark \
    --device cuda
```
> Kết quả so sánh Baseline vs Augmented sẽ tự động xuất ra:
> - `outputs/benchmark/benchmark_report.md` (Bảng Markdown đối chứng chuẩn bài báo)
> - `outputs/benchmark/benchmark_comparison.csv` (Bảng số liệu chi tiết)
> - `outputs/benchmark/benchmark_summary.json` (Dữ liệu số có cấu trúc)

---

## 📋 6 loại lỗi PCB chuẩn DeepPCB (Paper Standard)

1. `mouse_bite` (Vết cắn góc)
2. `spur` (Gờ lồi/via kim)
3. `missing_hole` (Thiếu lỗ khoan)
4. `short` (Ngắn mạch/chập)
5. `open_circuit` (Đứt mạch)
6. `spurious_copper` (Đồng thừa)
