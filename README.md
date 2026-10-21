# PCB Defect Image Synthesis & Quality Filter Pipeline

Triển khai hoàn chỉnh theo bài báo khoa học:
> **"A Synthetic Defect Image Generation Method for Enhanced Industrial Defect Detection Based on Inpainting Diffusion and Context-aware Mask Generation"**  
> *Journal of Intelligent Manufacturing (POSTECH, 2026)*

---

## Kiến trúc 4 bước cốt lõi của Pipeline

1. **Step 1 — Context-Aware Mask Generation (CAMG)**:
   - Backbone **DINOv2 ViT-B/14** (Frozen feature extractor).
   - Dự đoán phân loại patch-wise (7 classes) và kích thước bounding box ($\log w, \log h$).
   - Cơ chế lấy mẫu **Top-k Sampling (k = 5)** xác định vị trí khuyết tật tự nhiên nhất trên bề mặt bo mạch.

2. **Step 2 — Inpainting Diffusion + LoRA + Textual Inversion**:
   - Kiến trúc **Stable Diffusion Inpainting** (U-Net 9 kênh).
   - Parameter-Efficient Fine-Tuning: **LoRA ($r=8, \alpha=16$)** cho Attention và Output projections.
   - **6 tokens học riêng biệt** qua Textual Inversion cho 6 loại lỗi PCB (`<pcb_mouse_bite>`, v.v.).
   - Hàm mất mát kép **Dual-Loss** ($\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{def}} + \lambda \mathcal{L}_{\text{rec}}$, $\lambda = 0.1$).
   - 50 bước DDIM Scheduler kết hợp cơ chế **Latent Overwriting** bảo toàn 100% cấu trúc bo mạch nền sạch.

3. **Step 3 — Confidence-based Quality Filtering (YOLO11s Judge)**:
   - Huấn luyện mô hình phát hiện **YOLO11s** (100 epochs, AdamW optimizer, cosine lr scheduling).
   - Chấm điểm chất lượng đối chiếu mục tiêu (**Target-aware IoU matching $\ge 0.1$**).
   - Quy tắc lọc chất lượng: **Loại bỏ 25%** ảnh confidence thấp nhất, giữ lại 75% ảnh đạt chuẩn.
   - Tự động xuất nhãn bounding box chuẩn YOLO (`.txt`) và file cấu hình `augmented_dataset.yaml`.

4. **Step 4 — Downstream Benchmark Module**:
   - Đánh giá đối chứng trên 4 kiến trúc phát hiện khuyết tật: **YOLOv8s, YOLOv9s, YOLO11s, RT-DETR-l**.
   - Tái lập 2 bảng thực nghiệm cốt lõi của bài báo:
     - **Table 3 — In-Distribution (ID)**: 10% Low-Data thực tế vs Tập dữ liệu mở rộng.
     - **Table 6 — New Product Introduction (NPI)**: Đánh giá khả năng tổng quát hóa trên bo mạch mới (*Unseen Boards 10, 11, 12*).
   - Đo lường bộ ba chỉ số chuẩn mực: **Precision, Recall, mAP@50** (với ngưỡng đánh giá cố định: $\text{conf}=0.25, \text{IoU}=0.25$).
   - Chứng minh mức tăng vọt về **Recall** (lên tới **+21.5 điểm phần trăm**) giúp triệt tiêu hiện tượng bỏ sót lỗi trong sản xuất công nghiệp.

---

## Hướng dẫn cài đặt & Thiết lập dữ liệu

### 1. Clone mã nguồn (Siêu nhẹ < 2 MB)
```bash
git clone https://github.com/amebett1/diffusion.git
cd diffusion
pip install -r requirements.txt
```

### 2. Thiết lập dữ liệu (`data/`)
Dự án sử dụng tập dữ liệu **DeepPCB (PCB Defect Dataset)**. Người dùng tự tải tập dữ liệu theo đường link được cung cấp:

1. **Tải & Giải nén:** Tải file nén dataset và giải nén vào thư mục `data/pcb-defect-dataset/` sao cho cấu trúc như sau:
```text
data/
└── pcb-defect-dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

2. **Chạy tiền xử lý dữ liệu tự động (Tạo ID/NPI splits & sinh 799 ảnh phôi sạch `pcb_normal/`):**
```bash
python run_pipeline.py prepare_data
# Hoặc chạy: python pcb_pipeline/data_module.py && python setup_env_paths.py
```

> **Cơ chế tự động hóa:** Lệnh trên chỉ mất vài giây để:
> - Phân chia kịch bản **In-Distribution (ID - 10% Low-Data)** và **New Product Introduction (NPI - Unseen Boards)** theo đúng bài báo.
> - Tính toán thống kê kích thước Bounding Box ($Q_1, Q_3$) cho loss function.
> - Tự động áp dụng thuật toán Telea Inpainting để xóa sạch khuyết tật trên các bo mạch đại diện, tạo ra **799 ảnh phôi sạch chuẩn 600x600** trong thư mục `data/pcb_normal/`.
> - Chuẩn hóa toàn bộ đường dẫn tương đối để chạy được ngay trên bất kỳ máy tính nào (Windows, Linux, Colab) mà không bao giờ bị lỗi đường dẫn!

---

## Cách chạy Pipeline

### Cách 1: Chạy trên Kaggle với GPU 2x T4 (Khuyên dùng nhất — Có thể tắt máy tính)

Dự án tích hợp sẵn notebook [`run_on_kaggle.ipynb`](run_on_kaggle.ipynb) được tối ưu hóa cho môi trường Kaggle với **GPU 2x T4 (Dual GPU - 30GB VRAM)**.

#### 💡 Tại sao nên chọn Kaggle thay vì Google Colab?
- **Tắt máy tính hoàn toàn khi train**: Google Colab bắt buộc phải giữ tab trình duyệt luôn mở (`on-screen`). Với Kaggle, bạn chỉ cần bấm **`Save Version`** $\rightarrow$ chọn **`Save & Run All (Commit)`** $\rightarrow$ **tắt máy tính đi ngủ**! Máy chủ Kaggle sẽ chạy ngầm toàn bộ pipeline (tối đa 9–12 giờ) và tự động lưu toàn bộ trọng số, ảnh sinh và báo cáo đối chuẩn.
- **Tài nguyên GPU 2x T4**: Cung cấp miễn phí 30 giờ/tuần với cấu hình Dual GPU T4 x2 (30GB VRAM) hoặc P100 (16GB VRAM).
- **Cơ chế tự động khôi phục (`--resume`)**: Nếu phiên chạy bị gián đoạn, pipeline tự động nhận diện các giai đoạn đã hoàn tất và tiếp tục chính xác công đoạn đang dang dở.

#### Các bước thực hiện trên Kaggle:
1. **Tạo Notebook mới trên Kaggle**:
   - Truy cập [Kaggle Kernels](https://www.kaggle.com/code) $\rightarrow$ Chọn **New Notebook**.
   - Bấm **File** $\rightarrow$ **Import Notebook** $\rightarrow$ Tải lên file [`run_on_kaggle.ipynb`](run_on_kaggle.ipynb) từ repo.
2. **Cấu hình Accelerator GPU 2x T4 (Menu Settings bên phải)**:
   - **Accelerator**: Chọn **GPU T4 x2** (hoặc GPU P100).
   - **Internet**: Bật **On** (để clone mã nguồn và tải pretrained models).
   - **Persistence**: Chọn **Files only**.
3. **Chạy ngầm toàn bộ (One-Click Background Run)**:
   - Bấm nút **`Save Version`** ở góc trên cùng bên phải.
   - Chọn **`Save & Run All (Commit)`** $\rightarrow$ Bấm **Save**.
   - **Bạn có thể đóng trình duyệt và tắt máy tính!** Sau khi hoàn tất, kết quả sẽ sẵn sàng để tải về trong tab *Output*.

---

### Cách 2: Chạy trực tiếp trên Google Colab với GPU T4

Nếu bạn muốn chạy trên Google Colab, sử dụng notebook [`run_on_colab.ipynb`](run_on_colab.ipynb):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amebett1/diffusion/blob/main/run_on_colab.ipynb)

> **Lưu ý**: Khi chạy trên Colab, bạn cần giữ tab trình duyệt hoạt động trong suốt quá trình chạy.

---

### Cách 3: Chạy toàn bộ Pipeline bằng Command Line (One-Click Full & Resume)
Chạy toàn bộ quá trình sinh ảnh và tự động benchmark, tự động ghi nhận checkpoint và khôi phục khi cần:
```bash
python run_pipeline.py full \
    --image_dir data/pcb_normal \
    --real_defect_dir data/pcb-defect-dataset/train/images \
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
    --run_benchmark \
    --benchmark_models yolo11 \
    --benchmark_epochs 100 \
    --resume \
    --device cuda
```

Kiểm tra bảng trạng thái tiến trình của pipeline bất kỳ lúc nào:
```bash
python run_pipeline.py status --output_dir outputs/pcb_synthesis
```

### Cách 4: Chạy từng bước độc lập (Modular CLI Execution)

Bạn có thể chạy riêng lẻ từng công đoạn hoặc dùng cờ `--from_stage` / `--to_stage` trong lệnh `python run_pipeline.py full --from_stage 2b --to_stage 3`:

#### Bước 1: Sinh Mask bằng CAMG (Context-Aware Mask Generation)
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

#### Bước 2b: Sinh ảnh khuyết tật tổng hợp (SD Inpainting + Latent Overwriting)
```bash
python run_pipeline.py step2_generate \
    --mask_meta outputs/masks/mask_metadata.json \
    --lora_ckpt outputs/lora_checkpoints/epoch_010 \
    --output_dir outputs \
    --inpaint_steps 50 \
    --device cuda
```

#### Bước 3: Huấn luyện YOLO11 Judge & Lọc chất lượng (Top 75%)
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

#### Bước 4: Downstream Benchmark (Tái lập Table 3 & Table 6 của Bài Báo)
Huấn luyện và đánh giá đối chứng mô hình (mặc định chạy 1 mô hình `yolo11` để tối ưu thời gian GPU):
```bash
python run_pipeline.py step4 \
    --models yolo11 \
    --split id \
    --base_yaml data/id_setup.yaml \
    --aug_yaml outputs/pcb_synthesis/filtered/augmented_dataset.yaml \
    --epochs 100 \
    --batch_size 8 \
    --lr0 1e-4 \
    --conf 0.25 \
    --iou 0.25 \
    --output_dir outputs/benchmark \
    --device cuda
```

> **Ghi chú về tham số Step 4 chuẩn bài báo (Trang 8, Cột 2):**
> - `--models yolo11`: Mặc định chỉ chạy 1 mô hình (`yolo11`) để tiết kiệm thời gian (khoảng 40-50 phút trên Colab T4). Nếu muốn chạy đầy đủ 4 mô hình đối chứng theo bài báo, bạn có thể truyền: `--models yolo11 yolov8 yolov9 rtdetr`.
> - `--epochs 100`: Huấn luyện 100 epochs với optimizer AdamW và cosine learning rate decay.
> - `--batch_size 8` & `--lr0 1e-4`: Tốc độ học $10^{-4}$ và kích thước batch 8 chuẩn bài báo.
> - `--conf 0.25` & `--iou 0.25`: Ngưỡng đánh giá cố định để đo lường Precision và Recall.
> - Tự động áp dụng các phép tăng cường: xoay ngẫu nhiên (`degrees=10.0`), lật ảnh (`flip`), và biến đổi màu (`HSV`).

---

## Cấu trúc Thư mục Kết quả Đầu ra (`outputs/`)

```text
outputs/
├── pcb_synthesis/
│   ├── masks/                         # Binary masks từ Step 1
│   │   └── mask_metadata.json
│   ├── generated/                     # Ảnh khuyết tật tổng hợp từ Step 2
│   │   └── generated_metadata.json
│   ├── filtered/                      # Kết quả chọn lọc chất lượng từ Step 3
│   │   ├── accepted/                  # 75% ảnh đạt chuẩn chất lượng cao
│   │   ├── labels/                    # Nhãn bounding box chuẩn YOLO (.txt)
│   │   ├── accepted_metadata.json
│   │   ├── rejected_metadata.json
│   │   └── augmented_dataset.yaml     # File dataset mở rộng dùng cho Step 4
│   ├── lora_checkpoints/              # Trọng số LoRA và Textual Inversion
│   └── pipeline_report.json           # Báo cáo tổng kết quá trình sinh ảnh
└── benchmark/                         # Kết quả đánh giá đối chứng từ Step 4
    ├── benchmark_report.md            # Bảng báo cáo đối chứng Markdown (Table 3 / Table 6)
    ├── benchmark_comparison.csv       # File bảng biểu CSV chi tiết (P, R, mAP50, per-class)
    ├── benchmark_summary.json         # Dữ liệu số liệu JSON có cấu trúc
    └── runs/                          # Checkpoints tốt nhất (best.pt) của từng mô hình
```

---

## 6 loại lỗi PCB chuẩn DeepPCB (Paper Standard)

1. `mouse_bite` (Vết cắn góc viền)
2. `spur` (Gờ lồi kim loại / via kim)
3. `missing_hole` (Thiếu lỗ khoan via)
4. `short` (Đoản mạch / ngắn mạch)
5. `open_circuit` (Đứt đường dẫn mạch)
6. `spurious_copper` (Đồng thừa / ba via đồng)
