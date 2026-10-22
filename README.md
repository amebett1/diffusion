# PCB Defect Synthesis & Detection Pipeline

Pipeline 4 giai đoạn theo hướng tiếp cận của bài báo:

> **A Synthetic Defect Image Generation Method for Enhanced Industrial Defect Detection Based on Inpainting Diffusion and Context-aware Mask Generation**

Repo này là bản **resource-limited**, chạy được trên một GPU nhỏ. Không tuyên bố tái lập đầy đủ 5 mô hình hay toàn bộ bảng số của paper.

---

## 0. Chạy trên Kaggle bằng Save & Run All (khuyến nghị)

Không cần chạy trên máy cá nhân. Trên Windows, tạo code zip:

```powershell
python package_for_kaggle.py
```

Sau đó trên Kaggle:

1. Upload `diffusion_kaggle_code.zip` thành Dataset tên `diffusion-code`.
2. Upload `data/pcb-defect-dataset/` thành Dataset tên `pcb-defect-dataset`.
3. Mở `run_on_kaggle.ipynb`, **Add Input** cả hai dataset.
4. Bật **Internet** và chọn **GPU** (T4/P100).
5. **Save Version -> Save & Run All (Commit)**.

Notebook sẽ tự giải nén code zip vào `/kaggle/working/diffusion`, tải model, chạy
pipeline và lưu output. Có thể tắt máy sau khi Save; Kaggle chạy trên máy chủ.

Profile mặc định của notebook Kaggle là **resource-limited**, không phải full paper:
CAMG 10 epochs, LoRA 2 epochs, 400 ảnh synthetic, YOLO judge 50 epochs,
benchmark 50 epochs. Step 2 lưu checkpoint sau mỗi epoch và hỗ trợ `--resume`.

Xem hướng dẫn chi tiết, giới hạn thời gian và cách resume tại
[`KAGGLE_SETUP.md`](KAGGLE_SETUP.md).

---

## 1. Phạm vi thực tế của repo

- Chỉ benchmark **YOLO11s**. Các mô hình YOLOv8, YOLOv9, RT-DETR bị bỏ qua có chủ đích.
- Dataset local chỉ có **7 seen boards**: `01, 04, 05, 06, 07, 08, 09`.
  - Board `02, 03` không có trong dữ liệu đang dùng.
  - Unseen boards: `10, 11, 12`.
- Step 3 mặc định dùng YOLO11s judge. Nếu không có GPU/weights, code có thể chuyển sang heuristic Sobel/contrast; chế độ này được ghi rõ trong `pipeline_report.json` và **không được gọi là YOLO judge**.
- Kết quả benchmark phải được đọc theo đúng protocol:
  - `conf = 0.25`
  - NMS IoU = `0.7`
  - evaluate checkpoint `best.pt` được reload từ đĩa
  - Ultralytics dùng match IoU mặc định `0.5`; paper dùng match IoU `0.25`. Sai khác này được ghi trong báo cáo, không gán nhãn sai là NMS IoU.

---

## 2. Trạng thái setup hiện tại

Đã kiểm tra trên máy Windows hiện tại:

| Thành phần | Trạng thái |
|---|---|
| Python | 3.13.5 |
| PyTorch | 2.6.0 + CUDA 12.4 |
| GPU | RTX 4050 Laptop, 6 GB VRAM |
| diffusers | 0.41.0 |
| transformers | 5.7.0 |
| accelerate | 1.15.0 |
| Stable Diffusion Inpainting fp16 | đã cache |
| DINOv2 ViT-B/14 | đã cache |
| YOLO11s weights | có `yolo11s.pt` |
| Data prep | hợp lệ, 799 normals |
| CAMG production checkpoint | chưa có (sẽ train) |
| LoRA/TI production checkpoint | chưa có (sẽ train) |
| Benchmark thật | chưa chạy |

Chạy kiểm tra lại bất cứ lúc nào:

```bash
python verify_setup.py
```

Nếu cache model còn thiếu:

```bash
python verify_setup.py --download-models
```

---

## 3. Cài đặt

Khuyến nghị Python 3.10–3.13, có GPU NVIDIA và CUDA tương thích PyTorch.

```bash
python -m pip install -r requirements.txt
```

Nếu máy chưa có PyTorch CUDA, cài theo hướng dẫn chính thức của PyTorch trước, sau đó cài phần còn lại:

```bash
python -m pip install -r requirements.txt
```

Các package chính:

- `torch`, `torchvision`
- `diffusers`, `transformers`, `accelerate`, `safetensors`
- `ultralytics`
- `opencv-python`, `Pillow`, `numpy`, `tqdm`, `PyYAML`

---

## 4. Dữ liệu

Cấu trúc bắt buộc:

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

Tạo split và 799 ảnh phôi sạch:

```bash
python run_pipeline.py prepare_data
```

Nếu muốn tạo lại từ đầu:

```bash
python run_pipeline.py prepare_data --force
```

Kết quả hiện tại:

| Artifact | Số lượng |
|---|---:|
| ID train full | 5,339 |
| ID train low (10%) | 534 |
| ID val | 941 |
| NPI unseen val | 550 |
| PCB normal | 799 |
| Normal bằng clean crop | 163 |
| Normal bằng Telea đã verify | 636 |
| Telea bị reject | 336 |

Quy tắc chống leakage:

- ID train/val chỉ lấy từ **official train** của seen boards.
- NPI val chỉ lấy từ **official val/test** của unseen boards `10/11/12`.
- Không đưa ảnh unseen boards vào ID generation.
- NPI train dùng seen boards, không dùng unseen boards để train.

Kiểm tra nhanh:

```bash
python verify_setup.py
```

---

## 5. Chạy pipeline

### 5.1. Chạy một lần toàn bộ

```bash
python run_pipeline.py full \
    --output_dir outputs/pcb_synthesis \
    --run_benchmark \
    --resume \
    --device cuda
```

Profile nhanh cho Kaggle commit (không phải full paper):

```bash
python run_pipeline.py full \
    --output_dir outputs/pcb_synthesis \
    --camg_epochs 10 \
    --lora_epochs 2 \
    --num_samples 400 \
    --yolo_epochs 50 \
    --benchmark_models yolo11s \
    --benchmark_epochs 50 \
    --run_benchmark \
    --resume \
    --device cuda
```

Pipeline sẽ tự:

1. Kiểm tra/tạo data prep.
2. Train CAMG nếu chưa có `camg_best.pth` production.
3. Sinh mask.
4. Train LoRA + TI nếu chưa có checkpoint.
5. Sinh ảnh diffusion.
6. Lọc chất lượng bằng YOLO11s judge.
7. Benchmark YOLO11s baseline vs augmented nếu có `--run_benchmark`.

Xem trạng thái:

```bash
python run_pipeline.py status --output_dir outputs/pcb_synthesis
```

Chạy lại từ một stage:

```bash
python run_pipeline.py full --from_stage 2b --to_stage 3 --resume
```

`--from_stage` / `--to_stage` nhận: `0, 1, 2a, 2b, 3, 4`.

> `--resume` chỉ dùng cache nếu metadata chứng minh đúng generation version, đúng LoRA/TI checkpoint, đúng số bước DDIM và đúng guidance scale. Artifact cũ không đạt các điều kiện này sẽ bị bỏ qua hoặc archive.

---

### 5.2. Chạy từng giai đoạn

#### Step 0 — Data prep

```bash
python run_pipeline.py prepare_data
```

#### Step 1 — Train CAMG

```bash
python run_pipeline.py step1_train \
    --metadata_path data/train_metadata.json \
    --camg_epochs 10 \
    --camg_batch 32 \
    --camg_lr 1e-4 \
    --device cuda
```

Checkpoint mặc định:

```text
outputs/pcb_synthesis/camg_checkpoints/camg_best.pth
```

Smoke test rất nhanh (không phải model production):

```bash
python run_pipeline.py step1_train \
    --metadata_path scratch/camg_smoke/train_metadata_small.json \
    --camg_epochs 1 \
    --camg_batch 8 \
    --camg_out scratch/camg_smoke/checkpoints
```

#### Step 1 — Sinh mask

```bash
python run_pipeline.py step1 \
    --image_dir data/pcb_normal \
    --output_dir outputs/pcb_synthesis \
    --k 5 \
    --images_per_defect 1
```

Mask hợp lệ phải:

- nằm trong foreground PCB,
- không rơi vào mounting hole,
- không sát viền,
- có kích thước gần phân bố thật (W Q1/median/Q3 ≈ 21/27/36 px).

Nếu muốn sinh lại mask:

```bash
python run_pipeline.py step1 --force_masks
```

#### Step 2a — Train LoRA + Textual Inversion

```bash
python run_pipeline.py step2_train \
    --metadata_path data/train_metadata.json \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_epochs 2 \
    --lora_batch 32 \
    --device cuda
```

`2` là profile Kaggle; tăng nếu chạy local/đủ thời gian.

Lưu ý:

- `--lora_batch` là **gradient accumulation**, không phải batch vật lý. Batch vật lý luôn là 1 để phù hợp GPU 6 GB.
- Code tự dùng fp16 + attention slicing + gradient checkpointing trên CUDA.
- Checkpoint được lưu **sau mỗi epoch**, ví dụ với profile Kaggle 2 epochs:

```text
outputs/pcb_synthesis/lora_checkpoints/
├── epoch_001/
│   ├── pytorch_lora_weights.pt
│   ├── ti_embeddings.pt
│   ├── optimizer.pt
│   └── scaler.pt
├── epoch_002/
│   └── ...
└── training_state.json
```

`training_state.json` ghi số epoch đã hoàn tất. Nếu checkpoint mới có một phần,
lần chạy sau với `--resume` sẽ tự resume từ epoch đã lưu thay vì bỏ qua train.
Nếu checkpoint đã đủ số epoch, pipeline bỏ qua train. Muốn train lại từ đầu:

```bash
python run_pipeline.py step2_train --force_lora
```

#### Step 2b — Sinh ảnh lỗi

```bash
python run_pipeline.py step2_generate \
    --mask_meta outputs/pcb_synthesis/masks/mask_metadata.json \
    --output_dir outputs/pcb_synthesis \
    --num_samples 400 \
    --inpaint_steps 50 \
    --guidance 7.5 \
    --device cuda
```

`400` là profile Kaggle; local full có thể dùng `--num_samples 829` nếu đủ thời gian.
Code tự tìm checkpoint LoRA mới nhất; không cần hardcode tên `epoch_*`.

Code sẽ:

- dùng một noise schedule DDIM nhất quán,
- giữ background theo `z0` + noise cố định,
- feather biên mask 2–3 px,
- composite ở native resolution để giữ micro-trace.

#### Step 3 — Lọc chất lượng

```bash
python run_pipeline.py step3 \
    --gen_meta outputs/pcb_synthesis/generated/generated_metadata.json \
    --output_dir outputs/pcb_synthesis \
    --data_yaml data/id_setup.yaml \
    --yolo_size s \
    --yolo_epochs 50 \
    --reject_ratio 0.25 \
    --device cuda
```

Nếu chưa có YOLO judge weights, Step 3 sẽ train YOLO11s judge trên real low-data. Nếu không dùng được YOLO/GPU, code có thể fallback heuristic; kiểm tra `pipeline_report.json` để biết chế độ thực tế.

Kết quả:

```text
outputs/pcb_synthesis/filtered/
├── accepted/
├── rejected/
├── accepted_metadata.json
├── rejected_metadata.json
└── augmented_dataset.yaml
```

#### Step 4 — Benchmark YOLO11s

```bash
python run_pipeline.py step4 \
    --models yolo11s \
    --split id \
    --base_yaml data/id_setup.yaml \
    --aug_yaml outputs/pcb_synthesis/filtered/augmented_dataset.yaml \
    --output_dir outputs/pcb_synthesis \
    --epochs 50 \
    --batch_size 8 \
    --conf 0.25 \
    --iou 0.7 \
    --imgsz 600 \
    --device cuda
```

Chỉ `yolo11` / `yolo11s` được chấp nhận; các model khác bị log là skipped.

Báo cáo:

```text
outputs/pcb_synthesis/benchmark/
├── benchmark_summary.json
├── benchmark_comparison.csv
└── benchmark_report.md
```

Kết luận trong báo cáo được sinh theo delta thực tế: tăng, giảm hoặc gần như không đổi. Không có câu kết luận tích cực hardcode.

---

## 6. Tài nguyên và thời gian chạy

Máy local là RTX 4050 Laptop 6 GB:

- Step 1 CAMG: khoảng vài chục phút cho 10 epochs.
- Step 1 sinh 4,794 masks: khoảng vài chục phút.
- Step 2a LoRA 1 epoch trên 5,339 ảnh: khoảng 3 giờ trên GPU 6 GB.
- Step 2b 400 ảnh × 50 DDIM steps: khoảng vài chục phút.
- Step 3 judge 50 epochs + Step 4 benchmark 2 × 50 epochs: vài giờ.

Profile Kaggle (T4/P100) thường mất khoảng **6–9 giờ** cho toàn bộ pipeline,
tùy tốc độ GPU và việc model/dataset đã có cache hay chưa. Đây là ước tính, không
phải cam kết thời gian chạy.

Nếu chỉ muốn kiểm tra code end-to-end, dùng smoke test trước:

```bash
# CAMG smoke
python run_pipeline.py step1_train \
    --metadata_path scratch/camg_smoke/train_metadata_small.json \
    --camg_epochs 1 \
    --camg_out scratch/camg_smoke/checkpoints

# LoRA smoke (2 ảnh)
python run_pipeline.py step2_train \
    --metadata_path scratch/lora_smoke/train_metadata_small.json \
    --lora_epochs 1 \
    --lora_batch 1 \
    --output_dir scratch/lora_smoke/pipeline
```

> Smoke checkpoint chỉ để test code, không dùng cho benchmark.

---

## 7. Kiểm tra setup

```bash
python verify_setup.py
```

Script kiểm tra:

- package và CUDA/VRAM,
- 799 ảnh normal,
- split counts và leakage cơ bản,
- cache Stable Diffusion Inpainting fp16,
- cache DINOv2,
- YOLO11s weights,
- output active có mask của unseen boards hay không.

Nếu có lỗi, script in lệnh cần chạy để sửa.

---

## 8. Lưu ý về artifact cũ

Artifact cũ không hợp lệ đã được archive tại:

```text
outputs/_stale_invalid_pcb_synthesis_20261022/
```

Không dùng thư mục này cho benchmark. `outputs/pcb_synthesis/` hiện là output active.

Các lỗi cũ đã được loại bỏ:

- mask sinh ngoài foreground / vào mounting hole,
- mask quá lớn do cap `0.02–0.20`,
- dùng unseen boards trong ID generation,
- lẫn official val/test vào ID train,
- Step 2 dùng noise Gaussian mới ở mỗi bước,
- Step 4 đánh giá model trong RAM thay vì reload `best.pt`,
- dùng IoU 0.25 như NMS IoU,
- kết luận benchmark hardcode.

---

## 9. Troubleshooting

### `ModuleNotFoundError: diffusers`

```bash
python -m pip install "diffusers>=0.31.0" accelerate safetensors
```

### CUDA out of memory ở Step 2

- Đóng ứng dụng khác đang dùng GPU.
- Giữ `--lora_batch` làm gradient accumulation, không tăng batch vật lý.
- Thử `--num_samples` nhỏ hơn ở Step 2b.
- Chạy trên Colab/Kaggle nếu 6 GB không đủ.

### Không tìm thấy checkpoint

```bash
python run_pipeline.py status --output_dir outputs/pcb_synthesis
```

Pipeline `full` sẽ tự train nếu checkpoint production chưa có.

### Muốn kiểm tra lại data prep

```bash
python setup_env_paths.py
```

Script này chỉ kiểm tra, **không sửa đường dẫn**.

---

## 10. Cấu trúc thư mục chính

```text
pcb_pipeline/
├── data_module.py             # split, normal generation, stats
├── step1_mask_generator.py    # CAMG train + mask generation
├── step2_lora_inpainting.py   # LoRA/TI train + DDIM inpainting
├── step3_quality_filter.py    # YOLO11s judge + augmented YAML
└── step4_benchmark.py         # YOLO11s benchmark + report

run_pipeline.py                # CLI/orchestrator
verify_setup.py                # kiểm tra môi trường
setup_env_paths.py             # compatibility check, không sửa path
requirements.txt
```
