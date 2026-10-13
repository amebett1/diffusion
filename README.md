# PCB Defect Synthesis Pipeline

Pipeline sinh ảnh lỗi PCB tổng hợp dựa trên **DINOv2 + Stable Diffusion Inpainting + LoRA + YOLO11**.

## Kiến trúc tổng quan

```
data/pcb_normal/          data/pcb_real_defects/
      │                          │
      ▼                          ▼
┌─────────────┐          ┌───────────────────┐
│  STEP 1     │          │  LoRA Fine-tune   │
│  DINOv2     │          │  (LoRA rank=8     │
│  Mask Gen   │          │   alpha=16        │
│  + BBox     │          │   10 epoch        │
└──────┬──────┘          │   batch=32)       │
       │ mask_metadata   └────────┬──────────┘
       ▼                         │ lora_ckpt
┌─────────────────────────────────┴──────┐
│  STEP 2  — SD Inpainting + LoRA + TI  │
│  • Textual Inversion: <pcb-defect>     │
│  • Defect-aware loss (λ=2.0)          │
│  • 50 denoising steps (~4s/ảnh)       │
└───────────────────┬────────────────────┘
                    │ generated_metadata
                    ▼
          ┌──────────────────┐
          │  STEP 3          │
          │  YOLO11 Judge    │
          │  → Loại 25%      │
          │    confidence    │
          │    thấp nhất     │
          └────────┬─────────┘
                   │
                   ▼
         outputs/filtered/accepted/
         (~829 ảnh PCB tổng hợp)
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách dùng

### Chạy toàn bộ pipeline

```bash
python run_pipeline.py full \
    --image_dir       data/pcb_normal \
    --real_defect_dir data/pcb_real_defects \
    --output_dir      outputs/pcb_synthesis \
    --device          cuda
```

### Dùng LoRA checkpoint đã train sẵn

```bash
python run_pipeline.py full \
    --image_dir    data/pcb_normal \
    --lora_ckpt    outputs/lora_checkpoints/epoch_010 \
    --yolo_weights outputs/yolo_training/pcb_defect_judge/weights/best.pt \
    --output_dir   outputs/pcb_synthesis
```

### Chạy từng bước riêng

```bash
# Bước 1: Tạo mask
python run_pipeline.py step1 \
    --image_dir data/pcb_normal \
    --output_dir outputs

# Bước 2a: Train LoRA (cần real defect data)
python run_pipeline.py step2_train \
    --real_defect_dir data/pcb_real_defects \
    --output_dir outputs

# Bước 2b: Sinh ảnh
python run_pipeline.py step2_generate \
    --mask_meta  outputs/masks/mask_metadata.json \
    --lora_ckpt  outputs/lora_checkpoints/epoch_010 \
    --output_dir outputs

# Bước 3: Lọc ảnh
python run_pipeline.py step3 \
    --gen_meta     outputs/generated/generated_metadata.json \
    --yolo_weights outputs/yolo_training/.../best.pt \
    --output_dir   outputs/filtered
```

## Cấu trúc dữ liệu

### Input — Ảnh lỗi thực (để train LoRA + YOLO11)

```
data/pcb_real_defects/
├── scratch/
│   ├── img001.jpg
│   └── ...
├── missing_hole/
├── open_circuit/
├── short/
├── spur/
└── spurious_copper/
```

> **Lưu ý**: Thư mục `pcb_real_defects/` cần có thêm `train_metadata.json`:
> ```json
> [
>   {
>     "image_path": "data/pcb_real_defects/scratch/img001.jpg",
>     "mask_path":  "data/pcb_real_defects/scratch/img001_mask.png",
>     "defect_type": "scratch"
>   }
> ]
> ```

### Output

```
outputs/pcb_synthesis/
├── masks/
│   ├── *_mask.png          ← Binary masks từ Step 1
│   └── mask_metadata.json
├── generated/
│   ├── *.png               ← Ảnh lỗi tổng hợp từ Step 2
│   └── generated_metadata.json
├── filtered/
│   ├── accepted/
│   │   └── *.png           ← 75% ảnh chất lượng cao
│   ├── accepted_metadata.json
│   └── rejected_metadata.json
├── lora_checkpoints/
│   └── epoch_010/
│       ├── pytorch_lora_weights.safetensors
│       └── ti_embedding.pt
├── pipeline_config.json
└── pipeline_report.json
```

## Thiết lập theo bài báo

| Thành phần | Giá trị |
|---|---|
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Epochs | 10 |
| Batch size | 32 |
| Denoising steps | 50 |
| Thời gian tạo ảnh | ~4s/ảnh |
| Ảnh PCB tổng hợp | 829 ảnh |
| Lọc | 25% confidence thấp nhất |

## Loại lỗi PCB được hỗ trợ

- `scratch` — Trầy xước
- `missing_hole` — Thiếu lỗ
- `open_circuit` — Đứt mạch
- `short` — Ngắn mạch
- `spur` — Lồi bất thường
- `spurious_copper` — Đồng thừa
