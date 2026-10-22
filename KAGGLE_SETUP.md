# Chạy trên Kaggle bằng Save & Run All

Tài liệu này dành cho trường hợp bạn chạy pipeline trên **Kaggle Notebook**, dùng
**Save Version -> Save & Run All (Commit)**, tắt máy tính mà job vẫn chạy.

---

## 1. Chuẩn bị 2 Kaggle Datasets

### A. Code hiện tại: `diffusion-code`

Trên máy Windows, chạy:

```powershell
python package_for_kaggle.py
```

File được tạo:

```text
diffusion_kaggle_code.zip
```

Sau đó trên Kaggle:

1. Vào **Datasets -> New Dataset**.
2. Đặt tên dataset: `diffusion-code`.
3. Upload `diffusion_kaggle_code.zip`.
4. Create Dataset.

Notebook sẽ tự tìm file tại:

```text
/kaggle/input/diffusion-code/diffusion_kaggle_code.zip
```

và giải nén vào:

```text
/kaggle/working/diffusion
```

> Nếu không upload code zip, notebook có fallback clone GitHub. Bản GitHub có thể
> chưa có các fix mới trong repo local, nên **khuyến nghị dùng code zip**.

### B. Dataset PCB: `pcb-defect-dataset`

Upload `data/pcb-defect-dataset/` thành một Kaggle Dataset, ví dụ tên:

```text
pcb-defect-dataset
```

Sau đó trong notebook: **Add Input -> Your Datasets -> pcb-defect-dataset**.

Notebook tự quét `/kaggle/input` và copy/link dataset vào:

```text
/kaggle/working/diffusion/data/pcb-defect-dataset
```

Nếu không thấy dataset, notebook có fallback tải Google Drive bằng `gdown`, nhưng
cách ổn định nhất cho commit là dùng Kaggle Dataset.

---

## 2. Cấu hình notebook

Mở `run_on_kaggle.ipynb` trên Kaggle, kiểm tra:

- **Accelerator**: GPU T4 x2 hoặc P100.
- **Internet**: **On** (cần cho `pip install` và tải Stable Diffusion/DINOv2).
- **Add Input**:
  - `diffusion-code`
  - `pcb-defect-dataset`

Sau đó:

1. **Save Version**.
2. Chọn **Save & Run All (Commit)**.
3. Bấm **Save**.
4. Có thể đóng trình duyệt và tắt máy.

Kaggle sẽ chạy các cell theo thứ tự; không cần màn hình mở.

---

## 3. Profile chạy trên Kaggle

Notebook dùng profile resource-limited để cố gắng hoàn tất trong giới hạn session:

| Tham số | Giá trị |
|---|---:|
| CAMG epochs | 10 |
| LoRA epochs | 10 |
| Synthetic samples | 400 |
| YOLO judge epochs | 100 |
| Benchmark epochs | 100 |
| Conf | 0.25 |
| NMS IoU | 0.7 |
| imgsz | 600 |

Đây **không phải** cấu hình đầy đủ của paper (paper dùng nhiều epoch hơn).
Mục tiêu là có kết quả end-to-end trong một commit Kaggle.

Các ô **Phương án B** trong notebook chỉ là markdown tham khảo, không tự chạy
trong Save & Run All. Chỉ ô full pipeline (Phương án A) thực thi pipeline.

Nếu muốn chạy lâu hơn, sửa các giá trị trong cell full pipeline hoặc chạy từng
stage; nhưng phải kiểm tra giới hạn thời gian của Kaggle.

---

## 4. Output và resume

Output nằm tại:

```text
/kaggle/working/diffusion/outputs/pcb_synthesis/
```

Các artifact chính:

```text
camg_checkpoints/camg_best.pth
masks/mask_metadata.json
lora_checkpoints/epoch_002/pytorch_lora_weights.pt
lora_checkpoints/epoch_002/ti_embeddings.pt
generated/generated_metadata.json
filtered/augmented_dataset.yaml
benchmark/benchmark_report.md
```

`--resume` được bật trong notebook. Step 2 lưu checkpoint **sau mỗi epoch** và
`training_state.json`. Nếu commit bị timeout:

- Nếu timeout sau một epoch: lần chạy sau resume từ epoch đã lưu.
- Nếu timeout giữa epoch đầu: chưa có checkpoint, phải chạy lại Step 2.
- Các stage đã xong (mask, generated, filtered, benchmark) được dùng lại nếu
  metadata hợp lệ.

Kaggle commit mới thường có `/kaggle/working` mới. Để resume qua nhiều commit,
bạn cần giữ output cũ bằng cách tạo **Kaggle Dataset từ output** rồi Add Input,
hoặc chạy lại trong cùng một phiên. `--resume` không tự tìm output của commit cũ
nếu output đó không được add vào notebook.

---

## 5. Ước tính thời gian

Trên GPU T4/P100, profile trên thường mất khoảng **6–9 giờ**, tùy tốc độ GPU và
việc dataset đã có sẵn hay chưa. Các phần tốn thời gian nhất:

- tải Stable Diffusion Inpainting fp16 (~vài GB),
- Step 2a LoRA 2 epochs trên 5,339 ảnh,
- Step 3/4 YOLO11s 50 epochs.

Nếu Kaggle báo hết thời gian, xem mục resume và giảm thêm `--lora_epochs`,
`--num_samples`, `--yolo_epochs` hoặc `--benchmark_epochs`.

---

## 6. Lỗi thường gặp

### `No such file or directory: requirements.txt`

Code zip chưa được add hoặc giải nén sai thư mục. Kiểm tra:

```python
from pathlib import Path
print(list(Path("/kaggle/input").glob("**/diffusion_kaggle_code.zip")))
```

### `Cannot reach huggingface.co` hoặc `pip` lỗi mạng

Bật **Internet** trong notebook settings trước khi Save & Run All.

### `CUDA out of memory`

- Giữ `--lora_batch` là gradient accumulation, không tăng batch vật lý.
- Giảm `--camg_batch` xuống 16 hoặc 8.
- Giảm `--num_samples`.
- Dùng GPU T4/P100 thay vì CPU.

### Commit xong nhưng không thấy output

Vào **Output** của notebook version, tải `outputs/pcb_synthesis/`. Nếu cần giữ
lâu dài, tạo Kaggle Dataset từ thư mục output.
