"""
Step 1: Context-Aware Mask Generation (CAMG) Module
===================================================
Triển khai chuẩn theo bài báo:
"A Synthetic Defect Image Generation Method for Enhanced Industrial
Defect Detection Based on Inpainting Diffusion and Context-aware Mask Generation"
(Journal of Intelligent Manufacturing, 2026)

Cơ sở toán học:
  C(I) = P(L, T, S | Φ(I))
  - Φ(I): Patch embeddings từ frozen DINOv2 backbone (d=768)
  - L: Tọa độ patch (prior không gian)
  - T: Phân loại loại lỗi patch-wise (K lớp lỗi + 1 lớp 'no defect')
  - S: Hồi quy kích thước bounding box patch-wise (log ŵ, log ĥ)

Hàm mất mát:
  L_cls = - ∑ ∑ y_{i,k} log p_{i,k}
  L_box = ||log(w_i) - log(ŵ_i)||_2^2 + ||log(h_i) - log(ĥ_i)||_2^2  (chỉ tính trên positive patches)
  L_CAMG = L_cls + λ * L_box  (λ = 1.0)

Lấy mẫu huấn luyện:
  - Tỷ lệ 1 positive patch : 3 negative patches nhằm chống mất cân bằng dữ liệu.

Inference:
  - Dense defect probability map M ∈ R^{(K+2) × H' × W'}
  - Top-k sampling (k = 5) để chọn vị trí patch sinh lỗi cho từng class.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)

# Danh sách 6 loại lỗi PCB chuẩn theo DeepPCB & Paper
PCB_DEFECT_CLASSES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]
NO_DEFECT_CLASS = "no_defect"
ALL_CLASSES = PCB_DEFECT_CLASSES + [NO_DEFECT_CLASS]
NUM_CLASSES = len(ALL_CLASSES)  # 6 + 1 = 7


# ─────────────────────────────────────────────────────────
# 1. Kiến trúc mạng CAMG (Backbone + Heads)
# ─────────────────────────────────────────────────────────

class CAMGBackbone(nn.Module):
    """
    Frozen Vision Foundation Model backbone (DINOv2 ViT-B/14).
    Feature dimension d = 768, Patch size = 14.
    """

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patch_size = 14
        self.embed_dim = 768
        self.model_name = model_name

        logger.info(f"[CAMG] Đang tải backbone DINOv2: {model_name} trên {self.device}...")
        try:
            self.dino = torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception as e:
            logger.warning(f"Không thể tải từ torch.hub ({e}). Đang tạo dummy backbone cho CPU/offline fallback.")
            self.dino = None

        if self.dino is not None:
            # Đóng băng hoàn toàn trọng số foundation model
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x ∈ R^{B × 3 × H × W}
        Output: patch_tokens ∈ R^{B × d × H' × W'}
        """
        B, C, H, W = x.shape
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size

        if self.dino is not None:
            # Trích xuất patch embeddings từ intermediate layers
            out = self.dino.get_intermediate_layers(x, n=1, return_class_token=False)
            tokens = out[0]  # shape: (B, N_patches, d)
        else:
            # Fallback dummy embedding
            tokens = torch.randn(B, H_patch * W_patch, self.embed_dim, device=x.device)

        # Reshape sang dạng spatial feature map (B, d, H', W')
        tokens = tokens.permute(0, 2, 1).reshape(B, self.embed_dim, H_patch, W_patch)
        return tokens


class CAMGHeads(nn.Module):
    """
    Shared Linear Layer + Patch-wise Class Head + Patch-wise Size Head.
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, num_classes: int = 7):
        super().__init__()
        # Shared projection layer theo paper
        self.shared_proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )

        # Class Head: dự đoán xác suất lớp (K lỗi + 1 no-defect)
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

        # Size Head: dự đoán (log ŵ, log ĥ) chuẩn hóa
        self.size_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:  feat ∈ R^{B × 768 × H' × W'}
        Output:
          cls_logits ∈ R^{B × num_classes × H' × W'}
          size_pred  ∈ R^{B × 2 × H' × W'}  (log ŵ, log ĥ)
        """
        h = self.shared_proj(feat)
        cls_logits = self.cls_head(h)
        size_pred = self.size_head(h)
        return cls_logits, size_pred


class CAMGModel(nn.Module):
    """
    Mô hình tích hợp toàn diện của CAMG Module.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        num_classes: int = NUM_CLASSES,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone = CAMGBackbone(model_name=backbone_name, device=str(self.device))
        self.heads = CAMGHeads(in_dim=768, hidden_dim=512, num_classes=num_classes)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        feat = self.backbone(x)
        cls_logits, size_pred = self.heads(feat)
        return cls_logits, size_pred


# ─────────────────────────────────────────────────────────
# 2. Hàm mất mát CAMG (Cross-Entropy + Log-scale Size Loss)
# ─────────────────────────────────────────────────────────

class CAMGLoss(nn.Module):
    """
    LCAMG = Lcls + λ * Lbox (với λ = 1.0 theo Equation 4 trong paper).
    """

    def __init__(self, lambda_box: float = 1.0):
        super().__init__()
        self.lambda_box = lambda_box
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        cls_logits: torch.Tensor,     # (N, num_classes)
        cls_targets: torch.Tensor,    # (N,)
        size_preds: torch.Tensor,     # (N, 2)
        size_targets: torch.Tensor,   # (N, 2)
        is_positive: torch.Tensor,    # (N,) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Classification Loss (tính trên toàn bộ positive và negative patches)
        l_cls = self.ce_loss(cls_logits, cls_targets)

        # 2. Size Loss (chỉ tính trên positive patches)
        if is_positive.sum() > 0:
            pos_preds = size_preds[is_positive]
            pos_targets = size_targets[is_positive]
            l_box = F.mse_loss(pos_preds, pos_targets)
        else:
            l_box = torch.tensor(0.0, device=cls_logits.device)

        l_total = l_cls + self.lambda_box * l_box
        return l_total, l_cls, l_box


# ─────────────────────────────────────────────────────────
# 3. Patch Dataset với Tỷ lệ lấy mẫu 1 : 3 (Random Patch Sampling)
# ─────────────────────────────────────────────────────────

class CAMGPatchDataset(Dataset):
    """
    Dataset phục vụ huấn luyện CAMG:
    - Nhận danh sách metadata ảnh và bbox từ train_metadata.json
    - Thực hiện lấy mẫu 1 positive patch per defect instance và 3 negative patches
    """

    def __init__(
        self,
        metadata_path: Union[str, Path],
        image_size: int = 588,  # Bội số của 14: 588 = 14 x 42 patches
        patch_size: int = 14,
        neg_ratio: int = 3,
    ):
        self.metadata_path = Path(metadata_path)
        self.image_size = image_size
        self.patch_size = patch_size
        self.neg_ratio = neg_ratio
        self.grid_size = image_size // patch_size  # 42x42

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        img_path = rec["image_path"]
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(pil_img)

        # Phân tích các patch
        G = self.grid_size
        occupied_mask = np.zeros((G, G), dtype=bool)

        pos_patches = []
        for box in rec["boxes"]:
            cls_id = box["class_id"]
            cx = box["cx"]
            cy = box["cy"]
            w = max(box["w"], 1e-4)
            h = max(box["h"], 1e-4)

            # Tọa độ patch chứa tâm bbox
            pr = int(np.clip(cy * G, 0, G - 1))
            pc = int(np.clip(cx * G, 0, G - 1))
            occupied_mask[pr, pc] = True

            pos_patches.append({
                "patch_coord": (pr, pc),
                "class_id": cls_id,
                "log_size": [np.log(w), np.log(h)],
            })

        # Lấy mẫu negative patches từ vùng không lỗi (tỷ lệ 1 : 3)
        neg_patches = []
        free_coords = np.argwhere(~occupied_mask)
        num_neg_needed = max(1, len(pos_patches) * self.neg_ratio)

        if len(free_coords) > 0:
            chosen_indices = np.random.choice(
                len(free_coords),
                size=min(num_neg_needed, len(free_coords)),
                replace=False,
            )
            for ci in chosen_indices:
                pr, pc = free_coords[ci]
                neg_patches.append({
                    "patch_coord": (int(pr), int(pc)),
                    "class_id": NUM_CLASSES - 1,  # 6: no_defect
                    "log_size": [0.0, 0.0],
                })

        return {
            "image": img_tensor,
            "pos_patches": pos_patches,
            "neg_patches": neg_patches,
        }


# ─────────────────────────────────────────────────────────
# 4. CAMG Trainer
# ─────────────────────────────────────────────────────────

class CAMGTrainer:
    """
    Huấn luyện hai prediction heads của CAMG module (Backbone DINOv2 đóng băng).
    """

    def __init__(
        self,
        model: Optional[CAMGModel] = None,
        lr: float = 1e-4,
        device: str = "cuda",
        output_dir: str = "outputs/camg_checkpoints",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model or CAMGModel(device=str(self.device))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Chỉ tối ưu các tham số của heads
        self.optimizer = torch.optim.Adam(self.model.heads.parameters(), lr=lr)
        self.criterion = CAMGLoss(lambda_box=1.0)

    def train(
        self,
        metadata_path: str = "data/train_metadata.json",
        epochs: int = 10,
        batch_size: int = 32,
    ) -> str:
        """
        Vòng lặp huấn luyện CAMG 10 epochs theo bài báo.
        """
        logger.info(f"[CAMG Trainer] Khởi tạo huấn luyện: {epochs} epochs, batch_size={batch_size}")
        best_ckpt_path = self.output_dir / "camg_best.pth"

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_cls_loss = 0.0
            total_box_loss = 0.0
            accum_steps = 0
            sample_count = 0

            self.optimizer.zero_grad()

            for step, batch in enumerate(dataloader):
                img = batch["image"].unsqueeze(0).to(self.device)  # (1, 3, H, W)
                pos = batch["pos_patches"]                          # List[Dict]
                neg = batch["neg_patches"]                          # List[Dict]

                # Forward qua backbone và heads
                cls_logits, size_preds = self.model(img)  # (1, 7, G, G), (1, 2, G, G)

                patch_logits = []
                patch_cls_targets = []
                patch_size_preds = []
                patch_size_targets = []
                patch_is_pos = []

                # Thu thập positive patches (duyệt trực tiếp từng dict trong list)
                for p in pos:
                    pr, pc = p["patch_coord"]
                    cid = p["class_id"]
                    lw, lh = p["log_size"]

                    patch_logits.append(cls_logits[0, :, pr, pc])
                    patch_cls_targets.append(cid)
                    patch_size_preds.append(size_preds[0, :, pr, pc])
                    patch_size_targets.append([lw, lh])
                    patch_is_pos.append(True)

                # Thu thập negative patches (duyệt trực tiếp từng dict trong list)
                for n in neg:
                    pr, pc = n["patch_coord"]
                    cid = n["class_id"]

                    patch_logits.append(cls_logits[0, :, pr, pc])
                    patch_cls_targets.append(cid)
                    patch_size_preds.append(size_preds[0, :, pr, pc])
                    patch_size_targets.append([0.0, 0.0])
                    patch_is_pos.append(False)

                if len(patch_logits) == 0:
                    continue

                logits_tensor = torch.stack(patch_logits)
                targets_tensor = torch.tensor(patch_cls_targets, device=self.device, dtype=torch.long)
                sizes_tensor = torch.stack(patch_size_preds)
                size_targets_tensor = torch.tensor(patch_size_targets, device=self.device, dtype=torch.float32)
                is_pos_tensor = torch.tensor(patch_is_pos, device=self.device, dtype=torch.bool)

                loss, l_cls, l_box = self.criterion(
                    logits_tensor, targets_tensor, sizes_tensor, size_targets_tensor, is_pos_tensor
                )

                # Chia tỉ lệ loss trước khi backward để tránh nổ gradient khi tích lũy
                scaled_loss = loss / batch_size
                scaled_loss.backward()
                accum_steps += 1
                sample_count += 1

                # Cập nhật optimizer sau mỗi batch_size mẫu hoặc ở mẫu cuối
                if accum_steps % batch_size == 0 or step == len(dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.heads.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()
                total_cls_loss += l_cls.item()
                total_box_loss += l_box.item()

            avg_loss = total_loss / max(1, sample_count)
            avg_cls = total_cls_loss / max(1, sample_count)
            avg_box = total_box_loss / max(1, sample_count)

            logger.info(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f} (L_cls: {avg_cls:.4f}, L_box: {avg_box:.4f})")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.heads.state_dict(), best_ckpt_path)
                logger.info(f"  -> Đã lưu best checkpoint tại {best_ckpt_path}")

        logger.info(f"Hoàn tất huấn luyện CAMG! Checkpoint: {best_ckpt_path}")
        return str(best_ckpt_path)


# ─────────────────────────────────────────────────────────
# 5. CAMG Generator (Inference với Top-k Sampling, k=5)
# ─────────────────────────────────────────────────────────

class CAMGGenerator:
    """
    Module sinh mask lỗi context-aware:
    - Nhận ảnh bo mạch bình thường (Normal PCB Image)
    - Trích xuất Dense Probability Map M qua CAMG
    - Dùng Top-k sampling (k=5) để chọn vị trí và kích thước lỗi hợp lệ
    - Xuất Binary Mask và Bounding Box coordinates
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        k: int = 5,
        image_size: int = 588,
        patch_size: int = 14,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.k = k
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 42

        self.model = CAMGModel(device=str(self.device))
        if weights_path and Path(weights_path).exists():
            logger.info(f"[CAMG] Tải checkpoint trọng số từ {weights_path}")
            self.model.heads.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            logger.info("[CAMG] Khởi chạy ở chế độ evaluation với trọng số mặc định.")

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict_defect_map(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trả về:
          prob_map: (num_classes, G, G) xác suất lớp tại từng patch
          size_map: (2, G, G) kích thước (ŵ, ĥ) ước lượng tại từng patch
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        cls_logits, size_preds = self.model(img_tensor)

        # Softmax để lấy xác suất p_i
        probs = F.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy()  # (7, G, G)
        # Chuyển đổi log size -> kích thước chuẩn hóa (w, h)
        log_sizes = size_preds.squeeze(0).cpu().numpy()                # (2, G, G)
        # Clip log_size để tránh tràn số exp
        log_sizes = np.clip(log_sizes, -6.0, 0.0)
        sizes = np.exp(log_sizes)                                      # (2, G, G)

        return probs, sizes

    def generate_mask_for_defect(
        self,
        image: Image.Image,
        defect_type: str,
        orig_size: Tuple[int, int] = (600, 600),
    ) -> Dict:
        """
        Sinh 1 mask lỗi cho loại lỗi chỉ định bằng Top-k sampling (k=5).
        """
        if defect_type not in PCB_DEFECT_CLASSES:
            raise ValueError(f"Loại lỗi không hợp lệ: {defect_type}. Hỗ trợ: {PCB_DEFECT_CLASSES}")

        cls_id = PCB_DEFECT_CLASSES.index(defect_type)
        prob_map, size_map = self.predict_defect_map(image)

        # Lấy xác suất của class lỗi tương ứng: shape (G, G)
        c_prob = prob_map[cls_id]
        G = self.grid_size
        flat_probs = c_prob.flatten()

        # Thuật toán Top-k sampling (k = 5) theo bài báo
        k = min(self.k, len(flat_probs))
        top_k_indices = np.argsort(flat_probs)[-k:]  # k vị trí có xác suất cao nhất
        selected_idx = int(np.random.choice(top_k_indices))

        pr, pc = divmod(selected_idx, G)
        confidence = float(flat_probs[selected_idx])

        # Kích thước box dự đoán
        pred_w_norm = float(np.clip(size_map[0, pr, pc], 0.02, 0.20))
        pred_h_norm = float(np.clip(size_map[1, pr, pc], 0.02, 0.20))

        # Tọa độ tâm bbox trên ảnh gốc
        orig_w, orig_h = orig_size
        cx = (pc + 0.5) / G * orig_w
        cy = (pr + 0.5) / G * orig_h
        bw = pred_w_norm * orig_w
        bh = pred_h_norm * orig_h

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(orig_w, int(cx + bw / 2))
        y2 = min(orig_h, int(cy + bh / 2))

        # Tạo binary mask B_c (0 và 255)
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        mask_pil = Image.fromarray(mask)

        return {
            "defect_type": defect_type,
            "class_id": cls_id,
            "confidence": confidence,
            "patch_coord": (pr, pc),
            "bbox_pixel": [x1, y1, x2, y2],
            "bbox_norm": [cx / orig_w, cy / orig_h, bw / orig_w, bh / orig_h],
            "mask_image": mask_pil,
        }

    def generate_batch(
        self,
        image_dir: str | Path = "data/pcb_normal",
        output_dir: str | Path = "outputs/masks",
        defect_types: Optional[List[str]] = None,
        images_per_defect: int = 1,
    ) -> List[Dict]:
        """
        Sinh batch mask cho toàn bộ ảnh normal trong thư mục.
        """
        img_dir = Path(image_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        defect_types = defect_types or PCB_DEFECT_CLASSES
        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not images:
            logger.warning(f"Không tìm thấy ảnh nào trong {img_dir}")
            return []

        logger.info(f"[CAMG] Đang sinh masks cho {len(images)} ảnh, {len(defect_types)} loại lỗi...")
        records = []

        for img_path in images:
            pil_img = Image.open(img_path).convert("RGB")
            W, H = pil_img.size

            for dtype in defect_types:
                for rep in range(images_per_defect):
                    res = self.generate_mask_for_defect(pil_img, defect_type=dtype, orig_size=(W, H))

                    mask_filename = f"{img_path.stem}_{dtype}_{rep}_mask.png"
                    mask_save_path = out_dir / mask_filename
                    res["mask_image"].save(mask_save_path)

                    record = {
                        "normal_image_path": str(img_path.resolve()),
                        "mask_path": str(mask_save_path.resolve()),
                        "defect_type": dtype,
                        "class_id": res["class_id"],
                        "confidence": res["confidence"],
                        "bbox_pixel": res["bbox_pixel"],
                        "bbox_norm": res["bbox_norm"],
                    }
                    records.append(record)

        meta_path = out_dir / "mask_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info(f"Đã tạo {len(records)} masks. Metadata lưu tại: {meta_path}")
        return records


# ─────────────────────────────────────────────────────────
# Backward Compatibility Wrapper
# ─────────────────────────────────────────────────────────
class PCBMaskGenerator(CAMGGenerator):
    """Lớp bọc tương thích ngược với code cũ của pipeline."""
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    gen = CAMGGenerator()
    logger.info("CAMG Module đã được nạp thành công và sẵn sàng hoạt động.")
