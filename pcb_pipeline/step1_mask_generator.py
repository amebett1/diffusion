"""
Step 1: Context-Aware Mask Generation (CAMG) Module
===================================================
Triển khai theo bài báo:
"A Synthetic Defect Image Generation Method for Enhanced Industrial
Defect Detection Based on Inpainting Diffusion and Context-aware Mask Generation"
(Journal of Intelligent Manufacturing, 2026)

Các thay đổi quan trọng so với bản cũ
-------------------------------------
1. CAMGTrainer.train() thực sự tạo CAMGDataset + DataLoader từ metadata.
2. best_loss được khởi tạo bằng float("inf") trước vòng lặp epoch.
3. DINOv2 là bắt buộc: nếu không tải được backbone, chương trình dừng ngay
   với thông báo rõ ràng. Không còn sinh feature ngẫu nhiên (random dummy).
4. Negative sampling đánh dấu TOÀN BỘ patch giao với bbox lỗi là occupied,
   không chỉ patch chứa tâm bbox.
5. Bỏ bẫy np.clip(size, 0.02, 0.20). Kích thước mask được hiệu chỉnh theo
   phân phối thực của DeepPCB (Q1 ~ 20-22 px, median ~ 26-28 px, Q3 ~ 36 px,
   tức ~0.03-0.06 khi chuẩn hóa trên ảnh 600 px).
6. Vị trí sinh mask chỉ nằm trong foreground bo mạch; loại viền ảnh, góc
   trống/ít texture và lỗ ốc (mounting holes).

Ghi chú về kích thước:
  - size_head dự đoán log(w_norm), log(h_norm), với w_norm = w_px / image_width.
  - Khi inference, MaskSizePrior giới hạn kích thước trong một khoảng dựa
    trên thống kê dataset, KHÔNG dùng hằng số 0.20 như bản cũ.
"""

import os
import json
import math
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
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS_PATH = PROJECT_ROOT / "data" / "dataset_stats.json"
MASK_GENERATION_VERSION = "camg_v2_foreground_sizeprior"

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
# 0. Tiện ích ảnh: foreground, texture, mounting holes
# ─────────────────────────────────────────────────────────

def _read_gray(pil_img: Image.Image) -> np.ndarray:
    """Chuyển PIL RGB -> grayscale uint8."""
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    """
    Ước lượng vùng bo mạch (foreground) trong ảnh.

    Ảnh DeepPCB thường là một crop đầy bo mạch, nhưng một số ảnh có viền đen.
    Ngưỡng được chọn thích nghi theo phân vị nền thay vì Otsu (Otsu trên ảnh
    PCB chỉ tách các đường mạch sáng và bỏ qua phần substrate tối).
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    p2 = float(np.percentile(blur, 2))
    p50 = float(np.percentile(blur, 50))
    # Nền đen gần 0; substrate thường > 20. Nếu ảnh toàn bo mạch, p2 vẫn > 20.
    threshold = max(10.0, min(0.45 * p50, p2 + 8.0))
    fg = (blur > threshold).astype(np.uint8)

    # Đóng lỗ nhỏ do text/khoảng trống cục bộ trên bo mạch.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Nếu ước lượng quá nhỏ (ảnh tối bất thường), giữ vùng ảnh không quá tối.
    if fg.mean() < 0.05:
        fg = (blur > 12).astype(np.uint8)
    return fg.astype(bool)


def _detect_mounting_holes(gray: np.ndarray, rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Phát hiện lỗ ốc / mounting hole bằng HoughCircles + kiểm tra vành sáng.

    Trả về mask bool cùng kích thước ảnh. Chỉ đánh dấu các hình tròn có
    bán kính đủ lớn (>= 1.5% cạnh ngắn) và có cấu trúc vành sáng - tâm tối
    hoặc vành kim loại, tránh nhầm với các pad tròn nhỏ.
    """
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    hole_mask = np.zeros((h, w), dtype=np.uint8)

    # Mounting hole thường lớn hơn pad thông thường. Chỉ bắt các hình tròn
    # có bán kính >= 4.5% cạnh ngắn để tránh biến toàn bộ pad tròn thành
    # "mounting hole" (điều này sẽ làm mất các vị trí missing_hole hợp lệ).
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    min_radius = max(18, int(round(min_dim * 0.045)))
    max_radius = max(min_radius + 4, int(round(min_dim * 0.10)))
    if rgb is None:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    try:
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, min_dim // 18),
            param1=110,
            param2=34,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
    except cv2.error:
        circles = None

    if circles is None:
        return hole_mask.astype(bool)

    yy, xx = np.ogrid[:h, :w]
    for x, y, r in np.round(circles[0]).astype(int):
        if r < min_radius or r > max_radius:
            continue
        if x - r < 2 or y - r < 2 or x + r >= w - 2 or y + r >= h - 2:
            continue

        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        disk = dist <= max(2, int(r * 0.80))
        ring = (dist >= int(r * 1.05)) & (dist <= int(r * 1.40))
        if disk.sum() < 50 or ring.sum() < 40:
            continue

        disk_mean = float(gray[disk].mean())
        disk_std = float(gray[disk].std())
        ring_mean = float(gray[ring].mean())
        ring_std = float(gray[ring].std())
        disk_saturation = float(hsv[:, :, 1][disk].mean())

        # Vòng kim loại sáng, độ bão hòa thấp, bề mặt tương đối đồng nhất,
        # tương phản với nền bo mạch xung quanh.
        is_metal_disk = (
            disk_saturation < 60.0
            and disk_std < 45.0
            and disk_mean > 110.0
            and (abs(disk_mean - ring_mean) > 15.0 or ring_std > 12.0)
        )
        if not is_metal_disk:
            continue

        # Đánh dấu cả vùng lỗ và một viền an toàn.
        cv2.circle(hole_mask, (int(x), int(y)), int(r * 1.25) + 3, 255, thickness=-1)

    return hole_mask.astype(bool)


def _patch_statistics(gray: np.ndarray, grid: int, patch_size: int):
    """Tính foreground fraction, std và edge energy cho từng patch."""
    h, w = gray.shape[:2]
    expected = grid * patch_size
    if h != expected or w != expected:
        gray = cv2.resize(gray, (expected, expected), interpolation=cv2.INTER_AREA)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)

    patches = gray.reshape(grid, patch_size, grid, patch_size)
    edges = edge.reshape(grid, patch_size, grid, patch_size)
    patch_std = patches.std(axis=(1, 3))
    patch_edge = edges.mean(axis=(1, 3))
    return patch_std, patch_edge


def compute_valid_patch_mask(
    pil_img: Image.Image,
    image_size: int = 588,
    patch_size: int = 14,
    border_patches: int = 1,
) -> np.ndarray:
    """
    Trả về mask bool (G, G): True nếu patch nằm trong foreground bo mạch,
    không thuộc viền ảnh, không phải vùng trống ít texture và không trúng
    mounting hole.
    """
    grid = image_size // patch_size
    work = pil_img.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    gray = _read_gray(work)

    fg = _foreground_mask(gray)
    rgb = np.array(work)
    holes = _detect_mounting_holes(gray, rgb=rgb)

    patch_std, patch_edge = _patch_statistics(gray, grid, patch_size)
    fg_frac = fg.reshape(grid, patch_size, grid, patch_size).mean(axis=(1, 3))
    hole_frac = holes.reshape(grid, patch_size, grid, patch_size).mean(axis=(1, 3))

    valid = fg_frac >= 0.55
    # Vùng trống/đồng nhất: std thấp và ít cạnh. Bo mạch thật có texture cao.
    valid &= (patch_std >= 6.0) | (patch_edge >= 5.0)
    valid &= hole_frac < 0.08

    if border_patches > 0:
        valid[:border_patches, :] = False
        valid[-border_patches:, :] = False
        valid[:, :border_patches] = False
        valid[:, -border_patches:] = False

    # Không fallback vào vùng trung tâm: nếu ảnh không có patch foreground
    # hợp lệ (quá tối, toàn viền hoặc mounting hole), trả về toàn False để
    # generator bỏ qua ảnh đó thay vì đặt mask sai chỗ.
    if not valid.any():
        logger.warning(
            "Không tìm thấy patch foreground hợp lệ; bỏ ảnh này thay vì "
            "fallback vào vùng có thể là viền/mounting hole."
        )
    return valid


def _image_pixel_validity(
    pil_img: Image.Image,
    image_size: int = 588,
    patch_size: int = 14,
):
    """
    Trả về (foreground, holes) ở độ phân giải gốc để kiểm tra bbox pixel.
    """
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg = _foreground_mask(gray)
    holes = _detect_mounting_holes(gray, rgb=rgb)
    return fg, holes


# ─────────────────────────────────────────────────────────
# 1. Size prior theo phân phối DeepPCB
# ─────────────────────────────────────────────────────────

class MaskSizePrior:
    """
    Giới hạn kích thước mask theo phân phối bbox thực tế.

    Mặc định DeepPCB @ 600 px:
      Q1 ~ 21-22 px, median ~ 28 px, Q3 ~ 36 px
      => normalized Q1 ~ 0.036, median ~ 0.046, Q3 ~ 0.060

    Khoảng cho phép rộng hơn một chút để không "đóng băng" mọi mask về
    một kích thước, nhưng tuyệt đối không dùng cap 0.20 như bản cũ.
    """

    def __init__(self, stats_path: Union[str, Path] = DEFAULT_STATS_PATH, reference_size: int = 600):
        self.reference_size = int(reference_size)
        self.q1_w = 21.96 / reference_size
        self.q3_w = 36.0 / reference_size
        self.q1_h = 21.0 / reference_size
        self.q3_h = 36.0 / reference_size
        self.median_w = 27.96 / reference_size
        self.median_h = 27.96 / reference_size
        self.stats_loaded = False

        stats_file = Path(stats_path)
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                px = data.get("pixel_size", {})
                norm = data.get("normalized_size", {})

                if "width" in norm:
                    self.q1_w = float(norm["width"].get("q1", self.q1_w))
                    self.q3_w = float(norm["width"].get("q3", self.q3_w))
                if "height" in norm:
                    self.q1_h = float(norm["height"].get("q1", self.q1_h))
                    self.q3_h = float(norm["height"].get("q3", self.q3_h))

                if "width" in px:
                    self.median_w = float(px["width"].get("median", self.median_w * reference_size)) / reference_size
                    if "normalized_size" not in data:
                        self.q1_w = float(px["width"].get("q1", self.q1_w * reference_size)) / reference_size
                        self.q3_w = float(px["width"].get("q3", self.q3_w * reference_size)) / reference_size
                if "height" in px:
                    self.median_h = float(px["height"].get("median", self.median_h * reference_size)) / reference_size
                    if "normalized_size" not in data:
                        self.q1_h = float(px["height"].get("q1", self.q1_h * reference_size)) / reference_size
                        self.q3_h = float(px["height"].get("q3", self.q3_h * reference_size)) / reference_size

                self.stats_loaded = True
                logger.info(
                    "[MaskSizePrior] Q1/Q3/median (normalized): "
                    f"w={self.q1_w:.4f}/{self.q3_w:.4f}/{self.median_w:.4f}, "
                    f"h={self.q1_h:.4f}/{self.q3_h:.4f}/{self.median_h:.4f}"
                )
            except Exception as e:
                logger.warning(f"Không đọc được {stats_file}: {e}. Dùng prior mặc định DeepPCB.")

        # Khoảng an toàn: không nhỏ hơn ~2 px, không lớn hơn ~1.35 * Q3.
        self.min_w_norm = max(2.0 / 600.0, self.q1_w * 0.60)
        self.max_w_norm = min(0.12, self.q3_w * 1.35)
        self.min_h_norm = max(2.0 / 600.0, self.q1_h * 0.60)
        self.max_h_norm = min(0.12, self.q3_h * 1.35)

    def sanitize(
        self,
        pred_w_norm: float,
        pred_h_norm: float,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, bool]:
        """Chuẩn hóa dự đoán size_head, trả về (w_norm, h_norm, was_clamped)."""
        was_clamped = False

        if not np.isfinite(pred_w_norm) or pred_w_norm <= 0:
            pred_w_norm = self.median_w
            was_clamped = True
        if not np.isfinite(pred_h_norm) or pred_h_norm <= 0:
            pred_h_norm = self.median_h
            was_clamped = True

        min_w = max(self.min_w_norm, 2.0 / max(image_width, 1))
        min_h = max(self.min_h_norm, 2.0 / max(image_height, 1))
        max_w = max(min_w + 1e-4, self.max_w_norm)
        max_h = max(min_h + 1e-4, self.max_h_norm)

        w = float(np.clip(pred_w_norm, min_w, max_w))
        h = float(np.clip(pred_h_norm, min_h, max_h))
        if abs(w - pred_w_norm) > 1e-8 or abs(h - pred_h_norm) > 1e-8:
            was_clamped = True
        return w, h, was_clamped


# ─────────────────────────────────────────────────────────
# 2. Kiến trúc mạng CAMG (Backbone + Heads)
# ─────────────────────────────────────────────────────────

class CAMGBackbone(nn.Module):
    """
    Frozen Vision Foundation Model backbone (DINOv2 ViT-B/14).
    Feature dimension d = 768, Patch size = 14.

    Fail-fast: nếu DINOv2 không tải được (offline, thiếu cache, lỗi mạng),
    ném RuntimeError ngay. Không có fallback random feature.
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
            raise RuntimeError(
                "Không thể tải DINOv2 backbone "
                f"'{model_name}' từ torch.hub. CAMG không thể chạy nếu thiếu backbone.\n"
                "Cách khắc phục:\n"
                "  1) Kiểm tra kết nối mạng và chạy lại lệnh này để tải/cache DINOv2;\n"
                "  2) Hoặc tải trước trên máy có mạng rồi copy thư mục torch hub cache sang máy đích;\n"
                "  3) Hoặc cài đặt timm và thay model_name bằng một DINOv2 checkpoint local.\n"
                f"Chi tiết lỗi gốc: {e}"
            ) from e

        if self.dino is None:
            raise RuntimeError(
                f"torch.hub.load trả về None cho DINOv2 '{model_name}'. "
                "Không cho phép chạy CAMG với feature ngẫu nhiên."
            )

        # Đóng băng hoàn toàn trọng số foundation model
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x ∈ R^{B × 3 × H × W}, H/W là bội số của 14.
        Output: patch_tokens ∈ R^{B × d × H' × W'}
        """
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Kích thước ảnh ({H}x{W}) phải là bội số của patch_size={self.patch_size}."
            )

        H_patch = H // self.patch_size
        W_patch = W // self.patch_size

        out = self.dino.get_intermediate_layers(x, n=1, return_class_token=False)
        tokens = out[0]  # (B, N_patches, d)

        expected = H_patch * W_patch
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"Số patch token DINOv2 không khớp: nhận {tokens.shape[1]}, cần {expected}."
            )

        tokens = tokens.permute(0, 2, 1).reshape(B, self.embed_dim, H_patch, W_patch)
        return tokens


class CAMGHeads(nn.Module):
    """
    Shared Linear Layer + Patch-wise Class Head + Patch-wise Size Head.
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, num_classes: int = 7):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.size_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)

        # Khởi tạo size head quanh median thực tế (log ~ -3.1 cho 0.045),
        # giúp hội tụ nhanh và tránh những epoch đầu dự đoán size ~1.0.
        nn.init.normal_(self.size_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.size_head.bias, math.log(0.045))

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared_proj(feat)
        cls_logits = self.cls_head(h)
        size_pred = self.size_head(h)
        return cls_logits, size_pred


class CAMGModel(nn.Module):
    """Mô hình tích hợp CAMG: DINOv2 (frozen) + prediction heads."""

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

    def train(self, mode: bool = True):
        """Backbone luôn ở eval mode vì đã đóng băng."""
        super().train(mode)
        self.backbone.eval()
        if getattr(self.backbone, "dino", None) is not None:
            self.backbone.dino.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        feat = self.backbone(x)
        cls_logits, size_pred = self.heads(feat)
        return cls_logits, size_pred


# ─────────────────────────────────────────────────────────
# 3. Hàm mất mát CAMG
# ─────────────────────────────────────────────────────────

class CAMGLoss(nn.Module):
    """L_CAMG = L_cls + λ * L_box (λ = 1.0 theo Equation 4 trong paper)."""

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
        l_cls = self.ce_loss(cls_logits, cls_targets)

        if is_positive.sum() > 0:
            l_box = F.mse_loss(size_preds[is_positive], size_targets[is_positive])
        else:
            l_box = torch.tensor(0.0, device=cls_logits.device)

        l_total = l_cls + self.lambda_box * l_box
        return l_total, l_cls, l_box


# ─────────────────────────────────────────────────────────
# 4. Patch Dataset với 1 positive : 3 negative
# ─────────────────────────────────────────────────────────

def _collate_camg(batch: List[Dict]) -> Dict:
    """Collate giữ nguyên list patch của từng ảnh (batch_size = 1)."""
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "pos_patches": [b["pos_patches"] for b in batch],
        "neg_patches": [b["neg_patches"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


class CAMGDataset(Dataset):
    """
    Dataset huấn luyện CAMG.

    Mỗi ảnh tạo một "episode":
      - 1 positive patch cho mỗi defect box;
      - 3 negative patch trong vùng foreground hợp lệ.

    Tất cả patch giao với bất kỳ bbox lỗi nào đều bị đánh dấu occupied,
    sau đó dilate 1 patch để tránh negative nằm sát biên lỗi.
    """

    def __init__(
        self,
        metadata_path: Union[str, Path],
        image_size: int = 588,
        patch_size: int = 14,
        neg_ratio: int = 3,
        validate_images: bool = True,
    ):
        self.metadata_path = Path(metadata_path)
        self.image_size = image_size
        self.patch_size = patch_size
        self.neg_ratio = neg_ratio
        self.grid_size = image_size // patch_size

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata CAMG: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if validate_images:
            valid_records = []
            missing = 0
            for r in records:
                p = Path(str(r.get("image_path", "")))
                if p.exists():
                    valid_records.append(r)
                else:
                    missing += 1
            if missing:
                logger.warning(f"[CAMGDataset] Bỏ qua {missing} record vì không tìm thấy ảnh.")
            records = valid_records

        if not records:
            raise FileNotFoundError(
                f"Không có ảnh hợp lệ trong {self.metadata_path}. "
                "Hãy chạy 'python run_pipeline.py prepare_data' trước."
            )

        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        # Cache nhỏ chỉ chứa valid patch grid (42x42 bool), không chứa ảnh.
        self._valid_patch_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _get_valid_patches(self, pil_img: Image.Image, cache_key: str) -> np.ndarray:
        if cache_key in self._valid_patch_cache:
            return self._valid_patch_cache[cache_key]
        valid = compute_valid_patch_mask(
            pil_img,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        if len(self._valid_patch_cache) < 1024:
            self._valid_patch_cache[cache_key] = valid
        return valid

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        img_path = rec["image_path"]
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(pil_img)

        G = self.grid_size
        occupied = np.zeros((G, G), dtype=bool)
        pos_patches: List[Dict] = []
        seen_positive: set = set()

        for box in rec.get("boxes", []):
            cid = int(box.get("class_id", 0))
            cx = float(box.get("cx", 0.5))
            cy = float(box.get("cy", 0.5))
            w = max(float(box.get("w", 0.0)), 1e-4)
            h = max(float(box.get("h", 0.0)), 1e-4)

            x1, y1 = cx - w / 2.0, cy - h / 2.0
            x2, y2 = cx + w / 2.0, cy + h / 2.0

            # ── Đánh dấu MỌI patch giao với bbox, không chỉ patch tâm ──
            c0 = int(np.clip(np.floor(x1 * G), 0, G - 1))
            c1 = int(np.clip(np.ceil(x2 * G) - 1, 0, G - 1))
            r0 = int(np.clip(np.floor(y1 * G), 0, G - 1))
            r1 = int(np.clip(np.ceil(y2 * G) - 1, 0, G - 1))
            occupied[r0:r1 + 1, c0:c1 + 1] = True

            # Positive patch: patch trong bbox gần tâm nhất.
            pr = int(np.clip(np.floor(cy * G), r0, r1))
            pc = int(np.clip(np.floor(cx * G), c0, c1))
            key = (pr, pc)
            if key in seen_positive:
                continue
            seen_positive.add(key)

            pos_patches.append({
                "patch_coord": (pr, pc),
                "class_id": cid,
                "log_size": [math.log(w), math.log(h)],
            })

        # Loại negative nằm sát biên defect (1 patch) để tránh nhãn nhiễu.
        occupied_for_neg = cv2.dilate(
            occupied.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)

        valid = self._get_valid_patches(pil_img, str(img_path))
        free_coords = np.argwhere(valid & ~occupied_for_neg)

        neg_patches: List[Dict] = []
        num_neg_needed = max(1, len(pos_patches) * self.neg_ratio)
        if len(free_coords) > 0:
            n_take = min(num_neg_needed, len(free_coords))
            chosen = np.random.choice(len(free_coords), size=n_take, replace=False)
            for ci in chosen:
                pr, pc = free_coords[ci]
                neg_patches.append({
                    "patch_coord": (int(pr), int(pc)),
                    "class_id": NUM_CLASSES - 1,  # no_defect
                    "log_size": [0.0, 0.0],
                })

        return {
            "image": img_tensor,
            "pos_patches": pos_patches,
            "neg_patches": neg_patches,
            "image_path": str(img_path),
        }


# Backward-compatible alias
CAMGPatchDataset = CAMGDataset


# ─────────────────────────────────────────────────────────
# 5. CAMG Trainer
# ─────────────────────────────────────────────────────────

class CAMGTrainer:
    """Huấn luyện hai prediction heads; DINOv2 backbone đóng băng."""

    def __init__(
        self,
        model: Optional[CAMGModel] = None,
        lr: float = 1e-4,
        device: str = "cuda",
        output_dir: str = "outputs/camg_checkpoints",
        num_workers: int = 0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model or CAMGModel(device=str(self.device))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = int(num_workers)

        self.optimizer = torch.optim.Adam(self.model.heads.parameters(), lr=lr)
        self.criterion = CAMGLoss(lambda_box=1.0)

    def train(
        self,
        metadata_path: str = "data/train_metadata.json",
        epochs: int = 10,
        batch_size: int = 32,
    ) -> str:
        """
        Huấn luyện CAMG.

        batch_size ở đây là số ảnh tích lũy gradient trước mỗi lần optimizer.step().
        Mỗi ảnh được đưa qua DataLoader với batch vật lý = 1 vì số patch
        positive/negative thay đổi theo từng ảnh.
        """
        if epochs <= 0:
            raise ValueError("epochs phải > 0.")
        grad_accum_steps = max(1, int(batch_size))

        logger.info("=" * 70)
        logger.info(
            f"[CAMG Trainer] metadata={metadata_path} | epochs={epochs} | "
            f"grad_accum={grad_accum_steps} ảnh/step | device={self.device}"
        )
        logger.info("=" * 70)

        dataset = CAMGDataset(metadata_path=metadata_path)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=_collate_camg,
            drop_last=False,
        )

        if len(dataloader) == 0:
            raise RuntimeError("CAMG DataLoader rỗng. Kiểm tra lại metadata/ảnh.")

        best_ckpt_path = self.output_dir / "camg_best.pth"
        last_ckpt_path = self.output_dir / "camg_last.pth"
        best_loss = float("inf")

        self.model.train()
        self.model.backbone.eval()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_cls_loss = 0.0
            total_box_loss = 0.0
            sample_count = 0
            accum_steps = 0

            self.optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(
                dataloader,
                desc=f"Epoch [{epoch}/{epochs}] [CAMG]",
                leave=False,
                unit="ảnh",
            )

            for step, batch in enumerate(pbar):
                images = batch["image"].to(self.device)  # (1, 3, 588, 588)
                pos_batch = batch["pos_patches"]        # List[List[Dict]]
                neg_batch = batch["neg_patches"]        # List[List[Dict]]

                # Forward 1 lần cho cả batch vật lý (thường = 1 ảnh).
                cls_logits, size_preds = self.model(images)
                # cls_logits: (B, 7, G, G), size_preds: (B, 2, G, G)

                patch_logits = []
                patch_cls_targets = []
                patch_size_preds = []
                patch_size_targets = []
                patch_is_pos = []

                for b in range(images.shape[0]):
                    for p in pos_batch[b]:
                        pr, pc = p["patch_coord"]
                        patch_logits.append(cls_logits[b, :, pr, pc])
                        patch_cls_targets.append(int(p["class_id"]))
                        patch_size_preds.append(size_preds[b, :, pr, pc])
                        patch_size_targets.append([float(p["log_size"][0]), float(p["log_size"][1])])
                        patch_is_pos.append(True)

                    for n in neg_batch[b]:
                        pr, pc = n["patch_coord"]
                        patch_logits.append(cls_logits[b, :, pr, pc])
                        patch_cls_targets.append(int(n["class_id"]))
                        patch_size_preds.append(size_preds[b, :, pr, pc])
                        patch_size_targets.append([0.0, 0.0])
                        patch_is_pos.append(False)

                if len(patch_logits) == 0:
                    continue

                logits_tensor = torch.stack(patch_logits)
                targets_tensor = torch.tensor(patch_cls_targets, device=self.device, dtype=torch.long)
                sizes_tensor = torch.stack(patch_size_preds)
                size_targets_tensor = torch.tensor(
                    patch_size_targets, device=self.device, dtype=torch.float32
                )
                is_pos_tensor = torch.tensor(patch_is_pos, device=self.device, dtype=torch.bool)

                loss, l_cls, l_box = self.criterion(
                    logits_tensor,
                    targets_tensor,
                    sizes_tensor,
                    size_targets_tensor,
                    is_pos_tensor,
                )

                # Gradient accumulation: chia loss để tổng gradient tương đương batch lớn.
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()
                accum_steps += 1
                sample_count += 1

                if accum_steps % grad_accum_steps == 0 or step == len(dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.heads.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.item())
                total_cls_loss += float(l_cls.item())
                total_box_loss += float(l_box.item())
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "cls": f"{l_cls.item():.4f}",
                    "box": f"{l_box.item():.4f}",
                })

            avg_loss = total_loss / max(1, sample_count)
            avg_cls = total_cls_loss / max(1, sample_count)
            avg_box = total_box_loss / max(1, sample_count)

            logger.info(
                f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f} "
                f"(L_cls: {avg_cls:.4f}, L_box: {avg_box:.4f})"
            )

            # ── best_loss đã được khởi tạo = inf trước vòng lặp ──
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.heads.state_dict(), best_ckpt_path)
                logger.info(f"  -> Đã lưu best checkpoint tại {best_ckpt_path}")

        # Luôn lưu checkpoint cuối để debug/khôi phục.
        torch.save(self.model.heads.state_dict(), last_ckpt_path)
        if not best_ckpt_path.exists():
            torch.save(self.model.heads.state_dict(), best_ckpt_path)

        logger.info(f"Hoàn tất huấn luyện CAMG! Best checkpoint: {best_ckpt_path}")
        return str(best_ckpt_path)


# ─────────────────────────────────────────────────────────
# 6. CAMG Generator (Inference với Top-k Sampling, k=5)
# ─────────────────────────────────────────────────────────

class CAMGGenerator:
    """
    Sinh mask lỗi context-aware:
      - Dense probability map từ CAMG;
      - Top-k sampling (k=5) chỉ trên các patch foreground hợp lệ;
      - Size prior theo phân phối DeepPCB;
      - Không bao giờ chạy với trọng số random nếu không được phép rõ ràng.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        k: int = 5,
        image_size: int = 588,
        patch_size: int = 14,
        allow_random_weights: bool = False,
        stats_path: Union[str, Path] = DEFAULT_STATS_PATH,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.k = int(k)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.size_prior = MaskSizePrior(stats_path=stats_path)

        self.model = CAMGModel(device=str(self.device))

        if weights_path is None or not Path(weights_path).exists():
            msg = (
                "[CAMG] Không có checkpoint trọng số hợp lệ. "
                "CAMG sẽ KHÔNG chạy với prediction heads random vì kết quả mask "
                "sẽ vô nghĩa.\n"
                "Hãy chạy trước: python run_pipeline.py step1_train "
                "--metadata_path data/train_metadata.json --epochs 10\n"
                "hoặc dùng 'python run_pipeline.py full ...' để pipeline tự huấn luyện."
            )
            if not allow_random_weights:
                raise FileNotFoundError(msg)
            logger.warning(msg + "\n[CAMG] allow_random_weights=True: tiếp tục với trọng số random (chỉ dùng để debug).")
        else:
            logger.info(f"[CAMG] Tải checkpoint trọng số từ {weights_path}")
            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(weights_path, map_location=self.device)
            self.model.heads.load_state_dict(state)

        self.weights_path = str(Path(weights_path).resolve()) if weights_path else None
        self.allow_random_weights = bool(allow_random_weights)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def predict_defect_map(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trả về:
          prob_map: (num_classes, G, G)
          size_map: (2, G, G) kích thước normalized (w, h)
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        cls_logits, size_preds = self.model(img_tensor)

        probs = F.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy()
        log_sizes = size_preds.squeeze(0).cpu().numpy()
        # Chặn tràn số; không dùng cap 0.20. Việc giới hạn thực tế do MaskSizePrior.
        log_sizes = np.clip(log_sizes, -8.0, math.log(0.25))
        sizes = np.exp(log_sizes)
        return probs, sizes

    def _prepare_validity(self, image: Image.Image):
        valid_patches = compute_valid_patch_mask(
            image,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        fg_pixels, hole_pixels = _image_pixel_validity(
            image,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        return valid_patches, fg_pixels, hole_pixels

    def _box_is_valid(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        fg_pixels: np.ndarray,
        hole_pixels: np.ndarray,
        margin: int = 2,
    ) -> bool:
        h, w = fg_pixels.shape[:2]
        if x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin:
            return False
        if x2 <= x1 or y2 <= y1:
            return False

        box_fg = fg_pixels[y1:y2, x1:x2]
        box_hole = hole_pixels[y1:y2, x1:x2]
        if box_fg.size == 0:
            return False
        if float(box_fg.mean()) < 0.55:
            return False
        if float(box_hole.mean()) > 0.10:
            return False
        return True

    def generate_mask_for_defect(
        self,
        image: Image.Image,
        defect_type: str,
        orig_size: Optional[Tuple[int, int]] = None,
        validity: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Dict:
        """Sinh 1 mask lỗi cho loại lỗi chỉ định."""
        if defect_type not in PCB_DEFECT_CLASSES:
            raise ValueError(f"Loại lỗi không hợp lệ: {defect_type}. Hỗ trợ: {PCB_DEFECT_CLASSES}")

        if orig_size is None:
            orig_size = image.size
        orig_w, orig_h = int(orig_size[0]), int(orig_size[1])
        cls_id = PCB_DEFECT_CLASSES.index(defect_type)

        prob_map, size_map = self.predict_defect_map(image)
        if validity is None:
            validity = self._prepare_validity(image)
        valid_patches, fg_pixels, hole_pixels = validity

        c_prob = prob_map[cls_id]
        G = self.grid_size
        valid_flat = valid_patches.flatten()
        candidate_indices = np.flatnonzero(valid_flat)

        if candidate_indices.size == 0:
            raise ValueError(
                "Không có patch foreground hợp lệ để sinh mask "
                "(ảnh có thể trống, quá tối hoặc toàn mounting hole)."
            )

        # Top-k chỉ trên các vị trí hợp lệ.
        k_eff = min(max(1, self.k), candidate_indices.size)
        candidate_probs = c_prob.flatten()[candidate_indices]
        top_local = np.argsort(candidate_probs)[-k_eff:]
        top_indices = candidate_indices[top_local]

        # Paper dùng top-k sampling; ta thử ngẫu nhiên trong top-k và kiểm tra
        # ràng buộc foreground/hole cho từng ứng viên.
        chosen = None
        chosen_sizes = None
        order = np.random.permutation(top_indices)
        for flat_idx in order:
            pr, pc = divmod(int(flat_idx), G)
            w_norm, h_norm, was_clamped = self.size_prior.sanitize(
                float(size_map[0, pr, pc]),
                float(size_map[1, pr, pc]),
                orig_w,
                orig_h,
            )

            bw = max(2, int(round(w_norm * orig_w)))
            bh = max(2, int(round(h_norm * orig_h)))

            cx = (pc + 0.5) / G * orig_w
            cy = (pr + 0.5) / G * orig_h
            # Dịch tâm vào trong để bbox không chạm viền ảnh.
            cx = float(np.clip(cx, bw / 2.0 + 2, orig_w - bw / 2.0 - 2))
            cy = float(np.clip(cy, bh / 2.0 + 2, orig_h - bh / 2.0 - 2))

            x1 = max(0, int(round(cx - bw / 2.0)))
            y1 = max(0, int(round(cy - bh / 2.0)))
            x2 = min(orig_w, x1 + bw)
            y2 = min(orig_h, y1 + bh)
            x1 = max(0, x2 - bw)
            y1 = max(0, y2 - bh)

            if self._box_is_valid(x1, y1, x2, y2, fg_pixels, hole_pixels):
                chosen = (pr, pc, int(flat_idx), was_clamped)
                chosen_sizes = (w_norm, h_norm)
                break

        # Nếu top-k không ứng viên nào thỏa, dùng vị trí hợp lệ có xác suất cao
        # nhất và thu nhỏ bbox về median prior để đảm bảo nằm trong foreground.
        if chosen is None:
            best_local = int(np.argmax(candidate_probs))
            flat_idx = int(candidate_indices[best_local])
            pr, pc = divmod(flat_idx, G)
            w_norm = float(np.clip(self.size_prior.median_w, self.size_prior.min_w_norm, self.size_prior.max_w_norm))
            h_norm = float(np.clip(self.size_prior.median_h, self.size_prior.min_h_norm, self.size_prior.max_h_norm))
            bw = max(2, int(round(w_norm * orig_w)))
            bh = max(2, int(round(h_norm * orig_h)))
            cx = (pc + 0.5) / G * orig_w
            cy = (pr + 0.5) / G * orig_h
            cx = float(np.clip(cx, bw / 2.0 + 2, orig_w - bw / 2.0 - 2))
            cy = float(np.clip(cy, bh / 2.0 + 2, orig_h - bh / 2.0 - 2))
            x1 = max(0, int(round(cx - bw / 2.0)))
            y1 = max(0, int(round(cy - bh / 2.0)))
            x2 = min(orig_w, x1 + bw)
            y2 = min(orig_h, y1 + bh)
            x1 = max(0, x2 - bw)
            y1 = max(0, y2 - bh)
            if not self._box_is_valid(x1, y1, x2, y2, fg_pixels, hole_pixels):
                raise ValueError(
                    f"Không tìm được bbox foreground hợp lệ cho '{defect_type}' "
                    "sau khi thử top-k và median-size fallback."
                )
            chosen = (pr, pc, flat_idx, True)
            chosen_sizes = (w_norm, h_norm)
            logger.warning(
                f"[CAMG] Không có top-k ứng viên hợp lệ cho '{defect_type}'; "
                "dùng fallback median size tại patch foreground tốt nhất."
            )

        pr, pc, flat_idx, was_clamped = chosen
        w_norm, h_norm = chosen_sizes
        confidence = float(c_prob[pr, pc])

        bw = max(2, int(round(w_norm * orig_w)))
        bh = max(2, int(round(h_norm * orig_h)))
        cx = (pc + 0.5) / G * orig_w
        cy = (pr + 0.5) / G * orig_h
        cx = float(np.clip(cx, bw / 2.0 + 2, orig_w - bw / 2.0 - 2))
        cy = float(np.clip(cy, bh / 2.0 + 2, orig_h - bh / 2.0 - 2))
        x1 = max(0, int(round(cx - bw / 2.0)))
        y1 = max(0, int(round(cy - bh / 2.0)))
        x2 = min(orig_w, x1 + bw)
        y2 = min(orig_h, y1 + bh)

        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        mask_pil = Image.fromarray(mask)

        return {
            "defect_type": defect_type,
            "class_id": cls_id,
            "confidence": confidence,
            "patch_coord": (pr, pc),
            "bbox_pixel": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_norm": [
                float((x1 + x2) / 2.0 / orig_w),
                float((y1 + y2) / 2.0 / orig_h),
                float((x2 - x1) / orig_w),
                float((y2 - y1) / orig_h),
            ],
            "size_norm": [float(w_norm), float(h_norm)],
            "size_clamped": bool(was_clamped),
            "mask_image": mask_pil,
        }

    def generate_batch(
        self,
        image_dir: str | Path = "data/pcb_normal",
        output_dir: str | Path = "outputs/masks",
        defect_types: Optional[List[str]] = None,
        images_per_defect: int = 1,
    ) -> List[Dict]:
        """Sinh batch mask cho toàn bộ ảnh normal trong thư mục."""
        img_dir = Path(image_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        defect_types = defect_types or PCB_DEFECT_CLASSES
        images = sorted(
            list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        )
        if not images:
            logger.warning(f"Không tìm thấy ảnh nào trong {img_dir}")
            return []

        logger.info(
            f"[CAMG] Sinh masks cho {len(images)} ảnh, {len(defect_types)} loại lỗi, "
            f"{images_per_defect} lần/loại..."
        )
        records: List[Dict] = []
        total_masks = len(images) * len(defect_types) * images_per_defect
        pbar = tqdm(total=total_masks, desc="Sinh mask khuyết tật (CAMG Top-k)", unit="mask")
        skipped = 0

        for img_path in images:
            pil_img = Image.open(img_path).convert("RGB")
            W, H = pil_img.size

            # Tính foreground/hole một lần cho mỗi ảnh, dùng cho cả 6 loại lỗi.
            try:
                validity = self._prepare_validity(pil_img)
            except Exception as e:
                logger.warning(f"[CAMG] Bỏ qua {img_path.name}: không chuẩn bị được validity ({e}).")
                skipped += len(defect_types) * images_per_defect
                pbar.update(len(defect_types) * images_per_defect)
                continue

            for dtype in defect_types:
                for rep in range(images_per_defect):
                    try:
                        res = self.generate_mask_for_defect(
                            pil_img,
                            defect_type=dtype,
                            orig_size=(W, H),
                            validity=validity,
                        )
                    except Exception as e:
                        logger.warning(f"[CAMG] Bỏ qua mask {img_path.name}/{dtype}: {e}")
                        skipped += 1
                        pbar.update(1)
                        continue

                    mask_filename = f"{img_path.stem}_{dtype}_{rep}_mask.png"
                    mask_save_path = out_dir / mask_filename
                    res["mask_image"].save(mask_save_path)

                    records.append({
                        "normal_image_path": str(img_path.resolve()),
                        "mask_path": str(mask_save_path.resolve()),
                        "defect_type": dtype,
                        "class_id": res["class_id"],
                        "confidence": res["confidence"],
                        "patch_coord": list(res["patch_coord"]),
                        "bbox_pixel": res["bbox_pixel"],
                        "bbox_norm": res["bbox_norm"],
                        "size_norm": res["size_norm"],
                        "size_clamped": res["size_clamped"],
                    })
                    pbar.update(1)

        pbar.close()

        meta_path = out_dir / "mask_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        info = {
            "mask_version": MASK_GENERATION_VERSION,
            "weights_path": self.weights_path,
            "allow_random_weights": self.allow_random_weights,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "k": self.k,
            "size_prior": {
                "q1_w": self.size_prior.q1_w,
                "q3_w": self.size_prior.q3_w,
                "median_w": self.size_prior.median_w,
                "q1_h": self.size_prior.q1_h,
                "q3_h": self.size_prior.q3_h,
                "median_h": self.size_prior.median_h,
            },
            "total_attempted": total_masks,
            "total_saved": len(records),
            "total_skipped": skipped,
        }
        with open(out_dir / "mask_generation_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info(f"Đã tạo {len(records)} masks. Metadata: {meta_path}")
        if skipped:
            logger.warning(f"Đã bỏ qua {skipped}/{total_masks} mask do ảnh/vùng không hợp lệ.")
        return records


# ─────────────────────────────────────────────────────────
# Backward Compatibility Wrapper
# ─────────────────────────────────────────────────────────
class PCBMaskGenerator(CAMGGenerator):
    """Lớp bọc tương thích ngược với code cũ của pipeline."""
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(
        "CAMG module đã sẵn sàng. Hãy dùng:\n"
        "  python run_pipeline.py step1_train --metadata_path data/train_metadata.json\n"
        "  python run_pipeline.py step1 --image_dir data/pcb_normal "
        "--camg_weights outputs/pcb_synthesis/camg_checkpoints/camg_best.pth"
    )
