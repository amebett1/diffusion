"""
Step 1: Defect Mask Generation using DINOv2
--------------------------------------------
- Trích xuất đặc trưng hình ảnh từ DINOv2
- Xác định vị trí, kích thước phù hợp để đặt lỗi
- Dùng bounding box thay vì mask pixel-level chính xác
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# DINOv2 Feature Extractor
# ─────────────────────────────────────────────────────────
class DINOv2FeatureExtractor:
    """Trích xuất patch-level features từ DINOv2 để xác định vùng đặt lỗi."""

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading DINOv2 model: {model_name} on {self.device}")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

        self.patch_size = 14  # ViT-B/14
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_patch_features(self, image: Image.Image) -> np.ndarray:
        """
        Trả về ma trận attention map (H_patch x W_patch) dùng để
        ưu tiên đặt lỗi vào vùng có texture phức tạp (không phải nền).
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        # Lấy attention của [CLS] token với các patch
        outputs = self.model.get_intermediate_layers(x, n=1, return_class_token=True)
        patch_tokens = outputs[0][0].squeeze(0).cpu().numpy()  # (N_patches, D)

        n_patches = int(patch_tokens.shape[0] ** 0.5)
        # Dùng L2-norm của patch feature làm "saliency score"
        saliency = np.linalg.norm(patch_tokens, axis=-1).reshape(n_patches, n_patches)
        return saliency


# ─────────────────────────────────────────────────────────
# Bounding Box Proposal
# ─────────────────────────────────────────────────────────
class DefectBBoxProposer:
    """
    Đề xuất bounding box vị trí lỗi dựa trên:
      1. Saliency map từ DINOv2
      2. Ràng buộc kích thước tương đối với ảnh gốc
    """

    def __init__(
        self,
        min_area_ratio: float = 0.01,   # Lỗi tối thiểu 1% diện tích ảnh
        max_area_ratio: float = 0.15,   # Lỗi tối đa 15% diện tích ảnh
        n_proposals: int = 5,
        strategy: str = "saliency",     # "saliency" | "random" | "grid"
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.n_proposals = n_proposals
        self.strategy = strategy

    def propose(
        self,
        image: Image.Image,
        saliency_map: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """
        Trả về danh sách dict với keys: x, y, w, h (pixel coords), score.
        """
        W, H = image.size
        proposals = []

        if self.strategy == "saliency" and saliency_map is not None:
            proposals = self._saliency_proposals(saliency_map, W, H)
        elif self.strategy == "grid":
            proposals = self._grid_proposals(W, H)
        else:
            proposals = self._random_proposals(W, H)

        return proposals[:self.n_proposals]

    def _saliency_proposals(self, saliency: np.ndarray, W: int, H: int) -> list[dict]:
        """Đặt lỗi ưu tiên vào vùng có saliency cao (trace/pad trên PCB)."""
        # Resize saliency về kích thước ảnh
        sal_resized = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR)
        sal_norm = (sal_resized - sal_resized.min()) / (sal_resized.max() - sal_resized.min() + 1e-8)

        proposals = []
        for _ in range(self.n_proposals * 3):  # oversample rồi lọc
            # Sample một điểm theo phân phối saliency
            flat_probs = sal_norm.flatten()
            flat_probs = flat_probs / flat_probs.sum()
            idx = np.random.choice(len(flat_probs), p=flat_probs)
            cy, cx = divmod(idx, W)

            # Kích thước bbox ngẫu nhiên trong [min, max] area ratio
            area = random.uniform(self.min_area_ratio, self.max_area_ratio) * W * H
            aspect = random.uniform(0.5, 2.0)
            bw = int(np.sqrt(area * aspect))
            bh = int(np.sqrt(area / aspect))
            bw = max(16, min(bw, W // 2))
            bh = max(16, min(bh, H // 2))

            x = max(0, cx - bw // 2)
            y = max(0, cy - bh // 2)
            x = min(x, W - bw)
            y = min(y, H - bh)

            score = float(sal_norm[y: y + bh, x: x + bw].mean())
            proposals.append(dict(x=x, y=y, w=bw, h=bh, score=score))

        proposals.sort(key=lambda p: -p["score"])
        return proposals

    def _random_proposals(self, W: int, H: int) -> list[dict]:
        proposals = []
        for _ in range(self.n_proposals):
            area = random.uniform(self.min_area_ratio, self.max_area_ratio) * W * H
            aspect = random.uniform(0.5, 2.0)
            bw = int(np.sqrt(area * aspect))
            bh = int(np.sqrt(area / aspect))
            bw = max(16, min(bw, W // 2))
            bh = max(16, min(bh, H // 2))
            x = random.randint(0, W - bw)
            y = random.randint(0, H - bh)
            proposals.append(dict(x=x, y=y, w=bw, h=bh, score=1.0))
        return proposals

    def _grid_proposals(self, W: int, H: int) -> list[dict]:
        """Chia ảnh thành grid, đề xuất ở từng ô."""
        grid = int(np.ceil(np.sqrt(self.n_proposals)))
        cw, ch = W // grid, H // grid
        proposals = []
        for i in range(grid):
            for j in range(grid):
                cx = j * cw + cw // 2
                cy = i * ch + ch // 2
                bw = max(16, int(cw * random.uniform(0.3, 0.7)))
                bh = max(16, int(ch * random.uniform(0.3, 0.7)))
                x = max(0, min(cx - bw // 2, W - bw))
                y = max(0, min(cy - bh // 2, H - bh))
                proposals.append(dict(x=x, y=y, w=bw, h=bh, score=1.0))
        return proposals


# ─────────────────────────────────────────────────────────
# Mask Renderer
# ─────────────────────────────────────────────────────────
def bbox_to_mask(image_size: tuple[int, int], bbox: dict) -> np.ndarray:
    """Chuyển bbox dict → binary mask (H×W, uint8, 0/255)."""
    W, H = image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    mask[y: y + h, x: x + w] = 255

    # Optional: làm mờ viền để inpainting tự nhiên hơn
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


# ─────────────────────────────────────────────────────────
# MaskGenerator — entry point cho Step 1
# ─────────────────────────────────────────────────────────
class PCBMaskGenerator:
    """
    Kết hợp DINOv2 feature extractor + BBox proposer thành 1 interface.

    Sử dụng:
        gen = PCBMaskGenerator()
        result = gen.generate("board.jpg", defect_type="scratch")
        # result["mask"] -> PIL.Image (binary mask)
        # result["bbox"] -> dict(x,y,w,h,score)
    """

    def __init__(
        self,
        dino_model: str = "dinov2_vitb14",
        device: str = "cuda",
        strategy: str = "saliency",
        n_proposals: int = 5,
    ):
        self.extractor = DINOv2FeatureExtractor(dino_model, device)
        self.proposer = DefectBBoxProposer(strategy=strategy, n_proposals=n_proposals)

    def generate(
        self,
        image_path: str | Path,
        defect_type: str = "scratch",
        proposal_idx: int = 0,
    ) -> dict:
        """
        Args:
            image_path: Đường dẫn ảnh PCB gốc (không lỗi).
            defect_type: Loại lỗi (dùng để log / đặt tên file).
            proposal_idx: Chọn proposal thứ mấy trong danh sách đề xuất.

        Returns:
            dict với các keys:
              - image (PIL.Image): ảnh gốc
              - mask (PIL.Image):  binary mask (0/255, mode 'L')
              - bbox (dict):       {x, y, w, h, score}
              - defect_type (str)
        """
        image = Image.open(image_path).convert("RGB")
        saliency = self.extractor.extract_patch_features(image)
        proposals = self.proposer.propose(image, saliency)

        if not proposals:
            raise RuntimeError("Không thể tạo proposal cho ảnh này.")

        bbox = proposals[min(proposal_idx, len(proposals) - 1)]
        mask_arr = bbox_to_mask(image.size, bbox)
        mask_pil = Image.fromarray(mask_arr, mode="L")

        return dict(
            image=image,
            mask=mask_pil,
            bbox=bbox,
            defect_type=defect_type,
        )

    def generate_batch(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        defect_types: list[str] | None = None,
        images_per_defect: int = 1,
    ) -> list[dict]:
        """
        Sinh mask cho toàn bộ ảnh trong thư mục.

        Returns:
            Danh sách metadata records (list of dicts) để đưa vào Step 2.
        """
        if defect_types is None:
            defect_types = ["scratch", "missing_hole", "open_circuit",
                            "short", "spur", "spurious_copper"]

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for p in image_dir.glob(ext)
        )

        records = []
        for img_path in image_paths:
            for defect_type in defect_types:
                for idx in range(images_per_defect):
                    try:
                        result = self.generate(img_path, defect_type)
                        stem = f"{img_path.stem}_{defect_type}_{idx:03d}"
                        mask_path = output_dir / f"{stem}_mask.png"
                        result["mask"].save(mask_path)

                        record = {
                            "image_path": str(img_path),
                            "mask_path": str(mask_path),
                            "bbox": result["bbox"],
                            "defect_type": defect_type,
                        }
                        records.append(record)
                        logger.info(f"[Mask] {stem} bbox={result['bbox']}")

                    except Exception as e:
                        logger.warning(f"[Mask] Lỗi xử lý {img_path}: {e}")

        # Lưu metadata
        meta_path = output_dir / "mask_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(f"[Mask] Đã lưu {len(records)} records → {meta_path}")
        return records


# ─────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: PCB Mask Generator")
    parser.add_argument("--image_dir", required=True, help="Thư mục ảnh PCB gốc")
    parser.add_argument("--output_dir", default="outputs/masks", help="Thư mục lưu mask")
    parser.add_argument("--defect_types", nargs="+",
                        default=["scratch", "missing_hole", "open_circuit",
                                 "short", "spur", "spurious_copper"])
    parser.add_argument("--images_per_defect", type=int, default=1)
    parser.add_argument("--strategy", default="saliency",
                        choices=["saliency", "random", "grid"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    gen = PCBMaskGenerator(strategy=args.strategy, device=args.device)
    records = gen.generate_batch(
        args.image_dir,
        args.output_dir,
        args.defect_types,
        args.images_per_defect,
    )
    print(f"✅ Tạo được {len(records)} mask records.")
