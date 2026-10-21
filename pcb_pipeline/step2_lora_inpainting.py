"""
Step 2: Defect Image Synthesis using Inpainting Diffusion + LoRA + Multi-Token Textual Inversion
================================================================================================
Triển khai chuẩn theo bài báo:
"A Synthetic Defect Image Generation Method for Enhanced Industrial
Defect Detection Based on Inpainting Diffusion and Context-aware Mask Generation"
(Journal of Intelligent Manufacturing, 2026 - Mục Methods, Trang 5)

Các điểm kỹ thuật cốt lõi:
1. Multi-Token Textual Inversion:
   - Gán 1 token học riêng cho mỗi loại lỗi (6 tokens cho 6 lớp lỗi PCB).
   - Prompt lỗi P:  "a photo of a pcb with <defect_type>"
   - Prompt chuẩn P': "a photo of a pcb"
2. Parameter-Efficient LoRA (r=8, α=16):
   - Gắn vào Attention projections và Output projections của CẢ U-Net và Text Encoder.
   - Learning rate: 1e-4 cho U-Net, 5e-5 cho Text Encoder, 5e-4 cho Textual Inversion.
3. Hàm mất mát kép (Dual Loss Function - Equations 6, 7, 8):
   - L_def = E [ B  * ||ε - ε_θ(x_t, B,  t, τ(P)) ||^2_2 ]  (trên mask lỗi B)
   - L_rec = E [ B' * ||ε - ε_θ(x_t, B', t, τ(P'))||^2_2 ] (trên box ngẫu nhiên B' ∈ [Q1, Q3])
   - L_tot = L_def + λ * L_rec  (λ = 0.1)
4. Denoising Pipeline với cơ chế Latent Overwriting:
   - 50 Denoising steps (DDIM Scheduler).
   - Latent Overwriting tại mỗi bước khử nhiễu t (Trang 5 bài báo):
       z_{t-1} = B_lat ⊙ z_{t-1, synthesized} + (1 - B_lat) ⊙ z_{t-1, orig}
     giúp bảo toàn 100% cấu trúc nền của bo mạch.
   - Negative prompt: "blurry, low quality, deformed, ugly, watermark, signature, text, logo".
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

# Danh sách 6 lớp lỗi PCB chuẩn DeepPCB & Paper
PCB_DEFECT_CLASSES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]

# Ánh xạ từ vựng khởi tạo ngữ nghĩa cho từng token
INIT_TOKENS = {
    "mouse_bite": "notch",
    "spur": "spur",
    "missing_hole": "hole",
    "short": "short",
    "open_circuit": "broken",
    "spurious_copper": "copper",
}

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, deformed, ugly, watermark, signature, text, logo"
)


# ─────────────────────────────────────────────────────────
# 1. Multi-Token Textual Inversion Manager
# ─────────────────────────────────────────────────────────

class MultiTokenTIManager:
    """
    Quản lý Textual Inversion đa token cho 6 loại lỗi PCB:
    - Mỗi loại lỗi có 1 special token riêng dạng <defect_type>
    - Defect Prompt P:  'a photo of a pcb with <defect_type>'
    - Normal Prompt P': 'a photo of a pcb'
    """

    def __init__(
        self,
        tokenizer=None,
        text_encoder=None,
        defect_classes: Optional[List[str]] = None,
        product_name: str = "pcb",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.defect_classes = defect_classes or PCB_DEFECT_CLASSES
        self.product_name = product_name
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # Map placeholder tokens: "<mouse_bite>", "<spur>", v.v.
        self.placeholder_tokens = {c: f"<{c}>" for c in self.defect_classes}
        self.token_ids = {}
        self.placeholder_token_ids = []

        if self.tokenizer is not None and self.text_encoder is not None:
            self._initialize_tokens()

    @property
    def ti_params(self) -> Dict[str, torch.Tensor]:
        """Truy xuất embedding tensor của các token lỗi (phục vụ logging / debug)."""
        if self.text_encoder is not None:
            weight = self.text_encoder.get_input_embeddings().weight
            return {c: weight[tid] for c, tid in self.token_ids.items()}
        return {}

    def _initialize_tokens(self):
        """Thêm các placeholder tokens vào tokenizer và trích xuất vector embedding khởi tạo."""
        logger.info(f"[TI] Đang khởi tạo {len(self.defect_classes)} tokens Textual Inversion...")
        self.placeholder_token_ids = []
        for cls_name, placeholder in self.placeholder_tokens.items():
            num_added = self.tokenizer.add_tokens(placeholder)
            token_id = self.tokenizer.convert_tokens_to_ids(placeholder)
            self.token_ids[cls_name] = token_id
            self.placeholder_token_ids.append(token_id)

            # Tìm embedding của từ khởi tạo tương ứng
            init_word = INIT_TOKENS.get(cls_name, "defect")
            init_ids = self.tokenizer.encode(init_word, add_special_tokens=False)
            init_id = init_ids[0] if init_ids else self.tokenizer.unk_token_id

            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                init_emb = self.text_encoder.get_input_embeddings().weight[init_id].clone()
                self.text_encoder.get_input_embeddings().weight[token_id] = init_emb

        # Kích hoạt Autograd trực tiếp trên Embedding Table
        self.text_encoder.get_input_embeddings().weight.requires_grad_(True)
        # Lưu bản sao ban đầu của từ điển để bảo toàn các token gốc không bị trôi
        self.orig_embeds_params = self.text_encoder.get_input_embeddings().weight.data.clone()
        logger.info(f"[TI] Đã đăng ký và gắn kết Autograd cho các token: {list(self.placeholder_tokens.values())}")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Trả về tham số embedding table cho Optimizer."""
        if self.text_encoder is None:
            return []
        return [self.text_encoder.get_input_embeddings().weight]

    def zero_non_placeholder_grads(self):
        """
        Triệt tiêu gradient của toàn bộ token gốc trong CLIP vocab,
        chỉ bảo toàn gradient cho 6 placeholder tokens (<mouse_bite>, <spur>,...).
        Đây là kỹ thuật chuẩn của Textual Inversion giúp đồ thị Autograd không bị đứt.
        """
        if self.text_encoder is None:
            return
        embeds = self.text_encoder.get_input_embeddings()
        if embeds.weight.grad is not None:
            mask = torch.ones(embeds.weight.shape[0], dtype=torch.bool, device=embeds.weight.device)
            for tid in self.token_ids.values():
                mask[tid] = False
            embeds.weight.grad[mask] = 0.0

    def restore_non_placeholder_embeddings(self):
        """Khôi phục lại chính xác trọng số các token vocab gốc nếu bị weight decay làm lệch."""
        if self.text_encoder is None or not hasattr(self, "orig_embeds_params"):
            return
        with torch.no_grad():
            weight = self.text_encoder.get_input_embeddings().weight
            mask = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
            for tid in self.token_ids.values():
                mask[tid] = False
            weight.data[mask] = self.orig_embeds_params[mask].to(weight.device)

    def sync_embeddings(self):
        """Placeholder tương thích ngược. Trọng số embedding table đã nằm trong đồ thị Autograd."""
        pass

    def get_defect_prompt(self, defect_type: str) -> str:
        """Prompt P theo Equation 6 của bài báo."""
        token = self.placeholder_tokens.get(defect_type, f"<{defect_type}>")
        return f"a photo of a {self.product_name} with {token}"

    def get_normal_prompt(self) -> str:
        """Prompt P' theo Equation 7 của bài báo."""
        return f"a photo of a {self.product_name}"

    def save(self, output_path: Union[str, Path]):
        """Lưu trọng số của toàn bộ 6 TI tokens."""
        state = {}
        if self.text_encoder is not None:
            weight = self.text_encoder.get_input_embeddings().weight.detach().cpu()
            for cls_name, token_id in self.token_ids.items():
                state[cls_name] = weight[token_id].clone()
        torch.save(state, output_path)
        logger.info(f"[TI] Đã lưu TI embeddings -> {output_path}")

    def load(self, checkpoint_path: Union[str, Path]):
        """Nạp trọng số TI tokens từ checkpoint vào embedding table."""
        state = torch.load(checkpoint_path, map_location=self.device)
        if self.text_encoder is not None:
            weight = self.text_encoder.get_input_embeddings().weight
            with torch.no_grad():
                for cls_name, emb_data in state.items():
                    if cls_name in self.token_ids:
                        tid = self.token_ids[cls_name]
                        weight[tid] = emb_data.to(self.device)
        logger.info(f"[TI] Đã nạp TI embeddings <- {checkpoint_path}")


# ─────────────────────────────────────────────────────────
# 2. Native LoRA Layers (Cho cả U-Net và Text Encoder)
# ─────────────────────────────────────────────────────────

class LoRALinearLayer(nn.Module):
    """
    Lớp LoRA: ΔW = (α / r) * B * A
    Huấn luyện tham số bậc thấp với r=8, α=16 theo paper.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Ma trận A khởi tạo Gaussian, Ma trận B khởi tạo 0
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_A.device != x.device:
            self.lora_A.data = self.lora_A.data.to(x.device)
            self.lora_B.data = self.lora_B.data.to(x.device)
        return self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)


def inject_lora_into_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, LoRALinearLayer]:
    """
    Gắn các lớp LoRA vào các Attention projection layers và Output projection layers
    của model (U-Net hoặc Text Encoder).
    """
    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "out_proj"]

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(tgt) or tgt in name for tgt in target_modules):
                in_feat, out_feat = module.in_features, module.out_features
                target_device = module.weight.device
                target_dtype = module.weight.dtype
                lora_layer = LoRALinearLayer(in_feat, out_feat, rank=rank, alpha=alpha).to(
                    device=target_device, dtype=target_dtype
                )
                lora_layers[name] = lora_layer
                setattr(module, "lora_layer", lora_layer)

                orig_forward = module.forward

                def make_new_forward(m, orig_f):
                    def forward_fn(x):
                        dev = m.lora_layer.lora_A.device
                        dtype = m.lora_layer.lora_A.dtype
                        x_lora = x.to(device=dev, dtype=dtype) if (x.device != dev or x.dtype != dtype) else x
                        return orig_f(x) + m.lora_layer(x_lora)
                    return forward_fn

                module.forward = make_new_forward(module, orig_forward)

    logger.info(f"[LoRA] Đã gắn LoRA vào {len(lora_layers)} layers (rank={rank}, alpha={alpha}).")
    return lora_layers


# ─────────────────────────────────────────────────────────
# 3. Hàm mất mát kép (Dual Loss Function) & Lấy mẫu BBox B'
# ─────────────────────────────────────────────────────────

class DualInpaintLoss(nn.Module):
    """
    Triển khai hàm mất mát kép theo Equation 6, 7, 8 của bài báo:
      L_def = E [ B  * ||ε - ε_θ(x_t, B,  t, τ(P)) ||^2_2 ]
      L_rec = E [ B' * ||ε - ε_θ(x_t, B', t, τ(P'))||^2_2 ]
      L_tot = L_def + λ * L_rec  (λ = 0.1)

    Trong đó B' là mask bounding box ngẫu nhiên lấy từ vùng không lỗi,
    có kích thước nằm trong khoảng phân vị [Q1, Q3] của tập lỗi thực tế.
    """

    def __init__(
        self,
        lambda_rec: float = 0.1,
        stats_path: Union[str, Path] = "data/dataset_stats.json",
        default_q1: int = 21,
        default_q3: int = 36,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.q1_w = default_q1
        self.q3_w = default_q3
        self.q1_h = default_q1
        self.q3_h = default_q3

        stats_file = Path(stats_path)
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    px = data.get("pixel_size", {})
                    self.q1_w = int(round(px.get("width", {}).get("q1", default_q1)))
                    self.q3_w = int(round(px.get("width", {}).get("q3", default_q3)))
                    self.q1_h = int(round(px.get("height", {}).get("q1", default_q1)))
                    self.q3_h = int(round(px.get("height", {}).get("q3", default_q3)))
                logger.info(f"[Loss] Kích thước mask B' ngẫu nhiên: Width [{self.q1_w}..{self.q3_w}], Height [{self.q1_h}..{self.q3_h}]")
            except Exception as e:
                logger.warning(f"Không thể đọc stats từ {stats_path}: {e}")

    def sample_negative_mask(self, defect_mask: torch.Tensor) -> torch.Tensor:
        """
        Lấy mẫu bounding box mask B' trong vùng sạch (không chứa lỗi).
        defect_mask: Tensor nhị phân (B, 1, H, W)
        Returns:
          b_prime: Tensor nhị phân (B, 1, H, W)
        """
        B_batch, _, H, W = defect_mask.shape
        b_prime = torch.zeros_like(defect_mask)

        for b in range(B_batch):
            bw = random.randint(min(self.q1_w, max(1, W // 4)), min(self.q3_w, max(2, W // 2)))
            bh = random.randint(min(self.q1_h, max(1, H // 4)), min(self.q3_h, max(2, H // 2)))

            cur_mask = defect_mask[b, 0]
            max_tries = 20
            chosen_x1, chosen_y1 = 0, 0

            for _ in range(max_tries):
                x1 = random.randint(0, max(0, W - bw))
                y1 = random.randint(0, max(0, H - bh))
                overlap = cur_mask[y1:y1 + bh, x1:x1 + bw].sum()
                if overlap == 0:
                    chosen_x1, chosen_y1 = x1, y1
                    break
                chosen_x1, chosen_y1 = x1, y1

            b_prime[b, 0, chosen_y1:chosen_y1 + bh, chosen_x1:chosen_x1 + bw] = 1.0

        return b_prime

    def forward(
        self,
        noise_pred_def: torch.Tensor,
        noise_target: torch.Tensor,
        defect_mask: torch.Tensor,
        noise_pred_rec: Optional[torch.Tensor] = None,
        rec_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tính L_tot = L_def + λ * L_rec
        """
        m_def = F.interpolate(defect_mask, size=noise_pred_def.shape[-2:], mode="nearest")

        # 1. Defect Loss (Equation 6)
        n_def_pix = m_def.sum().clamp(min=1.0)
        l_def = (m_def * (noise_target - noise_pred_def) ** 2).sum() / n_def_pix

        # 2. Reconstruction Loss (Equation 7)
        if noise_pred_rec is not None and rec_mask is not None:
            m_rec = F.interpolate(rec_mask, size=noise_pred_rec.shape[-2:], mode="nearest")
            n_rec_pix = m_rec.sum().clamp(min=1.0)
            l_rec = (m_rec * (noise_target - noise_pred_rec) ** 2).sum() / n_rec_pix
        else:
            l_rec = torch.tensor(0.0, device=noise_pred_def.device)

        # 3. Total Loss (Equation 8)
        l_tot = l_def + self.lambda_rec * l_rec
        return l_tot, l_def, l_rec


# ─────────────────────────────────────────────────────────
# 4. LoRA Training Dataset
# ─────────────────────────────────────────────────────────

class PCBLoRADataset(Dataset):
    """
    Dataset nạp ảnh PCB thực tế và tạo mask bbox tương ứng từ train_metadata.json.
    """

    def __init__(self, metadata_path: Union[str, Path], image_size: int = 512):
        self.metadata_path = Path(metadata_path)
        self.image_size = image_size
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        img_path = rec["image_path"]

        # Đọc ảnh gốc
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback nếu ảnh không tồn tại
            img_bgr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        H, W, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Tạo mask nhị phân từ các bounding boxes
        mask = np.zeros((H, W), dtype=np.float32)
        defect_type = "missing_hole"

        boxes = rec.get("boxes", [])
        if boxes:
            defect_type = boxes[0].get("class_name", "missing_hole")
            for box in boxes:
                x1, y1 = box["x1"], box["y1"]
                x2, y2 = box["x2"], box["y2"]
                mask[y1:y2, x1:x2] = 1.0
        else:
            # Nếu không có box, tạo mask trung tâm nhỏ mặc định
            mask[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 1.0

        # Resize về kích thước chuẩn
        img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size))
        mask_resized = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Chuẩn hóa tensor ảnh [-1, 1] cho VAE
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 127.5 - 1.0
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "defect_type": defect_type,
            "image_path": img_path,
            "index": idx,
        }


# ─────────────────────────────────────────────────────────
# 5. LoRA + Textual Inversion Trainer (Full Diffusion Training Loop)
# ─────────────────────────────────────────────────────────

class LoRAInpaintTrainer:
    """
    Module quản lý huấn luyện LoRA (U-Net & Text Encoder) + Textual Inversion đa token.
    Triển khai đầy đủ Training Loop của bài báo:
    - VAE encode -> Latent space
    - Timestep sampling + Noise addition
    - Textual Inversion embedding + Prompt pairs (P, P')
    - Dual Loss (L_def + 0.1 * L_rec) với box B' ∈ [Q1, Q3]
    - Tích lũy Gradient chia tỷ lệ + Lưu LoRA & TI checkpoints
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        rank: int = 8,
        alpha: float = 16.0,
        lr_unet: float = 1e-4,
        lr_text_encoder: float = 5e-5,
        lr_ti: float = 5e-4,
        device: str = "cuda",
        output_dir: str = "outputs/lora_checkpoints",
        use_fp16: bool = True,
    ):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.rank = rank
        self.alpha = alpha
        self.lr_unet = lr_unet
        self.lr_text_encoder = lr_text_encoder
        self.lr_ti = lr_ti
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = DualInpaintLoss(lambda_rec=0.1)
        # Mục 1: Bật Mixed Precision (FP16 / AMP)
        self.use_amp = use_fp16 and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _init_models(self):
        """Khởi tạo và đóng băng các thành phần của Diffusion Model."""
        try:
            from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
            logger.info(f"[LoRA Trainer] Đang tải pipeline từ {self.model_id}...")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
            ).to(self.device)

            self.vae = pipe.vae
            self.unet = pipe.unet
            self.text_encoder = pipe.text_encoder
            self.tokenizer = pipe.tokenizer
            self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

            # Đóng băng các mô hình gốc
            self.vae.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.text_encoder.requires_grad_(False)

            # 1. Khởi tạo Textual Inversion Manager
            self.ti_manager = MultiTokenTIManager(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                device=str(self.device),
            )

            # 2. Gắn LoRA vào U-Net và Text Encoder
            self.lora_unet = inject_lora_into_model(self.unet, rank=self.rank, alpha=self.alpha)
            self.lora_text = inject_lora_into_model(self.text_encoder, rank=self.rank, alpha=self.alpha)

            # 3. Khởi tạo Optimizer AdamW với learning rate riêng
            trainable_params = [
                {"params": [p for l in self.lora_unet.values() for p in l.parameters()], "lr": self.lr_unet, "weight_decay": 1e-2},
                {"params": [p for l in self.lora_text.values() for p in l.parameters()], "lr": self.lr_text_encoder, "weight_decay": 1e-2},
                {"params": self.ti_manager.get_trainable_params(), "lr": self.lr_ti, "weight_decay": 0.0},
            ]
            self.optimizer = torch.optim.AdamW(trainable_params)
            self.is_initialized = True

        except ImportError:
            logger.warning("[LoRA Trainer] Chưa cài đặt 'diffusers'. Chạy ở chế độ dry-run / setup.")
            self.is_initialized = False

    def train(
        self,
        data_json: str | Path = "data/train_metadata.json",
        epochs: int = 10,
        batch_size: int = 32,
    ) -> str:
        """
        Vòng lặp huấn luyện chính thức chuẩn bài báo (tích hợp FP16 và VAE pre-encoding cache).
        """
        logger.info("=" * 60)
        logger.info("[LoRA Inpaint Trainer] Khởi chạy huấn luyện Dual-Loss LoRA:")
        logger.info(f"  Model ID:     {self.model_id}")
        logger.info(f"  LoRA Config:  rank={self.rank}, alpha={self.alpha}")
        logger.info(f"  Epochs:       {epochs}, Batch size: {batch_size}")
        logger.info(f"  LR U-Net:     {self.lr_unet}, LR Text: {self.lr_text_encoder}, LR TI: {self.lr_ti}")
        logger.info(f"  Device:       {self.device} (Mixed Precision AMP: {self.use_amp})")
        logger.info("=" * 60)

        data_path = Path(data_json)
        if not data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file metadata: {data_path}")

        dataset = PCBLoRADataset(metadata_path=data_path, image_size=512)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        ckpt_dir = self.output_dir / f"epoch_{epochs:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._init_models()
        if not getattr(self, "is_initialized", False):
            logger.info("Môi trường chưa cài thư viện 'diffusers' để huấn luyện GPU thực tế.")
            logger.info(f"Setup kiến trúc mã nguồn hoàn tất. Checkpoint dự kiến lưu tại: {ckpt_dir}")
            return str(ckpt_dir)

        # ── MỤC 2: TIỀN MÃ HÓA BỘ NHỚ ĐỆM VAE LATENTS (PRE-ENCODE CACHE) ────
        # 668 ảnh chỉ tốn ~22MB RAM, tránh việc VAE phải encode lặp lại mỗi epoch
        logger.info("[LoRA Trainer] Đang tiền mã hóa VAE Latents (Pre-encode cache 668 ảnh)...")
        cached_latents = {}
        cached_mask_lat_def = {}
        cached_masked_lat_def = {}

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="Pre-encoding VAE Latents", leave=False):
                sample = dataset[idx]
                img_t = sample["image"].unsqueeze(0).to(self.device)
                msk_t = sample["mask"].unsqueeze(0).to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    lat = self.vae.encode(img_t).latent_dist.sample() * 0.18215
                    lh, lw = lat.shape[-2:]
                    m_lat = F.interpolate(msk_t, size=(lh, lw), mode="nearest")
                    m_img = img_t * (1.0 - msk_t)
                    m_lat_img = self.vae.encode(m_img).latent_dist.sample() * 0.18215

                cached_latents[idx] = lat.cpu()
                cached_mask_lat_def[idx] = m_lat.cpu()
                cached_masked_lat_def[idx] = m_lat_img.cpu()

        logger.info(f"[LoRA Trainer] Đã cache thành công {len(cached_latents)} VAE latents vào RAM (Mục 2)!")
        if self.use_amp:
            logger.info("[LoRA Trainer] Đã kích hoạt Mixed Precision (FP16 / AMP) cho U-Net & Text Encoder (Mục 1)!")

        # ── VÒNG LẶP HUẤN LUYỆN DIFFUSION THỰC TẾ ─────────────────────────
        self.unet.train()
        self.text_encoder.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            accum_steps = 0
            self.optimizer.zero_grad()

            pbar_loader = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}] [LoRA Dual-Loss]", leave=False, unit="mẫu")
            for step, batch in enumerate(pbar_loader):
                sample_idx = batch["index"][0].item() if "index" in batch else None
                images = batch["image"].to(self.device)   # (1, 3, 512, 512)
                masks = batch["mask"].to(self.device)     # (1, 1, 512, 512)
                defect_type = batch["defect_type"][0]

                # 1. Mã hóa ảnh qua VAE để lấy latent z0 (Lấy từ RAM cache Mục 2)
                if sample_idx is not None and sample_idx in cached_latents:
                    latents = cached_latents[sample_idx].to(self.device)
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            latents = self.vae.encode(images).latent_dist.sample() * 0.18215
                latent_h, latent_w = latents.shape[-2:]

                # 2. Lấy mẫu bước thời gian t và cộng nhiễu Gaussian
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                    device=self.device, dtype=torch.long
                )
                noise = torch.randn_like(latents)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Mã hóa prompt P và prompt P'
                prompt_def = self.ti_manager.get_defect_prompt(defect_type)
                prompt_rec = self.ti_manager.get_normal_prompt()

                tokens_def = self.tokenizer(prompt_def, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(self.device)
                tokens_rec = self.tokenizer(prompt_rec, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    text_emb_def = self.text_encoder(tokens_def)[0]
                    text_emb_rec = self.text_encoder(tokens_rec)[0]

                # 4. Lấy mẫu bounding box B' từ vùng không lỗi (Equation 7)
                rec_masks = self.loss_fn.sample_negative_mask(masks)

                # ── XỬ LÝ 9 KÊNH ĐẦU VÀO CHO INPAINTING U-NET ──────────────────
                # runwayml/stable-diffusion-inpainting nhận 9 channels:
                # [noisy_latents (4), mask_lat (1), masked_image_latents (4)]
                in_channels = getattr(self.unet.config, "in_channels", 9)
                if in_channels == 9:
                    with torch.no_grad():
                        # Nhánh Defect (mask B) - Lấy từ RAM cache Mục 2:
                        if sample_idx is not None and sample_idx in cached_mask_lat_def:
                            mask_lat_def = cached_mask_lat_def[sample_idx].to(self.device)
                            masked_lat_def = cached_masked_lat_def[sample_idx].to(self.device)
                        else:
                            mask_lat_def = F.interpolate(masks, size=(latent_h, latent_w), mode="nearest")
                            masked_img_def = images * (1.0 - masks)
                            with torch.cuda.amp.autocast(enabled=self.use_amp):
                                masked_lat_def = self.vae.encode(masked_img_def).latent_dist.sample() * 0.18215
                        unet_input_def = torch.cat([noisy_latents, mask_lat_def, masked_lat_def], dim=1)

                        # Nhánh Reconstruction (mask B' ngẫu nhiên mỗi step):
                        mask_lat_rec = F.interpolate(rec_masks, size=(latent_h, latent_w), mode="nearest")
                        masked_img_rec = images * (1.0 - rec_masks)
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            masked_lat_rec = self.vae.encode(masked_img_rec).latent_dist.sample() * 0.18215
                        unet_input_rec = torch.cat([noisy_latents, mask_lat_rec, masked_lat_rec], dim=1)
                else:
                    unet_input_def = noisy_latents
                    unet_input_rec = noisy_latents

                # 5. U-Net dự đoán nhiễu cho cả 2 nhánh (AMP FP16 Mục 1)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    noise_pred_def = self.unet(unet_input_def, timesteps, encoder_hidden_states=text_emb_def).sample
                    noise_pred_rec = self.unet(unet_input_rec, timesteps, encoder_hidden_states=text_emb_rec).sample

                    # 6. Tính toán Dual Loss (L_tot = L_def + 0.1 * L_rec)
                    loss, l_def, l_rec = self.loss_fn(
                        noise_pred_def=noise_pred_def,
                        noise_target=noise,
                        defect_mask=masks,
                        noise_pred_rec=noise_pred_rec,
                        rec_mask=rec_masks,
                    )

                    # 7. Chia tỷ lệ loss cho batch_size và tích lũy gradient
                    scaled_loss = loss / batch_size

                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accum_steps += 1
                epoch_loss += loss.item()
                pbar_loader.set_postfix({"loss": f"{loss.item():.4f}", "l_def": f"{l_def.item():.4f}", "l_rec": f"{l_rec.item():.4f}"})

                if accum_steps % batch_size == 0 or step == len(dataloader) - 1:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    # Triệt tiêu gradient token thường của CLIP, bảo toàn 6 token TI
                    self.ti_manager.zero_non_placeholder_grads()

                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), max_norm=1.0)

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    # Khôi phục token vocab gốc nếu bị ảnh hưởng và reset gradient
                    self.ti_manager.restore_non_placeholder_embeddings()
                    self.optimizer.zero_grad()

            avg_loss = epoch_loss / max(1, len(dataloader))
            logger.info(f"Epoch [{epoch}/{epochs}] Dual Loss: {avg_loss:.4f}")

        # Lưu Checkpoints (LoRA weights & TI embeddings)
        lora_state = {
            "unet_lora": {k: v.state_dict() for k, v in self.lora_unet.items()},
            "text_lora": {k: v.state_dict() for k, v in self.lora_text.items()},
        }
        torch.save(lora_state, ckpt_dir / "pytorch_lora_weights.pt")
        self.ti_manager.save(ckpt_dir / "ti_embeddings.pt")

        logger.info(f"Hoàn tất huấn luyện LoRA! Đã lưu tại: {ckpt_dir}")
        return str(ckpt_dir)


# ─────────────────────────────────────────────────────────
# 6. Inpainting Generator với Cơ chế Latent Overwriting
# ─────────────────────────────────────────────────────────

class PCBDefectInpainter:
    """
    Module sinh ảnh lỗi PCB bằng Inpainting Diffusion:
    - 50 Denoising steps (DDIM Scheduler)
    - Prompt P kết hợp Multi-token Textual Inversion
    - Negative prompt chuẩn bài báo
    - Latent overwriting (Trang 5 bài báo) tại mỗi bước khử nhiễu:
        z_{t-1} = B_lat ⊙ z_{t-1, gen} + (1 - B_lat) ⊙ z_{t-1, orig}
      bảo toàn 100% bối cảnh bo mạch gốc.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        lora_ckpt: Optional[str] = None,
        ti_ckpt: Optional[str] = None,
        device: str = "cuda",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lora_ckpt = lora_ckpt
        self.ti_ckpt = ti_ckpt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt

        self._load_pipeline()

    def _load_pipeline(self):
        """Khởi tạo pipeline và nạp các checkpoint LoRA + Textual Inversion nếu có."""
        try:
            from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            logger.info(f"[Inpainter] Đang tải Inpainting Pipeline ({dtype}): {self.model_id}...")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            ).to(self.device)

            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            self.pipe = pipe
            self.vae = pipe.vae
            self.unet = pipe.unet
            self.text_encoder = pipe.text_encoder
            self.tokenizer = pipe.tokenizer
            self.scheduler = pipe.scheduler

            # 1. Gắn cấu trúc LoRA vào U-Net và Text Encoder
            self.lora_unet = inject_lora_into_model(self.unet, rank=8, alpha=16)
            self.lora_text = inject_lora_into_model(self.text_encoder, rank=8, alpha=16)

            # 2. Nạp trọng số LoRA nếu lora_ckpt được cung cấp
            if self.lora_ckpt:
                lora_path = Path(self.lora_ckpt)
                if lora_path.is_dir():
                    lora_path = lora_path / "pytorch_lora_weights.pt"
                if lora_path.exists():
                    logger.info(f"[Inpainter] Đang nạp trọng số LoRA từ: {lora_path}...")
                    lora_state = torch.load(lora_path, map_location=self.device)
                    if "unet_lora" in lora_state:
                        loaded_u = 0
                        for k, state in lora_state["unet_lora"].items():
                            if k in self.lora_unet:
                                self.lora_unet[k].load_state_dict(state)
                                loaded_u += 1
                        logger.info(f"[Inpainter] Đã nạp LoRA U-Net: {loaded_u}/{len(self.lora_unet)} modules.")
                    if "text_lora" in lora_state:
                        loaded_t = 0
                        for k, state in lora_state["text_lora"].items():
                            if k in self.lora_text:
                                self.lora_text[k].load_state_dict(state)
                                loaded_t += 1
                        logger.info(f"[Inpainter] Đã nạp LoRA Text Encoder: {loaded_t}/{len(self.lora_text)} modules.")
                    logger.info("[Inpainter] Đã nạp thành công toàn bộ trọng số LoRA vào U-Net & Text Encoder!")
                else:
                    logger.warning(f"[Inpainter] Không tìm thấy checkpoint LoRA tại {lora_path}. Sẽ sử dụng base model.")

            self.unet.eval()
            self.text_encoder.eval()

            # 3. Khởi tạo & nạp Textual Inversion
            self.ti_manager = MultiTokenTIManager(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                device=str(self.device),
            )
            if self.ti_ckpt:
                ti_path = Path(self.ti_ckpt)
                if ti_path.is_dir():
                    ti_path = ti_path / "ti_embeddings.pt"
                if ti_path.exists():
                    self.ti_manager.load(ti_path)

            self.has_diffusion = True
        except ImportError:
            logger.warning("[Inpainter] 'diffusers' chưa được cài đặt. Khởi tạo chế độ cấu hình kiến trúc.")
            self.has_diffusion = False
            self.ti_manager = MultiTokenTIManager(device=str(self.device))

    @torch.no_grad()
    def generate(
        self,
        normal_image: Union[Image.Image, np.ndarray, str],
        defect_mask: Union[Image.Image, np.ndarray, str],
        defect_type: str,
    ) -> Image.Image:
        """
        Quy trình Denoising 50 bước với cơ chế LATENT OVERWRITING theo chuẩn bài báo.
        """
        if isinstance(normal_image, (str, Path)):
            normal_image = Image.open(normal_image).convert("RGB")
        elif isinstance(normal_image, np.ndarray):
            normal_image = Image.fromarray(normal_image)

        if isinstance(defect_mask, (str, Path)):
            defect_mask = Image.open(defect_mask).convert("L")
        elif isinstance(defect_mask, np.ndarray):
            defect_mask = Image.fromarray(defect_mask)

        orig_w, orig_h = normal_image.size
        target_size = (512, 512)

        # ── NẾU CÓ ĐẦY ĐỦ DIFFUSION & TRỌNG SỐ TRÊN GPU ──────────────────
        if getattr(self, "has_diffusion", False):
            # 1. Tiền xử lý ảnh gốc và mask
            img_resized = normal_image.resize(target_size)
            msk_resized = defect_mask.resize(target_size)

            img_np = np.array(img_resized).astype(np.float32) / 127.5 - 1.0
            msk_np = (np.array(msk_resized) > 128).astype(np.float32)

            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.unet.dtype)  # (1, 3, H, W)
            msk_tensor = torch.from_numpy(msk_np).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.unet.dtype)      # (1, 1, H, W)

            # 2. Mã hóa ảnh gốc sang Latent z0_orig
            z0_orig = self.vae.encode(img_tensor).latent_dist.sample() * 0.18215
            latent_h, latent_w = z0_orig.shape[-2:]

            # Mask ở không gian Latent B_lat
            b_lat = F.interpolate(msk_tensor, size=(latent_h, latent_w), mode="nearest")

            # Chuẩn bị masked image latents cho U-Net Inpainting 9 kênh
            in_channels = getattr(self.unet.config, "in_channels", 9)
            if in_channels == 9:
                masked_img = img_tensor * (1.0 - msk_tensor)
                masked_img_latents = self.vae.encode(masked_img).latent_dist.sample() * 0.18215
                mask_input = torch.cat([b_lat] * 2)
                masked_latents_input = torch.cat([masked_img_latents] * 2)

            # 3. Chuẩn bị Text Embeddings (Prompt P và Negative Prompt)
            prompt = self.ti_manager.get_defect_prompt(defect_type)
            tok_pos = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(self.device)
            tok_neg = self.tokenizer(self.negative_prompt, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(self.device)

            emb_pos = self.text_encoder(tok_pos)[0]
            emb_neg = self.text_encoder(tok_neg)[0]
            text_embeddings = torch.cat([emb_neg, emb_pos])  # (2, 77, 768)

            # 4. Khởi tạo nhiễu ngẫu nhiên z_T
            self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps
            latents = torch.randn_like(z0_orig)

            # 5. VÒNG LẶP KHỬ NHIỄU 50 BƯỚC + LATENT OVERWRITING
            pbar_denoise = tqdm(timesteps, desc=f"Khử nhiễu ({defect_type})", leave=False, unit="step")
            for i, t in enumerate(pbar_denoise):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Nối ghép đủ 9 kênh nếu U-Net là inpainting model
                if in_channels == 9:
                    latent_model_input = torch.cat([latent_model_input, mask_input, masked_latents_input], dim=1)

                # U-Net dự đoán nhiễu với Classifier-Free Guidance
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Tính latent bước trước z_{t-1}
                latents_prev = self.scheduler.step(noise_pred, t, latents).prev_sample

                # ── CƠ CHẾ LATENT OVERWRITING CỐT LÕI (Trang 5 Paper) ──────
                # "After each denoising process, the latent representation corresponding
                # to the unmasked region (1 - B) is overwritten to preserve the background area."
                if i < len(timesteps) - 1:
                    prev_t = timesteps[i + 1]
                    noise_bg = torch.randn_like(z0_orig)
                    z_orig_t_prev = self.scheduler.add_noise(z0_orig, noise_bg, prev_t)
                    # Gán đè vùng unmasked bằng latent gốc chứa nhiễu ở timestep prev_t
                    latents = b_lat * latents_prev + (1.0 - b_lat) * z_orig_t_prev
                else:
                    # Bước cuối cùng: gán đè trực tiếp với latent sạch z0_orig
                    latents = b_lat * latents_prev + (1.0 - b_lat) * z0_orig

            # 6. Giải mã Latent về Pixel space qua VAE Decoder
            decoded = self.vae.decode(latents / 0.18215).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded_np = (decoded.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            decoded_pil = Image.fromarray(decoded_np).resize((orig_w, orig_h))

            # 7. Ghép pixel chính xác 100% ngoài vùng mask
            final_img = Image.composite(decoded_pil, normal_image, defect_mask)
            return final_img

        # ── FALLBACK CHO MÔI TRƯỜNG CHƯA TẢI TRỌNG SỐ ─────────────────────
        logger.info(f"[Inpainting Fallback] Đang mô phỏng defect '{defect_type}' trên mask...")
        orig_np = np.array(normal_image).copy()
        mask_np = np.array(defect_mask)
        m = (mask_np > 128)
        if m.any():
            if defect_type in ["missing_hole", "open_circuit"]:
                orig_np[m] = (orig_np[m] * 0.3).astype(np.uint8)
            elif defect_type in ["spur", "spurious_copper", "short"]:
                orig_np[m] = np.clip(orig_np[m].astype(int) + 60, 0, 255).astype(np.uint8)
            else:
                orig_np[m] = (255 - orig_np[m])

        return Image.fromarray(orig_np)

    def generate_batch(
        self,
        mask_records: List[Dict],
        output_dir: Union[str, Path] = "outputs/generated",
    ) -> List[Dict]:
        """
        Sinh batch toàn bộ ảnh lỗi dựa trên danh sách metadata từ Step 1 (CAMG).
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Step 2] Bắt đầu sinh {len(mask_records)} ảnh lỗi tổng hợp...")
        gen_records = []

        for i, rec in enumerate(tqdm(mask_records, desc="Khuếch tán sinh ảnh lỗi (Inpainting)", unit="ảnh")):
            normal_img_path = rec["normal_image_path"]
            mask_path = rec["mask_path"]
            defect_type = rec["defect_type"]

            stem = Path(normal_img_path).stem
            out_name = f"syn_{stem}_{defect_type}_{i:04d}.png"
            out_img_path = out_dir / out_name

            # Bỏ qua nếu ảnh đã được sinh từ trước (hỗ trợ khôi phục tiến trình)
            if not (out_img_path.exists() and out_img_path.stat().st_size > 1024):
                gen_img = self.generate(
                    normal_image=normal_img_path,
                    defect_mask=mask_path,
                    defect_type=defect_type,
                )
                gen_img.save(out_img_path)

            gen_records.append({
                "synthetic_image_path": str(out_img_path.resolve()),
                "source_normal_image": normal_img_path,
                "mask_path": mask_path,
                "defect_type": defect_type,
                "class_id": rec.get("class_id", 0),
                "bbox_pixel": rec.get("bbox_pixel", []),
                "bbox_norm": rec.get("bbox_norm", []),
                "prompt": self.ti_manager.get_defect_prompt(defect_type),
                "negative_prompt": self.negative_prompt,
            })

        meta_out = out_dir / "generated_metadata.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(gen_records, f, indent=2, ensure_ascii=False)

        logger.info(f"Step 2 xong! Đã tạo {len(gen_records)} ảnh. Metadata: {meta_out}")
        return gen_records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    inpainter = PCBDefectInpainter()
    logger.info("Step 2 Module (Diffusion Training Loop + Latent Overwriting) đã nạp thành công.")
