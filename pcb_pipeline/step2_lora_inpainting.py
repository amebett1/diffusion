"""
Step 2: Defect Image Synthesis using Inpainting Diffusion + LoRA + Multi-Token Textual Inversion
================================================================================================
Triển khai theo bài báo:
"A Synthetic Defect Image Generation Method for Enhanced Industrial
Defect Detection Based on Inpainting Diffusion and Context-aware Mask Generation"
(Journal of Intelligent Manufacturing, 2026 - Mục Methods, Trang 5)

Các điểm kỹ thuật cốt lõi
------------------------
1. Multi-Token Textual Inversion cho 6 loại lỗi PCB.
2. LoRA (r=8, alpha=16) trên Attention/Output projections của U-Net và Text Encoder.
3. Dual Loss: L_def + 0.1 * L_rec, với B' lấy trong vùng không lỗi và
   kích thước theo phân vị Q1-Q3 của DeepPCB.
4. Denoising 50 bước với DDIM + Latent Overwriting nhất quán.

Các sửa lỗi quan trọng so với bản cũ
------------------------------------
- Latent Overwriting dùng MỘT noise nền cố định (noise_bg) cho toàn bộ
  quỹ đạo khử nhiễu. Bản cũ gọi torch.randn_like() mới ở từng bước, làm
  nền bị đổi nhiễu liên tục và vỡ vùng tiếp giáp B / (1-B).
- Composite cuối dùng alpha mask được feather 2-3 px, chỉ trong vùng mask
  dilation; ngoài vùng đó pixel gốc được giữ nguyên 100%.
- Ảnh được xử lý ở độ phân giải gần native (bội số của 8, tối đa cấu hình)
  thay vì luôn resize 600 -> 512 -> 600. Với ảnh 600x600, VAE/UNet chạy
  trực tiếp 600x600 để không làm mờ micro-trace.
- Fail-fast nếu thiếu diffusers hoặc checkpoint LoRA/TI được yêu cầu nhưng
  không tồn tại; không âm thầm chạy base model/fallback rồi coi như đã tái lập.
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
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS_PATH = PROJECT_ROOT / "data" / "dataset_stats.json"

# Danh sách 6 lớp lỗi PCB chuẩn DeepPCB & Paper
PCB_DEFECT_CLASSES = [
    "mouse_bite",
    "spur",
    "missing_hole",
    "short",
    "open_circuit",
    "spurious_copper",
]

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

        self.placeholder_tokens = {c: f"<{c}>" for c in self.defect_classes}
        self.token_ids: Dict[str, int] = {}
        self.placeholder_token_ids: List[int] = []
        self.ti_parameters = nn.ParameterDict()
        self._embedding_forward_patched = False

        if self.tokenizer is not None and self.text_encoder is not None:
            self._initialize_tokens()

    @property
    def ti_params(self) -> Dict[str, torch.Tensor]:
        return dict(self.ti_parameters)

    def _initialize_tokens(self):
        """Thêm placeholder tokens và khởi tạo embedding từ từ ngữ gần nghĩa."""
        logger.info(f"[TI] Đang khởi tạo {len(self.defect_classes)} tokens Textual Inversion...")

        for cls_name, placeholder in self.placeholder_tokens.items():
            self.tokenizer.add_tokens(placeholder)
            self.token_ids[cls_name] = self.tokenizer.convert_tokens_to_ids(placeholder)
            self.placeholder_token_ids.append(self.token_ids[cls_name])

        # Resize một lần sau khi đã thêm tất cả token.
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        embedding = self.text_encoder.get_input_embeddings()
        weight = embedding.weight

        with torch.no_grad():
            for cls_name in self.placeholder_tokens:
                init_word = INIT_TOKENS.get(cls_name, "defect")
                init_ids = self.tokenizer.encode(init_word, add_special_tokens=False)
                init_id = init_ids[0] if init_ids else self.tokenizer.unk_token_id
                token_id = self.token_ids[cls_name]
                weight[token_id] = weight[init_id].clone()

        # Chỉ 6 vector token là tham số huấn luyện, ở fp32.
        # Không dùng cả bảng embedding 49k dòng: tránh gradient fp16 và tốn VRAM.
        weight.requires_grad_(False)
        self.ti_parameters = nn.ParameterDict({
            cls_name: nn.Parameter(weight[token_id].detach().float().clone())
            for cls_name, token_id in self.token_ids.items()
        })

        # Patch token_embedding.forward: lấy bảng gốc (frozen) rồi thay 6 hàng
        # bằng tham số TI. Gradient chỉ chảy vào 6 vector fp32.
        if not self._embedding_forward_patched:
            manager = self

            def ti_embedding_forward(input_ids):
                out = F.embedding(input_ids, embedding.weight)
                for cls_name, token_id in manager.token_ids.items():
                    vec = manager.ti_parameters[cls_name].to(
                        device=out.device, dtype=out.dtype
                    )
                    out = torch.where(
                        (input_ids == token_id).unsqueeze(-1),
                        vec.view(1, 1, -1),
                        out,
                    )
                return out

            embedding.forward = ti_embedding_forward
            self._embedding_forward_patched = True

        logger.info(
            f"[TI] Đã đăng ký các token: {list(self.placeholder_tokens.values())}"
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self.ti_parameters.values())

    def zero_non_placeholder_grads(self):
        # Bảng embedding gốc đã frozen; không cần zero gradient.
        return

    def restore_non_placeholder_embeddings(self):
        # Bảng embedding gốc không bao giờ bị optimizer cập nhật.
        return

    def sync_embeddings(self):
        return

    def get_defect_prompt(self, defect_type: str) -> str:
        token = self.placeholder_tokens.get(defect_type, f"<{defect_type}>")
        return f"a photo of a {self.product_name} with {token}"

    def get_normal_prompt(self) -> str:
        return f"a photo of a {self.product_name}"

    def save(self, output_path: Union[str, Path]):
        state = {
            cls_name: param.detach().cpu().clone()
            for cls_name, param in self.ti_parameters.items()
        }
        torch.save(state, output_path)
        logger.info(f"[TI] Đã lưu TI embeddings -> {output_path}")

    def load(self, checkpoint_path: Union[str, Path]):
        state = torch.load(checkpoint_path, map_location=self.device)
        loaded = 0
        with torch.no_grad():
            for cls_name, emb_data in state.items():
                if cls_name in self.ti_parameters:
                    param = self.ti_parameters[cls_name]
                    param.data.copy_(
                        emb_data.to(device=param.device, dtype=param.dtype)
                    )
                    loaded += 1
        logger.info(f"[TI] Đã nạp {loaded}/{len(self.token_ids)} TI embeddings <- {checkpoint_path}")


# ─────────────────────────────────────────────────────────
# 2. Native LoRA Layers
# ─────────────────────────────────────────────────────────

class LoRALinearLayer(nn.Module):
    """ΔW = (α / r) * B * A, với r=8, α=16 theo paper."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Giữ tham số LoRA ở fp32 để GradScaler/AdamW hoạt động ổn định,
        # chỉ cast tạm sang dtype của activation khi chạy forward.
        a = self.lora_A
        b = self.lora_B
        if a.device != x.device:
            a = a.to(x.device)
            b = b.to(x.device)
        if a.dtype != x.dtype:
            a = a.to(x.dtype)
            b = b.to(x.dtype)
        return self.scale * F.linear(F.linear(x, a), b)


def inject_lora_into_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, LoRALinearLayer]:
    """Gắn LoRA vào Attention/Output projection layers."""
    if target_modules is None:
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",
            "q_proj", "k_proj", "v_proj", "out_proj",
        ]

    lora_layers: Dict[str, LoRALinearLayer] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(tgt) or tgt in name for tgt in target_modules
        ):
            lora_layer = LoRALinearLayer(
                module.in_features,
                module.out_features,
                rank=rank,
                alpha=alpha,
            ).to(device=module.weight.device, dtype=torch.float32)
            lora_layers[name] = lora_layer
            setattr(module, "lora_layer", lora_layer)

            orig_forward = module.forward

            def make_new_forward(m, orig_f):
                def forward_fn(x):
                    return orig_f(x) + m.lora_layer(x)
                return forward_fn

            module.forward = make_new_forward(module, orig_forward)

    logger.info(f"[LoRA] Đã gắn LoRA vào {len(lora_layers)} layers (rank={rank}, alpha={alpha}).")
    return lora_layers


# ─────────────────────────────────────────────────────────
# 3. Dual Loss & Negative Box Sampling
# ─────────────────────────────────────────────────────────

class DualInpaintLoss(nn.Module):
    """
    L_def = E[B * ||ε - ε_θ(x_t, B, t, τ(P))||²]
    L_rec = E[B' * ||ε - ε_θ(x_t, B', t, τ(P'))||²]
    L_tot = L_def + λ * L_rec, λ = 0.1.

    B' là box ngẫu nhiên trong vùng không lỗi, kích thước lấy theo Q1-Q3
    của DeepPCB (được scale theo kích thước ảnh hiện tại, mặc định 600 px).
    """

    def __init__(
        self,
        lambda_rec: float = 0.1,
        stats_path: Union[str, Path] = DEFAULT_STATS_PATH,
        default_q1: int = 21,
        default_q3: int = 36,
        reference_size: int = 600,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.reference_size = int(reference_size)
        self.q1_w = float(default_q1)
        self.q3_w = float(default_q3)
        self.q1_h = float(default_q1)
        self.q3_h = float(default_q3)

        stats_file = Path(stats_path)
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                px = data.get("pixel_size", {})
                self.q1_w = float(px.get("width", {}).get("q1", default_q1))
                self.q3_w = float(px.get("width", {}).get("q3", default_q3))
                self.q1_h = float(px.get("height", {}).get("q1", default_q1))
                self.q3_h = float(px.get("height", {}).get("q3", default_q3))
                logger.info(
                    f"[Loss] Kích thước B' (tại {self.reference_size}px): "
                    f"W [{self.q1_w:.1f}..{self.q3_w:.1f}], "
                    f"H [{self.q1_h:.1f}..{self.q3_h:.1f}]"
                )
            except Exception as e:
                logger.warning(f"Không thể đọc stats từ {stats_path}: {e}")

    def sample_negative_mask(self, defect_mask: torch.Tensor) -> torch.Tensor:
        """
        Lấy mẫu B' trong vùng sạch, không overlap defect mask.
        Không bao giờ trả về một box overlap chỉ vì hết max_tries.
        """
        b_batch, _, H, W = defect_mask.shape
        b_prime = torch.zeros_like(defect_mask)
        scale_w = W / float(self.reference_size)
        scale_h = H / float(self.reference_size)

        for b in range(b_batch):
            cur_mask = defect_mask[b, 0]
            chosen = None

            for attempt in range(2):
                # Lần 1: Q1-Q3; lần 2: thu nhỏ một nửa nếu vùng sạch quá ít.
                shrink = 0.5 if attempt == 1 else 1.0
                bw = int(round(random.uniform(self.q1_w, self.q3_w) * scale_w * shrink))
                bh = int(round(random.uniform(self.q1_h, self.q3_h) * scale_h * shrink))
                bw = max(4, min(bw, max(4, W - 4)))
                bh = max(4, min(bh, max(4, H - 4)))

                for _ in range(30):
                    x1 = random.randint(2, max(2, W - bw - 2))
                    y1 = random.randint(2, max(2, H - bh - 2))
                    if cur_mask[y1:y1 + bh, x1:x1 + bw].sum() == 0:
                        chosen = (x1, y1, bw, bh)
                        break
                if chosen is not None:
                    break

            if chosen is None:
                logger.debug("[Loss] Không tìm được B' sạch; bỏ qua L_rec cho mẫu này.")
                continue

            x1, y1, bw, bh = chosen
            b_prime[b, 0, y1:y1 + bh, x1:x1 + bw] = 1.0

        return b_prime

    def forward(
        self,
        noise_pred_def: torch.Tensor,
        noise_target: torch.Tensor,
        defect_mask: torch.Tensor,
        noise_pred_rec: Optional[torch.Tensor] = None,
        rec_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m_def = F.interpolate(defect_mask, size=noise_pred_def.shape[-2:], mode="nearest")
        n_def_pix = m_def.sum().clamp(min=1.0)
        l_def = (m_def * (noise_target - noise_pred_def) ** 2).sum() / n_def_pix

        if noise_pred_rec is not None and rec_mask is not None:
            m_rec = F.interpolate(rec_mask, size=noise_pred_rec.shape[-2:], mode="nearest")
            n_rec_pix = m_rec.sum().clamp(min=1.0)
            l_rec = (m_rec * (noise_target - noise_pred_rec) ** 2).sum() / n_rec_pix
        else:
            l_rec = torch.tensor(0.0, device=noise_pred_def.device)

        l_tot = l_def + self.lambda_rec * l_rec
        return l_tot, l_def, l_rec


# ─────────────────────────────────────────────────────────
# 4. LoRA Training Dataset
# ─────────────────────────────────────────────────────────

class PCBLoRADataset(Dataset):
    """Nạp ảnh PCB thực tế và tạo mask bbox từ train_metadata.json."""

    def __init__(self, metadata_path: Union[str, Path], image_size: int = 512):
        self.metadata_path = Path(metadata_path)
        self.image_size = int(image_size)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        # Chỉ giữ record có ít nhất một bbox lỗi; ảnh normal không đóng góp L_def.
        self.records = [r for r in records if r.get("boxes")]
        if not self.records:
            raise ValueError(
                f"Không có record nào chứa bbox lỗi trong {self.metadata_path}. "
                "Không thể huấn luyện inpainting."
            )
        logger.info(f"[LoRA Dataset] Dùng {len(self.records)}/{len(records)} record có bbox lỗi.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        img_path = rec["image_path"]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")

        H, W, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = np.zeros((H, W), dtype=np.float32)
        boxes = rec.get("boxes", [])
        defect_type = boxes[0].get("class_name", "missing_hole")

        for box in boxes:
            x1 = max(0, min(int(box["x1"]), W - 1))
            y1 = max(0, min(int(box["y1"]), H - 1))
            x2 = max(x1 + 1, min(int(box["x2"]), W))
            y2 = max(y1 + 1, min(int(box["y2"]), H))
            mask[y1:y2, x1:x2] = 1.0

        img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

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
# 5. LoRA + Textual Inversion Trainer
# ─────────────────────────────────────────────────────────

class LoRAInpaintTrainer:
    """
    Huấn luyện LoRA (U-Net & Text Encoder) + Multi-Token Textual Inversion.
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

        self.loss_fn = DualInpaintLoss(lambda_rec=0.1, stats_path=DEFAULT_STATS_PATH)
        self.use_amp = bool(use_fp16 and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.is_initialized = False

    def _init_models(self):
        """Khởi tạo diffusion pipeline; fail-fast nếu thiếu diffusers."""
        try:
            from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
        except ImportError as e:
            raise RuntimeError(
                "Chưa cài đặt 'diffusers'. Không thể huấn luyện LoRA inpainting.\n"
                "Cài đặt bằng: pip install 'diffusers>=0.25.0' 'transformers>=4.37.0' accelerate\n"
                f"Chi tiết: {e}"
            ) from e

        dtype = torch.float16 if self.use_amp else torch.float32
        logger.info(f"[LoRA Trainer] Đang tải pipeline từ {self.model_id} ({dtype})...")
        try:
            load_kwargs = {
                "torch_dtype": dtype,
                "use_safetensors": True,
            }
            if dtype == torch.float16:
                # Repo SD-inpainting chỉ có fp16 safetensors cho UNet/VAE/TextEncoder.
                # Không có variant này, diffusers sẽ tải nhầm fp32 (rất chậm và tốn VRAM).
                load_kwargs["variant"] = "fp16"
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                **load_kwargs,
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Không tải được diffusion model '{self.model_id}'.\n"
                "Kiểm tra kết nối mạng, dung lượng đĩa và quyền truy cập Hugging Face.\n"
                f"Chi tiết: {e}"
            ) from e

        self.vae = pipe.vae
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.vae_scaling = float(getattr(self.vae.config, "scaling_factor", 0.18215))

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Tối ưu cho GPU 6 GB: attention slicing + gradient checkpointing.
        # Không đổi kiến trúc hay loss; chỉ giảm activation memory.
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            self.vae.enable_slicing()
        except Exception:
            pass
        try:
            self.unet.enable_gradient_checkpointing()
        except Exception as exc:
            logger.warning(f"[LoRA Trainer] Không bật được gradient checkpointing: {exc}")

        self.ti_manager = MultiTokenTIManager(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=str(self.device),
        )

        self.lora_unet = inject_lora_into_model(self.unet, rank=self.rank, alpha=self.alpha)
        self.lora_text = inject_lora_into_model(self.text_encoder, rank=self.rank, alpha=self.alpha)

        trainable_params = [
            {
                "params": [p for layer in self.lora_unet.values() for p in layer.parameters()],
                "lr": self.lr_unet,
                "weight_decay": 1e-2,
            },
            {
                "params": [p for layer in self.lora_text.values() for p in layer.parameters()],
                "lr": self.lr_text_encoder,
                "weight_decay": 1e-2,
            },
            {
                "params": self.ti_manager.get_trainable_params(),
                "lr": self.lr_ti,
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(trainable_params)
        self.is_initialized = True

    def _save_checkpoint(self, epoch: int, target_epochs: int, complete: bool) -> Path:
        """Lưu checkpoint sau mỗi epoch để Kaggle commit có thể resume."""
        ckpt_dir = self.output_dir / f"epoch_{epoch:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        lora_state = {
            "config": {
                "model_id": self.model_id,
                "rank": int(self.rank),
                "alpha": float(self.alpha),
            },
            "unet_lora": {k: v.state_dict() for k, v in self.lora_unet.items()},
            "text_lora": {k: v.state_dict() for k, v in self.lora_text.items()},
        }
        torch.save(lora_state, ckpt_dir / "pytorch_lora_weights.pt")
        self.ti_manager.save(ckpt_dir / "ti_embeddings.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        if self.use_amp:
            torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")

        state = {
            "format": 1,
            "model_id": self.model_id,
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "completed_epochs": int(epoch),
            "target_epochs": int(target_epochs),
            "complete": bool(complete),
        }
        (self.output_dir / "training_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        return ckpt_dir

    def train(
        self,
        data_json: str | Path = "data/train_metadata.json",
        epochs: int = 10,
        batch_size: int = 32,
        start_epoch: int = 0,
    ) -> str:
        """Vòng lặp huấn luyện LoRA + TI với VAE pre-encoding cache.

        start_epoch > 0 dùng để resume từ checkpoint epoch đã lưu. Điều này cần
        cho Kaggle Save & Run All: nếu session bị timeout, lần chạy sau có thể
        tiếp tục thay vì train lại từ đầu.
        """
        if epochs <= 0:
            raise ValueError("epochs phải > 0.")
        start_epoch = max(0, min(int(start_epoch), int(epochs)))
        grad_accum_steps = max(1, int(batch_size))

        logger.info("=" * 60)
        logger.info("[LoRA Inpaint Trainer] Khởi chạy huấn luyện Dual-Loss LoRA:")
        logger.info(f"  Model ID:     {self.model_id}")
        logger.info(f"  LoRA Config:  rank={self.rank}, alpha={self.alpha}")
        logger.info(
            f"  Epochs:       {epochs} (start={start_epoch}), "
            f"Grad accumulation: {grad_accum_steps}"
        )
        logger.info(f"  LR U-Net:     {self.lr_unet}, LR Text: {self.lr_text_encoder}, LR TI: {self.lr_ti}")
        logger.info(f"  Device:       {self.device} (AMP: {self.use_amp})")
        logger.info("=" * 60)

        data_path = Path(data_json)
        if not data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file metadata: {data_path}")

        dataset = PCBLoRADataset(metadata_path=data_path, image_size=512)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        self._init_models()

        if start_epoch > 0:
            resume_dir = self.output_dir / f"epoch_{start_epoch:03d}"
            if not (resume_dir / "pytorch_lora_weights.pt").exists() or not (
                resume_dir / "ti_embeddings.pt"
            ).exists():
                raise FileNotFoundError(f"Thiếu checkpoint để resume: {resume_dir}")
            lora_state = torch.load(
                resume_dir / "pytorch_lora_weights.pt",
                map_location=self.device,
            )
            for key, layer in self.lora_unet.items():
                if key in lora_state.get("unet_lora", {}):
                    layer.load_state_dict(lora_state["unet_lora"][key])
            for key, layer in self.lora_text.items():
                if key in lora_state.get("text_lora", {}):
                    layer.load_state_dict(lora_state["text_lora"][key])
            self.ti_manager.load(resume_dir / "ti_embeddings.pt")
            optimizer_path = resume_dir / "optimizer.pt"
            if optimizer_path.exists():
                self.optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=self.device)
                )
            scaler_path = resume_dir / "scaler.pt"
            if self.use_amp and scaler_path.exists():
                self.scaler.load_state_dict(
                    torch.load(scaler_path, map_location=self.device)
                )
            logger.info(f"[LoRA Trainer] Resume từ {resume_dir}")

        if start_epoch >= epochs:
            logger.info(f"[LoRA Trainer] Đã đủ {epochs} epoch; không train thêm.")
            return str(self.output_dir / f"epoch_{epochs:03d}")

        # ── Pre-encode VAE latents ────────────────────────────────────────
        logger.info(f"[LoRA Trainer] Pre-encoding {len(dataset)} VAE latents...")
        cached_latents: Dict[int, torch.Tensor] = {}
        cached_mask_lat_def: Dict[int, torch.Tensor] = {}
        cached_masked_lat_def: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="Pre-encoding VAE Latents", leave=False):
                sample = dataset[idx]
                img_t = sample["image"].unsqueeze(0).to(self.device)
                msk_t = sample["mask"].unsqueeze(0).to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    lat = self.vae.encode(img_t).latent_dist.sample() * self.vae_scaling
                    lh, lw = lat.shape[-2:]
                    m_lat = F.interpolate(msk_t, size=(lh, lw), mode="nearest")
                    m_img = img_t * (1.0 - msk_t)
                    m_lat_img = self.vae.encode(m_img).latent_dist.sample() * self.vae_scaling

                cached_latents[idx] = lat.cpu().float()
                cached_mask_lat_def[idx] = m_lat.cpu().float()
                cached_masked_lat_def[idx] = m_lat_img.cpu().float()

        logger.info(f"[LoRA Trainer] Đã cache {len(cached_latents)} VAE latents.")

        self.unet.train()
        self.text_encoder.train()

        for epoch in range(start_epoch + 1, epochs + 1):
            epoch_loss = 0.0
            accum_steps = 0
            self.optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(
                dataloader,
                desc=f"Epoch [{epoch}/{epochs}] [LoRA Dual-Loss]",
                leave=False,
                unit="ảnh",
            )
            for step, batch in enumerate(pbar):
                sample_idx = int(batch["index"][0].item())
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                defect_type = batch["defect_type"][0]

                # 1. Latent z0 từ cache
                latents = cached_latents[sample_idx].to(self.device, dtype=self.unet.dtype)
                latent_h, latent_w = latents.shape[-2:]

                # 2. Timestep + noise
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=self.device,
                    dtype=torch.long,
                )
                noise = torch.randn_like(latents)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Prompt P và P'
                prompt_def = self.ti_manager.get_defect_prompt(defect_type)
                prompt_rec = self.ti_manager.get_normal_prompt()

                tokens_def = self.tokenizer(
                    prompt_def, padding="max_length", max_length=77, return_tensors="pt"
                ).input_ids.to(self.device)
                tokens_rec = self.tokenizer(
                    prompt_rec, padding="max_length", max_length=77, return_tensors="pt"
                ).input_ids.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    text_emb_def = self.text_encoder(tokens_def)[0]
                    text_emb_rec = self.text_encoder(tokens_rec)[0]

                # 4. B' cho reconstruction loss
                rec_masks = self.loss_fn.sample_negative_mask(masks).to(self.device)

                # 5. 9-channel inpainting input
                in_channels = int(getattr(self.unet.config, "in_channels", 9))
                if in_channels == 9:
                    mask_lat_def = cached_mask_lat_def[sample_idx].to(self.device, dtype=self.unet.dtype)
                    masked_lat_def = cached_masked_lat_def[sample_idx].to(self.device, dtype=self.unet.dtype)
                    unet_input_def = torch.cat([noisy_latents, mask_lat_def, masked_lat_def], dim=1)

                    mask_lat_rec = F.interpolate(rec_masks, size=(latent_h, latent_w), mode="nearest")
                    masked_img_rec = images * (1.0 - rec_masks)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            masked_lat_rec = self.vae.encode(masked_img_rec).latent_dist.sample() * self.vae_scaling
                    unet_input_rec = torch.cat([noisy_latents, mask_lat_rec, masked_lat_rec], dim=1)
                else:
                    unet_input_def = noisy_latents
                    unet_input_rec = noisy_latents

                # 6. U-Net dự đoán nhiễu cho 2 nhánh
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    noise_pred_def = self.unet(
                        unet_input_def, timesteps, encoder_hidden_states=text_emb_def
                    ).sample
                    noise_pred_rec = self.unet(
                        unet_input_rec, timesteps, encoder_hidden_states=text_emb_rec
                    ).sample

                    loss, l_def, l_rec = self.loss_fn(
                        noise_pred_def=noise_pred_def,
                        noise_target=noise,
                        defect_mask=masks,
                        noise_pred_rec=noise_pred_rec,
                        rec_mask=rec_masks,
                    )
                    scaled_loss = loss / grad_accum_steps

                # 7. Backward + gradient accumulation
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accum_steps += 1
                epoch_loss += float(loss.item())
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "l_def": f"{l_def.item():.4f}",
                    "l_rec": f"{l_rec.item():.4f}",
                })

                if accum_steps % grad_accum_steps == 0 or step == len(dataloader) - 1:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    self.ti_manager.zero_non_placeholder_grads()

                    trainable = [
                        p for group in self.optimizer.param_groups
                        for p in group["params"] if p.requires_grad
                    ]
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.ti_manager.restore_non_placeholder_embeddings()
                    self.optimizer.zero_grad(set_to_none=True)

            avg_loss = epoch_loss / max(1, len(dataloader))
            logger.info(f"Epoch [{epoch}/{epochs}] Dual Loss: {avg_loss:.4f}")

            # Lưu sau MỖI epoch để lần chạy Kaggle sau có thể resume.
            ckpt_dir = self._save_checkpoint(
                epoch, epochs, complete=(epoch == epochs)
            )
            logger.info(f"[LoRA Trainer] Đã lưu checkpoint: {ckpt_dir}")

        final_dir = self.output_dir / f"epoch_{epochs:03d}"
        logger.info(f"Hoàn tất huấn luyện LoRA! Đã lưu tại: {final_dir}")
        return str(final_dir)


# ─────────────────────────────────────────────────────────
# 6. Inpainting Generator: DDIM + consistent latent overwriting
# ─────────────────────────────────────────────────────────

class PCBDefectInpainter:
    """
    Sinh ảnh lỗi PCB bằng inpainting diffusion.

    Latent Overwriting (RePaint/DDIM):
        z_{t-1} = B_lat * z_{t-1,gen} + (1 - B_lat) * q(z0, t-1)
    với q(z0, t-1) dùng CÙNG một noise_bg cho mọi timestep, không phải
    torch.randn_like() mới ở từng bước.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        lora_ckpt: Optional[str] = None,
        ti_ckpt: Optional[str] = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        device: str = "cuda",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        max_side: int = 768,
        feather_sigma: float = 2.5,
        attention_slicing: bool = True,
        allow_fallback: bool = False,
    ):
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lora_ckpt = str(lora_ckpt) if lora_ckpt else None
        self.ti_ckpt = str(ti_ckpt) if ti_ckpt else None
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.negative_prompt = negative_prompt
        self.max_side = int(max_side)
        self.feather_sigma = float(feather_sigma)
        self.attention_slicing = bool(attention_slicing)
        self.allow_fallback = bool(allow_fallback)
        self.ti_loaded = False
        self.using_lora = False
        self._last_info: Dict = {}

        self._load_pipeline()

    def _resolve_checkpoint(self, ckpt: Optional[str], default_name: str) -> Optional[Path]:
        if not ckpt:
            return None
        p = Path(ckpt)
        if p.is_dir():
            p = p / default_name
        return p

    def _load_pipeline(self):
        try:
            from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
        except ImportError as e:
            if not self.allow_fallback:
                raise RuntimeError(
                    "Chưa cài đặt 'diffusers'. Không thể sinh ảnh inpainting thật.\n"
                    "Cài đặt: pip install 'diffusers>=0.25.0' 'transformers>=4.37.0' accelerate\n"
                    "Nếu chỉ muốn chạy fallback mô phỏng để test pipeline, hãy đặt "
                    "allow_fallback=True (kết quả KHÔNG phải diffusion thật).\n"
                    f"Chi tiết: {e}"
                ) from e
            logger.warning("[Inpainter] Thiếu diffusers; dùng fallback mô phỏng vì allow_fallback=True.")
            self.has_diffusion = False
            self.ti_manager = MultiTokenTIManager(device=str(self.device))
            return

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        logger.info(f"[Inpainter] Đang tải Inpainting Pipeline ({dtype}): {self.model_id}...")
        try:
            load_kwargs = {
                "torch_dtype": dtype,
                "safety_checker": None,
                "use_safetensors": True,
            }
            if dtype == torch.float16:
                load_kwargs["variant"] = "fp16"
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                **load_kwargs,
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Không tải được inpainting model '{self.model_id}'. "
                "Kiểm tra mạng/cache Hugging Face và dung lượng VRAM.\n"
                f"Chi tiết: {e}"
            ) from e

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            clip_sample=False,
            set_alpha_to_one=False,
        )
        if self.attention_slicing:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

        self.pipe = pipe
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
        self.vae_scaling = float(getattr(self.vae.config, "scaling_factor", 0.18215))

        # 1. Đọc checkpoint trước để lấy đúng rank/alpha đã dùng lúc train.
        # Nếu checkpoint cũ không có config, suy ra rank từ shape lora_A.
        lora_state = None
        lora_path = self._resolve_checkpoint(self.lora_ckpt, "pytorch_lora_weights.pt")
        if self.lora_ckpt:
            if lora_path is None or not lora_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy LoRA checkpoint: {lora_path}. "
                    "Hãy chạy 'python run_pipeline.py step2_train ...' trước."
                )
            lora_state = torch.load(lora_path, map_location=self.device)
            cfg = lora_state.get("config", {}) if isinstance(lora_state, dict) else {}
            if cfg.get("rank") is not None:
                self.lora_rank = int(cfg["rank"])
            elif isinstance(lora_state, dict) and lora_state.get("unet_lora"):
                first_state = next(iter(lora_state["unet_lora"].values()))
                if "lora_A" in first_state:
                    self.lora_rank = int(first_state["lora_A"].shape[0])
            if cfg.get("alpha") is not None:
                self.lora_alpha = float(cfg["alpha"])
            logger.info(
                f"[Inpainter] LoRA checkpoint rank={self.lora_rank}, "
                f"alpha={self.lora_alpha}"
            )

        # 2. Gắn cấu trúc LoRA (phải cùng rank/alpha với lúc train)
        self.lora_unet = inject_lora_into_model(
            self.unet, rank=self.lora_rank, alpha=self.lora_alpha
        )
        self.lora_text = inject_lora_into_model(
            self.text_encoder, rank=self.lora_rank, alpha=self.lora_alpha
        )

        # 3. Nạp trọng số LoRA.
        if lora_state is not None:
            logger.info(f"[Inpainter] Nạp LoRA từ {lora_path}...")
            if "unet_lora" in lora_state:
                loaded = 0
                for k, state in lora_state["unet_lora"].items():
                    if k in self.lora_unet:
                        self.lora_unet[k].load_state_dict(state)
                        loaded += 1
                if loaded != len(self.lora_unet):
                    raise RuntimeError(
                        f"LoRA U-Net chỉ nạp {loaded}/{len(self.lora_unet)} modules. "
                        "Checkpoint không khớp kiến trúc."
                    )
                logger.info(f"[Inpainter] LoRA U-Net: {loaded}/{len(self.lora_unet)} modules.")
            if "text_lora" in lora_state:
                loaded = 0
                for k, state in lora_state["text_lora"].items():
                    if k in self.lora_text:
                        self.lora_text[k].load_state_dict(state)
                        loaded += 1
                if loaded != len(self.lora_text):
                    raise RuntimeError(
                        f"LoRA Text Encoder chỉ nạp {loaded}/{len(self.lora_text)} modules. "
                        "Checkpoint không khớp kiến trúc."
                    )
                logger.info(f"[Inpainter] LoRA Text Encoder: {loaded}/{len(self.lora_text)} modules.")
            self.using_lora = True
        else:
            logger.warning(
                "[Inpainter] Không có lora_ckpt: đang dùng base model. "
                "Kết quả sẽ KHÔNG đại diện cho cấu hình LoRA của bài báo."
            )

        self.unet.eval()
        self.text_encoder.eval()

        # 3. Textual Inversion
        self.ti_manager = MultiTokenTIManager(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=str(self.device),
        )
        ti_path = self._resolve_checkpoint(self.ti_ckpt, "ti_embeddings.pt")
        if self.ti_ckpt:
            if ti_path is None or not ti_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy TI checkpoint: {ti_path}. "
                    "Hãy chạy 'python run_pipeline.py step2_train ...' trước."
                )
            self.ti_manager.load(ti_path)
            self.ti_loaded = True
        else:
            logger.warning(
                "[Inpainter] Không có ti_ckpt: dùng prompt từ ngữ tự nhiên thay vì token TI đã học."
            )
            self.ti_manager.placeholder_tokens = {
                c: c.replace("_", " ") for c in PCB_DEFECT_CLASSES
            }

        self.has_diffusion = True

    @torch.no_grad()
    def _encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(img_tensor).latent_dist.sample() * self.vae_scaling

    def _work_size(self, width: int, height: int) -> Tuple[int, int]:
        """
        Chọn kích thước làm việc:
          - bội số của 8 (VAE);
          - gần native nhất có thể, tối đa max_side;
          - tối thiểu 256 để UNet ổn định.
        Với ảnh 600x600, kết quả là 600x600 (không resize).
        """
        scale = min(1.0, self.max_side / float(max(width, height)))
        work_w = max(256, int(round(width * scale / 8.0)) * 8)
        work_h = max(256, int(round(height * scale / 8.0)) * 8)
        work_w = min(work_w, self.max_side)
        work_h = min(work_h, self.max_side)
        return work_w, work_h

    @staticmethod
    def _feather_alpha(mask: np.ndarray, sigma: float, expand_px: int = 4) -> np.ndarray:
        """
        Tạo alpha mềm từ mask nhị phân:
          - Gaussian blur sigma 2-3 px;
          - chỉ giữ alpha trong vùng mask được dilate expand_px;
          - ngoài vùng đó alpha = 0 => pixel gốc được giữ nguyên tuyệt đối.
        """
        binary = (mask > 127).astype(np.float32)
        if binary.max() <= 0:
            return binary

        if sigma > 0:
            alpha = cv2.GaussianBlur(binary, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            alpha = binary

        k = max(1, int(expand_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        expanded = cv2.dilate(binary, kernel, iterations=1)
        alpha = np.clip(alpha * expanded, 0.0, 1.0)

        # Giữ tâm mask gần như chắc chắn 1.0 để lỗi không bị pha loãng.
        alpha = np.maximum(alpha, cv2.erode(binary, np.ones((3, 3), np.uint8)) * 0.98)
        return alpha.astype(np.float32)

    @staticmethod
    def _composite(original: np.ndarray, generated: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        a = alpha[..., None]
        out = generated.astype(np.float32) * a + original.astype(np.float32) * (1.0 - a)
        return np.clip(out, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def generate(
        self,
        normal_image: Union[Image.Image, np.ndarray, str],
        defect_mask: Union[Image.Image, np.ndarray, str],
        defect_type: str,
    ) -> Image.Image:
        """Sinh một ảnh lỗi, giữ nguyên pixel ngoài vùng mask."""
        if isinstance(normal_image, (str, Path)):
            normal_image = Image.open(normal_image).convert("RGB")
        elif isinstance(normal_image, np.ndarray):
            normal_image = Image.fromarray(normal_image).convert("RGB")
        else:
            normal_image = normal_image.convert("RGB")

        if isinstance(defect_mask, (str, Path)):
            defect_mask = Image.open(defect_mask).convert("L")
        elif isinstance(defect_mask, np.ndarray):
            defect_mask = Image.fromarray(defect_mask).convert("L")
        else:
            defect_mask = defect_mask.convert("L")

        orig_w, orig_h = normal_image.size
        mask_np_orig = np.array(defect_mask, dtype=np.uint8)
        if not np.any(mask_np_orig > 127):
            logger.warning("[Inpainter] Mask rỗng; trả lại ảnh gốc.")
            self._last_info = {"generation_mode": "no_mask", "work_size": [orig_w, orig_h]}
            return normal_image.copy()

        if getattr(self, "has_diffusion", False):
            work_w, work_h = self._work_size(orig_w, orig_h)
            img_work = normal_image if (work_w, work_h) == (orig_w, orig_h) else normal_image.resize(
                (work_w, work_h), Image.Resampling.LANCZOS
            )
            msk_work = defect_mask if (work_w, work_h) == (orig_w, orig_h) else defect_mask.resize(
                (work_w, work_h), Image.Resampling.NEAREST
            )

            img_np = np.array(img_work).astype(np.float32) / 127.5 - 1.0
            msk_np = (np.array(msk_work) > 127).astype(np.float32)

            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(
                device=self.device, dtype=self.unet.dtype
            )
            msk_tensor = torch.from_numpy(msk_np).unsqueeze(0).unsqueeze(0).to(
                device=self.device, dtype=self.unet.dtype
            )

            # 1. Latent gốc
            z0_orig = self._encode_image(img_tensor)
            latent_h, latent_w = z0_orig.shape[-2:]
            b_lat = F.interpolate(msk_tensor, size=(latent_h, latent_w), mode="nearest")
            b_lat = (b_lat > 0.5).to(z0_orig.dtype)

            # 2. 9-channel conditioning
            in_channels = int(getattr(self.unet.config, "in_channels", 9))
            if in_channels == 9:
                masked_img = img_tensor * (1.0 - msk_tensor)
                masked_img_latents = self._encode_image(masked_img)
                mask_input = torch.cat([b_lat] * 2)
                masked_latents_input = torch.cat([masked_img_latents] * 2)

            # 3. Text embeddings (CFG)
            prompt = self.ti_manager.get_defect_prompt(defect_type)
            tok_pos = self.tokenizer(
                prompt, padding="max_length", max_length=77, return_tensors="pt"
            ).input_ids.to(self.device)
            tok_neg = self.tokenizer(
                self.negative_prompt, padding="max_length", max_length=77, return_tensors="pt"
            ).input_ids.to(self.device)

            emb_pos = self.text_encoder(tok_pos)[0]
            emb_neg = self.text_encoder(tok_neg)[0]
            text_embeddings = torch.cat([emb_neg, emb_pos])

            # 4. Khởi tạo z_T và MỘT noise nền cố định cho toàn bộ quỹ đạo.
            # Vùng lỗi bắt đầu từ Gaussian; vùng nền bắt đầu từ q(z0, T)
            # dùng đúng noise_bg sẽ được dùng lại ở mọi bước overwrite.
            self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps
            noise_bg = torch.randn_like(z0_orig)
            z0_at_T = self.scheduler.add_noise(z0_orig, noise_bg, timesteps[0])
            latents = b_lat * torch.randn_like(z0_orig) + (1.0 - b_lat) * z0_at_T

            pbar = tqdm(
                timesteps,
                desc=f"Khử nhiễu ({defect_type})",
                leave=False,
                unit="step",
            )
            for i, t in enumerate(pbar):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if in_channels == 9:
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_input, masked_latents_input], dim=1
                    )

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)

                latents_prev = self.scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample

                # ── Latent Overwriting nhất quán ──────────────────────
                # Dùng cùng noise_bg để tạo q(z0, prev_t). Không lấy mẫu
                # Gaussian mới ở từng bước.
                if i < len(timesteps) - 1:
                    prev_t = timesteps[i + 1]
                    prev_t_tensor = torch.as_tensor(
                        prev_t, device=self.device, dtype=torch.long
                    )
                    z0_prev = self.scheduler.add_noise(z0_orig, noise_bg, prev_t_tensor)
                    latents = b_lat * latents_prev + (1.0 - b_lat) * z0_prev
                else:
                    # t = 0: vùng nền là latent sạch tuyệt đối.
                    latents = b_lat * latents_prev + (1.0 - b_lat) * z0_orig

            # 5. Decode
            decoded = self.vae.decode(latents / self.vae_scaling).sample
            decoded = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
            decoded_np = (
                decoded.detach().squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0
            ).astype(np.uint8)
            decoded_pil = Image.fromarray(decoded_np)

            if (work_w, work_h) != (orig_w, orig_h):
                decoded_pil = decoded_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

            # 6. Feather + composite ở native resolution
            sigma = self.feather_sigma
            # Với ảnh rất lớn, tăng nhẹ bán kính feather theo tỷ lệ.
            if max(orig_w, orig_h) > 800:
                sigma = max(sigma, min(4.0, sigma * max(orig_w, orig_h) / 800.0))

            orig_np = np.array(normal_image, dtype=np.uint8)
            gen_np = np.array(decoded_pil.convert("RGB"), dtype=np.uint8)
            alpha = self._feather_alpha(mask_np_orig, sigma=sigma, expand_px=4)
            final_np = self._composite(orig_np, gen_np, alpha)

            lora_file = self._resolve_checkpoint(
                self.lora_ckpt, "pytorch_lora_weights.pt"
            )
            ti_file = self._resolve_checkpoint(self.ti_ckpt, "ti_embeddings.pt")
            self._last_info = {
                "generation_mode": "diffusion",
                "generation_version": "ddim_consistent_v2",
                "model_id": self.model_id,
                "lora_checkpoint": str(lora_file.resolve()) if lora_file else None,
                "ti_checkpoint": str(ti_file.resolve()) if ti_file else None,
                "inpaint_steps": int(self.num_inference_steps),
                "guidance_scale": float(self.guidance_scale),
                "work_size": [int(work_w), int(work_h)],
                "orig_size": [int(orig_w), int(orig_h)],
                "feather_sigma": float(sigma),
                "latent_overwrite": "consistent_single_noise_ddim",
                "using_lora": bool(self.using_lora),
                "ti_loaded": bool(self.ti_loaded),
            }
            return Image.fromarray(final_np)

        # ── Fallback mô phỏng (chỉ khi allow_fallback=True) ────────────────
        logger.warning(
            f"[Inpainting Fallback] Mô phỏng defect '{defect_type}' trên mask. "
            "Đây KHÔNG phải kết quả diffusion thật."
        )
        orig_np = np.array(normal_image).copy()
        m = mask_np_orig > 127
        if m.any():
            if defect_type in ["missing_hole", "open_circuit"]:
                orig_np[m] = (orig_np[m] * 0.3).astype(np.uint8)
            elif defect_type in ["spur", "spurious_copper", "short"]:
                orig_np[m] = np.clip(orig_np[m].astype(int) + 60, 0, 255).astype(np.uint8)
            else:
                orig_np[m] = 255 - orig_np[m]

        self._last_info = {
            "generation_mode": "fallback_simulation",
            "work_size": [int(orig_w), int(orig_h)],
            "feather_sigma": 0.0,
            "latent_overwrite": "none",
            "using_lora": False,
            "ti_loaded": False,
        }
        return Image.fromarray(orig_np)

    def generate_batch(
        self,
        mask_records: List[Dict],
        output_dir: Union[str, Path] = "outputs/generated",
    ) -> List[Dict]:
        """Sinh batch ảnh lỗi từ mask_metadata.json của Step 1."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Step 2] Bắt đầu sinh {len(mask_records)} ảnh lỗi tổng hợp...")
        gen_records: List[Dict] = []
        failures = 0

        # Resume chỉ được phép dùng lại ảnh nếu metadata cũ chứng minh ảnh đó
        # thật sự được sinh bằng diffusion + LoRA + TI + latent overwrite nhất
        # quán, đúng checkpoint và đúng prompt/steps. Nếu không, phải sinh lại.
        current_lora = self._resolve_checkpoint(
            self.lora_ckpt, "pytorch_lora_weights.pt"
        )
        current_ti = self._resolve_checkpoint(self.ti_ckpt, "ti_embeddings.pt")
        current_lora = str(current_lora.resolve()) if current_lora else None
        current_ti = str(current_ti.resolve()) if current_ti else None

        cached_by_name: Dict[str, Dict] = {}
        existing_meta = out_dir / "generated_metadata.json"
        if existing_meta.exists():
            try:
                with open(existing_meta, "r", encoding="utf-8") as f:
                    old_records = json.load(f)
                for old in old_records:
                    p = Path(old.get("synthetic_image_path", ""))
                    if (
                        p.exists()
                        and old.get("generation_mode") == "diffusion"
                        and old.get("generation_version") == "ddim_consistent_v2"
                        and old.get("latent_overwrite") == "consistent_single_noise_ddim"
                        and old.get("using_lora")
                        and old.get("ti_loaded")
                        and old.get("lora_checkpoint") == current_lora
                        and old.get("ti_checkpoint") == current_ti
                        and int(old.get("inpaint_steps", -1)) == int(self.num_inference_steps)
                        and abs(float(old.get("guidance_scale", -1.0)) - float(self.guidance_scale)) < 1e-6
                        and old.get("model_id") == self.model_id
                    ):
                        cached_by_name[p.name] = old
            except Exception as e:
                logger.warning(f"[Step 2] Không đọc được metadata cũ để resume: {e}")

        def _file_sig(path: Union[str, Path]):
            try:
                st = Path(path).stat()
                return int(st.st_size), int(st.st_mtime_ns)
            except OSError:
                return None

        for i, rec in enumerate(tqdm(mask_records, desc="Khuếch tán sinh ảnh lỗi", unit="ảnh")):
            normal_img_path = rec["normal_image_path"]
            mask_path = rec["mask_path"]
            defect_type = rec["defect_type"]

            stem = Path(normal_img_path).stem
            out_name = f"syn_{stem}_{defect_type}_{i:04d}.png"
            out_img_path = out_dir / out_name

            try:
                cached = cached_by_name.get(out_img_path.name)
                normal_sig = _file_sig(normal_img_path)
                mask_sig = _file_sig(mask_path)
                if cached is not None:
                    cached_normal_sig = (
                        cached.get("source_normal_size"),
                        cached.get("source_normal_mtime_ns"),
                    )
                    cached_mask_sig = (
                        cached.get("mask_size"),
                        cached.get("mask_mtime_ns"),
                    )
                    if (
                        normal_sig is None
                        or mask_sig is None
                        or cached_normal_sig != normal_sig
                        or cached_mask_sig != mask_sig
                        or not out_img_path.exists()
                        or out_img_path.stat().st_size <= 1024
                    ):
                        cached = None

                if cached is not None:
                    info = cached
                else:
                    gen_img = self.generate(
                        normal_image=normal_img_path,
                        defect_mask=mask_path,
                        defect_type=defect_type,
                    )
                    gen_img.save(out_img_path)
                    info = dict(self._last_info) if self._last_info else {}
            except Exception as e:
                failures += 1
                logger.warning(f"[Step 2] Bỏ qua {normal_img_path} / {defect_type}: {e}")
                continue

            gen_records.append({
                "synthetic_image_path": str(out_img_path.resolve()),
                "source_normal_image": normal_img_path,
                "mask_path": mask_path,
                "source_normal_size": normal_sig[0] if normal_sig else None,
                "source_normal_mtime_ns": normal_sig[1] if normal_sig else None,
                "mask_size": mask_sig[0] if mask_sig else None,
                "mask_mtime_ns": mask_sig[1] if mask_sig else None,
                "defect_type": defect_type,
                "class_id": rec.get("class_id", 0),
                "bbox_pixel": rec.get("bbox_pixel", []),
                "bbox_norm": rec.get("bbox_norm", []),
                "prompt": self.ti_manager.get_defect_prompt(defect_type),
                "negative_prompt": self.negative_prompt,
                "generation_mode": info.get("generation_mode", "unknown"),
                "generation_version": info.get("generation_version", "unknown"),
                "model_id": info.get("model_id", self.model_id),
                "lora_checkpoint": info.get("lora_checkpoint", current_lora),
                "ti_checkpoint": info.get("ti_checkpoint", current_ti),
                "inpaint_steps": info.get("inpaint_steps", self.num_inference_steps),
                "guidance_scale": info.get("guidance_scale", self.guidance_scale),
                "work_size": info.get("work_size", []),
                "feather_sigma": info.get("feather_sigma", self.feather_sigma),
                "latent_overwrite": info.get("latent_overwrite", "unknown"),
                "using_lora": info.get("using_lora", self.using_lora),
                "ti_loaded": info.get("ti_loaded", self.ti_loaded),
            })

        meta_out = out_dir / "generated_metadata.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(gen_records, f, indent=2, ensure_ascii=False)

        logger.info(f"Step 2 xong! Đã tạo {len(gen_records)} ảnh. Metadata: {meta_out}")
        if failures:
            logger.warning(f"Đã bỏ qua {failures}/{len(mask_records)} mẫu do lỗi sinh ảnh.")
        return gen_records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(
        "Step 2 module đã sẵn sàng. Dùng:\n"
        "  python run_pipeline.py step2_train --metadata_path data/train_metadata.json\n"
        "  python run_pipeline.py step2_generate --mask_meta outputs/pcb_synthesis/masks/mask_metadata.json "
        "--lora_ckpt outputs/pcb_synthesis/lora_checkpoints/epoch_010"
    )
