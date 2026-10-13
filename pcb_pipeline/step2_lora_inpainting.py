"""
Step 2: Defect Image Synthesis using Stable Diffusion Inpainting + LoRA
------------------------------------------------------------------------
- Huấn luyện LoRA (rank=8, alpha=16) trên ảnh lỗi thực
- Textual Inversion để embed khái niệm "lỗi PCB"
- Inpainting chỉ trong vùng mask, giữ nguyên vùng ngoài
- Loss function kết hợp reconstruction + defect promotion
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# HuggingFace / Diffusers
from diffusers import (
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTokenizer, CLIPTextModel

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# LoRA Utilities
# ─────────────────────────────────────────────────────────

def add_lora_to_unet(unet: UNet2DConditionModel, rank: int = 8, alpha: float = 16.0):
    """
    Gắn LoRA attention processors vào UNet.
    rank=8, alpha=16 theo thiết lập trong bài.
    """
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    # Đặt scale = alpha / rank
    lora_layers = AttnProcsLayers(unet.attn_processors)
    return lora_layers


# ─────────────────────────────────────────────────────────
# Textual Inversion Token
# ─────────────────────────────────────────────────────────

class TextualInversionManager:
    """
    Quản lý embedding của token đặc biệt đại diện cho "lỗi PCB".
    Token: <pcb-defect>
    """

    PLACEHOLDER = "<pcb-defect>"

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        init_token: str = "defect",
        device: str = "cuda",
    ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self._add_token(init_token)

    def _add_token(self, init_token: str):
        """Thêm placeholder token vào vocabulary."""
        num_added = self.tokenizer.add_tokens(self.PLACEHOLDER)
        self.placeholder_id = self.tokenizer.convert_tokens_to_ids(self.PLACEHOLDER)

        # Khởi tạo embedding từ token gốc
        init_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(init_token)[0]
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            init_emb = self.text_encoder.get_input_embeddings().weight[init_id].clone()
            self.text_encoder.get_input_embeddings().weight[self.placeholder_id] = init_emb

        # Chỉ train embedding của token mới
        self.text_encoder.get_input_embeddings().weight.requires_grad_(False)
        self.ti_param = torch.nn.Parameter(
            self.text_encoder.get_input_embeddings().weight[self.placeholder_id]
            .clone()
            .to(self.device)
        )
        logger.info(f"[TI] Thêm token '{self.PLACEHOLDER}' (id={self.placeholder_id})")

    def update_embedding(self):
        """Đồng bộ ti_param vào embedding table sau mỗi bước train."""
        with torch.no_grad():
            self.text_encoder.get_input_embeddings().weight[
                self.placeholder_id
            ] = self.ti_param.data

    def get_prompt(self, defect_type: str) -> str:
        return f"a photo of {self.PLACEHOLDER} {defect_type} on PCB circuit board"

    def save(self, path: str | Path):
        torch.save({"ti_embedding": self.ti_param.data}, path)
        logger.info(f"[TI] Đã lưu embedding → {path}")

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.ti_param.data = ckpt["ti_embedding"]
        self.update_embedding()
        logger.info(f"[TI] Đã load embedding ← {path}")


# ─────────────────────────────────────────────────────────
# Defect-aware Loss
# ─────────────────────────────────────────────────────────

def defect_aware_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lambda_defect: float = 2.0,
) -> torch.Tensor:
    """
    Loss kết hợp:
      - Reconstruction loss (toàn ảnh): giữ vùng ngoài mask không đổi
      - Defect loss (trong mask): nhân trọng số lambda_defect để tập trung tạo lỗi
    
    Args:
        model_pred: Noise dự đoán từ UNet  (B, C, H, W)
        target:     Noise ground truth      (B, C, H, W)
        mask:       Binary mask (0/1)       (B, 1, H, W) — 1 = vùng lỗi
        lambda_defect: Hệ số tăng cường loss trong vùng mask
    """
    # Resize mask về spatial dim của latent
    mask_resized = F.interpolate(mask, size=model_pred.shape[-2:], mode="nearest")

    # L2 loss toàn ảnh
    loss_recon = F.mse_loss(model_pred * (1 - mask_resized), target * (1 - mask_resized))

    # L2 loss trong mask (× lambda để tập trung tạo lỗi)
    n_mask_pixels = mask_resized.sum().clamp(min=1)
    loss_defect = (lambda_defect * (model_pred - target) ** 2 * mask_resized).sum() / n_mask_pixels

    return loss_recon + loss_defect


# ─────────────────────────────────────────────────────────
# LoRA Fine-tuning Trainer
# ─────────────────────────────────────────────────────────

class LoRAInpaintTrainer:
    """
    Huấn luyện LoRA + Textual Inversion trên tập ảnh lỗi PCB thực.

    Config mặc định theo bảng trong ảnh:
        rank=8, alpha=16, epochs=10, batch_size=32
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        rank: int = 8,
        alpha: float = 16.0,
        lr_lora: float = 1e-4,
        lr_ti: float = 5e-4,
        device: str = "cuda",
        output_dir: str = "outputs/lora_checkpoints",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.rank = rank
        self.alpha = alpha
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[LoRA] Loading base model: {model_id}")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        # Freeze base weights
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Add LoRA
        self.lora_layers = add_lora_to_unet(self.unet, rank=rank, alpha=alpha)

        # Add Textual Inversion
        self.ti_manager = TextualInversionManager(
            self.tokenizer, self.text_encoder, device=str(self.device)
        )

        # Optimizers
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.lora_layers.parameters(), "lr": lr_lora},
                {"params": [self.ti_manager.ti_param], "lr": lr_ti},
            ]
        )

    # ── Dataset loading ──────────────────────────────────────────────────
    def _load_dataset(self, data_json: str | Path):
        """
        data_json: JSON file với list of dicts:
          {image_path, mask_path, defect_type}
        (output của Step 1 hoặc tập ảnh lỗi thực)
        """
        with open(data_json, encoding="utf-8") as f:
            records = json.load(f)
        return records

    @staticmethod
    def _preprocess_batch(records: list[dict], device: torch.device, dtype: torch.dtype):
        """
        Tải ảnh + mask từ danh sách records, trả về tensors.
        Returns:
            images:   (B, 3, 512, 512) trong [-1, 1]
            masks:    (B, 1, 512, 512) trong [0, 1]
            prompts:  list[str]
        """
        from torchvision.transforms.functional import to_tensor, resize

        imgs, msks, prompts = [], [], []
        placeholder = TextualInversionManager.PLACEHOLDER

        for rec in records:
            img = Image.open(rec["image_path"]).convert("RGB").resize((512, 512))
            msk = Image.open(rec["mask_path"]).convert("L").resize((512, 512))
            defect_type = rec.get("defect_type", "defect")

            imgs.append(to_tensor(img) * 2 - 1)          # [-1, 1]
            msks.append(to_tensor(msk))                   # [0, 1]
            prompts.append(
                f"a photo of {placeholder} {defect_type} on PCB circuit board"
            )

        return (
            torch.stack(imgs).to(device, dtype),
            torch.stack(msks).to(device, dtype),
            prompts,
        )

    # ── Training loop ────────────────────────────────────────────────────
    def train(
        self,
        data_json: str | Path,
        epochs: int = 10,
        batch_size: int = 32,
        noise_steps: int = 1000,
        save_every: int = 2,
    ):
        """
        Huấn luyện chính.

        Args:
            data_json:   Đường dẫn tới metadata JSON (từ Step 1 hoặc real defects)
            epochs:      Số epoch (mặc định 10)
            batch_size:  Kích thước batch (mặc định 32)
            noise_steps: Số bước noise scheduler
            save_every:  Lưu checkpoint mỗi N epoch
        """
        records = self._load_dataset(data_json)
        n = len(records)
        logger.info(f"[LoRA] Tập huấn luyện: {n} samples | epochs={epochs} | batch={batch_size}")

        dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(noise_steps)

        for epoch in range(1, epochs + 1):
            np.random.shuffle(records)
            epoch_loss = 0.0
            n_batches = 0

            for start in tqdm(range(0, n, batch_size), desc=f"Epoch {epoch}/{epochs}"):
                batch = records[start: start + batch_size]
                images, masks, prompts = self._preprocess_batch(batch, self.device, dtype)

                with torch.no_grad():
                    # Encode ảnh vào latent space
                    latents = self.vae.encode(images).latent_dist.sample() * 0.18215
                    # Encode text prompt
                    text_inputs = self.tokenizer(
                        prompts, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt"
                    ).to(self.device)
                    encoder_hidden = self.text_encoder(text_inputs.input_ids)[0]

                # Sample noise + timestep
                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_steps, (latents.shape[0],), device=self.device).long()
                noisy_latents = scheduler.add_noise(latents, noise, t)

                # Mask-conditioned latent (inpainting)
                mask_latent = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
                masked_latents = latents * (1 - mask_latent)
                inpaint_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)

                # Forward
                model_pred = self.unet(
                    inpaint_input.to(dtype),
                    t,
                    encoder_hidden_states=encoder_hidden,
                ).sample

                # Defect-aware loss
                loss = defect_aware_loss(model_pred.float(), noise.float(), masks, lambda_defect=2.0)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lora_layers.parameters(), 1.0)
                self.optimizer.step()
                self.ti_manager.update_embedding()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"[LoRA] Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

            if epoch % save_every == 0 or epoch == epochs:
                self._save_checkpoint(epoch)

        logger.info("[LoRA] ✅ Huấn luyện hoàn tất!")

    def _save_checkpoint(self, epoch: int):
        ckpt_dir = self.output_dir / f"epoch_{epoch:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.lora_layers.save_pretrained(ckpt_dir)
        self.ti_manager.save(ckpt_dir / "ti_embedding.pt")
        logger.info(f"[LoRA] 💾 Checkpoint → {ckpt_dir}")

    def load_checkpoint(self, ckpt_dir: str | Path):
        ckpt_dir = Path(ckpt_dir)
        self.unet.load_attn_procs(ckpt_dir)
        ti_path = ckpt_dir / "ti_embedding.pt"
        if ti_path.exists():
            self.ti_manager.load(ti_path)
        logger.info(f"[LoRA] ✅ Load checkpoint ← {ckpt_dir}")


# ─────────────────────────────────────────────────────────
# Inference: Sinh ảnh lỗi
# ─────────────────────────────────────────────────────────

class PCBDefectInpainter:
    """
    Sinh ảnh lỗi PCB bằng SD Inpainting + LoRA.

    Thiết lập mặc định:
        num_inference_steps = 50   (~4 giây/ảnh theo bảng)
        guidance_scale = 7.5
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        lora_ckpt: Optional[str | Path] = None,
        ti_ckpt: Optional[str | Path] = None,
        device: str = "cuda",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        dtype = torch.float16 if "cuda" in str(self.device) else torch.float32

        logger.info(f"[Inpaint] Loading pipeline: {model_id}")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()

        # Load LoRA weights
        if lora_ckpt:
            self.pipe.unet.load_attn_procs(lora_ckpt)
            logger.info(f"[Inpaint] LoRA loaded ← {lora_ckpt}")

        # Load TI embedding
        if ti_ckpt:
            ti_path = Path(ti_ckpt)
            if ti_path.exists():
                num_added = self.pipe.tokenizer.add_tokens(
                    TextualInversionManager.PLACEHOLDER
                )
                placeholder_id = self.pipe.tokenizer.convert_tokens_to_ids(
                    TextualInversionManager.PLACEHOLDER
                )
                self.pipe.text_encoder.resize_token_embeddings(
                    len(self.pipe.tokenizer)
                )
                ckpt = torch.load(ti_path, map_location=self.device)
                with torch.no_grad():
                    self.pipe.text_encoder.get_input_embeddings().weight[
                        placeholder_id
                    ] = ckpt["ti_embedding"].to(dtype)
                logger.info(f"[Inpaint] TI embedding loaded ← {ti_path}")

    def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        defect_type: str = "scratch",
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Sinh 1 ảnh lỗi.

        Args:
            image:       Ảnh PCB gốc (RGB)
            mask:        Mask vùng lỗi (L, 0/255)
            defect_type: Loại lỗi
            seed:        Random seed để tái hiện kết quả

        Returns:
            Ảnh lỗi đã được inpainting (PIL.Image RGB)
        """
        placeholder = TextualInversionManager.PLACEHOLDER
        prompt = f"a photo of {placeholder} {defect_type} on PCB circuit board, realistic, high quality"
        neg_prompt = "blurry, low quality, distorted, unrealistic, cartoon"

        generator = None
        if seed is not None:
            generator = torch.Generator(device=str(self.device)).manual_seed(seed)

        image_resized = image.resize((512, 512))
        mask_resized = mask.resize((512, 512))

        result = self.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]

        # Phục hồi về kích thước gốc
        result = result.resize(image.size, Image.LANCZOS)
        return result

    def generate_batch(
        self,
        mask_metadata: list[dict],
        output_dir: str | Path,
        seeds: Optional[list[int]] = None,
    ) -> list[dict]:
        """
        Sinh ảnh lỗi cho toàn bộ records từ Step 1.

        Returns:
            Updated records có thêm key 'generated_path'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for i, rec in enumerate(tqdm(mask_metadata, desc="[Inpaint] Generating")):
            try:
                image = Image.open(rec["image_path"]).convert("RGB")
                mask = Image.open(rec["mask_path"]).convert("L")
                seed = seeds[i] if seeds else None

                gen_img = self.generate(image, mask, rec.get("defect_type", "defect"), seed)

                stem = Path(rec["image_path"]).stem
                out_path = output_dir / f"{stem}_{rec['defect_type']}_{i:05d}.png"
                gen_img.save(out_path)

                new_rec = {**rec, "generated_path": str(out_path)}
                results.append(new_rec)
                logger.info(f"[Inpaint] ✓ {out_path.name}")

            except Exception as e:
                logger.warning(f"[Inpaint] Lỗi {rec.get('image_path')}: {e}")

        # Lưu metadata
        meta_path = output_dir / "generated_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"[Inpaint] 💾 {len(results)} ảnh → {output_dir}")
        return results


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    # Train
    tr = subparsers.add_parser("train", help="Fine-tune LoRA + TI")
    tr.add_argument("--data_json",  required=True)
    tr.add_argument("--model_id",   default="runwayml/stable-diffusion-inpainting")
    tr.add_argument("--epochs",     type=int,   default=10)
    tr.add_argument("--batch_size", type=int,   default=32)
    tr.add_argument("--rank",       type=int,   default=8)
    tr.add_argument("--alpha",      type=float, default=16.0)
    tr.add_argument("--output_dir", default="outputs/lora_checkpoints")
    tr.add_argument("--device",     default="cuda")

    # Generate
    gn = subparsers.add_parser("generate", help="Sinh ảnh lỗi PCB")
    gn.add_argument("--mask_meta",  required=True, help="mask_metadata.json từ Step 1")
    gn.add_argument("--lora_ckpt",  default=None)
    gn.add_argument("--ti_ckpt",    default=None)
    gn.add_argument("--output_dir", default="outputs/generated")
    gn.add_argument("--steps",      type=int,   default=50)
    gn.add_argument("--guidance",   type=float, default=7.5)
    gn.add_argument("--device",     default="cuda")

    args = parser.parse_args()

    if args.cmd == "train":
        trainer = LoRAInpaintTrainer(
            model_id=args.model_id,
            rank=args.rank,
            alpha=args.alpha,
            device=args.device,
            output_dir=args.output_dir,
        )
        trainer.train(args.data_json, epochs=args.epochs, batch_size=args.batch_size)

    elif args.cmd == "generate":
        with open(args.mask_meta, encoding="utf-8") as f:
            mask_meta = json.load(f)
        inpainter = PCBDefectInpainter(
            lora_ckpt=args.lora_ckpt,
            ti_ckpt=args.ti_ckpt,
            device=args.device,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
        )
        inpainter.generate_batch(mask_meta, args.output_dir)

    else:
        parser.print_help()
