#!/usr/bin/env python3
"""LoRA finetuning of parakeet-tdt-0.6b-v3 for Arabic Quran ASR."""

import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.utils import exp_manager


def apply_lora(model, rank: int, alpha: int, dropout: float):
    """Freeze encoder and apply LoRA to encoder attention+MLP layers."""
    # Freeze entire encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Identify target linear layers in encoder for LoRA
    lora_targets = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
    lora_layers = {}

    for name, module in model.encoder.named_modules():
        if isinstance(module, torch.nn.Linear) and any(t in name for t in lora_targets):
            lora_layers[name] = module

    if not lora_layers:
        # Fallback: apply to all Linear layers in encoder
        print("  Warning: named LoRA targets not found, applying to all encoder Linear layers")
        for name, module in model.encoder.named_modules():
            if isinstance(module, torch.nn.Linear):
                lora_layers[name] = module

    # Replace with LoRA wrappers
    for name, module in lora_layers.items():
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.encoder
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)

        lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_module)

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA applied to {len(lora_layers)} layers")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


class LoRALinear(torch.nn.Module):
    """LoRA wrapper around a frozen Linear layer."""

    def __init__(self, original: torch.nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight)
        torch.nn.init.zeros_(self.lora_B.weight)

        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return original_out + lora_out


def build_dataloader_config(manifest_path: str, batch_size: int, shuffle: bool):
    """Build NeMo dataloader config dict."""
    return {
        "manifest_filepath": manifest_path,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": 4,
        "pin_memory": True,
        "sample_rate": 16000,
        "trim_silence": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Finetune parakeet-tdt-0.6b-v3 with LoRA for Arabic Quran ASR")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/parakeet-arabic"))
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (default: 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--val-check-interval", type=int, default=500)
    parser.add_argument("--save-top-k", type=int, default=3)
    args = parser.parse_args()

    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_rank

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pretrained model ---
    print("Loading pretrained parakeet-tdt-0.6b-v3...")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

    # --- Swap tokenizer and rebuild decoder ---
    print("Swapping tokenizer (rebuilds decoder/joint)...")
    tokenizer_model = str(args.tokenizer_dir / "tokenizer.model")
    model.change_vocabulary(new_tokenizer_dir=str(args.tokenizer_dir), new_tokenizer_type="bpe")

    # --- Apply LoRA ---
    print("Applying LoRA to encoder...")
    apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # --- Configure SpecAugment ---
    if hasattr(model, "spec_augmentation") and model.spec_augmentation is not None:
        spec_cfg = model.cfg.get("spec_augment", {})
        if spec_cfg:
            spec_cfg["freq_masks"] = 2
            spec_cfg["time_masks"] = 2
            print("  SpecAugment: freq_masks=2, time_masks=2")

    # --- Configure data ---
    print("Configuring dataloaders...")
    train_dl_cfg = build_dataloader_config(str(args.train_manifest.resolve()), args.batch_size, shuffle=True)
    val_dl_cfg = build_dataloader_config(str(args.val_manifest.resolve()), args.batch_size, shuffle=False)

    model.setup_training_data(OmegaConf.create({"train_ds": train_dl_cfg}))
    model.setup_validation_data(OmegaConf.create({"validation_ds": val_dl_cfg}))

    # --- Configure optimizer ---
    print("Configuring optimizer...")
    optim_cfg = {
        "name": "adamw",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": int(args.max_steps * args.warmup_ratio),
            "min_lr": 1e-6,
        },
    }
    model.setup_optimization(OmegaConf.create({"optim": optim_cfg}))

    # --- Precision ---
    precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    print(f"  Precision: {precision}")

    # --- Trainer ---
    print("Setting up trainer...")
    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=args.gradient_clip,
        precision=precision,
        enable_progress_bar=True,
        default_root_dir=str(args.output_dir),
    )

    # --- Experiment manager (checkpointing + logging) ---
    exp_cfg = {
        "exp_dir": str(args.output_dir),
        "name": "parakeet-arabic-lora",
        "checkpoint_callback_params": {
            "monitor": "val_loss",
            "mode": "min",
            "save_top_k": args.save_top_k,
            "always_save_nemo": True,
        },
        "create_tensorboard_logger": True,
    }
    exp_manager.exp_manager(trainer, OmegaConf.create({"exp_manager": exp_cfg}))

    # --- Train ---
    print(f"Starting training for {args.max_steps} steps...")
    trainer.fit(model)

    # --- Save final model ---
    final_path = args.output_dir / "final_model.nemo"
    model.save_to(str(final_path))
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
