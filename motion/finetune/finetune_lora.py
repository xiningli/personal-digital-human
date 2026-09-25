"""LoRA fine-tune of EMAGE audio->motion on the owner's profile data.

Standalone replacement for pantomatrix/train_emage_audio.py (which imports wandb and
diffusers and wants DDP). Manual LoRA — no peft: every nn.Linear whose leaf name is one of
TARGETS (the transformer decoder stacks and all projections, per
models/emage_audio/modeling_emage_audio.py:215-263) gets wrapped in LoRALinear. The packed
in_proj_weight of nn.MultiheadAttention is a raw Parameter no LoRA lib can target either,
so nn.Linear coverage is equal. Base weights stay frozen; the WavEncoder BatchNorms stay
in eval() so their running stats do not drift on ~15 min of one speaker.

Train step mirrors train_val_fn (train_emage_audio.py:130-183): GT axis-angle -> 6D ->
motion_vq.map2index/map2latent, then a seed forward (first 4 frames unmasked) and one
fully-masked forward, rec + cls losses with the yaml weights except lf=0 (profile GT
expressions are all zeros, so the face latent has nothing to reconstruct). The no-audio
forward is skipped. AdamW on the LoRA params only, grad clip 0.99.

    ../.venv/bin/python finetune_lora.py --bs 16 --steps 3000 --lr 1e-4
    ../.venv/bin/python finetune_lora.py --mix-meta data/beat2_s20_l64.json --mix-ratio 0.4 --out outputs_mix

--mix-meta mixes a second dataset (the BEAT2 subset from prepare_beat2.py) into every
batch as an anti-overfitting regularizer: --mix-ratio of each batch's items come from it,
the rest from the profile windows, via two independent shuffled iterators. Every
--eval-every steps the model is eval()'d on the held-out profile test windows and the rec
losses are logged as test_rec_seed/test_rec_audio — the train/test gap is the overfitting
signal the mixing is meant to shrink.

Saves <out>/lora_only.pt (A/B weights) and <out>/emage_lora_merged/ (deltas merged back
into the base model, save_pretrained — point generate.py's EMAGE_MODEL_PATH at it).
Progress lands in <out>/train_log.jsonl, one line per LOG_EVERY steps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "pantomatrix"
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import emage_utils.rotation_conversions as rc  # noqa: E402
from datasets.beat2 import BEAT2DatasetEamgeFootContact  # noqa: E402
from models.emage_audio import (  # noqa: E402
    EmageAudioModel,
    EmageVAEConv,
    EmageVQModel,
    EmageVQVAEConv,
)

HF = "H-Liu1997/emage_audio"
META = Path(__file__).resolve().parent / "data" / "profiles_s20_l64.json"
SEED_FRAMES = 4
# Leaf names of the nn.Linear modules that get a LoRA adapter.
TARGETS = {
    "linear1", "linear2", "out_proj", "fc1", "fc2",
    "moton_proj", "audio_body_motion_proj", "audio_face_motion_proj",
    "motion_out_proj_upper", "motion_out_proj_hands", "motion_out_proj_lower",
    "face_out_proj",
}
# Loss weights from configs/emage_audio.yaml, with lf=0 (GT expressions are zeros).
LU, LL, LH, LF = 3.0, 3.0, 3.0, 0.0
CU, CL, CH, CF = 1.0, 1.0, 1.0, 0.0
LOG_EVERY = 20


class LoRALinear(nn.Module):
    """y = base(x) + dropout(x) @ A^T @ B^T * alpha/r; base stays frozen."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge(self):
        self.base.weight.data += (self.lora_B @ self.lora_A) * self.scaling


class LoRAWeight(nn.Module):
    """Weight-space LoRA for nn.MultiheadAttention.out_proj: MHA reads out_proj.weight
    directly in F.linear instead of calling the module, so a wrapper's forward would never
    run. A weight parametrization is seen by both paths. (No dropout here — weight-level.)"""

    def __init__(self, weight: torch.Tensor, r: int, alpha: float):
        super().__init__()
        out_f, in_f = weight.shape
        self.lora_A = nn.Parameter(torch.randn(r, in_f, device=weight.device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=weight.device))
        self.scaling = alpha / r

    def forward(self, w):
        return w + (self.lora_B @ self.lora_A) * self.scaling


def apply_lora(model: nn.Module, r: int, alpha: float, dropout: float) -> int:
    from torch.nn.utils import parametrize

    count = 0
    for name, module in list(model.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in TARGETS or not isinstance(module, nn.Linear):
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        if isinstance(parent, nn.MultiheadAttention):
            parametrize.register_parametrization(module, "weight", LoRAWeight(module.weight, r, alpha))
        else:
            setattr(parent, leaf, LoRALinear(module, r=r, alpha=alpha, dropout=dropout).to(module.weight.device))
        count += 1
    return count


def merge_and_unwrap(model: nn.Module) -> None:
    from torch.nn.utils import parametrize

    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            module.merge()
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], module.base)
        elif parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


def get_rec_loss(pred, gt, lu, ll, lh, lf):
    return (
        lu * F.mse_loss(pred["rec_upper"], gt["upper"])
        + ll * F.mse_loss(pred["rec_lower"], gt["lower"])
        + lh * F.mse_loss(pred["rec_hands"], gt["hands"])
        + lf * F.mse_loss(pred["rec_face"], gt["face"])
    )


def get_cls_loss(pred, gt, cu, cl, ch, cf, cls_fn):
    return (
        cu * cls_fn(F.log_softmax(pred["cls_upper"], dim=2).permute(0, 2, 1), gt["upper"])
        + cl * cls_fn(F.log_softmax(pred["cls_lower"], dim=2).permute(0, 2, 1), gt["lower"])
        + ch * cls_fn(F.log_softmax(pred["cls_hands"], dim=2).permute(0, 2, 1), gt["hands"])
        + cf * cls_fn(F.log_softmax(pred["cls_face"], dim=2).permute(0, 2, 1), gt["face"])
    )


def forward_losses(model, motion_vq, batch, device, cls_fn):
    """One train_val_fn-style pass: seed forward + fully-masked forward, rec+cls losses."""
    motion_gt = batch["motion"].to(device).float()
    audio = batch["audio"].to(device).float()
    expressions_gt = batch["expressions"].to(device).float()
    trans = batch["trans"].to(device).float()
    foot_contact = batch["foot_contact"].to(device).float()

    bs, t, jc = motion_gt.shape
    motion_6d = rc.axis_angle_to_rotation_6d(motion_gt.reshape(bs, t, jc // 3, 3)).reshape(bs, t, -1)
    with torch.no_grad():
        latent_index = motion_vq.map2index(motion_6d, expressions_gt, tar_contact=foot_contact, tar_trans=trans)
        latent = motion_vq.map2latent(motion_6d, expressions_gt, tar_contact=foot_contact, tar_trans=trans)
    masked_motion = torch.cat([motion_6d, trans, foot_contact], dim=-1)
    speaker_id = torch.zeros(bs, 1).long().to(device)

    mask = torch.ones_like(masked_motion)
    mask[:, :SEED_FRAMES] = 0
    pred_seed = model(audio, speaker_id, masked_motion=masked_motion, mask=mask, use_audio=True)
    loss_dict = {
        "rec_seed": get_rec_loss(pred_seed, latent, LU, LL, LH, LF),
        "cls_seed": get_cls_loss(pred_seed, latent_index, CU, CL, CH, CF, cls_fn),
    }
    mask = torch.ones_like(masked_motion)
    pred_full = model(audio, speaker_id, masked_motion=masked_motion, mask=mask, use_audio=True)
    loss_dict["rec_audio"] = get_rec_loss(pred_full, latent, LU, LL, LH, LF)
    loss_dict["cls_audio"] = get_cls_loss(pred_full, latent_index, CU, CL, CH, CF, cls_fn)
    return loss_dict


def evaluate(model, motion_vq, loader, device, cls_fn):
    """Mean rec losses over the held-out profile test windows (overfitting signal)."""
    model.eval()
    sums, cnt = {}, 0
    with torch.no_grad():
        for batch in loader:
            for k, v in forward_losses(model, motion_vq, batch, device, cls_fn).items():
                sums[k] = sums.get(k, 0.0) + float(v)
            cnt += 1
    model.train()
    model.audio_encoder_face.eval()  # frozen BatchNorm running stats
    model.audio_encoder_body.eval()
    return {f"test_{k}": round(v / cnt, 4) for k, v in sums.items()}


def cycle(loader):
    while True:
        yield from loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--mix-meta", type=str, default=None,
                    help="second meta JSON (e.g. BEAT2 subset); each batch draws --mix-ratio of its items from it")
    ap.add_argument("--mix-ratio", type=float, default=0.4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    torch.manual_seed(222)
    np.random.seed(222)
    device = torch.device("cuda")

    cfg = OmegaConf.load(REPO / "configs" / "emage_audio.yaml")
    cfg.data.meta_paths = [str(META)]
    train_set = BEAT2DatasetEamgeFootContact(cfg, "train")
    test_set = BEAT2DatasetEamgeFootContact(cfg, "test")
    loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, drop_last=True,
                        num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False, drop_last=False,
                             num_workers=2, persistent_workers=False)
    print(f"train windows: {len(train_set)}, batches/epoch: {len(loader)}, test windows: {len(test_set)}", flush=True)

    mix_iter = None
    if args.mix_meta:
        cfg.data.meta_paths = [args.mix_meta]
        mix_set = BEAT2DatasetEamgeFootContact(cfg, "train")
        mix_loader = DataLoader(mix_set, batch_size=args.bs, shuffle=True, drop_last=True,
                                num_workers=4, persistent_workers=True)
        mix_iter = cycle(mix_loader)
        print(f"mix windows: {len(mix_set)} at ratio {args.mix_ratio}", flush=True)

    part = lambda name: EmageVQVAEConv.from_pretrained(HF, subfolder=f"emage_vq/{name}").to(device)  # noqa: E731
    motion_vq = EmageVQModel(
        face_model=part("face"), upper_model=part("upper"),
        lower_model=part("lower"), hands_model=part("hands"),
        global_model=EmageVAEConv.from_pretrained(HF, subfolder="emage_vq/global").to(device),
    ).to(device).eval()
    for p in motion_vq.parameters():
        p.requires_grad = False

    model = EmageAudioModel.from_pretrained(HF).to(device)
    for p in model.parameters():
        p.requires_grad = False
    n_lora = apply_lora(model, r=args.rank, alpha=args.alpha, dropout=0.05)
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    n_train = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA on {n_lora} linears — trainable {n_train/1e6:.2f}M / {n_total/1e6:.1f}M params", flush=True)

    model.train()
    model.audio_encoder_face.eval()  # frozen BatchNorm running stats
    model.audio_encoder_body.eval()

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    cls_fn = nn.NLLLoss().to(device)
    log_f = open(log_path, "a")

    step = 0
    started = time.monotonic()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            if mix_iter is not None:
                k_mix = round(args.bs * args.mix_ratio)
                mix_batch = next(mix_iter)
                batch = {key: torch.cat([batch[key][: args.bs - k_mix], mix_batch[key][:k_mix]])
                         for key in batch}
            optimizer.zero_grad()
            loss_dict = forward_losses(model, motion_vq, batch, device, cls_fn)
            loss = sum(loss_dict.values())
            torch.nn.utils.clip_grad_norm_(lora_params, 0.99)
            loss.backward()
            optimizer.step()
            step += 1

            rec = None
            if step % LOG_EVERY == 0 or step == 1:
                rec = {
                    "step": step,
                    **{k: round(float(v.detach()), 4) for k, v in loss_dict.items()},
                    "all": round(float(loss.detach()), 4),
                    "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                    "elapsed_s": round(time.monotonic() - started, 1),
                }
            if step % args.eval_every == 0 or step == args.steps:
                test_metrics = evaluate(model, motion_vq, test_loader, device, cls_fn)
                rec = rec or {"step": step}
                rec.update(test_metrics)
            if rec:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(rec, flush=True)

    log_f.close()
    lora_sd = {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}
    torch.save({"lora": lora_sd, "rank": args.rank, "alpha": args.alpha}, out_dir / "lora_only.pt")
    merge_and_unwrap(model)
    model.save_pretrained(out_dir / "emage_lora_merged")
    print(f"done: {step} steps in {time.monotonic() - started:.0f}s -> {out_dir}/emage_lora_merged", flush=True)


if __name__ == "__main__":
    main()
