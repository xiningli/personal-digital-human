"""Objective metrics for generated motion, so a change can be tested instead of eyeballed.

The voice side already works this way: every take passes WER and speaker similarity before a
human is asked to listen. This is the motion equivalent. Run it on a generated npz and the
numbers either hold or they do not.

    .venv/bin/python metrics.py <motion.npz> <audio.wav>

Measured on 2026-09-19 against deliberately degraded copies of a real EMAGE clip, which is
how the thresholds below were chosen and how one metric was disqualified:

| sample        | BC     | diversity | float  | skate | FGD   |
|---------------|--------|-----------|--------|-------|-------|
| EMAGE, real   | 0.789  | 11.49     | 1.2cm  | 0.78  | 6.685 |
| frozen        | 0.000  | 0.0001    | 0.0    | 0.00  | 9.212 |
| time-shuffled | 0.871  | 11.49     | 1.2cm  | 10.02 | 6.780 |
| jittered      | 0.852  | 15.93     | 5.2cm  | 21.50 | 7.279 |

- **foot skate / float**, lower is better. The sharpest discriminator by far: 13x between
  real and time-shuffled, 27x against jitter. Purely kinematic models know nothing about the
  ground, and this is the defect that actually shipped, so these are hard gates.
- **diversity (L1div)**, higher is better. Absolutely catches a frozen body, 0.0001 against
  11.49, and catches nothing else. Hard gate against the statue failure.
- **FGD**, lower is better. Frechet distance to real BEAT2 motion in a learned latent space,
  the field's standard human-likeness proxy. It ranks the four correctly, but real against
  time-shuffled is 6.685 against 6.780, a 1.4% margin: a 5.8s clip does not fill the
  encoder's 240-frame window and has to be tiled, which erases most of the temporal
  structure FGD would otherwise punish. Report and compare it across models over many clips;
  do not gate a single clip on it.
- **beat consistency (BC)**: **do not use as a quality measure.** Time-shuffled motion (0.871)
  and pure jitter (0.852) both score HIGHER than the real clip (0.789), because BC rewards
  sharp velocity changes near audio onsets and noise has those everywhere. It detects only
  the degenerate case of no motion at all, which diversity already covers.

None of these is naturalness. They are necessary conditions a human rater would also fail a
clip for, which is what makes them useful as tests; the arena is still what decides whether
motion looks like the owner.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "pantomatrix"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

TOOLS = REPO / "emage_evaltools"
SMPLX_DIR = TOOLS / "smplx_models"
# SMPL-X joint indices for the feet; toes are what touch the floor.
LEFT_TOE, RIGHT_TOE = 10, 11
FPS = 30


def joint_positions(poses: np.ndarray, betas: np.ndarray, device: torch.device) -> np.ndarray:
    """SMPL-X forward kinematics: (frames, 55, 3) joint positions in metres."""
    import smplx

    model = smplx.create(
        str(SMPLX_DIR), model_type="smplx", gender="NEUTRAL_2020",
        use_face_contour=False, num_betas=300, num_expression_coeffs=100,
        ext="npz", use_pca=False,
    ).to(device).eval()
    frames = poses.shape[0]
    p = torch.from_numpy(poses).float().to(device)
    b = torch.from_numpy(betas[:300]).float().to(device).unsqueeze(0).repeat(frames, 1)
    with torch.no_grad():
        out = model(
            betas=b, transl=torch.zeros(frames, 3, device=device),
            global_orient=p[:, 0:3], body_pose=p[:, 3:66],
            jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72], reye_pose=p[:, 72:75],
            left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165],
            expression=torch.zeros(frames, 100, device=device),
            return_verts=False,
        )
    return out.joints[:, :55].detach().cpu().numpy()


def foot_metrics(joints: np.ndarray) -> dict[str, float]:
    """Float: how far the lower foot sits off the floor. Skate: how far a planted foot slides.

    The floor is taken as the lowest toe height over the clip, so a clip that is uniformly
    offset is not punished twice; what is measured is variation the eye reads as floating.
    """
    toes = joints[:, [LEFT_TOE, RIGHT_TOE], :]          # frames, 2, 3
    height = toes[:, :, 1]
    floor = height.min()
    lower = height.min(axis=1) - floor                   # per frame, the nearer foot
    planted = height - floor < 0.03                      # within 3 cm of the floor
    step = np.linalg.norm(np.diff(toes[:, :, [0, 2]], axis=0), axis=2)  # horizontal travel
    sliding = step[planted[1:]]
    return {
        "foot_float_mean_m": float(lower.mean()),
        "foot_float_max_m": float(lower.max()),
        "foot_skate_mean_m": float(sliding.mean()) if sliding.size else 0.0,
    }


def fgd_against_reference(poses: np.ndarray, device: torch.device) -> float | None:
    """Frechet distance to real human motion, in the latent space of the BEAT2 autoencoder.

    This is the one metric here that measures human-likeness rather than a necessary
    condition, and the one that needs real motion to compare against: clips under
    reference/beat2/ (downloaded from the BEAT2 dataset). Both sides are cut into the
    encoder's 240-frame windows, so a short generated clip still contributes.
    """
    refs = sorted((REPO / "reference" / "beat2").glob("*.npz"))
    if not refs:
        return None
    from emage_evaltools.mertic import FGD
    from emage_utils import rotation_conversions as rc

    evaluator = FGD(download_path=str(TOOLS) + "/")
    win = 240

    def windows(p: np.ndarray) -> torch.Tensor | None:
        n = p.shape[0] // win
        if n == 0:
            return None
        chunk = p[: n * win].reshape(n, win, 55, 3)
        t = torch.from_numpy(chunk).float().to(device)
        return rc.axis_angle_to_rotation_6d(t).reshape(n, win, 55 * 6)

    gen = windows(poses)
    if gen is None:  # pad a short clip by looping it to one window
        rep = int(np.ceil(win / poses.shape[0]))
        gen = windows(np.tile(poses, (rep, 1))[:win])
    for ref in refs:
        real = windows(np.load(ref, allow_pickle=True)["poses"])
        if real is None:
            continue
        take = min(real.shape[0], 8)
        evaluator.update(gen[: take].float(), real[: take].float())
    return float(evaluator.compute())


# Chosen from the table above: comfortably clear of the real clip, comfortably inside every
# degraded one. They fail a build; FGD is reported but never gates a single clip.
GATES = {"diversity_min": 1.0, "foot_skate_max_cm": 3.0, "foot_float_mean_max_cm": 3.0}


def gate_overrides(argv: list[str]) -> list[str]:
    """--<gate> X overrides GATES (key dashes, e.g. --foot-float-mean-max-cm 4.5), e.g. for
    footage whose feet are not visible (black trousers on a black stage) where foot float is
    an estimate, not a fact. Returns the remaining (positional) args."""
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        key = a[2:].replace("-", "_") if a.startswith("--") else ""
        if key in GATES and i + 1 < len(argv):
            try:
                GATES[key] = float(argv[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        if not a.startswith("--"):
            positional.append(a)
        i += 1
    return positional


def main() -> None:
    check = "--check" in sys.argv
    args = gate_overrides(sys.argv[1:])
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(2)
    npz, wav = Path(args[0]), Path(args[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(npz, allow_pickle=True)
    poses, betas = data["poses"], data["betas"]
    frames = poses.shape[0]
    print(f"{npz.name}: {frames} frames ({frames / FPS:.2f}s at {FPS} fps), {poses.shape[1] // 3} joints")

    joints = joint_positions(poses, betas, device)
    flat = joints.reshape(frames, -1)

    from emage_evaltools.mertic import BC, L1div

    bc = BC(download_path=str(TOOLS) + "/", sigma=0.3, order=7)
    audio_beat = bc.load_audio(str(wav), t_start=0, t_end=int(frames / FPS * 16000))
    motion_beat = bc.load_motion(flat, t_start=0, t_end=frames, pose_fps=FPS, without_file=True)
    bc.compute(audio_beat, motion_beat, length=frames, pose_fps=FPS)

    div = L1div()
    div.compute(flat)

    feet = foot_metrics(joints)
    fgd = fgd_against_reference(poses, device)
    print()
    print(f"  beat consistency   {bc.avg():.4f}   (higher is better)")
    print(f"  diversity          {div.avg():.4f}   (higher is better; ~0 means frozen)")
    print(f"  foot float mean    {feet['foot_float_mean_m'] * 100:.1f} cm  (lower is better)")
    print(f"  foot float max     {feet['foot_float_max_m'] * 100:.1f} cm")
    print(f"  foot skate mean    {feet['foot_skate_mean_m'] * 100:.2f} cm/frame while planted")
    print()
    if fgd is None:
        print("  FGD                unavailable: put real-motion npz files in reference/beat2/")
    else:
        print(f"  FGD                {fgd:.3f}   (reported, not gated; see the header)")

    if not check:
        return
    failures = []
    if div.avg() < GATES["diversity_min"]:
        failures.append(f"diversity {div.avg():.4f} < {GATES['diversity_min']} (the body is frozen)")
    if feet["foot_skate_mean_m"] * 100 > GATES["foot_skate_max_cm"]:
        failures.append(f"foot skate {feet['foot_skate_mean_m'] * 100:.2f} cm > {GATES['foot_skate_max_cm']} (planted feet slide)")
    if feet["foot_float_mean_m"] * 100 > GATES["foot_float_mean_max_cm"]:
        failures.append(f"foot float {feet['foot_float_mean_m'] * 100:.1f} cm > {GATES['foot_float_mean_max_cm']} (the body hovers)")
    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        raise SystemExit(1)
    print("  PASS  every gate")


if __name__ == "__main__":
    main()
