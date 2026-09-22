"""extract.py <video.mp4> [out.npz] — video -> SMPL-X motion in the EMAGE npz contract.

Drives GVHMR headless (static camera, no DPVO, no rendering), then repacks its
predicted SMPL-X params into the format motion/retarget.py consumes:

    poses             (T, 55, 3) axis-angle: global_orient + body_pose + zeros(22..54)
    betas             (300,) frame-0 betas zero-padded (GVHMR predicts 10)
    mocap_frame_rate  real fps of the input video

The full GVHMR prediction (incl. per-frame betas/transl, camera params) is kept as
<out>.raw.pt next to the npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GVHMR = HERE / "GVHMR"
sys.path.insert(0, str(GVHMR))
import os

import hydra
from hydra import compose, initialize_config_module

# With static_cam=true, GVHMR's world frame has the person facing -Z while the
# three.js stage expects the avatar to face +Z toward the camera (avatar at origin,
# camera at z=+4.35). A constant yaw pre-rotation of the world frame fixes facing.
WORLD_YAW_FIX = np.pi

from hmr4d.configs import register_store_gvhmr
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: F401  (registers the demo model store)
from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda  # noqa: F401
from hmr4d.utils.preproc import Extractor, Tracker, VitPoseExtractor
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer


def init_config():
    """Hydra initialization, once per process. A context manager: compose_cfg() and
    hydra.utils.instantiate() must run inside it. Split from compose_cfg so
    extract_shots.py can compose one config per shot in a single process."""
    return initialize_config_module(version_base="1.3", config_module="hmr4d.configs")


def compose_cfg(video_path: Path, output_root: Path):
    register_store_gvhmr()
    cfg = compose(
        config_name="demo",
        overrides=[
            f"video_name={video_path.stem}",
            "static_cam=true",
            "use_dpvo=false",
            "verbose=false",
            f"output_root={output_root}",
        ],
    )
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def build_cfg(video_path: Path, output_root: Path):
    with init_config():
        return compose_cfg(video_path, output_root)


def copy_video(video_path: Path, cfg) -> None:
    """GVHMR's demo re-encodes the input; keep the same behaviour."""
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=23)
        for img in reader:
            writer.write_frame(img)
        writer.close()
        reader.close()


@torch.no_grad()
def run_preprocess(cfg) -> None:
    paths = cfg.paths
    video_path = cfg.video_path
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
        torch.cuda.empty_cache()
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, torch.load(paths.bbx)["bbx_xys"])
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
        torch.cuda.empty_cache()
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, torch.load(paths.bbx)["bbx_xys"])
        torch.save(vit_features, paths.vit_features)
        del extractor
        torch.cuda.empty_cache()


def load_data_dict(cfg) -> dict:
    length, width, height = get_video_lwh(cfg.video_path)
    R_w2c = torch.eye(3).repeat(length, 1, 1)  # static camera
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    return {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(cfg.paths.bbx)["bbx_xys"],
        "kp2d": torch.load(cfg.paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(cfg.paths.vit_features),
    }


def load_model(cfg):
    """The GVHMR demo model, eval mode on CUDA. Expensive (2.7 GB HMR2 backbone):
    extract_shots.py loads it once and predicts every shot with the same instance."""
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    return model.eval().cuda()


@torch.no_grad()
def predict(model, cfg) -> dict:
    return detach_to_cpu(model.predict(load_data_dict(cfg), static_cam=True))


def world_fix(global_orient: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """With static_cam=true, GVHMR's world frame has the person facing -Z while the
    three.js stage expects the avatar to face +Z toward the camera. Pre-rotate the
    world frame (global_orient and transl) by a constant yaw so output faces +Z."""
    from scipy.spatial.transform import Rotation

    R_fix = Rotation.from_rotvec(np.array([0.0, WORLD_YAW_FIX, 0.0])).as_matrix()
    R_glob = Rotation.from_rotvec(global_orient).as_matrix()      # (T, 3, 3)
    fixed_orient = Rotation.from_matrix(R_fix[None] @ R_glob).as_rotvec().astype(np.float32)
    fixed_transl = (R_fix[None] @ transl[..., None])[..., 0].astype(np.float32)
    return fixed_orient, fixed_transl


def video_fps(video_path: Path) -> int:
    import cv2

    return round(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS)) or 30


def save_npz(out_npz: Path, global_orient: np.ndarray, body_pose: np.ndarray,
             betas10: np.ndarray, transl: np.ndarray, fps: int) -> np.ndarray:
    """Repack SMPL-X sequences (already world-fixed) into the EMAGE npz contract
    motion/retarget.py consumes. `betas10` is one (10,) shape vector for the clip —
    frame 0's for a single-shot extract, the per-shot median for extract_shots.
    Returns the (T, 55, 3) poses for the sanity report."""
    T = global_orient.shape[0]
    poses55 = np.zeros((T, 55, 3), dtype=np.float32)
    poses55[:, 0] = global_orient
    poses55[:, 1:22] = body_pose.reshape(T, 21, 3)
    # joints 22-54 (jaw, eyes, hands) stay zero: GVHMR predicts no face/fingers

    # EMAGE convention is a neutral body (all existing motion/out/*.npz have betas=0), and
    # retarget.py hard-codes a zero-beta rest pose, so the npz carries zero betas for
    # consistency. GVHMR's predicted shape is kept in `betas_gvhmr` and in the raw .pt.
    betas300 = np.zeros(300, dtype=np.float32)
    betas_gvhmr = np.zeros(300, dtype=np.float32)
    betas_gvhmr[: betas10.shape[0]] = betas10

    # EMAGE contract (see motion/out/*.npz): poses stored flat (T, 165)
    np.savez(
        out_npz,
        poses=poses55.reshape(T, 165),
        betas=betas300,
        betas_gvhmr=betas_gvhmr,
        expressions=np.zeros((T, 100), dtype=np.float32),
        trans=transl,
        model="smplx",
        gender="neutral",
        mocap_frame_rate=fps,
    )
    return poses55


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    video_path = Path(sys.argv[1]).resolve()
    assert video_path.exists(), f"video not found: {video_path}"
    out_npz = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else video_path.with_suffix(".npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(GVHMR)  # GVHMR config/paths are relative to its repo root

    cfg = build_cfg(video_path, HERE / "outputs")
    copy_video(video_path, cfg)
    run_preprocess(cfg)

    Log.info("[HMR4D] Predicting")
    model = load_model(cfg)
    pred = predict(model, cfg)

    p = pred["smpl_params_global"]
    global_orient = p["global_orient"].numpy()            # (T, 3)
    body_pose = p["body_pose"].numpy()                    # (T, 63)
    betas = p["betas"].numpy()                            # (T, 10)
    transl = p["transl"].numpy()                          # (T, 3)

    global_orient, transl = world_fix(global_orient, transl)
    # keep the raw sidecar in the same (fixed) world frame
    p["global_orient"] = torch.from_numpy(global_orient)
    p["transl"] = torch.from_numpy(transl)
    torch.save(pred, out_npz.with_suffix(".raw.pt"))

    fps = video_fps(video_path)
    poses55 = save_npz(out_npz, global_orient, body_pose, betas[0], transl, fps)
    T = global_orient.shape[0]

    # ---- sanity report ----
    assert np.isfinite(poses55).all(), "non-finite values in poses!"
    aa_step = poses55[1:, :22] - poses55[:-1, :22]
    ang_speed = np.linalg.norm(aa_step, axis=-1).mean() * fps  # rad/s over 22 actuated joints
    print(f"OK  {out_npz}")
    print(f"    frames={T}  fps={fps}  poses=(T, 165) flat, 55 joints x 3")
    print(f"    finite=True  mean joint angular speed={np.degrees(ang_speed):.1f} deg/s")
    print(f"    raw GVHMR output: {out_npz.with_suffix('.raw.pt')}")


if __name__ == "__main__":
    main()
