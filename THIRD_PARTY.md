# Third-party assets

- **Avatar** (`public/assets/model-clips.glb`, ignored): the owner's MetaPerson (AvatarSDK)
  export with nine Mixamo (Adobe) animation clips retargeted onto it in `~/digital-human/personal-site`.
  Copied here by `scripts/import-avatar.sh`, never committed: the owner's reading of
  Mixamo's terms is that animations may be used inside a personal or commercial project but
  not redistributed as standalone assets, and a public repository would be the latter. Not
  legal advice; the terms are re-checked before any public launch.
- **Speech** (`public/audio/arena/`, ignored): the owner's cloned voice, synthesized by
  Personal Voice Clone Studio (CosyVoice3, FunAudioLLM, Apache-2.0 code) from a fine-tune
  on the owner's own recordings. Personal data; never committed.
- **three.js** (MIT) renders the avatar; **Next.js** (MIT) serves the app.
- **Motion extraction** (`motion/extract/`, heavyweight parts ignored): GVHMR (Zhejiang
  University 3D Vision Group, all-rights-reserved research license, citation required)
  predicts SMPL-X motion from monocular video, using its pretrained checkpoints plus
  HMR2.0 / ViTPose / YOLOv8x weights fetched from the project's Google Drive (each carries
  its own license), and the SMPL-X body model (Max Planck, research only, already on disk
  for EMAGE). Source videos are the uploader's copyright; extracted motion is used for
  research/development here, and any clip published with the site needs its own clearance.
- **EMAGE** and its eval tools under `motion/` (see `motion/README.md`): code and weights
  as published by their authors; generated motion is not redistributed as a dataset.
