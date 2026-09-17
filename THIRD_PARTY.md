# Third-party assets

- **Avatar** (`public/assets/model-clips.glb`, ignored): the owner's MetaPerson (AvatarSDK)
  export with nine Mixamo (Adobe) animation clips retargeted onto it in `~/personal-site`.
  Copied here by `scripts/import-avatar.sh`, never committed: the owner's reading of
  Mixamo's terms is that animations may be used inside a personal or commercial project but
  not redistributed as standalone assets, and a public repository would be the latter. Not
  legal advice; the terms are re-checked before any public launch.
- **Speech** (`public/audio/arena/`, ignored): the owner's cloned voice, synthesized by
  Personal Voice Clone Studio (CosyVoice3, FunAudioLLM, Apache-2.0 code) from a fine-tune
  on the owner's own recordings. Personal data; never committed.
- **three.js** (MIT) renders the avatar; **Next.js** (MIT) serves the app.
