# Personal Digital Human — Claude Code Guide

Next.js 16 app on port 3020: a motion arena that ranks body-language policies for the
owner's digital human by pairwise preference (docs/protocol.md). Sibling of
`~/digital-human/personal-voice-clone-studio` (same methodology, same toolchain) and consumer of
`~/digital-human/personal-site` (the avatar asset and the rules in `lib/pose.ts`, which mirror the site's
`src/pose.ts`).

## Running

```bash
bash scripts/import-avatar.sh   # public/assets/model-clips.glb from the site; ignored by git; also repairs the tracks (clean- + fix-avatar-motion.mjs)
npm run dev                     # http://0.0.0.0:3020 (no HTTPS needed: no microphone here)
```

Needs a voice service for speech: `STUDIO_TTS_URL` in `.env.local` (default the studio's backend on 127.0.0.1:8010, or an exported bundle on another host). Reference clips come from the service's `GET /v1/profiles`. Ports 3000/8000 belong to another
project; never kill what listens there.

## Contract

`lib/types.ts` is the single source of truth. Routes under `app/api/arena/*` (random,
rounds, votes, stats, export), `app/api/report`, `app/api/tts` (studio health). Pure logic:
`lib/policies.ts` (design space, sampler, line bank), `lib/stats.ts` (Bradley-Terry,
bootstrap, reliability), `lib/pose.ts` (the motion rules), `lib/tts.ts` (studio client with
content-hash cache). Rendering: `components/MotionStage.tsx` (one candidate, three.js) and
`components/ArenaPanel.tsx` (the round, shared audio, votes).

Keep `lib/pose.ts` in step with `personal-site/src/pose.ts`: the arena ranks what the site
ships, so a rule that changes in one must change in the other.

## Data (ignored by git)

`data/` (rounds, votes), `public/audio/arena/` (the owner's voice), `public/assets/*.glb`.
Never commit or upload them.

## Verification

```bash
npm test && npx tsc --noEmit && npx eslint .
```
