# Personal Digital Human

Preference learning for the body language of a personal digital human — the motion
counterpart of [Personal Voice Clone Studio](../personal-voice-clone-studio). The avatar
says one line in the owner's cloned voice two ways; the owner picks the way that moves more
like them; a Bradley-Terry table with bootstrap intervals ranks the ways; the decided pairs
export as training data for a gesture policy.

## Run

```bash
bash scripts/import-avatar.sh     # copies the avatar with the Mixamo clips from ~/personal-site (never committed)
npm install
npm run dev                       # http://<host>:3020/arena
```

Speech comes from a voice service: the studio's backend on this machine
(`bash backend/run.sh` in `~/personal-voice-clone-studio`), or a bundle the studio's
**Export** tab pushed to another host — set `STUDIO_TTS_URL` in `.env.local` (this
checkout uses `http://p520:8010`, a resident service on the owner's second machine, so this
card's memory stays free). The arena asks the service for its reference clips
(`GET /v1/profiles`) and never needs the studio's files. Without a ready service a round
cannot be drawn and the page says so.

## What a round is

- one line from a bank of the kinds of sentence the digital human says, spoken once by the
  studio with the emotion the site would use for it, cached by content hash;
- two motion policies — a Mixamo clip underneath or the procedural body, and the amplitude
  of the head, beat and sway rules on top — drawn by inverse frequency;
- both rendered live on the same avatar from the same audio, side by side.

Method, trial types, ranking and export: [docs/protocol.md](docs/protocol.md).

## API

| Method | Route | Body / result |
|---|---|---|
| POST | `/api/arena/random` | `{category?: "any" \| greeting \| answer \| explaining \| thinking \| delight, trial?: "test" \| "repeat", repeatOf?}` → round |
| GET / DELETE | `/api/arena/rounds`, `/api/arena/rounds/[id]` | rounds |
| GET / POST | `/api/arena/votes` | `{roundId, winnerId \| null, ratings?: {[candidateId]: {naturalness, fit, likeness}}, notes?, candidateNotes?}` |
| GET | `/api/arena/stats?voiceModel=` | Bradley-Terry table, reliability, trial counts |
| GET | `/api/arena/export` | `preference-pairs.jsonl` (`?format=json`) |
| GET | `/api/report` | Markdown report per voice checkpoint |
| GET | `/api/tts` | voice studio health |

## Data (ignored by git)

`data/arena-rounds.json`, `data/arena-votes.jsonl`, `public/audio/arena/*.wav` (the owner's
cloned voice), `public/assets/model-clips.glb` (Mixamo clips, see `THIRD_PARTY.md`).

## Verification

```bash
npm test && npx tsc --noEmit && npx eslint .
```
