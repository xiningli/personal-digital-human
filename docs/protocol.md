# Protocol: preference learning for a digital human's body language

The same method as Personal Voice Clone Studio's arena, applied to motion: the owner
watches their avatar say one line two ways and picks the way that moves more like them.
The choice is the data; everything else is bookkeeping around it.

## 0. Principles

- **One variable per round.** Both candidates say the same line, from the same audio file,
  on the same avatar, with the same seed. Only the motion policy differs.
- **Forced choice first, ratings second.** A/B with "no preference" is required; the three
  5-point scales (natural, fits the words, like me) are optional and reported as means.
- **Partition by voice.** The audio's checkpoint (`voiceModel`) is a hard partition key,
  as `model` is in the studio: a motion that fits one delivery need not fit another.
- **Measure the rater.** 15 % of rounds re-serve a decided pair under fresh labels; agreement
  with the earlier vote is the test-retest reliability, reported next to every table.
- **Nothing personal in git.** Rounds, votes, synthesized speech and the avatar with the
  clips baked in are ignored; the export is a file the owner produces on purpose.

## 1. Candidates: motion policies

A policy is what the site's avatar can do today (`lib/pose.ts`, the rules of ADR-028 with
their amplitudes exposed), described by:

| field | meaning |
|---|---|
| `clip` | Mixamo clip playing underneath (`Talking`, `Telling A Secret`) or `null` for the procedural body alone |
| `headLift` | how much the head lifts with loudness, 0 / 1 / 1.6 (1 = the site's default) |
| `beats` | beat gestures per loudness rise, 0 / 1 / 1.8 (0 = none) |
| `sway` | slow head turn and torso sway, 0.5 / 1 / 1.6 |
| `ease` | easing time-constant scale (1 for now) |

With a clip underneath, beats are off except at the site default, so the grid stays small
(2 clips × 9 + 1 procedural × 27 = 45 policies) and the clip's arms are not doubled. A
policy is identified by its id, and the ranking is over ids, so a learned policy — the
output of a co-speech model such as EMAGE or DiffSHEG, delivered as a `MotionTrack` of
joint rotations — joins the same table as `source: "model"` without changing the method.

## 2. Rounds and trials

`POST /api/arena/random` draws a line from the bank (`lib/policies.ts`; categories
greeting, answer, explaining, thinking, delight — the kinds of sentence the digital human
says), has the voice studio speak it once with the emotion the site would use for that
category (cached by content hash), and draws two policies by inverse frequency over how
often each has been served. Labels are shuffled.

- `test` [85 %]: a fresh line and pair. Decided votes feed the Bradley-Terry ranking.
- `repeat` [15 %]: a decided test round re-served with fresh labels, the same audio, seed
  and policies. Counted for reliability, never for the ranking.

## 3. Ranking

Bradley-Terry strengths by maximum likelihood (MM iteration, half-win prior against a
virtual opponent) on decided test votes, mean-centred, with 95 % bootstrap intervals over
matches [1000 resamples, seeded]. Reported per policy: score, CI, wins, losses, rounds,
mean ratings. `GET /api/arena/stats?voiceModel=` for one checkpoint, `all` to pool.

## 4. Export and what it trains

`GET /api/arena/export` writes `preference-pairs.jsonl`: one line per decided pair with the
line, audio path, emotion, voice checkpoint, avatar, seed, both policies, ratings and notes.
Two uses:

1. **Choose the site's policy now.** The top of the table is what `personal-site` should
   ship as its default body language, per voice checkpoint.
2. **Train a preference model later.** The pairs are DPO-style data for a gesture policy:
   given audio (and the words), prefer the chosen motion over the rejected one. With
   generated candidates from a co-speech model, the same pairs fine-tune that model on the
   owner's taste, which is the plan of the site's ADR-016.

Per-candidate notes are written for a reader who cannot see the round: "hands: too busy;
want them still on the short words" — attribute, what was seen, what is wanted.

## 5. Profiles: motion learned from video

A profile is a person whose body language the owner admires, captured from a video clip:
the owner pastes a YouTube URL (optionally with start and duration seconds) or uploads a
file, and the server runs the extraction pipeline of `motion/extract/` on it — fetch and
trim to 30 fps, GVHMR with a static camera to SMPL-X poses, retarget onto the avatar's rig,
then the same two checks every generated track passes (`verify_retarget.py --check` for the
transfer, `metrics.py --check` for floating feet, skating and a frozen body). Between them,
`motion/refine_track.py` recenters the track onto the avatar's own idea of elegant: GVHMR
tracks arrive with the forearms chronically bent (~85 deg off rest where the capture clips
sit at ~40) and the hips twisting, so the arms and torso chain are recentered onto the
time-mean pose of the asset's "Talking" clip, the hips' dynamic component is shrunk, and
the estimated legs are converged toward the rest pose. Fingers are synthesized:
GVHMR has no hand keypoints, so a raw track freezes all 40 finger bones; the pass gives
each one the "Talking" clip's time-mean local rotation (a natural speaking hand, not the
flat bind pose) plus a slow deterministic drift — spectral noise brick-walled to
0.3-0.8 Hz, 3-6 deg peak per bone, fixed seed, seamless under looping — and up to 4 deg
of opening while the wrist moves fast. A clipping guard rides on the
recentering: per-frame FK compares each hand joint against body capsules whose surfaces
are calibrated from the asset's own clips (the closest any approved capture ever comes),
and wherever the recentered arm would cross that line the recentering weight fades to
zero over a ±9-frame Hann window — the raw pose, measured safe there, takes over — with
an escape rotation on the forearm for the rare frame the raw pose itself penetrates. The
run refuses to finish quietly if any frame still ends up deep (d < 0.7 × surface). The
raw track is kept as `track.unrefined.json` next to the profile; a refine failure never
fails the profile by itself.

What the player finally serves is gated separately, because every check above fired on
the upstream npz while refine rewrote the track afterwards: `motion/check_track.py`
(`--check`, gate overrides in metrics.py's style) FKs the refined track.json on the glb
skeleton — pure rotation, bind translations, exactly the player's `trackToClip`
convention — and fails the profile on four measurements: hand clipping against the same
calibrated capsule surfaces the guard uses (deep penetration must be zero frames, touch
at most 20), the lower toe's world height (mean must sit in [0.02, 0.08] m around the
sole offset the runtime grounds to), planted-foot horizontal drift (mean at most
0.25 cm/frame, calibrated as ~2x the worst built-in clip's baseline; `--calibrate`
prints the per-clip baseline), and peak quaternion angular velocity (max 600 deg/s,
against teleport/seizure flicker). It is also what judges the unrefined track when
refine itself failed, since the gate runs after the last step that mutates the track.
Clips of about
30 s with one person in view, camera not moving, work best. Ready profiles join the arena's
candidate pool as `source: "profile"` and are drawn, weighted and ranked exactly like mocap
clips and generated tracks.

A profile's motion is decoupled from the audio of the video it came from. Every candidate
in a round moves to the round's shared audio — the profile's track plays under the same line
as its opponent — so the comparison stays blind and fair: what is ranked is how the person
moves, not what they happened to be saying.

The audio is not thrown away, though: after the quality gates, `motion/extract/segment.py`
transcribes it (faster-whisper, word timestamps grouped into sentences) and cuts the track
into sentence-level gesture segments, each with a semantic embedding
(sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, 384-d, L2-normalized).
The segments contract is versioned JSON:

```json
{ "version": 1, "fps": 30, "frames": 900,
  "embedModel": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "segments": [ { "i": 0, "startS": 0.32, "endS": 2.87, "startFrame": 10, "endFrame": 86,
                  "text": "...", "embedding": [384 floats] } ] }
```

`data/profiles/<id>/segments.json` keeps the embeddings (server-side runtime matching only);
`public/motion/profile-<id>.segments.json` is the same structure with `embedding` dropped,
for frontend preview and slicing. Segmentation is additive and never gates readiness: a
speechless clip or a whisper failure yields a ready profile with `hasSegments` unset.

The extracted audio (16 kHz mono, `data/profiles/<id>/audio.wav`) is served by
`GET /api/profiles/<id>/audio` — `audio/wav`, streaming, with byte-range support — so the
profiles page can play a profile's motion with the voice it was extracted from: the whole
take looping in sync, or one sentence's frame range against its startS..endS.

Extraction quality is judged by a human, not only by the gates: `/profiles/<id>` is the
imitation-eval page. It plays the source clip (`GET /api/profiles/<id>/video`, streamed with
byte-range support, served as-is) and the extracted track on the avatar side by side, on one
transport — the video's `currentTime` is the single master clock and the avatar's track
action follows it, hard-corrected when the drift exceeds 80 ms, with an optional 0.5x mode
for frame-by-frame checks. The default flow is **per-segment** (逐段评测): the page picks the
first unrated sentence segment, loops it on both sides until the rater acts, and shows the
progress (已评 n/N) on a bar plus per-chip ticks with the latest score. A submission
(`POST /api/profiles/<id>/eval` with `segment` = the segment's `i`, bounds-checked against
the public segments JSON) auto-advances to the next unrated segment; clicking any chip jumps
to that segment, so re-rating a rated one just appends a new line. The alternative tab
(整段对比) is the original whole-clip loop whose submissions carry no segment. Both score
three 1-5 dimensions — likeness (像不像本人), timing (节奏同步), naturalness (自然度) — with an
optional note; each submission appends one line to `data/profile-evals.jsonl` (gitignored,
like the votes). `GET` keeps the two kinds apart: the top-level `count`/`means` cover
whole-clip evals, while `perSegment[i]` carries the submission count and the means of the
**latest** submission for segment `i` (re-ratings supersede, they do not average).

The limitations are honest ones, inherited from the pipeline. GVHMR predicts only the 22
body joints: the fingers, jaw and eyes stay in the bind pose, so a profile's hands are
relaxed where the real person's may not be. And in a medium shot — the usual framing of a
talk — the legs are out of frame or barely visible, so leg motion is the least reliable part
of what is extracted; the foot metrics gate the worst of it, and the arena ranks the rest.

## References

- Bradley & Terry (1952), rank analysis of incomplete block designs.
- Hunter (2004), MM algorithms for generalized Bradley-Terry models.
- Rafailov et al. (2023), Direct Preference Optimization.
- Liu et al. (2024), EMAGE: co-speech gesture generation via expressive masked audio gesture modeling.
- Chen et al. (2024), DiffSHEG: a diffusion-based approach for real-time speech-driven holistic 3D expression and gesture generation.
