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

## References

- Bradley & Terry (1952), rank analysis of incomplete block designs.
- Hunter (2004), MM algorithms for generalized Bradley-Terry models.
- Rafailov et al. (2023), Direct Preference Optimization.
- Liu et al. (2024), EMAGE: co-speech gesture generation via expressive masked audio gesture modeling.
- Chen et al. (2024), DiffSHEG: a diffusion-based approach for real-time speech-driven holistic 3D expression and gesture generation.
