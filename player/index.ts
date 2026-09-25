/**
 * The digital human, as one importable piece. personal-site consumes this from the sibling
 * checkout (see its vite.config alias) instead of keeping a second copy of the same code.
 */
export { AvatarStage, CLIP_FOR, OUTFIT_MATERIAL, POCKET_AREA, PRINT_AREA, stripIsland, trackToClip, type AvatarOptions, type AvatarState, type MotionState, type MotionTrack } from './avatar';
export { groundOffset, SOLE_BELOW_TOE, blink, speechFace } from './ground';
