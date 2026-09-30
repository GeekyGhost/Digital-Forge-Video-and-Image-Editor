/**
 * VFX Phase 5: the Clip Editor VFX surface's "Insert character" card (never shown in Photo Edit).
 * A character picture + a driving clip (this plate, or another clip) -> Wan Animate 2 -> SAM 2 +
 * BiRefNet matte -> an element, placed over the plate aligned (when the plate drove it), with an
 * optional Relight step and an optional clean plate patch that paints the actor out around it.
 */
import { useState } from "react";
import { Eraser, UserRound } from "lucide-react";
import type { MediaAsset } from "../../domain/types";
import type { AnimateStatus } from "../../services/characterService";
import type { ElementEntry } from "../../services/elementService";

export type CharacterOptions = {
  name: string;
  imageId: string;
  prompt: string;
  /** "plate", or the id of another video asset to drive with. */
  drive: string;
  /** "mask/entry" id of the plate's actor matte, or "" (then BiRefNet alone). */
  matteId: string;
  steps: number;
  seed: number;
  area: number;
  relight: boolean;
  /** Build a clean plate patch after placing the character, so the actor does not show around it. */
  hideActor: boolean;
};

export function CharacterPanel(props: {
  status?: AnimateStatus;
  images: MediaAsset[];
  videos: MediaAsset[];
  mattes: Array<{ id: string; label: string }>;
  plateName?: string;
  plateFrames?: number;
  elements: ElementEntry[];
  busy: boolean;
  onGenerate: (options: CharacterOptions) => void;
  onPlace: (element: ElementEntry, relight: boolean) => void;
  /** Paint the actor out of this plate (a clean plate patch) with this actor matte. */
  onHideActor: (matteId: string) => void;
}) {
  const { status, images, videos, mattes } = props;
  const [name, setName] = useState("Character");
  const [imageId, setImageId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [drive, setDrive] = useState("plate");
  const [matteId, setMatteId] = useState("");
  const [steps, setSteps] = useState(10);
  const [seed, setSeed] = useState(42);
  const [area, setArea] = useState(512 * 512);
  const [relight, setRelight] = useState(false);
  const [hideActor, setHideActor] = useState(true);
  const image = images.find((item) => item.id === imageId);
  const matte = mattes.find((item) => item.id === matteId)?.id ?? "";
  const characters = props.elements.filter((element) => element.manifest.source === "wan-animate" || element.manifest.source === "clean-plate");
  const ready = Boolean(status?.ready);
  return <div className="photo-character">
    <p className="photo-note">
      Wan Animate 2 animates a character picture with the motion of a driving clip, on a plain
      background; SAM 2 and BiRefNet cut it out. Driven by this plate, it lands exactly over the
      actor. {status?.licence}
    </p>
    {status && !status.ready && <p className="photo-note warn">{status.detail}</p>}
    <label className="studio-field"><span>Name</span><input value={name} onChange={(event) => setName(event.target.value)} /></label>
    <label className="studio-field"><span>Character picture</span>
      <select value={imageId} onChange={(event) => setImageId(event.target.value)}>
        <option value="">Pick from Project Media…</option>
        {images.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select>
    </label>
    <label className="studio-field"><span>Describe the character</span>
      <textarea rows={2} value={prompt} placeholder="an armoured knight in a red cloak, full body"
        onChange={(event) => setPrompt(event.target.value)} />
    </label>
    <label className="studio-field"><span>Driving clip</span>
      <select value={drive} onChange={(event) => setDrive(event.target.value)}>
        <option value="plate">This plate{props.plateName ? ` (${props.plateName}, ${props.plateFrames ?? "?"} f)` : ""}</option>
        {videos.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select>
    </label>
    {drive === "plate" && <label className="studio-field"><span>Actor matte (SAM 2)</span>
      <select value={matte} onChange={(event) => setMatteId(event.target.value)}>
        <option value="">None: BiRefNet alone</option>
        {mattes.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select>
    </label>}
    {drive === "plate" && !mattes.length && <small className="photo-note">
      Tip: Click-select the actor on the plate (Masks) first; the character is then tracked by SAM 2
      from the actor's position, which holds through spins and crossings.
    </small>}
    <div className="photo-buttons">
      <label className="studio-field"><span>Steps</span><input type="number" min={1} max={50} value={steps}
        onChange={(event) => setSteps(Math.max(1, Math.min(50, Number(event.target.value) || 10)))} /></label>
      <label className="studio-field"><span>Seed</span><input type="number" value={seed}
        onChange={(event) => setSeed(Math.max(0, Math.floor(Number(event.target.value) || 0)))} /></label>
      <label className="studio-field"><span>Size</span>
      <select value={area} onChange={(event) => setArea(Number(event.target.value))}>
          <option value={384 * 384}>384² (fast)</option>
          <option value={512 * 512}>512² (16 GB preset)</option>
          <option value={640 * 640}>640² (slow)</option>
        </select>
      </label>
    </div>
    <label className="character-check"><input type="checkbox" checked={relight} onChange={(event) => setRelight(event.target.checked)} />
      Add a Relight step to match the plate</label>
    {drive === "plate" && matte && <label className="character-check"><input type="checkbox" checked={hideActor} onChange={(event) => setHideActor(event.target.checked)} />
      Paint the actor out around the character (camera must not move)</label>}
    <button className="studio-run" type="button" disabled={!ready || props.busy || !image || !name.trim()}
      title={!ready ? status?.detail : !image ? "Pick a character picture" : undefined}
      onClick={() => props.onGenerate({ name: name.trim(), imageId, prompt, drive, matteId: matte, steps, seed, area, relight, hideActor: hideActor && drive === "plate" && Boolean(matte) })}>
      <UserRound size={15} /> Animate and insert
    </button>
    {drive === "plate" && matte && <button type="button" disabled={props.busy}
      title="Build the background behind the actor from the frames around this shot and lay it over the plate where the actor is (locked-off camera only). For a character already placed."
      onClick={() => props.onHideActor(matte)}>
      <Eraser size={13} /> Paint the actor out
    </button>}
    {characters.length > 0 && <div className="photo-nodes">
      {characters.map((element) => <div key={element.manifest.id} className="photo-node">
        <button className="photo-node-main" type="button" disabled={props.busy} onClick={() => props.onPlace(element, relight)}
          title="Add over the plate (aligned when this plate drove it)">
          <i>{element.manifest.source === "clean-plate" ? <Eraser size={12} /> : <UserRound size={12} />}</i>
          <span><strong>{element.manifest.name}</strong>
            <small>{element.manifest.frameCount} f · {element.manifest.width}×{element.manifest.height} · {element.manifest.source === "clean-plate" ? "clean plate · " : ""}place</small></span>
        </button>
      </div>)}
    </div>}
  </div>;
}

export type SheetOptions = { layerId?: string; first: number; last: number; step: number; scale: number; padding: number; trim: boolean; loop: boolean };

/** VFX Phase 5: spritesheet export (atlas PNG + Godot 4 SpriteFrames .tres + JSON), beside Render VFX. */
export function SpritesheetPanel(props: {
  layers: Array<{ id: string; name: string }>;
  frames: number;
  busy: boolean;
  onExport: (options: SheetOptions) => void;
}) {
  const [layerId, setLayerId] = useState("");
  const [first, setFirst] = useState(0);
  const [last, setLast] = useState<number>();
  const [step, setStep] = useState(1);
  const [scale, setScale] = useState(0.5);
  const [trim, setTrim] = useState(true);
  const [loop, setLoop] = useState(true);
  const end = Math.min(props.frames - 1, last ?? props.frames - 1);
  const count = Math.max(0, Math.floor((end - first) / Math.max(1, step)) + 1);
  const number = (value: string, fallback: number) => Number.isFinite(Number(value)) && value !== "" ? Math.floor(Number(value)) : fallback;
  return <div className="photo-character">
    <label className="studio-field"><span>Source</span>
      <select value={layerId} onChange={(event) => setLayerId(event.target.value)}>
        <option value="">Whole composition</option>
        {props.layers.map((layer) => <option key={layer.id} value={layer.id}>{layer.name} alone</option>)}
      </select>
    </label>
    <div className="photo-buttons">
      <label className="studio-field"><span>From</span><input type="number" min={0} max={props.frames - 1} value={first}
        onChange={(event) => setFirst(Math.max(0, Math.min(props.frames - 1, number(event.target.value, 0))))} /></label>
      <label className="studio-field"><span>To</span><input type="number" min={0} max={props.frames - 1} value={end}
        onChange={(event) => setLast(Math.max(0, Math.min(props.frames - 1, number(event.target.value, props.frames - 1))))} /></label>
      <label className="studio-field"><span>Every</span><input type="number" min={1} value={step}
        onChange={(event) => setStep(Math.max(1, number(event.target.value, 1)))} /></label>
      <label className="studio-field"><span>Scale</span>
        <select value={scale} onChange={(event) => setScale(Number(event.target.value))}>
          {[0.25, 0.5, 0.75, 1].map((value) => <option key={value} value={value}>{value * 100}%</option>)}
        </select>
      </label>
    </div>
    <label className="character-check"><input type="checkbox" checked={trim} onChange={(event) => setTrim(event.target.checked)} />
      Trim to the frames' alpha (one box for all, so the pivot holds)</label>
    <label className="character-check"><input type="checkbox" checked={loop} onChange={(event) => setLoop(event.target.checked)} />
      Loop in Godot</label>
    <button type="button" disabled={props.busy || count < 1}
      onClick={() => props.onExport({ layerId: layerId || undefined, first, last: end, step, scale, padding: 2, trim, loop })}>
      Export spritesheet · {count} frame{count === 1 ? "" : "s"}
    </button>
    <small className="photo-note">Saved to exports\spritesheets\ as an atlas PNG, a Godot 4 SpriteFrames .tres and a JSON sidecar.</small>
  </div>;
}
