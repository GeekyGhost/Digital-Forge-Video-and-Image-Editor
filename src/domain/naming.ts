/**
 * Readable names for generated media.
 *
 * Every generator used to name its output with a timestamp and a UUID -- `zit-20260903-185500-
 * 3f9a2b1c….png`, or `H3-image-1757012345678.mp4` from the save dialog. Those names are unique,
 * sortable and completely opaque: a Project Media grid full of them tells you nothing about which
 * picture is which. The app knows the prompt at the moment it writes the file and throws it away.
 * These helpers put the prompt back into the name.
 *
 * The slug is deliberately lossy. Five content words is enough to recognise a picture in a grid,
 * and anything longer stops fitting in a media tile, so this trades completeness for scanability.
 * Uniqueness is not this module's job: the Rust side appends a counter when a name is already
 * taken, and the save dialog lets the user decide about overwriting.
 */

/**
 * Words that cost characters without helping you recognise the file. Dropped wherever they appear,
 * not just at the front, so "remove the car from the driveway" slugs to "remove-car-driveway".
 */
const NOISE = new Set([
  "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or", "by",
  "is", "are", "be", "this", "that", "it", "its", "as", "from", "into", "over", "under",
  "very", "highly", "extremely", "detailed", "quality", "masterpiece",
]);

/**
 * Lowercase, hyphenated, ASCII-only fragment of `text`.
 *
 * `maxLength` is checked against the joined result rather than each word, because the thing that
 * has to fit is the finished name, and one long word should not be silently truncated into
 * something unrecognisable -- it is dropped instead, unless it would leave the slug empty.
 */
export function slugify(text: string, maxWords = 5, maxLength = 44): string {
  const words = text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return "";
  // A prompt made entirely of noise words still deserves a name, so fall back to the raw words.
  const meaningful = words.filter((word) => !NOISE.has(word));
  const source = meaningful.length ? meaningful : words;
  const kept: string[] = [];
  for (const word of source) {
    if (kept.length >= maxWords) break;
    if ([...kept, word].join("-").length > maxLength) break;
    kept.push(word);
  }
  return kept.length ? kept.join("-") : source[0].slice(0, maxLength);
}

/** `20260903-185500`. Only used when there is no prompt to name the file after. */
export function timeStamp(when = new Date()): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${when.getFullYear()}${pad(when.getMonth() + 1)}${pad(when.getDate())}`
    + `-${pad(when.getHours())}${pad(when.getMinutes())}${pad(when.getSeconds())}`;
}

/**
 * A default filename for a save dialog: `H3-image-rusted-lighthouse-dusk.mp4`.
 *
 * The prefix stays in front so a folder still groups by what made the file; the slug carries the
 * meaning. With no usable hint this degrades to the old timestamp behaviour rather than producing
 * a name that collides on every run.
 */
export function suggestedFileName(prefix: string, hint: string, extension: string): string {
  const slug = slugify(hint);
  return `${prefix}-${slug || timeStamp()}.${extension}`;
}

/**
 * The name to offer when the user asks to rename an existing asset back to something meaningful.
 * Generated assets carry their prompt in `origin`, so this recovers a good name for files that
 * were created before any of this existed.
 */
export function nameFromPrompt(prompt: string | undefined, extension: string): string | undefined {
  const slug = prompt ? slugify(prompt) : "";
  return slug ? `${slug}.${extension}` : undefined;
}

/**
 * True for the machine-generated names the app used to write: a prefix, a timestamp and a UUID.
 *
 * Used to decide whether to offer "name from prompt" on an asset. Files made before prompt-derived
 * naming existed are exactly the ones worth offering it for, and a name the user has already fixed
 * should not keep nagging them, so the offer disappears the moment the name stops looking like this.
 */
export function looksGenerated(name: string): boolean {
  return /-\d{8}-\d{6}-[0-9a-f]{6,}/i.test(name) || /[0-9a-f]{16,}/i.test(name);
}
