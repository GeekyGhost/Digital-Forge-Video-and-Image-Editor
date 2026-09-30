import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";

/**
 * Click-to-rename label, used by every place that shows a media name: the Image Studio media grid,
 * the layer list and the Video Editor's media bin.
 *
 * Editing is controlled by the parent rather than held here, for two reasons. Only one field in a
 * list should be open at a time, and the media bin opens this one from a context menu rather than
 * from a click on the label -- a tile there already uses double-click to add to the timeline.
 *
 * The rendered element is a plain `<strong>` until asked to edit, so a grid of a hundred tiles
 * costs a hundred spans and no inputs. Commit on Enter or blur, abandon on Escape. The app's global
 * shortcut handler ignores events from inputs, so typing "s" here cannot split a clip -- but the
 * keydown is stopped anyway so a future handler cannot reintroduce that.
 */
export function RenameField({ value, editing, onEditing, onCommit, className, placeholder }: {
  value: string;
  editing: boolean;
  onEditing: (editing: boolean) => void;
  onCommit: (next: string) => void;
  className?: string;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState(value);
  const input = useRef<HTMLInputElement>(null);
  // Escape leaves the draft behind; reseeding on every open is what makes cancelling actually cancel.
  useEffect(() => {
    if (!editing) return;
    setDraft(value);
    const element = input.current;
    if (!element) return;
    element.focus();
    // Selecting the stem rather than the whole name means typing replaces "zit-a1b2c3" and keeps
    // ".png", which is what people want when the extension is the only part worth keeping.
    const stem = value.lastIndexOf(".");
    element.setSelectionRange(0, stem > 0 ? stem : value.length);
  }, [editing, value]);

  if (!editing) {
    return <strong className={className} title={value}>{value}</strong>;
  }

  const commit = () => { onCommit(draft); onEditing(false); };
  const onKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    event.stopPropagation();
    if (event.key === "Enter") { event.preventDefault(); commit(); }
    else if (event.key === "Escape") { event.preventDefault(); onEditing(false); }
  };

  return <input
    ref={input}
    className={`rename-field${className ? " " + className : ""}`}
    value={draft}
    placeholder={placeholder}
    spellCheck={false}
    aria-label="Rename"
    onChange={(event) => setDraft(event.target.value)}
    onKeyDown={onKeyDown}
    onBlur={commit}
    onClick={(event) => event.stopPropagation()}
    onDoubleClick={(event) => event.stopPropagation()}
    onPointerDown={(event) => event.stopPropagation()}
  />;
}
