export type TransitionCategory =
  | "Essentials"
  | "Wipes"
  | "Slides & Motion"
  | "Shapes & Iris"
  | "Slices & Wind"
  | "Stylized";

export type TransitionDefinition = {
  id: string;
  name: string;
  category: TransitionCategory;
  description: string;
  defaultDuration: number;
  minDuration: number;
  keywords: string[];
};

const transition = (
  id: string,
  name: string,
  category: TransitionCategory,
  description: string,
  defaultDuration = 0.5,
  minDuration = 0.08,
  keywords: string[] = [],
): TransitionDefinition => ({ id, name, category, description, defaultDuration, minDuration, keywords });

export const TRANSITIONS: TransitionDefinition[] = [
  transition("none", "Hard Cut", "Essentials", "Immediate edit with no visual overlap.", 0),
  transition("fade", "Cross Dissolve", "Essentials", "Smoothly blends the outgoing and incoming shots."),
  transition("dissolve", "Film Dissolve", "Essentials", "A textured dissolve with a more organic blend."),
  transition("fadeblack", "Dip to Black", "Essentials", "Fades the outgoing shot through black."),
  transition("fadewhite", "Dip to White", "Essentials", "Fades the outgoing shot through white."),
  transition("fadegrays", "Luma Gray Fade", "Essentials", "Desaturates through gray while blending."),
  transition("fadefast", "Fast Dissolve", "Essentials", "A dissolve weighted toward a quick handoff.", 0.3),
  transition("fadeslow", "Slow Dissolve", "Essentials", "A dissolve with a softer, lingering handoff.", 0.8),
  transition("hblur", "Blur Dissolve", "Essentials", "Adds horizontal blur through the edit."),
  transition("pixelize", "Pixelate", "Essentials", "Breaks both shots into pixels during the transition."),

  transition("wipeleft", "Wipe Left", "Wipes", "Reveals the incoming shot from right to left."),
  transition("wiperight", "Wipe Right", "Wipes", "Reveals the incoming shot from left to right."),
  transition("wipeup", "Wipe Up", "Wipes", "Reveals the incoming shot from bottom to top."),
  transition("wipedown", "Wipe Down", "Wipes", "Reveals the incoming shot from top to bottom."),
  transition("wipetl", "Corner Wipe · Top Left", "Wipes", "Diagonal corner reveal toward the top-left."),
  transition("wipetr", "Corner Wipe · Top Right", "Wipes", "Diagonal corner reveal toward the top-right."),
  transition("wipebl", "Corner Wipe · Bottom Left", "Wipes", "Diagonal corner reveal toward the bottom-left."),
  transition("wipebr", "Corner Wipe · Bottom Right", "Wipes", "Diagonal corner reveal toward the bottom-right."),
  transition("diagtl", "Diagonal · Top Left", "Wipes", "Straight diagonal transition toward the top-left."),
  transition("diagtr", "Diagonal · Top Right", "Wipes", "Straight diagonal transition toward the top-right."),
  transition("diagbl", "Diagonal · Bottom Left", "Wipes", "Straight diagonal transition toward the bottom-left."),
  transition("diagbr", "Diagonal · Bottom Right", "Wipes", "Straight diagonal transition toward the bottom-right."),

  transition("slideleft", "Push Left", "Slides & Motion", "Pushes the outgoing shot left as the next enters."),
  transition("slideright", "Push Right", "Slides & Motion", "Pushes the outgoing shot right as the next enters."),
  transition("slideup", "Push Up", "Slides & Motion", "Pushes the outgoing shot upward."),
  transition("slidedown", "Push Down", "Slides & Motion", "Pushes the outgoing shot downward."),
  transition("smoothleft", "Smooth Slide Left", "Slides & Motion", "A softened horizontal slide to the left."),
  transition("smoothright", "Smooth Slide Right", "Slides & Motion", "A softened horizontal slide to the right."),
  transition("smoothup", "Smooth Slide Up", "Slides & Motion", "A softened vertical slide upward."),
  transition("smoothdown", "Smooth Slide Down", "Slides & Motion", "A softened vertical slide downward."),
  transition("coverleft", "Cover Left", "Slides & Motion", "The incoming shot covers the outgoing shot from the right."),
  transition("coverright", "Cover Right", "Slides & Motion", "The incoming shot covers the outgoing shot from the left."),
  transition("coverup", "Cover Up", "Slides & Motion", "The incoming shot covers from below."),
  transition("coverdown", "Cover Down", "Slides & Motion", "The incoming shot covers from above."),
  transition("revealleft", "Reveal Left", "Slides & Motion", "The outgoing shot moves left to reveal the next."),
  transition("revealright", "Reveal Right", "Slides & Motion", "The outgoing shot moves right to reveal the next."),
  transition("revealup", "Reveal Up", "Slides & Motion", "The outgoing shot moves up to reveal the next."),
  transition("revealdown", "Reveal Down", "Slides & Motion", "The outgoing shot moves down to reveal the next."),

  transition("circlecrop", "Circle Crop", "Shapes & Iris", "A circular crop grows between shots."),
  transition("rectcrop", "Rectangle Crop", "Shapes & Iris", "A rectangular crop reveals the incoming shot."),
  transition("circleopen", "Iris Open", "Shapes & Iris", "A circular iris opens onto the next shot."),
  transition("circleclose", "Iris Close", "Shapes & Iris", "A circular iris closes into the next shot."),
  transition("radial", "Clock Wipe", "Shapes & Iris", "A radial sweep rotates around the frame."),
  transition("distance", "Distance Field", "Shapes & Iris", "A soft geometric distance-field reveal."),
  transition("vertopen", "Vertical Open", "Shapes & Iris", "Opens vertically from the center."),
  transition("vertclose", "Vertical Close", "Shapes & Iris", "Closes vertically toward the center."),
  transition("horzopen", "Horizontal Open", "Shapes & Iris", "Opens horizontally from the center."),
  transition("horzclose", "Horizontal Close", "Shapes & Iris", "Closes horizontally toward the center."),

  transition("hlslice", "Slice · Left", "Slices & Wind", "Horizontal slices move toward the left."),
  transition("hrslice", "Slice · Right", "Slices & Wind", "Horizontal slices move toward the right."),
  transition("vuslice", "Slice · Up", "Slices & Wind", "Vertical slices move upward."),
  transition("vdslice", "Slice · Down", "Slices & Wind", "Vertical slices move downward."),
  transition("hlwind", "Wind · Left", "Slices & Wind", "Staggered bands sweep toward the left."),
  transition("hrwind", "Wind · Right", "Slices & Wind", "Staggered bands sweep toward the right."),
  transition("vuwind", "Wind · Up", "Slices & Wind", "Staggered bands sweep upward."),
  transition("vdwind", "Wind · Down", "Slices & Wind", "Staggered bands sweep downward."),

  transition("squeezeh", "Squeeze Horizontal", "Stylized", "Compresses the frame horizontally through the cut."),
  transition("zoomin", "Zoom In", "Stylized", "Zooms into the incoming shot."),
];

export const TRANSITION_CATEGORIES: TransitionCategory[] = [
  "Essentials", "Wipes", "Slides & Motion", "Shapes & Iris", "Slices & Wind", "Stylized",
];

export const transitionById = (id?: string): TransitionDefinition =>
  TRANSITIONS.find((item) => item.id === id) ?? TRANSITIONS[0];

export const isTransition = (id?: string): boolean => Boolean(id) && id !== "none";

export type ResolvedTransition = { id: string; duration: number };

export const NO_TRANSITION: ResolvedTransition = { id: "none", duration: 0 };

type OutgoingSide = { transitionOut?: string; transitionOutDuration?: number } | undefined;
type IncomingSide = { transitionIn?: string; transitionDuration?: number } | undefined;

/**
 * A cut between two clips carries at most one rendered transition. The incoming clip's
 * "in" choice wins when both sides are set, and the longer of the two durations is used so
 * neither side is truncated. Returns `none` when the cut is hard on both sides.
 */
export const resolveJunction = (outgoing: OutgoingSide, incoming: IncomingSide): ResolvedTransition => {
  const outId = isTransition(outgoing?.transitionOut) ? outgoing!.transitionOut! : undefined;
  const inId = isTransition(incoming?.transitionIn) ? incoming!.transitionIn! : undefined;
  if (!outId && !inId) return NO_TRANSITION;
  const id = inId ?? outId!;
  const definition = transitionById(id);
  const durations = [
    outId ? outgoing?.transitionOutDuration ?? definition.defaultDuration : 0,
    inId ? incoming?.transitionDuration ?? definition.defaultDuration : 0,
  ];
  return { id, duration: Math.max(definition.minDuration, ...durations) };
};

/** The head of a track fades in from black; the tail fades out to black. */
export const resolveEdge = (id: string | undefined, duration: number | undefined): ResolvedTransition => {
  if (!isTransition(id)) return NO_TRANSITION;
  const definition = transitionById(id);
  return { id: definition.id, duration: Math.max(definition.minDuration, duration ?? definition.defaultDuration) };
};

export const filterTransitions = (query: string): TransitionDefinition[] => {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return TRANSITIONS;
  return TRANSITIONS.filter((item) =>
    [item.name, item.id, item.category, item.description, ...item.keywords]
      .some((value) => value.toLowerCase().includes(normalized)),
  );
};
