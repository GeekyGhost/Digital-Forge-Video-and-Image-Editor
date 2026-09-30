/**
 * Which typefaces this machine actually has.
 *
 * Two routes, best first. Chromium's Local Font Access API returns the real list, everything
 * installed, with no guessing -- but it is permissioned and a webview may simply refuse. So the
 * fallback is measurement: render a string in a candidate family and in a known generic, and if the
 * widths differ the family resolved to something real. That is exact for a name that is present and
 * silent about names nobody has, which is the right way round: the list below is long, and the ones
 * a given machine lacks quietly disappear rather than being offered and then not used.
 *
 * Nothing is bundled. Shipping typefaces means shipping their licences, and the machine already has
 * hundreds -- this finds them instead.
 */

export type FontGroup = { name: string; families: string[] };

/**
 * Names worth asking about.
 *
 * Weighted towards what Windows and Office install, since that is where this runs, with the common
 * Adobe and Google families people add on top. Ordered inside each group by how likely they are to
 * be the one someone wants.
 */
const CANDIDATES: FontGroup[] = [
  { name: "Sans", families: [
    "Inter", "Segoe UI", "Segoe UI Variable", "Arial", "Arial Black", "Arial Narrow", "Helvetica",
    "Helvetica Neue", "Calibri", "Candara", "Corbel", "Verdana", "Tahoma", "Trebuchet MS",
    "Franklin Gothic Book", "Franklin Gothic Medium", "Century Gothic", "Gill Sans MT", "Futura",
    "Avenir", "Avenir Next", "Lato", "Open Sans", "Roboto", "Roboto Condensed", "Montserrat",
    "Poppins", "Nunito", "Nunito Sans", "Raleway", "Source Sans Pro", "Source Sans 3", "Work Sans",
    "Barlow", "Rubik", "Manrope", "Outfit", "DM Sans", "Karla", "Mulish", "Oswald", "Bebas Neue",
    "Anton", "Archivo", "Archivo Black", "Public Sans", "Noto Sans", "PT Sans", "Ubuntu", "Cabin",
    "Fira Sans", "Titillium Web", "Exo 2", "Quicksand", "Josefin Sans", "Comfortaa", "Jost",
    "Space Grotesk", "Sora", "Figtree", "Plus Jakarta Sans", "Lexend", "Urbanist", "Bahnschrift",
    "Malgun Gothic", "Microsoft JhengHei", "Microsoft YaHei", "Yu Gothic", "Meiryo",
  ] },
  { name: "Serif", families: [
    "Times New Roman", "Georgia", "Garamond", "EB Garamond", "Cambria", "Constantia", "Palatino",
    "Palatino Linotype", "Book Antiqua", "Bookman Old Style", "Baskerville", "Baskerville Old Face",
    "Century Schoolbook", "Sitka Text", "Sitka Heading", "Cormorant Garamond", "Playfair Display",
    "Merriweather", "Lora", "PT Serif", "Noto Serif", "Source Serif Pro", "Crimson Text",
    "Libre Baskerville", "Spectral", "Bitter", "Vollkorn", "Cardo", "Neuton", "Arvo", "Gelasio",
    "Minion Pro", "Adobe Garamond Pro", "Perpetua", "Rockwell", "Bodoni MT", "Didot",
  ] },
  { name: "Display and slab", families: [
    "Impact", "Haettenschweiler", "Agency FB", "Bank Gothic", "Copperplate Gothic Bold",
    "Cooper Black", "Stencil", "Showcard Gothic", "Broadway", "Britannic Bold", "Eras Bold ITC",
    "Bauhaus 93", "Berlin Sans FB", "Harlow Solid Italic", "Jokerman", "Ravie", "Snap ITC",
    "Alfa Slab One", "Bungee", "Fredoka", "Fredoka One", "Righteous", "Russo One", "Titan One",
    "Passion One", "Bowlby One SC", "Ultra", "Chewy", "Lilita One", "Staatliches", "Monoton",
    "Black Ops One", "Press Start 2P", "Silkscreen", "Special Elite", "Rye", "Alegreya Sans SC",
  ] },
  { name: "Comic and cartoon", families: [
    "Comic Sans MS", "Comic Neue", "Bangers", "Luckiest Guy", "Boogaloo", "Chelsea Market",
    "Patrick Hand", "Gochi Hand", "Schoolbell", "Coming Soon", "Architects Daughter", "Rock Salt",
    "Permanent Marker", "Shadows Into Light", "Amatic SC", "Gloria Hallelujah", "Indie Flower",
    "Kalam", "Cabin Sketch", "Sniglet", "Baloo 2", "Modak", "Concert One", "Kranky", "Creepster",
    "Nosifer", "Freckle Face", "Faster One", "Sedgwick Ave", "Life Savers", "Bubblegum Sans",
  ] },
  { name: "Script and hand", families: [
    "Brush Script MT", "Segoe Script", "Segoe Print", "Ink Free", "Lucida Handwriting",
    "Bradley Hand ITC", "Freestyle Script", "Kunstler Script", "Edwardian Script ITC",
    "French Script MT", "Palace Script MT", "Vivaldi", "Mistral", "Pristina", "Rage Italic",
    "Script MT Bold", "Dancing Script", "Great Vibes", "Pacifico", "Satisfy", "Sacramento",
    "Parisienne", "Allura", "Tangerine", "Yellowtail", "Lobster", "Lobster Two", "Courgette",
    "Caveat", "Kaushan Script", "Marck Script", "Homemade Apple",
  ] },
  { name: "Monospace", families: [
    "DM Mono", "Cascadia Code", "Cascadia Mono", "Consolas", "Courier New", "Lucida Console",
    "JetBrains Mono", "Fira Code", "Source Code Pro", "IBM Plex Mono", "Roboto Mono",
    "Space Mono", "Inconsolata", "Ubuntu Mono", "Anonymous Pro", "Share Tech Mono", "VT323",
    "Overpass Mono", "Red Hat Mono", "Courier Prime", "OCR A Extended",
  ] },
];

/** Every name this build knows to ask about, in the order it asks. */
export const CANDIDATE_GROUPS = CANDIDATES;

const PROBE = "MWAmwil@#0Ogq—1Il";
const GENERICS = ["monospace", "serif", "sans-serif"] as const;

/**
 * Does this machine resolve `family` to something of its own?
 *
 * Measured against all three generics, because a family that happens to be metrically identical to
 * one of them -- Arial and Helvetica on some machines, Liberation clones on others -- would look
 * absent if only one was checked.
 */
function present(context: CanvasRenderingContext2D, family: string, base: number[]): boolean {
  for (let index = 0; index < GENERICS.length; index++) {
    context.font = `72px "${family}", ${GENERICS[index]}`;
    if (Math.abs(context.measureText(PROBE).width - base[index]) > 0.5) return true;
  }
  return false;
}

function byMeasurement(): FontGroup[] {
  const context = window.document.createElement("canvas").getContext("2d");
  if (!context) return CANDIDATES;
  const base = GENERICS.map((generic) => {
    context.font = `72px ${generic}`;
    return context.measureText(PROBE).width;
  });
  return CANDIDATES
    .map((group) => ({
      name: group.name,
      families: group.families.filter((family) => present(context, family, base)),
    }))
    .filter((group) => group.families.length > 0);
}

/**
 * Sort a full system list into the same groups.
 *
 * By name, which is a heuristic and openly one: a typeface's category is not in its metadata, and
 * the alternative is one flat list of several hundred names in installation order.
 */
function classify(families: string[]): FontGroup[] {
  const groups: Record<string, string[]> = {
    Sans: [], Serif: [], "Display and slab": [], "Comic and cartoon": [],
    "Script and hand": [], Monospace: [], Other: [],
  };
  const known = new Map<string, string>();
  for (const group of CANDIDATES) {
    for (const family of group.families) known.set(family.toLowerCase(), group.name);
  }
  for (const family of families) {
    const lower = family.toLowerCase();
    const named = known.get(lower);
    if (named) { groups[named].push(family); continue; }
    if (/mono|code|console|courier|terminal|typewriter/.test(lower)) groups.Monospace.push(family);
    else if (/comic|cartoon|bang|toon|kid|doodle/.test(lower)) groups["Comic and cartoon"].push(family);
    else if (/script|hand|brush|marker|pen|calligra|signature|casual|print/.test(lower)) groups["Script and hand"].push(family);
    else if (/display|poster|black|stencil|slab|headline|title|deco/.test(lower)) groups["Display and slab"].push(family);
    else if (/serif|times|georgia|garamond|roman|book|minion|caslon|didot|bodoni/.test(lower) && !/sans/.test(lower)) groups.Serif.push(family);
    else if (/sans|gothic|grotesk|helvetica|arial|segoe|calibri|verdana|tahoma/.test(lower)) groups.Sans.push(family);
    else groups.Other.push(family);
  }
  return Object.entries(groups)
    .map(([name, list]) => ({ name, families: [...new Set(list)].sort((a, b) => a.localeCompare(b)) }))
    .filter((group) => group.families.length > 0);
}

type LocalFont = { family: string };
type WithLocalFonts = Window & { queryLocalFonts?: () => Promise<LocalFont[]> };

let cached: Promise<FontGroup[]> | undefined;

export function availableFonts(): Promise<FontGroup[]> {
  cached ??= (async () => {
    const query = (window as WithLocalFonts).queryLocalFonts;
    if (typeof query === "function") {
      try {
        const found = await query.call(window);
        const families = [...new Set(found.map((font) => font.family))];
        if (families.length > 8) return classify(families);
      } catch {
        // Refused, or no permission. Measurement still finds several hundred, so this is a quiet
        // downgrade rather than something to put in front of the user.
      }
    }
    return byMeasurement();
  })();
  return cached;
}

/** Ask again — after the user installs something, or drops a font into the project. */
export function forgetFonts(): void {
  cached = undefined;
}
