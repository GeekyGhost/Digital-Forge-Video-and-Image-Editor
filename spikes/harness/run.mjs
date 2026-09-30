// Drives the spike page in headless Chromium against server.mjs and prints the results JSON.
//   node spikes/harness/run.mjs [url] [spikes]
// Uses the globally installed playwright (container only). SwiftShader provides WebGL2, so GPU
// numbers here are correctness checks, not performance.
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH ?? "/opt/node22/lib/node_modules/playwright");
const [url = "http://localhost:4173/spikes/index.html", which = "1"] = process.argv.slice(2);

const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=swiftshader", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage();
page.on("console", (message) => { if (message.text().startsWith("[spike]")) console.error(message.text()); });
page.on("pageerror", (error) => console.error("PAGE ERROR", error));
await page.goto(`${url}?backend=http&auto=${which}${process.env.SPIKE_QUERY ? `&${process.env.SPIKE_QUERY}` : ""}`);
await page.waitForFunction(() => (window).__spikeResults, null, { timeout: 30 * 60 * 1000, polling: 1000 });
const results = await page.evaluate(() => (window).__spikeResults);
console.log(JSON.stringify(results, null, 2));
await browser.close();
