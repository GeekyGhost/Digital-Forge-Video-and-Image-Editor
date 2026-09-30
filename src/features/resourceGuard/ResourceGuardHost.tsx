import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { AlertTriangle } from "lucide-react";
import { useEffect, useState } from "react";
import {
  commitLine, formatGb, holderTag, shortOnCommit, vramLine,
  type PreflightDecision, type PreflightPrompt,
} from "../../domain/resourceGuard";
import "./resourceGuard.css";

/**
 * The resource guard's warning (VFX Phase 0, task 0.1). Mounted once beside <App/>.
 *
 * Rust holds the AI gate and waits while this is open, so no other job can start underneath it.
 * Every choice is the user's: nothing here closes another program. "Release this app's worker"
 * only stops a model worker this app itself started.
 */
export default function ResourceGuardHost() {
  const [queue, setQueue] = useState<PreflightPrompt[]>([]);
  const [sending, setSending] = useState(false);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void listen<PreflightPrompt>("ai-preflight", (event) => {
      setQueue((items) => [...items, event.payload]);
    }).then((stop) => {
      if (disposed) { stop(); return; }
      unlisten = stop;
      void invoke("ai_preflight_ready", { ready: true }).catch(() => undefined);
    }).catch(() => undefined);
    return () => {
      disposed = true;
      unlisten?.();
      void invoke("ai_preflight_ready", { ready: false }).catch(() => undefined);
    };
  }, []);

  const prompt = queue[0];
  if (!prompt) return null;
  const { verdict } = prompt;

  const decide = async (decision: PreflightDecision) => {
    setSending(true);
    try {
      await invoke("ai_preflight_decide", { id: prompt.id, decision });
    } catch {
      // The job already stopped waiting (timeout); there is nothing left to answer.
    } finally {
      setSending(false);
      setQueue((items) => items.slice(1));
    }
  };

  const vramShort = verdict.vramFreeGb !== null && verdict.vramFreeGb < verdict.vramNeededGb && verdict.commitReclaimableGb === 0;
  return <div className="modal-backdrop">
    <section className="settings-dialog resource-guard-dialog" role="alertdialog" aria-modal="true" aria-labelledby="resource-guard-title">
      <div className="dialog-heading"><div>
        <strong id="resource-guard-title"><AlertTriangle size={17} /> Not enough free memory for {verdict.label}</strong>
        <small>Starting anyway may end in a crash (0xC0000005) rather than a clean out-of-memory error.</small>
      </div></div>

      <ul className="resource-guard-meters">
        <li className={shortOnCommit(verdict) ? "short" : ""}><span>Commit</span><b>{commitLine(verdict)}</b></li>
        <li className={vramShort ? "short" : ""}><span>VRAM</span><b>{vramLine(verdict)}</b></li>
        <li><span>RAM</span><b>{formatGb(verdict.ramFreeGb)} free of {formatGb(verdict.ramTotalGb)}</b></li>
      </ul>

      {verdict.holders.length > 0 && <table className="resource-guard-holders">
        <thead><tr><th>Process</th><th>PID</th><th>Commit</th><th /></tr></thead>
        <tbody>{verdict.holders.map((holder) => <tr key={holder.pid} className={holder.kind}>
          <td>{holder.name}</td><td>{holder.pid}</td><td>{formatGb(holder.privateGb)}</td><td>{holderTag(holder)}</td>
        </tr>)}</tbody>
      </table>}

      <small className="resource-guard-footnote">
        Nothing has been closed. Requirement: {verdict.requirementSource}. Checked in {verdict.measureMs.toFixed(0)} ms.
      </small>

      <div className="dialog-actions">
        <button autoFocus disabled={sending} onClick={() => void decide("cancel")}>Cancel</button>
        {verdict.hasReleasableAppWorker && <button disabled={sending} onClick={() => void decide("release-app-workers")}
          title="Stops the resident LTX retake worker this app started. Nothing else is touched.">Release this app's worker</button>}
        <button className="primary" disabled={sending} onClick={() => void decide("continue")}>Continue anyway</button>
      </div>
      {queue.length > 1 && <small className="resource-guard-footnote">{queue.length - 1} more waiting.</small>}
    </section>
  </div>;
}
