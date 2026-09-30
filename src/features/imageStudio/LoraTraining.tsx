import { convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { AlertTriangle, Check, FolderOpen, Pause, Play, RefreshCw, Sparkles, Square, TestTube2, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  recommendedTrainingSteps, suggestedTriggerToken, trainingReadiness, type LoraProfile,
} from "../../domain/loraTraining";
import {
  cancelZImageLoraTraining, captionZImageLoraImages, inspectZImageLora, pauseZImageLoraTraining,
  resumeZImageLoraTraining, startZImageLoraTraining, zImageLoraTrainingStatus,
  type AiRuntimeStatus, type CaptionedLoraImage, type LoraTrainingJob, type ZImageLoraInfo,
} from "../../services/modelService";

const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

export function LoraTraining({ projectId, runtime, setStatus, onTestCheckpoint }: {
  projectId: string; runtime?: AiRuntimeStatus; setStatus: (message: string) => void;
  onTestCheckpoint?: (path: string) => void | Promise<void>;
}) {
  const [profile, setProfile] = useState<LoraProfile>("character");
  const [trainingModel, setTrainingModel] = useState<"base" | "turbo">("turbo");
  const [backend, setBackend] = useState<"qwen" | "gemma">("qwen");
  const [name, setName] = useState("");
  const [trigger, setTrigger] = useState("");
  const [className, setClassName] = useState("person");
  const [paths, setPaths] = useState<string[]>([]);
  const [images, setImages] = useState<CaptionedLoraImage[]>([]);
  const [resolution, setResolution] = useState<512 | 768 | 1024>(1024);
  const [steps, setSteps] = useState(500);
  const [rank, setRank] = useState(16);
  const [alpha, setAlpha] = useState(16);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [seed, setSeed] = useState(42);
  const [checkpointEvery, setCheckpointEvery] = useState(100);
  const [checkpointLimit, setCheckpointLimit] = useState(5);
  const [initialLora, setInitialLora] = useState<ZImageLoraInfo>();
  const [captioning, setCaptioning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [job, setJob] = useState<LoraTrainingJob>();
  const [failure, setFailure] = useState<string>();
  const modelReady = trainingModel === "turbo" ? runtime?.zitImageReady : runtime?.zimageBaseReady;
  const modelName = trainingModel === "turbo" ? "Z-Image-Turbo" : "Z-Image Base";
  const readiness = useMemo(() => trainingReadiness({ name, trigger, className, images }), [name, trigger, className, images]);
  const recommendedTrigger = suggestedTriggerToken(name);
  const triggerDiffersFromRecommendation = Boolean(recommendedTrigger && trigger.trim()
    && trigger.trim().toLocaleLowerCase() !== recommendedTrigger.toLocaleLowerCase());

  useEffect(() => {
    if (!job || !["running", "pausing", "canceling"].includes(job.state)) return;
    const timer = window.setInterval(() => {
      void zImageLoraTrainingStatus(job.id).then(setJob).catch((error) => setFailure(errorText(error)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.state]);

  useEffect(() => {
    const saved = window.localStorage.getItem(`aiv-zimage-training-${projectId}`);
    if (!saved || job) return;
    void zImageLoraTrainingStatus(saved).then((restored) => { setJob(restored); setSteps(restored.total); })
      .catch(() => window.localStorage.removeItem(`aiv-zimage-training-${projectId}`));
  }, [projectId]);

  useEffect(() => {
    if (!job) return;
    const key = `aiv-zimage-training-${projectId}`;
    if (["running", "pausing", "paused", "canceling"].includes(job.state)) window.localStorage.setItem(key, job.id);
    else window.localStorage.removeItem(key);
  }, [job?.id, job?.state, projectId]);

  useEffect(() => {
    if (job?.state === "complete") setStatus(`LoRA ready: ${job.outputPath}`);
    if (job?.state === "failed") setStatus("LoRA training failed");
  }, [job?.state]);

  const chooseImages = async () => {
    const selected = await open({ multiple: true, filters: [{ name: "Training images", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"] }] });
    const chosen = typeof selected === "string" ? [selected] : selected ?? [];
    if (!chosen.length) return;
    const unique = [...new Set(chosen)];
    setPaths(unique); setImages([]); setFailure(undefined);
    setSteps(recommendedTrainingSteps(unique.length, profile));
    setStatus(`Selected ${unique.length} LoRA training images`);
  };

  const caption = async () => {
    if (!paths.length || !trigger.trim() || !className.trim()) return;
    setCaptioning(true); setFailure(undefined); setStatus(`Captioning ${paths.length} images with ${backend === "qwen" ? "Qwen" : "Gemma"}…`);
    try {
      const result = await captionZImageLoraImages({ imagePaths: paths, profile, trigger, className, backend });
      setImages(result.items);
      setStatus(`Captioned ${result.items.length} training images; review them before training`);
    } catch (error) { setFailure(errorText(error)); setStatus("Dataset captioning failed"); }
    finally { setCaptioning(false); }
  };

  const updateCaption = (path: string, caption: string) => setImages((current) => current.map((image) => image.imagePath === path ? { ...image, caption } : image));
  const removeImage = (path: string) => {
    setPaths((current) => current.filter((item) => item !== path));
    setImages((current) => current.filter((item) => item.imagePath !== path));
  };

  const startTraining = async () => {
    if (!readiness.ready) return;
    setStarting(true); setFailure(undefined); setStatus(`Starting isolated ${modelName} LoRA training…`);
    try {
      const started = await startZImageLoraTraining({
        projectId, name, profile, trigger, className, images: readiness.usable,
        resolution, steps, rank, alpha, learningRate, seed, checkpointEvery, checkpointLimit,
        initialLoraPath: initialLora?.folder, trainingModel,
      });
      setJob(started); setStatus("Z-Image LoRA training is running");
    } catch (error) { setFailure(errorText(error)); setStatus("Could not start LoRA training"); }
    finally { setStarting(false); }
  };

  const chooseInitialLora = async () => {
    const root = runtime?.root.replace(/[\\/]+$/, "");
    const selected = await open({
      multiple: false, directory: true,
      defaultPath: root ? `${root}/models/loras/z-image` : undefined,
      title: "Choose an existing Z-Image LoRA folder to continue",
    });
    if (typeof selected !== "string") return;
    try {
      const info = await inspectZImageLora(selected);
      setInitialLora(info);
      if (info.rank) setRank(info.rank);
      if (info.alpha) setAlpha(info.alpha);
      setFailure(undefined);
      setStatus(`New training phase will start from ${info.name}`);
    } catch (error) { setFailure(errorText(error)); }
  };

  const pause = async () => {
    if (!job) return;
    try { setJob(await pauseZImageLoraTraining(job.id)); setStatus("Saving a full checkpoint, then releasing the GPU…"); }
    catch (error) { setFailure(errorText(error)); }
  };

  const resume = async () => {
    if (!job) return;
    setStarting(true); setFailure(undefined);
    try { setJob(await resumeZImageLoraTraining(job.id, steps)); setStatus(`Resuming from checkpoint ${job.step}`); }
    catch (error) { setFailure(errorText(error)); }
    finally { setStarting(false); }
  };

  const cancel = async () => {
    if (!job) return;
    try { setJob(await cancelZImageLoraTraining(job.id)); }
    catch (error) { setFailure(errorText(error)); }
  };

  const changeProfile = (value: LoraProfile) => {
    setProfile(value); setImages([]); setSteps(recommendedTrainingSteps(paths.length, value));
    if (value === "style" && className === "person") setClassName("image");
    if ((value === "subject" || value === "character") && className === "image") setClassName("person");
  };

  const changeName = (value: string) => {
    const previousSuggestion = suggestedTriggerToken(name);
    const nextSuggestion = suggestedTriggerToken(value);
    setName(value);
    if (!trigger || trigger === previousSuggestion) {
      setTrigger(nextSuggestion);
      setImages([]);
    }
  };

  return <section className="lora-training-workspace">
    <header className="lora-training-header">
      <div><strong>Z-Image LoRA Training</strong><span>{trainingModel === "turbo" ? "Caption, review, then train for Turbo's 8-step generation" : "Caption, review, then train against the undistilled Base model"}</span></div>
      <div className="lora-readiness">
        <b className={runtime?.zimageCaptionerReady ? "ready" : "not-ready"}>CAPTIONER {runtime?.zimageCaptionerReady ? "READY" : "MISSING"}</b>
        <b className={runtime?.zimageTrainerReady && modelReady ? "ready" : "not-ready"}>TRAINER {runtime?.zimageTrainerReady && modelReady ? "READY" : "MISSING"}</b>
      </div>
    </header>

    {(!runtime?.zimageTrainerReady || !modelReady) && <div className="lora-setup-note"><AlertTriangle size={18} /><div>
      <strong>Training components are not fully installed.</strong>
      <span>{!runtime?.zimageTrainerReady
        ? <>Run <code>Setup-ZImage-Trainer.bat</code> (it also downloads the Turbo training adapter).</>
        : trainingModel === "turbo"
          ? <>Z-Image-Turbo is missing; install it through Image Studio setup.</>
          : <>Install Base with <code>.venv\Scripts\python.exe scripts\models.py install z-image-base --accept-licenses</code>.</>}</span>
    </div></div>}

    <div className="lora-training-columns">
      <aside className="lora-settings">
        <label><span>Train for</span><select value={trainingModel} onChange={(event) => setTrainingModel(event.target.value as "base" | "turbo")} disabled={Boolean(job && ["running", "pausing", "paused", "canceling"].includes(job.state))}><option value="turbo">Turbo · 8-step Fast mode (training adapter)</option><option value="base">Base · Creative mode, CFG</option></select></label>
        <label><span>LoRA name</span><input value={name} onChange={(event) => changeName(event.target.value)} placeholder="Captain Dicio" /></label>
        <div className="studio-grid">
          <label><span>Profile</span><select value={profile} onChange={(event) => changeProfile(event.target.value as LoraProfile)}><option value="character">Designed character</option><option value="subject">Person identity</option><option value="style">Visual style</option><option value="product">Product / object</option><option value="concept">Clothing / concept</option></select></label>
          <label><span>Caption model</span><select value={backend} onChange={(event) => setBackend(event.target.value as "qwen" | "gemma")}><option value="qwen">Qwen 2511 encoder</option><option value="gemma">Gemma 3 detailed</option></select></label>
        </div>
        <div className="studio-grid">
          <label><span>Trigger token</span><input value={trigger} placeholder="Created from the LoRA name" onChange={(event) => { setTrigger(event.target.value); setImages([]); }} /></label>
          <label><span>Class noun</span><input value={className} onChange={(event) => { setClassName(event.target.value); setImages([]); }} /></label>
        </div>
        {triggerDiffersFromRecommendation && <small className="lora-trigger-warning"><AlertTriangle size={12} /> Prefer the rare token <code>{recommendedTrigger}</code>. Names and ordinary words can keep their base-model meaning and overpower the trained identity.</small>}
        <button onClick={() => void chooseImages()} disabled={captioning || Boolean(job && ["running", "pausing", "canceling"].includes(job.state))}><FolderOpen size={15} /> Choose images</button>
        <button className="studio-run" onClick={() => void caption()} disabled={!runtime?.zimageCaptionerReady || !paths.length || captioning || !trigger.trim() || !className.trim()}>
          {captioning ? <RefreshCw className="spin" size={15} /> : <Sparkles size={15} />} Auto-caption {paths.length || "dataset"}
        </button>
        <small>{profile === "character" ? "Designed character keeps signature appearance bound to the trigger; add several clear close or medium views." : "Qwen is the default. Captions remain editable and training cannot begin until they pass validation."}</small>

        <h3>Training preset</h3>
        <div className="studio-grid">
          <label><span>Resolution</span><select value={resolution} onChange={(event) => setResolution(Number(event.target.value) as 512 | 768 | 1024)}><option value={512}>512 · fastest</option><option value={768}>768 · 16 GB balanced</option><option value={1024}>1024 · identity recommended</option></select></label>
          <label><span>Steps</span><input type="number" min={100} max={10000} value={steps} onChange={(event) => setSteps(Number(event.target.value))} /></label>
          <label><span>Rank</span><input type="number" min={4} max={128} value={rank} onChange={(event) => setRank(Number(event.target.value))} /></label>
          <label><span>Alpha</span><input type="number" min={1} max={256} value={alpha} onChange={(event) => setAlpha(Number(event.target.value))} /></label>
          <label><span>Learning rate</span><input type="number" min={0.000001} max={0.01} step={0.00001} value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} /></label>
          <label><span>Seed</span><input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label>
          <label><span>Checkpoint every</span><input type="number" min={25} max={2000} step={25} value={checkpointEvery} onChange={(event) => setCheckpointEvery(Number(event.target.value))} /></label>
          <label><span>Keep checkpoints</span><input type="number" min={1} max={50} value={checkpointLimit} onChange={(event) => setCheckpointLimit(Number(event.target.value))} /></label>
        </div>
        <div className="lora-continue-source">
          <span>Starting LoRA · optional</span>
          {initialLora ? <div className="zimage-lora-details"><strong>{initialLora.name}</strong><small>{initialLora.rank ? `Rank ${initialLora.rank}` : "External adapter"}{initialLora.alpha ? ` · alpha ${initialLora.alpha}` : ""}</small><button type="button" onClick={() => setInitialLora(undefined)}><X size={13} /> Clear</button></div>
            : <button type="button" onClick={() => void chooseInitialLora()}><FolderOpen size={14} /> Continue from LoRA folder</button>}
          <small>This starts a new phase from adapter weights. For an exact optimizer/scheduler continuation, pause and resume the active job below.</small>
        </div>
        {!readiness.ready && images.length > 0 && <ul className="lora-problems">{readiness.problems.map((problem) => <li key={problem}>{problem}</li>)}</ul>}
        <button className="studio-run" onClick={() => void startTraining()} disabled={!runtime?.zimageTrainerReady || !modelReady || !readiness.ready || starting || Boolean(job && ["running", "pausing", "paused", "canceling"].includes(job.state))}>
          {starting ? <RefreshCw className="spin" size={15} /> : <Check size={15} />} {initialLora ? "Continue as new Z-Image LoRA" : `Train ${modelName} LoRA`}
        </button>
      </aside>

      <div className="lora-dataset-review">
        <div className="lora-review-heading"><strong>Dataset review</strong><span>{images.length ? `${readiness.usable.length} usable / ${images.length} captioned` : `${paths.length} selected`}</span></div>
        <div className="lora-caption-grid">{(images.length ? images : paths.map((path) => ({ imagePath: path } as CaptionedLoraImage))).map((image) => <article key={image.imagePath} className={image.qualityFlags?.length ? "has-warning" : ""}>
          <img src={convertFileSrc(image.imagePath)} alt="LoRA training reference" />
          <button className="lora-remove-image" onClick={() => removeImage(image.imagePath)} title="Remove from dataset"><Trash2 size={13} /></button>
          {image.caption !== undefined ? <>
            <textarea value={image.caption} onChange={(event) => updateCaption(image.imagePath, event.target.value)} rows={4} />
            <div className="lora-image-meta"><span>{image.width} × {image.height}</span><span>{Math.round(image.confidence * 100)}% confidence</span></div>
            {image.tags.length > 0 && <small className="lora-tags">{image.tags.slice(0, 8).join(" · ")}</small>}
            {image.qualityFlags.length > 0 && <small className="lora-flags"><AlertTriangle size={12} /> {image.qualityFlags.join(" · ")}</small>}
          </> : <small>Awaiting caption</small>}
        </article>)}</div>
      </div>
    </div>

    {job && <div className={`lora-job lora-job-${job.state}`}>
      <div><strong>{job.state === "complete" ? "Training complete" : job.state === "failed" ? "Training failed" : job.state === "paused" ? "Training paused" : job.state === "pausing" ? "Saving pause checkpoint" : "Training Z-Image LoRA"}</strong><span>{job.message}</span></div>
      <progress max={1} value={job.progress} /><b>{job.step} / {job.total}</b>
      {job.outputPath && <code>{job.outputPath}</code>}
      {job.checkpoints.length > 0 && <div className="lora-checkpoint-list"><strong>Saved checkpoints</strong>{job.checkpoints.map((checkpoint) => <div key={checkpoint.path}><code>Step {checkpoint.step}</code><button type="button" onClick={() => void onTestCheckpoint?.(checkpoint.path)} disabled={!onTestCheckpoint || !["paused", "complete", "failed", "canceled"].includes(job.state)}><TestTube2 size={13} /> Test in Image Studio</button></div>)}</div>}
      {job.error && <pre>{job.error}</pre>}
      {job.state === "running" && <button onClick={() => void pause()}><Pause size={13} /> Save & pause</button>}
      {job.state === "paused" && <button onClick={() => void resume()} disabled={starting || steps <= job.step}><Play size={13} /> Resume to {steps} total steps</button>}
      {job.state === "paused" && <button onClick={() => void cancel()}><Square size={13} /> End session · keep checkpoints</button>}
      {["running", "pausing", "canceling"].includes(job.state) && <button onClick={() => void cancel()} disabled={job.state === "canceling"}><Square size={13} /> {job.state === "canceling" ? "Stopping…" : "Cancel"}</button>}
    </div>}
    {failure && <pre className="worker-failure">{failure}</pre>}
  </section>;
}
