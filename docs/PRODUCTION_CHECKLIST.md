# SEETHRU — Production Readiness Checklist

Generated 2026-07-15 from a multi-agent audit of the codebase against the project spec.
Severity: **[BLOCKER]** = broken/wrong today · **[CORE]** = required for production · **[POLISH]** = quality.

> **Read this first.** Three separate problems below would silently invalidate every number you
> produce, and one means your GPU is currently doing nothing. They are cheap to fix now and
> unfixable-after-the-fact once results are written up. Start at Phase 0.

---

## The five things that matter most

1. **Your GPU is idle.** Installed torch is `2.12.0+cpu` — zero CUDA DLLs. Nothing you run today touches the GPU.
2. **`val` and `test` splits are silently empty.** `default_identity_fn` collapses every identity to one value, so `n_val = round(0.15 * 1) = 0`, and the dataset returns `len == 0` without raising. Early stopping on val loss could never fire.
3. **FaceForensics++ has a two-identity leak.** `033_097.mp4` contains *both* identities; the code keys on `033` only. The model trains on a face it is then tested on.
4. **Explainability needs a retrain, not just plumbing.** The default `fusion="concat"` produces no branch weights, so branch attribution has no data source. Decide this *before* you train.
5. **Cross-dataset AUC will be ~0.65–0.75, and that is a success.** Not a bug. See Phase 4.

---

## Phase 0 — Stop the bleeding (~1–2 hours, unblocks everything)

- [ ] **[BLOCKER] Rebuild the venv with a CUDA build of torch.** `venv/pyvenv.cfg` points at
      `C:\Users\maith\...\Python311` — a different machine's user. The venv is unrunnable, and its
      torch is `+cpu` (`venv/Lib/site-packages/torch/version.py` → `cuda=None`). Delete it, recreate,
      install torch from the PyTorch CUDA index, and **verify `torch.cuda.is_available()` is True**
      before anything else. (`venv/` is already gitignored and untracked — a plain delete is safe.)
- [ ] **[BLOCKER] Create `.dockerignore`** (`venv/ data/ .git/ notebooks/ node_modules/ **/__pycache__/ *.pt *.pth`).
      Docker ignores `.gitignore`, so `COPY . .` currently ships the ~2.8 GB venv into both images.
      Build context 3 GB → <5 MB. Highest ROI file in the repo.
- [ ] **[BLOCKER] Delete `opencv-python` from `requirements.txt`,** keep only `opencv-python-headless`.
      Both are installed (4.11.0.86 each), both own the `cv2` namespace, last writer wins.
      Then drop `libgl1` from both Dockerfiles (headless doesn't need it); keep `libglib2.0-0`.
- [ ] **[BLOCKER] `docker compose up` fails today** — it runs `uvicorn backend.main:app` but `backend/`
      holds only `.gitkeep`. Same for `ml/train.py` (referenced in compose, doesn't exist). Comment
      the services out until Phase 6, or the README's quickstart is a lie.
- [ ] **[CORE] Add `LICENSE`.** README references one that doesn't exist. Apache-2.0 recommended
      (patent grant). **Code only** — see the EULA note below.
- [ ] **[CORE] EULA firewall — decide now, it constrains deployment.** FF++ (TUM) and Celeb-DF (UAlbany)
      are signed, research-only, no-redistribution agreements. Therefore: never commit frames/crops;
      **do not bake trained weights into a public Docker image** (that is redistribution of derived work);
      ship weights under research-only terms with a NOTICE of provenance. Apache-2.0 covers the code
      only — state this explicitly in `LICENSE` and `README.md`.

---

## Phase 1 — Data integrity (do this before you train, not after)

These are the bugs that produce impressive numbers that mean nothing.

- [ ] **[BLOCKER] Fix `default_identity_fn`** (`data/dataset_manager.py:52-66`). The regex `^([A-Za-z0-9]+)`
      grabs the leading alphanumeric run, not "the token before the first separator" as documented.
      **Verified:** all 80 dummy images → identity `'person'` → train=80, val=0, test=0, no error raised.
      The identity-split code path has **never once been exercised.**
- [ ] **[BLOCKER] Make `DeepfakeDataset` raise, not return empty.** Assert `len(split) > 0`,
      `n_identities >= 3`, and val/test non-empty at construction. A silent empty split is the
      failure mode that wasted the most time here.
- [ ] **[BLOCKER] Fix the FF++ two-identity leak** (`ml/preprocessing/video_processor.py:177,246`).
      `path.stem.split("_")[0]` on `033_097` returns the *target* only, but the swapped-in face is the
      *source* (`097`). **Simulated on a realistic FF++ layout: 828/4000 fakes land in train while the
      source identity's real video sits in val/test.** Fix: **use the official FF++ `splits/*.json`
      (720/140/140 identity pairs)** — the problem is already solved upstream, and it makes your numbers
      comparable to published baselines. For Celeb-DF, key on the *set* of both ids (`frozenset{id3,id5}`)
      and split by connected components of the co-occurrence graph.
- [ ] **[BLOCKER] `data/data_stats.py:123-136` reports a false green.** Its identity-separation check is
      vacuous when val/test are empty (∅ ∩ ∅ = ∅ → prints "OK"). Assert non-empty splits first.
- [ ] **[BLOCKER] Celeb-DF test set is 86.4% fake** (890 real / 5639 fake) — `prepare_datasets.py:89` skips
      `_balance_5050`. An always-"fake" model scores 86.4% on your headline cross-dataset metric.
      Also: filter to the **official `List_of_testing_videos.txt` (518 videos)** or your number is
      incomparable to every published result.
- [ ] **[CORE] Balance train only, never test.** `_balance_5050` currently downsamples FF++ fakes
      4000→1000, discarding 75% of your fake data, and does it to test too — destroying the
      per-method breakdown. AUC is prevalence-insensitive; balancing test buys nothing.
      Use `WeightedRandomSampler` + class-weighted loss on train instead.
- [ ] **[CORE] Write `data/audit_splits.py`** and run it in CI and before every training run: assert
      identity sets pairwise-disjoint **counting both ids of every fake**, no video path in two splits,
      no real video whose identity appears as either member of a fake pair in another split.
- [ ] **[CORE] Fix `create_dummy_dataset.py`** to emit FF++/Celeb-DF-shaped names (`033_097.mp4`,
      `id3_id5_0001.mp4`) so the dummy set actually smoke-tests the real split logic. Today it tests nothing.
- [ ] **[POLISH] Negative control:** train on shuffled labels — AUC must land at 0.50 ± 0.05.
      Anything higher means the pipeline leaks the label through a path you haven't found.

---

## Phase 2 — Training pipeline (the #1 blocker: it does not exist)

**File layout:** `ml/config.py` · `ml/engine.py` · `ml/checkpoint.py` · `ml/train.py` · `ml/evaluate.py` ·
`ml/utils/{seed,logging,metrics,tensorboard}.py` · `ml/configs/{image,video,smoke}.yaml` ·
artifacts → `runs/<name>/{best.pt,last.pt,config.yaml,train.log,tb/}`

### Model fixes the loop depends on
- [ ] **[BLOCKER] Replace `nn.BatchNorm1d` → `nn.LayerNorm` in `ml/models/fusion.py:33,37`.**
      Two independent reasons: (a) at batch 8 the BN statistics are noise, and **gradient accumulation
      does not fix this** — BN normalizes per micro-batch, so 2×8 is still 8; (b) **verified: a batch of 1
      in `train()` raises `ValueError: Expected more than 1 value per channel`**, and DataLoader defaults
      to `drop_last=False`, so a trailing 1-sample batch kills a run at a random epoch. LayerNorm is
      batch-independent and drop-in for `(B, C)`. If BN stays, `drop_last=True` is mandatory.
- [ ] **[BLOCKER] `VideoClassifier` will OOM.** `classifier.py:146` flattens `B*T` into the backbone.
      **Measured: ~150 MB/frame of fp32 activations → B=8, T=16 ≈ 19.3 GB for the spatial branch alone**,
      before the frequency branch (also on `B*T`), gradients, and optimizer state. Use video
      `batch_size=2–4` + AMP + gradient checkpointing, or precompute spatial features.
- [ ] **[CORE] Thread `dropout` through fusion.** It's hardcoded to 0.4 at `fusion.py:35,39`; the
      constructor's `dropout=` only reaches the final head. The spec's "dropout 0.3–0.5" is
      currently unreachable from config.
- [ ] **[CORE] Return `manipulation` + `identity` from dataset items.** `video_processor.py:519` collects
      `manipulation` at line 180 and then discards it — per-method metrics are impossible without this.

### Config, seeding, reproducibility
- [ ] **[CORE] `ml/config.py`: frozen dataclasses + YAML overlay.** Not Hydra — its multirun/CLI buys
      nothing for two runs and hijacks output dirs; dataclasses give typed defaults and an `asdict()`
      that drops straight into the checkpoint. Every spec number lives here, nowhere else.
- [ ] **[CORE] `ml/utils/seed.py`:** seed random/numpy/torch/cuda; `cudnn.benchmark=True`.
      **Do not default `use_deterministic_algorithms(True)`** — cuDNN's BiLSTM backward has no
      deterministic kernel and will raise. Expose it as a flag for smoke runs only.
- [ ] **[CORE] Log the resolved config + `git rev-parse HEAD` + dirty flag** at startup.

### The loop (`ml/engine.py`)
- [ ] **[CORE] AMP: bf16 when `torch.cuda.is_bf16_supported()`, else fp16 + GradScaler.** bf16 specifically
      because `frequency_branch.py:37` (`log(mag + 1e-8)`) and `fusion.py:113` (`-inf` mask) are
      exponent-range hazards, and bf16 needs no scaler. Wrap `log_magnitude_spectrum` in
      `autocast(enabled=False)` to make the fp32 FFT explicit rather than incidental.
- [ ] **[CORE] Scheduler: 1-epoch linear warmup → cosine to 1e-6, stepped per-iteration.** Not
      ReduceLROnPlateau: with 15–30 epochs and early-stop patience 5, its patience must be ≤2 to react
      at all — too coarse, and it muddies resume state.
- [ ] **[CORE] Freeze schedule:** epochs 0–2 freeze `model.spatial.features` (train freq+fusion+head @1e-3),
      then unfreeze with param groups backbone 1e-5 / rest 1e-4. **Also call `model.spatial.eval()` while
      frozen** — `requires_grad=False` does *not* stop EfficientNet's BN from updating running stats,
      so a "frozen" backbone silently drifts.
- [ ] **[CORE] Windows DataLoader:** `spawn` re-imports `__main__`, so `ml/train.py` **must** guard
      `if __name__ == "__main__": main()` or workers fork-bomb. Use `num_workers=4,
      persistent_workers=True, pin_memory=True, drop_last=True`.
- [ ] **[CORE] Gradient accumulation** → image effective batch 32 (16×2); scale loss `1/accum`; under fp16
      `scaler.unscale_()` **before** `clip_grad_norm_(1.0)`.
- [ ] **[POLISH] `torch.compile` default off.** Triton-on-Windows is fragile and the cuDNN LSTM
      graph-breaks. Keep the flag, measure before trusting it.

### Checkpointing (`ml/checkpoint.py`)
- [ ] **[CORE] Payload:** model/optimizer/scheduler/scaler state, epoch, global_step, config **as a plain
      dict**, git SHA + dirty, metrics, RNG states, class_names, image_size, norm stats, arch, fusion,
      EER threshold. **torch ≥2.6 defaults `torch.load(weights_only=True)`** — a dataclass or `Path` in
      the payload makes every load raise.
- [ ] **[CORE] `load_for_inference(path, map_location) -> (model, meta)`** rebuilds the arch from stored
      config so `backend/` imports **zero** training code. This is the only function the backend calls.
- [ ] **[CORE] `best.pt` (monitor val loss per spec) + `last.pt` each epoch;** early stop patience 5;
      restore best before the final test pass. Write to tmp + `os.replace` so a crash can't corrupt a
      checkpoint. `--resume last.pt` restores model/opt/sched/scaler/epoch/step/RNG.

### Metrics (`ml/utils/metrics.py`)
- [ ] **[CORE] Accuracy is the wrong headline.** Val is 50:50 only *because* of downsampling (not the
      field prior), argmax freezes the threshold at 0.5 when the product needs a tunable one, and one
      number hides that NeuralTextures sits near chance while Deepfakes hits ~99%. Compute:
      **AUC-ROC** (primary), **AP**, **EER + its threshold** (ship in checkpoint meta), **confusion
      matrix**, **per-manipulation recall @ EER threshold**, and **video-level** beside frame-level.
- [ ] **[CORE] TensorBoard over W&B** — built into torch, offline/Docker-safe, no account. A solo
      project needs no hosted sweeps. Add `scikit-learn` to requirements (**it's absent — you cannot
      compute a single metric today**).

### Smoke test (before ANY real run)
- [ ] **[BLOCKER] `ml/configs/smoke.yaml` + `--smoke`:** 10 samples (5/5), **`val_transform` on train**
      (the random crop/blur/noise/JPEG makes 10 samples unmemorizable), 100 steps, `pretrained=false`,
      deterministic. **Assert train loss <0.05 and acc 100%.** If it can't overfit 10 samples the loop
      is broken — don't discover that at epoch 12.
- [ ] **[CORE] Smoke also asserts:** val loader non-empty (catches Phase 1), one accum cycle changes
      weights, no NaN through the FFT `log` under bf16/fp16, batch-of-1 survives the fusion norm,
      checkpoint round-trips under `weights_only=True`.

### Two-stage training
- [ ] **[CORE] Stage 1** `train.py --config configs/image.yaml` → `runs/image/best.pt`.
      **Stage 2** adds `--init-from runs/image/best.pt`.
- [ ] **[CORE] Transfer is clean** because `VideoClassifier` subclasses `DeepfakeClassifier`:
      `spatial.*`, `frequency.*`, `fusion.*`, `classifier.*` are identical module paths, so
      `video.load_state_dict(img_sd, strict=False)` moves all four. **Assert `missing == temporal.*`
      and `unexpected == []`** — a silent typo here means you trained from scratch and never found out.
- [ ] **[CORE] Caveat:** stage 1 runs fusion with temporal **zeros**, so MLP input dims 1664:2176 never
      saw a gradient. Feeding real temporal features there shifts the fused distribution hard.
      Mitigate: stage 2 epochs 0–1 freeze spatial+frequency, train temporal+fusion+head @1e-4.

---

## Phase 3 — Real data (start the EULA requests TODAY — they are the long pole)

- [ ] **[BLOCKER] Request FF++ now.** Google Form EULA (github.com/ondyari/FaceForensics) → they email
      the download script. **Turnaround 2 days–2 weeks, manual, sometimes unanswered.** c23 ≈25–40 GB.
      Never download raw/c0 (1.6–2 TB). Their server runs ~1–5 MB/s; wrap the script in a retry/resume loop.
- [ ] **[BLOCKER] Request Celeb-DF v2 now.** Email the release agreement (UB Media Forensics). Same
      2-day–2-week wait. 6529 videos, ≈12–16 GB, 59 identities.
- [ ] **[CORE] Minimum viable combo: FF++ c23 (train/val/test) + Celeb-DF v2 (test-only).**
      ≈40–55 GB raw + ≈28 GB processed → **budget 100 GB.** That already buys in-dataset AUC +
      per-method breakdown + the cross-dataset headline: a complete, defensible story.
- [ ] **[POLISH] Skip the DFDC train set** (470 GB / 119k videos — days per epoch for little gain).
      If you want a 2nd cross-dataset point, take only the public test set (~4–5 GB, Kaggle, instant access).
- [ ] **[POLISH] Kaggle "Deepfake Faces" is smoke-test only** — filenames are video hashes, so
      identity separation silently degenerates to a random split. Never report a number from it.

### Preprocessing at scale (the compute bottleneck)
- [ ] **[BLOCKER] `prepare_datasets.py` will OOM before it writes anything.** `process_dataset`
      accumulates every sequence in one list, then pickles it whole. **Computed: 3.4 GB for `ff_train`,
      15.7 GB for `celebdf_test`, resident in RAM** before the first byte is written — plus a copy
      during `pickle.dump`. Then `DeepfakeVideoDataset` reloads the whole thing per DataLoader worker
      (Windows spawn = full copy → 8 workers × 5 GB = dead).
- [ ] **[BLOCKER] Replace pickle with per-video `.npy` (uint8) + a Parquet manifest**
      (`path,label,identity,manip,split,dataset,n_missing`). Chosen over LMDB/HDF5/WebDataset: 5k–13k
      files is small, resumability is free (file exists = skip), no map_size/fork-safety pain, no new
      deps, Windows-safe. (Pickle is also arbitrary-code-execution on load.)
- [ ] **[CORE] Do NOT JPEG-encode the crops to save disk.** The model keys on compression artifacts;
      re-encoding overwrites the exact signal you're training on.
- [ ] **[BLOCKER] Swap the face detector for the offline pass.** `retina-face` is the serengil TF
      package: single-image, **no batching**, and it fights PyTorch for VRAM in-process. Use batched
      RetinaFace (`batch_face`) or **InsightFace SCRFD via `onnxruntime-gpu`** at batch 32–64.
- [ ] **[CORE] Kill the random seeks.** `_read_indices` (`video_processor.py:349-359`) calls
      `cap.set(CAP_PROP_POS_FRAMES)` per frame; on H.264 every seek re-decodes from the prior keyframe
      (~50–200 ms) **and lands on the nearest keyframe, so you aren't sampling the indices you asked for.**
      Sequential decode keeping the wanted indices is both faster and exact.
- [ ] **[CORE] Cost math:** FF++ = 5000 videos × 16 frames = 80k detections. As written (serial + CPU TF
      RetinaFace + seeks): **20–40 h.** Fixed (8-way multiprocessing + sequential decode + GPU batched
      detector): **1.5–3 h**; Celeb-DF 1–2 h.
- [ ] **[CORE] Resumability + failure log:** skip videos whose `.npy` exists; append every failure to
      `data/processed/failures.csv` with a reason. Today `process_dataset` swallows skips into a counter —
      you won't know *which* 400 videos vanished or whether they were disproportionately real vs fake
      (a silent label-prior shift).
- [ ] **[BLOCKER] Fix per-clip augmentation.** `video_processor.py:521-525` applies the transform
      **independently to each frame** — each of 16 frames draws its own RandomCrop/Rotate/Flip
      (~half a clip flipped). The BiLSTM learns augmentation jitter instead of deepfake flicker.
      Use `A.ReplayCompose` / `additional_targets` for one parameter draw per clip.
- [ ] **[BLOCKER] Fix the train/val geometry mismatch.** Train is `Resize(256)→RandomCrop(224)` (87.5% FOV);
      val is `Resize(224)` (100% FOV) — different zoom *and* different resample ratio. The FFT branch keys
      on resampling artifacts, so its input distribution shifts at eval. Use `Resize(256)→CenterCrop(224)` at val.
- [ ] **[BLOCKER] `A.Rotate(limit=10, border_mode=0)` runs *after* `RandomCrop`,** so 70% of train images
      get guaranteed black corner wedges that never appear at val — a strong label-independent spectral
      artifact fed straight into the FFT branch. Use `border_mode=cv2.BORDER_REFLECT_101`, or rotate before cropping.
- [ ] **[CORE] Make the FaceDetector import lazy** (`ml/preprocessing/__init__.py:3`). Importing
      `ml.preprocessing` just to get `build_train_transform` pulls in **all of TensorFlow (~600 MB RSS +
      a CUDA context)** — in every DataLoader worker and in the backend.
- [ ] **[CORE] `face_detector.py:81,87` — `confidence_threshold` is a lie below 0.9.**
      `RetinaFace.detect_faces()` is called without `threshold=`, so it defaults to 0.9 internally;
      the later filter can only *tighten*. Pass it through.

---

## Phase 4 — Train & evaluate (set expectations before you run)

- [ ] **[CORE] In-dataset (FF++ c23 → FF++ c23): expect video-level AUC 0.97–0.995.** Easy; proves
      almost nothing about generalization.
- [ ] **[BLOCKER] Cross-dataset (FF++ → Celeb-DF, official 518-video list) is THE headline.
      Expect AUC 0.65–0.75.** Published baselines: Xception 0.653, EfficientNet-B4 ~0.64–0.69,
      RECCE 0.687. Only blending-artifact augmentation (SBI 0.93, Face X-ray 0.74, FTCN 0.87) breaks 0.80.
      **If you see >0.95 cross-dataset, you have a leak or a bug — not a result.**
      The 30-point drop is expected physics, not failure.
- [ ] **[CORE] Per-manipulation breakdown** (each fake subset vs the *same* real set):
      Deepfakes/Face2Face/FaceSwap ~0.98–0.99, **NeuralTextures ~0.90–0.95 — reliably your worst**
      (it only edits the mouth region). A single averaged number hides this.
- [ ] **[BLOCKER] Select the threshold on val, never test.** Pick once on FF++ val, then apply that
      *frozen* threshold to FF++ test and to Celeb-DF. Re-tuning on Celeb-DF is the single most common
      way projects accidentally report a fantasy number.
- [ ] **[CORE] Aggregate frames→video by mean of probabilities.** (Max is noise-sensitive — avoid.)
      Note your architecture is clip-level today, so clip == video.
- [ ] **[CORE] Compression robustness: train c23 → test c40** (+8–10 GB download).
      **Expect 0.97–0.99 → 0.86–0.92.** Cheapest credibility win available.
- [ ] **[POLISH] Bootstrap 95% CIs** over videos (n=518 → CI ≈ ±0.04 AUC). Without them, 0.71 vs 0.73
      is noise being reported as progress.
- [ ] **[POLISH] Calibration:** reliability diagram + Brier/ECE. Expect the FF++-trained model to be
      badly overconfident on Celeb-DF; temperature fit **on val** is the honest fix.

---

## Phase 5 — Explainability (the headline feature — and it needs a decision before you train)

- [ ] **[BLOCKER] Decide fusion mode BEFORE training.** `classifier.py:57` defaults `fusion="concat"`;
      `FeatureFusion` has **no** branch weights. Only `AttentionFusion` returns them. Branch attribution
      — arguably your strongest explainability feature — requires the production checkpoint to be
      *trained* with `fusion="attention"`. This is a retrain, not plumbing.
      (Counter-argument from the training audit: concat transfers better between stages. **Pick one
      deliberately; don't discover the conflict after a 20-hour run.**)
- [ ] **[BLOCKER] Both "already done" hooks are unreachable.** `forward` calls `self.fusion(s, f, None)`
      without `return_weights=True`; `VideoClassifier.forward` calls `self.temporal(seq)` without
      `return_attention=True`. Add `_forward_impl(x, collect_aux)` + `forward_explain(x) -> (logits, aux)`.
- [ ] **[BLOCKER] Frame→seconds is impossible today.** `extract_frames` computes
      `indices = np.linspace(...)` then **discards them**, and never reads `cv2.CAP_PROP_FPS`.
      Thread `{fps, source_indices, total_frames, duration_s}` through to the output. Also surface the
      last-frame padding so duplicated frames aren't drawn as real timestamps.
- [ ] **[BLOCKER] The spec's 0.6 attention threshold cannot fire.** `TemporalAttention` returns a
      **softmax over T** (rows sum to 1); at T=16, uniform is 0.0625 and a raw weight essentially never
      reaches 0.6 — you'd flag zero frames forever. Normalize: flag on `w / w.max() >= 0.6`, and report
      `w * T` ("3.2× more attention than average") as the human-readable number.
- [ ] **[CORE] GradCAM: hand-roll it (~15 lines); drop `grad-cam` from requirements.**
      Verified in the installed source: `BaseCAM.get_target_width_height` reads a 5D input as a *3D conv
      volume* → garbage for video; and it assumes activation-batch == input-batch, which
      `VideoClassifier`'s `(B,T,…)→(B*T,…)` flatten violates (16 CAMs for 1 input). It also silently
      mutates your model (`self.model = model.eval()`). The video path is structurally incompatible.
- [ ] **[CORE] Target layer: `model.spatial.features[-1]`** — verified as `features[8]`, the final
      `Conv2dNormActivation(384→1536, k=1, SiLU)`, output `(B,1536,7,7)`. Hook the whole block (post-BN/SiLU),
      not `features[8][0]`. 7×7 upsampled to 224 is inherently coarse — say so in the UI.
- [ ] **[CORE] Exploit the flatten:** one backward on a `(1,16,…)` clip yields all 16 frame CAMs in a
      single `(16,1536,7,7)` capture. No per-frame loop.
- [ ] **[CORE] Skip frequency GradCAM.** The frequency CNN's H/W axes are FFT coordinates, not image
      coordinates; overlaying that CAM on a face is actively misleading.
- [ ] **[BLOCKER] The `@torch.no_grad()` gotcha.** Keep `predict()` decorated as the fast path; add an
      undecorated `predict_with_explanation(x)` wrapping `with torch.enable_grad():`.
      **Silent-failure guard:** if the backbone is frozen and the input doesn't require grad, the grad
      hook never fires and you get an *empty* gradient list — not an error. Call
      `input.requires_grad_(True)` and assert gradients were captured.
- [ ] **[CORE] Frequency explainability that actually convinces:** high-frequency energy ratio as **one
      number, one sentence** ("87% more high-frequency energy than a typical real face"), plus a radial
      power profile against real/fake reference bands (compute those means once over the training set,
      ship as `.npz` — the curve alone is meaningless). The raw spectrum heatmap is **decorative** —
      keep it as the small panel, not the headline.
- [ ] **[CORE] Per-frame scores don't exist yet.** Reuse the already-computed `spatial_seq`/`frequency_seq`
      and run fusion+head per frame with `temporal=None` → `(B,T)` fake-probabilities for ~free.
      Document honestly: trained on clip-level labels, so per-frame is an uncalibrated proxy.
- [ ] **[CORE] "Feature evolution" — honest verdict.** original/frequency/attention/prediction are real.
      `cv2.Sobel`+`Canny` panels labelled "what the model sees" are **pure theater** — the model never
      computes them. Salvage it: source **edge** from `spatial.features[0]` (stem, 112×112, genuinely an
      edge/blob detector) and **texture** from a mid stage (`features[2]`/`features[3]`). Same visual,
      actually true.
- [ ] **[BLOCKER] `render.py` must call `matplotlib.use("Agg")` before any pyplot import** — the default
      backend in a FastAPI worker attempts a GUI and will crash or leak figures. `plt.close(fig)` in `finally`.
- [ ] **[BLOCKER] Degenerate-map guard.** Min-max normalizing turns a dead CAM into **amplified float
      noise that looks exactly like a real explanation.** Assert `cam.max() - cam.min() > 1e-6`, else
      return `None` + `degenerate=True`. A confident-looking fake heatmap is worse than no heatmap.
- [ ] **[CORE] Weight randomization test (Adebayo et al. 2018)** — progressively randomize
      `spatial.features`; the CAM must degrade toward noise. If it doesn't, your CAM is an edge detector,
      not an explanation. The only test that actually validates the headline feature.
- [ ] **[CORE] Class sensitivity:** assert `corr(cam(x, target=0), cam(x, target=1)) < 0.99` — 2-class
      heads produce near-mirror maps that trivially pass a `!=` check.

---

## Phase 6 — Backend (`backend/` is empty; compose already references it)

- [ ] **[BLOCKER] Add `python-multipart`** — FastAPI `UploadFile` **raises at request time without it**,
      so every upload endpoint 500s. Also: `pydantic-settings`, `filetype`, `prometheus-client`,
      `slowapi`, `httpx`, `pytest-asyncio`.
- [ ] **[BLOCKER] Never serve an untrained head.** `pretrained=True` only loads ImageNet into
      `SpatialBranch`; fusion/classifier/temporal are randomly initialized — the API would emit random
      verdicts at ~50% confidence. Default `SEETHRU_ALLOW_UNTRAINED=false` → `/ready` 503s until a
      checkpoint loads.
- [ ] **[BLOCKER] Never call `classifier.predict()` from the server.** It does `self.eval()`/`self.train()` —
      mutating *shared* module state, and it doesn't restore on exception. Race under a threaded server.
      Call `.eval()` once at startup, then `forward()` only.
- [ ] **[BLOCKER] Load two models:** `VideoClassifier` for the verdict, `ImageClassifier` for per-frame
      timeline scores (VideoClassifier returns one logit pair per 16-frame clip). Budget VRAM for both.
- [ ] **[BLOCKER] Video jobs: in-process `asyncio.Queue` + one worker + `SQLiteJobStore`. Not Celery.**
      On a single GPU a Celery worker is a *second* CUDA context plus a second TF/RetinaFace allocation
      (~2–3 GB idle VRAM) on the same card, for a queue whose steady-state depth is ~1 — concurrency must
      be 1 regardless. Keep it behind a `JobStore` protocol so swapping to Celery is one implementation.
      **This mandates `--workers 1`** (job submitted to worker A, polled from B → spurious 404) and
      **no `--reload`** (the reloader kills in-flight jobs). Fix `docker-compose.yml` accordingly.
- [ ] **[CORE] File layout:** `backend/{__init__,main}.py` · `core/{config,logging,errors,metrics}.py` ·
      `api/routes/{health,predict,jobs,artifacts,model}.py` · `services/{registry,inference,explain,uploads,jobs}.py` ·
      `schemas/` · `tests/`.
- [ ] **[CORE] Endpoints:** `GET /health` (**no GPU, no model, no disk** — else orchestrators kill the pod
      mid-load) · `GET /ready` · `POST /v1/predict/image` (200 sync) · `POST /v1/predict/video`
      (**202 + job_id**) · `GET /v1/jobs/{id}` · `GET /v1/jobs/{id}/result` · `DELETE /v1/jobs/{id}` ·
      `GET /v1/model/info` · `GET /v1/artifacts/{job_id}/{name}`.
- [ ] **[CORE] Warmup in lifespan:** one dummy image forward, one clip forward, **plus a dummy
      `RetinaFace.detect_faces`** — it lazily builds and globally caches its TF model on first call
      (multi-second stall, not thread-safe).
- [ ] **[BLOCKER] Upload size cap BEFORE buffering.** Check `Content-Length` (cheap, but spoofable/absent
      under chunked encoding), then **stream `file.read(chunk)` and abort at cumulative cap** — that's the
      authoritative check. Starlette spools to disk past 1 MB, so a 10 GB body silently fills the container.
      Caps: image 15 MB, video 200 MB.
- [ ] **[BLOCKER] Magic-byte validation** via `filetype`/`python-magic` against an allowlist. Extension and
      client `Content-Type` are **not evidence**.
- [ ] **[BLOCKER] `VideoProcessor.extract_frames` calls `_read_all` whenever `CAP_PROP_FRAME_COUNT <= 0`** —
      a crafted/VFR file with a broken header decodes the *entire* video into a RAM list. Direct OOM DoS.
      Gate every video with `ffprobe` (subprocess, `timeout=10s`) first: reject if duration missing or
      `> 60s`, no video stream, >1 video stream, or dims over cap.
- [ ] **[BLOCKER] Decompression bombs:** set `Image.MAX_IMAGE_PIXELS` explicitly (PIL's default only
      *warns*), `LOAD_TRUNCATED_IMAGES=False`, `verify()` then reopen. **`cv2.imread` has no bomb guard.**
- [ ] **[BLOCKER] Path traversal:** never use the client filename for a path. Write to
      `{TMP}/{uuid4().hex}{validated_ext}`; keep the sanitized original as a display string only.
- [ ] **[BLOCKER] No face detected is the common case** → `422 {error_code:"no_face_detected"}`,
      **not** a real/fake guess. The model is only meaningful on aligned crops.
- [ ] **[CORE] Multiple faces:** image → analyze **all**, return a `faces[]` array each with its own
      verdict/heatmap/bbox; never silently pick face 0. Video → `VideoProcessor` hardcodes `detected[0]`;
      document single-subject-only and emit `warnings:["multi_face_video_first_subject_only"]`.
- [ ] **[CORE] Also handle:** `insufficient_faces` (>8/16 frames lack a face), `unreadable_media` (422),
      `gpu_busy` (503 + `Retry-After`, catch `torch.cuda.OutOfMemoryError` → `empty_cache()`),
      `model_not_ready` (503).
- [ ] **[CORE] Artifact URLs, not base64.** A video result carries 16 timeline heatmaps + spectrum + CAM
      ≈ 1.6 MB of base64 in one JSON blob — fully buffered, JSON-parsed, uncacheable. Return relative
      paths; JSON stays <10 KB and PNGs drop straight into `<img src>` with `Cache-Control: immutable`.
- [ ] **[BLOCKER] Do not ship an uncalibrated softmax labelled "confidence."** Either temperature-scale
      on val and set `calibrated: true`, or name the field `raw_score` and set `calibrated: false`.
- [ ] **[CORE] Three-way verdict — `real | fake | uncertain`** with an explicit margin band. A binary
      forced verdict on a near-0.5 margin is the single most misleading thing this API could do.
- [ ] **[CORE] Non-strippable `disclaimer` on every response:** `not_forensic_evidence: true`,
      model_version, trained_on, known_limitations. Detectors trained on FF++ generalize poorly OOD —
      the API must say so **in-band**, not in docs the caller won't read.
- [ ] **[CORE] Ops:** JSON logging + request-id contextvar, `/metrics` (inference latency histogram by
      stage, queue depth, GPU mem, `uploads_rejected_total{reason}`), pydantic-settings with
      `SEETHRU_` prefix, graceful shutdown, `response_model` on every route.
- [ ] **[CORE] Tests: use `with TestClient(app) as c:`** — bare `TestClient(app)` does **not** run
      lifespan, so the registry is never populated and every test 503s. Fixture a tiny fake model so
      tests never import TF or need weights or a GPU.

---

## Phase 7 — Frontend (`frontend/` is empty; compose has no frontend service)

- [ ] **[CORE] Stack:** Vite + React + TypeScript, Node 20. TanStack Query v5 (job polling makes it
      near-essential), react-router v6, **axios everywhere** (`fetch` cannot report upload progress),
      Tailwind + shadcn/ui (Radix a11y for free, copy-in not a runtime dep).
- [ ] **[CORE] No chart library.** The timeline (16 pts, scrubber-synced), radial spectrum (polar bins),
      and attribution bar (3 segments) are all unconventional — `d3-scale` + hand-rolled SVG costs less
      than bending Recharts/visx.
- [ ] **[BLOCKER] Job id in the URL.** On 202 → `navigate('/analyze/' + job_id, {replace:true})` **before**
      rendering progress. Refresh survival + deep-linkable + resumes polling cold.
- [ ] **[CORE] Polling:** `refetchInterval` **1s for first 10s → 2s to 60s → 5s thereafter**; stop on
      terminal state; cap total at 10 min → `timeout` state.
- [ ] **[CORE] The 5-minute wait is the hardest UX problem here.** Determinate bar **only if** the backend
      sends `progress` — else indeterminate, never a fake ETA. Show the user's own poster frame, a stage
      list (decode → sample → detect → score → explain) lit by progress band, monotonic elapsed time,
      and update `document.title` so a backgrounded tab shows state.
- [ ] **[BLOCKER] The bbox/heatmap coordinate problem is worse than the usual scaled-`<img>` bug.**
      Heatmaps are computed on the **224×224 aligned crop**; bboxes are in **original image space**, and
      alignment can *rotate*. No client-side stretching maps one to the other correctly. **Backend decision
      required:** return original-space composited heatmaps, or ship the per-face 2×3 affine.
      Until then, overlay on the crop only.
- [ ] **[CORE] For the bbox layer, render SVG with `viewBox="0 0 naturalW naturalH"` + `preserveAspectRatio`
      matching the img's `object-fit`** — then bbox coords need **zero** JS math and survive every resize.
- [ ] **[CORE] Opacity slider default 0.5, clamped 0.4–0.6** per spec, with a side-by-side ⇄ overlay toggle.
- [ ] **[BLOCKER] Timeline cannot be a continuous curve.** 16 uniform probes across the whole duration =
      one sample per ~19s on a 5-min video. Render **discrete sample marks on a seconds axis**; shade a
      span only where ≥2 consecutive samples clear threshold, and label it "sampled region", not
      "detected span". Request an `interpolated: bool` per entry — `_interpolate_missing` copies a
      neighbour's crop into faceless frames, and those were never measured.
- [ ] **[BLOCKER] Uncalibrated is the DEFAULT state, not an edge case** (there is no calibration code in
      `ml/` at all). When `calibrated === false`, **suppress the percentage entirely** — render a 3-band
      qualitative scale (weak/moderate/strong signal) + "this score is uncalibrated — it is not a
      probability". Raw score only in a collapsed "Technical detail" drawer.
- [ ] **[BLOCKER] `uncertain` is a first-class, visually equal state** — slate/neutral, not a grey
      footnote under a red one. **No red "FAKE 97%" hero, ever.**
- [ ] **[CORE] Label the attribution bar "what the model leaned on", never "70% frequency evidence."**
      Those weights are branch *gates* (softmax over 3 learned scalars), not evidence shares. For images
      the temporal branch is masked to exactly 0 — render it as **structurally absent** ("not applicable
      to stills"), never "0% temporal", which reads as a measurement.
- [ ] **[CORE] `no_face_detected` is a first-class empty state,** not an error toast. `gpu_busy` /
      `model_not_ready` → auto-retry with backoff, "still queued" — recoverable, not failure.
      Network loss mid-poll → **keep last known state on screen** with a "reconnecting…" chip; the job is
      still running server-side. Never wipe progress to an error page.
- [ ] **[CORE] A11y:** verdict never color-alone (icon + word + fill pattern); `aria-live="polite"` on
      progress and verdict; timeline marks are keyboard-navigable `<button>`s; **generated alt text** on
      heatmaps ("Heatmap for face 1 — strongest activation around <region>"). `alt="heatmap"` on the
      product's core artifact is a fail.
- [ ] **[BLOCKER] Vite inlines `VITE_*` at BUILD time** → a compose `environment:` on the frontend does
      *nothing*. Fix by design: **relative `/v1` baseURL + nginx `proxy_pass http://backend:8000`**.
      Same-origin → no env var in prod, no CORS, one portable image.
- [ ] **[CORE] `frontend/nginx.conf`:** SPA fallback, `client_max_body_size 200m` (**must match the video
      cap or uploads die at the proxy with an opaque 413**), `proxy_read_timeout 300s`, assets immutable,
      `index.html` no-store. Multi-stage Dockerfile (`node:20-alpine` → `nginx:alpine`) + a `frontend`
      service in compose.
- [ ] **[POLISH] Testing:** MSW handlers covering 202→QUEUED→RUNNING→SUCCEEDED, `no_face_detected` 422,
      `gpu_busy` 503, mid-poll network drop; fixtures for calibrated **and uncalibrated** results.
      Vitest priority: coords mapping at ≥3 container sizes · `ConfidenceBand` suppresses % when
      uncalibrated · polling backoff + terminal stop. One Playwright happy path.

---

## Phase 8 — Engineering hygiene, CI, deploy

- [ ] **[CORE] Dependencies: use `uv`** (`pyproject.toml` + `uv.lock`). Not for speed — only uv expresses
      *per-package indexes*: `[[tool.uv.index]] explicit=true` + `[tool.uv.sources]` pins torch to the
      CUDA index and **nothing else**. pip-tools has no equivalent (`--extra-index-url` makes pip pick the
      highest version *across* indexes — a dependency-confusion footgun with PyTorch's shadow packages).
- [ ] **[CORE] Groups:** prod (fastapi, uvicorn, torch, torchvision, opencv-headless, pillow, numpy) ·
      train (albumentations==1.4.3, retina-face, tf-keras, scipy, matplotlib) · dev (pytest, ruff, mypy,
      pip-audit, pre-commit). Only prod goes in the backend image.
- [ ] **[CORE] `albumentations==1.4.3` is a LOAD-BEARING pin** — `var_limit=`, `quality_lower=`,
      `quality_upper=` are renamed/removed in ≥1.4.14 (`std_range`, `quality_range`). Don't bump without
      editing `augmentation.py:50-53`. **Guard it with a canary test, not a comment.**
- [ ] **[CORE] numpy: pin `==1.26.4` deliberately.** `matplotlib==3.8.0` is numpy-1 ABI, but TF/retina-face
      is the real laggard. One ABI world beats two. Move matplotlib to dev (it's only a lazy import).
      **Exit criterion:** when RetinaFace/TF is dropped, go numpy 2.1.x.
- [ ] **[CORE] `python:3.11-slim` is fine for GPU** — torch's CUDA wheels vendor their own CUDA/cuDNN and
      work on slim with `--gpus all`. A CUDA base image is only needed for TF/nvcc, and costs ~2 GB.
      Targets: backend-cpu ≤1.8 GB, ml-gpu ≤8 GB.
- [ ] **[CORE] compose: use `deploy.resources.reservations.devices: [{capabilities: [gpu]}]`.**
      **`runtime: nvidia` will NOT work on Docker Desktop/WSL2** (that's legacy nvidia-docker2).
      Add `restart: unless-stopped`, `mem_limit`, `pids_limit`, and
      `logging: json-file {max-size: 10m, max-file: 3}` — **Docker's default is unbounded and will fill the disk.**
- [ ] **[CORE] TF/torch VRAM contention:** `TF_FORCE_GPU_ALLOW_GROWTH=true`, `TF_CPP_MIN_LOG_LEVEL=2`,
      `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **Better fix:** RetinaFace only runs in the
      offline pass, so keep TF out of the *serving* process entirely — and file an ADR to replace it with
      SCRFD/YOLOv8n-face on onnxruntime (deletes TF, ~600 MB, the whole contention class, and ~10× CPU speedup).
- [ ] **[CORE] Windows/WSL2 GPU passthrough is a real setup step.** Document in `docs/runbook.md`:
      NVIDIA driver on the **Windows host only** (never inside WSL), Docker Desktop ≥4.x WSL2 backend,
      verify with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- [ ] **[CORE] Non-root in both Dockerfiles** (`useradd -u 10001`), `HEALTHCHECK` on `/health`,
      copy requirements before source for layer caching.

### The 5 tests that matter, in order (`tests/` at root — there are ZERO today)
- [ ] **[CORE] 1. `test_no_identity_leakage`** — train/val/test identity sets pairwise disjoint **and all
      three non-empty**. Catches the live `default_identity_fn` bug. **It will fail on the first run —
      that's the point.** Highest-value test in the repo.
- [ ] **[CORE] 2. `test_split_determinism`** — same seed → identical splits; different seed → different.
- [ ] **[CORE] 3. `test_model_shapes`** — `pretrained=False` is **how CI stays offline** (no weight
      download, no CUDA). Lift the existing `__main__` asserts into pytest and delete them.
- [ ] **[CORE] 4. `test_transforms`** — output shape/dtype/normalization **+ the albumentations API canary**.
- [ ] **[CORE] 5. `test_api`** — httpx TestClient + fake model via `dependency_overrides`.
- [ ] **[CORE] `conftest.py` must *generate* fixtures** into `tmp_path` — `data/dummy/` is gitignored,
      so CI will never have it. Coverage target **70%** (not 90% — most of this tree is nn.Module plumbing).
- [ ] **[CORE] GPU tier:** `@pytest.mark.gpu`, deselected by default (`addopts = "-m 'not gpu'"`).

### CI
- [ ] **[CORE] `.github/workflows/ci.yml`:** `setup-uv` with cache → ruff check → ruff format --check →
      mypy → pytest. py3.11 only (a torch matrix is expensive for ~zero signal).
      **CPU-only** — resolve torch from the cpu index (~200 MB vs ~5 GB).
- [ ] **[CORE] mypy realism for a torch codebase:** `ignore_missing_imports = true`,
      `disallow_untyped_defs` on `ml/`+`backend/` only. **Do not enable `--strict`** — torch/cv2/albumentations
      stubs will bury you. The code is already annotated, so this is nearly free.
- [ ] **[CORE] `.pre-commit-config.yaml`:** ruff, ruff-format, **`check-added-large-files --maxkb=1024`**
      (this is what actually stops a 50 MB `.pth` landing in git), detect-private-key, nbstripout.
- [ ] **[CORE] Branch protection on `main`** + dependabot (pip + actions) + Trivy image scan + `pip-audit`.
      Note: the remote is **`github.com/Harshittgupta/SeeThru`** — a shared repo, so add `CODEOWNERS`.
- [ ] **[POLISH] Nightly self-hosted GPU runner** on your own box (`runs-on: [self-hosted, gpu]`) running
      `pytest -m gpu` + a 1-epoch smoke train. Cheap, since the GPU already exists.

### Weights & deployment
- [ ] **[CORE] Weights → Hugging Face Hub, not Git LFS.** GitHub LFS gives 1 GB storage + **1 GB
      bandwidth/mo** free — CI pulling a 50 MB ckpt burns that in ~20 runs. HF is free/unlimited for
      public models, git-lfs versioned, has a native model card, and `hf_hub_download(revision=<sha>)`
      pins + caches. **Keep it gated/private until the EULA question is settled.**
- [ ] **[CORE] Weights at runtime: download-at-startup, not baked.** Pin by revision SHA + verify SHA256
      into a named volume. Baking couples model version to image tag, forces a 7 GB rebuild per
      experiment, and — decisively — **a public image containing FF++-derived weights is redistribution.**
- [ ] **[CORE] Checkpoint naming:** `seethru-{arch}-{dataset}-{date}-{gitsha7}-auc{x.xx}.pt`.
      Always `torch.load(..., weights_only=True)` — pickle is RCE.
- [ ] **[CORE] Untrusted media is the real attack surface.** ffmpeg/OpenCV/Pillow parsing attacker bytes
      is a live RCE class (cf. libwebp CVE-2023-4863). Magic-byte allowlist + re-encode before decode +
      hard caps + subprocess timeouts, and run the decode worker as its own service with
      `network_mode: none`, `read_only: true`, `cap_drop: [ALL]`, `no-new-privileges`, tmpfs scratch.
- [ ] **[CORE] Deployment — the honest numbers.** CPU inference: EfficientNet-B3 @224² ≈ 40–80 ms/frame,
      but **RetinaFace (ResNet50) on CPU is 0.5–2 s per 1080p frame and dominates** → ~0.6–2 s/image
      (demo-usable); a 32-frame clip = 20–60 s (**not** viable over sync HTTP).
  - **Portfolio/demo → local GPU + Cloudflare Tunnel, $0/mo** (~$3–10/mo electricity; free tier, TLS,
    no port-forward). Caveats: uptime = your PC's uptime, and the **free tier caps request bodies at
    100 MB** → large video uploads need presigned direct-to-R2.
  - **Real service → Modal.** Scale-to-zero, per-second billing (A10G ≈ $1.10/hr *while running*),
    $30/mo free credits, 5–20 s cold start. At ~100 req/mo × 3 s ≈ **$0.09/mo**.
  - **Rejected:** a 24/7 rented GPU VM is absurd here — RunPod 4090 **$245–500/mo**, Vast 3090
    $145–250/mo, Lambda A10 $540/mo, HF persistent T4 ≈ $290/mo. HF ZeroGPU needs Pro ($9/mo) and is
    Gradio-shaped.

### Docs
- [ ] **[CORE] `MODEL_CARD.md`:** intended use, training data **+ EULA provenance**, metrics *per dataset*,
      the **cross-dataset generalization gap published honestly**, failure modes, bias audit by skin
      tone/lighting, explicit **"not for forensic or legal use"**.
- [ ] **[CORE] Rewrite `README.md` honestly** — it currently advertises a React frontend and a
      `docker compose up` that both don't exist. Status table + real quickstart.
- [ ] **[POLISH]** `CONTRIBUTING.md`, `SECURITY.md`, `.env.example`, `data/README.md` (how to obtain
      FF++/Celeb-DF + EULA links + "we ship no data"), `docs/{architecture,api,runbook}.md`,
      `docs/adr/` (0001-uv, 0002-hf-weights, 0003-drop-retinaface, 0004-modal-vs-local).

---

## Where the spec (the PDF) is wrong or over-promises

- **"Attention Threshold: 0.6"** — impossible against a softmax over 16 frames (uniform = 0.0625).
  Needs normalization. See Phase 5.
- **"Feature Evolution: Original → Edge → Texture → Frequency"** — the edge/texture rungs are theater
  unless sourced from real stem/mid-stage activations. The model never computes Sobel/Canny.
- **"GradCAM Target Layer: Last convolution layer"** — under-specified for a two-branch model.
  Frequency-branch GradCAM is actively misleading (FFT axes ≠ image axes).
- **"Batch Size: 8–16"** — incompatible with the `BatchNorm1d` in the fusion MLP. See Phase 2.
- **"Initial Recommendation: Feature Concatenation"** — conflicts with the spec's own explainability
  goals, since concat produces no branch weights. Pick deliberately.
- **Spec is silent on calibration** — yet it promises a "confidence score" as a headline output.
  A raw softmax is not a probability.
- **Spec's "human-in-the-loop adaptive retraining" (§10) is the right call** and well-judged — keep it.

---

## Honest timeline

- **Phase 0**: 1–2 hours. Do it today.
- **Phase 3 EULA requests**: send today — **2 days–2 weeks of waiting** is the long pole.
  Do Phases 0–2 while blocked.
- **Preprocessing compute**: ~1 day once fixed (20–40 h if not).
- **Disk**: budget 100 GB.
- **The two things that would silently invalidate every number you produce**: the FF++ `033_097`
  two-identity leak, and threshold-tuning on test. Both are cheap now, unfixable after the fact.
