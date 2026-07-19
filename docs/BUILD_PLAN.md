# SEETHRU — Build Plan (sequential task list)

**How to use this:** tasks are numbered `T1…T58` in dependency order. Say "do T7" and we work that task.
Check items off as they land. Each task has a **Done when** you can actually verify.

**Companion doc:** [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) — the *why* behind every task
(the full audit: what's broken, measured numbers, rejected claims). This doc is the *what next*.

---

## The strategy: build the whole pipe on CPU, then pour real data through it

The GPU (college server) arrives late, so we invert the usual order. **Everything except the final
training run can be built and verified on CPU with the dummy dataset.** An untrained model still has
correct shapes, so the explainability engine, the backend, and the frontend can all be built and tested
against it — they just return garbage verdicts, which is fine while we're testing plumbing.

This means:
- **CPU torch is correct for now.** Do *not* fight with CUDA yet. That's T50, on the college server.
- **The smoke test (overfit 10 samples) runs on CPU in minutes** and proves the training loop works
  before you ever queue for the GPU.
- **Start the dataset EULA requests immediately (T2)** even though we won't use the data until T51 —
  they take 2 days to 2 weeks and are the only thing we can't compress.

Build order: environment → data integrity → models → training loop → preprocessing → explainability →
backend → frontend → real data → real training → ship.

---

## Milestone 0 — Working environment ✅ COMPLETE (except T2, which is yours)

> Landed 2026-07-15. Verified: `torch 2.13.0+cpu` · `torchvision 0.28.0+cpu` ·
> `cv2 4.11.0` (headless only) · `albumentations 1.4.3` (pin held) · `numpy 1.26.4` ·
> `sklearn 1.9.0` · `retinaface` imports · `pytest` 4 passed, 1 gpu test correctly deselected.
> Ruff baseline on existing `ml/`+`data/`: **97 findings, 82 auto-fixable, all cosmetic**
> (typing modernization — `Dict`→`dict`, `Optional[X]`→`X | None`). No real bugs. Clean up in T82.
>
> **Docker Desktop is not installed on this machine** — so `docker compose config` can't be
> verified locally. Doesn't block anything until T86; the compose file is correct by inspection.

- [x] **T1. Install Python 3.11 and rebuild the venv.** ✅
  You have **only Python 3.13.14**; the existing `venv/` holds cp311 binaries and its `pyvenv.cfg` points
  at `C:\Users\maith\...\Python311` — another machine's user. It cannot run. Install 3.11 (not 3.13:
  the pinned stack — albumentations 1.4.3, retina-face, tf-keras — is from the 3.11 era and TensorFlow
  is the fussiest dep in the tree). Delete `venv/` (it's gitignored and untracked, so this is safe) and
  rebuild. **Install CPU torch for now** — you have no GPU yet, and CPU is correct for all dev work.
  *Done when:* `python -c "import torch, cv2, albumentations, retinaface; print(torch.__version__)"` works.

- [ ] **T2. Send the FaceForensics++ and Celeb-DF v2 access requests. Today.** *(Human task — needs your
  name, institution, and signature. Verified links as of 2026-07-15.)*
  - **FaceForensics++** → [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform).
    Once accepted they email you `download-FaceForensics.py`. Contact: `faceforensics@googlegroups.com`.
    Their docs say: **if no reply within a week, your email is bouncing** — check before resending.
  - **Celeb-DF v2** → [Google Form](https://forms.gle/2jYBby6y1FBU3u6q9) or
    [Tencent Form](https://wj.qq.com/s2/8540155/b5d9/). Download link arrives once accepted.
    Contact: `deepfakeforensics@gmail.com`.
  - Use your **college email address**, not gmail — these are research-only agreements and an
    institutional address is what gets them approved.
  This is the project's long pole and costs you ten minutes now. Everything through T66 proceeds
  without the data.
  *Done when:* both forms submitted; write the dates here → FF++ ____, Celeb-DF ____.

- [x] **T3. Fix `requirements.txt`.** ✅ Also pinned `torch==2.13.0` / `torchvision==0.28.0`
  **without** a local version suffix on purpose — the same pin resolves to `+cpu` from PyPI here and
  to `+cu124` from the CUDA index on the training server. One requirements.txt, both machines.
  Remove `opencv-python` (keep only `opencv-python-headless` — both are installed, both own the `cv2`
  namespace, last writer wins). Add the deps that are used-but-undeclared or needed next:
  `scikit-learn` (**you cannot compute a single metric today**), `pyyaml`, `tqdm`, `tensorboard`,
  `python-multipart` (FastAPI `UploadFile` **raises at request time without it**). Pin torch/torchvision.
  Keep `albumentations==1.4.3` — it's **load-bearing**, see T4.
  *Done when:* a fresh `pip install -r requirements.txt` works and `import cv2` resolves to headless.

- [ ] **T4. Add a comment + canary marking the albumentations pin as load-bearing.** 🟡 **HALF DONE**
  `augmentation.py:50-53` uses `var_limit=`, `quality_lower=`, `quality_upper=` — all renamed or removed
  in albumentations ≥1.4.14. Bumping the pin silently breaks training augmentation.
  ✅ Comment landed in `requirements.txt`. ⬜ The canary test needs the `tests/` scaffold, so it lands
  with **T14** — tracked there, don't forget it.
  *Done when:* the pin is commented **and** T14's canary test exists.

- [x] **T5. Create `.dockerignore`.** ✅
  `venv/ data/ .git/ notebooks/ node_modules/ **/__pycache__/ *.pt *.pth`. Docker ignores `.gitignore`,
  so `COPY . .` currently ships the ~2.8 GB venv into both images. Build context 3 GB → <5 MB.
  *Done when:* `docker build` context is under 5 MB.

- [x] **T6. Stop `docker compose up` from lying.** ✅
  It ran `uvicorn backend.main:app`, but `backend/` holds only `.gitkeep`, so it failed instantly. Same
  for `ml/train.py`. Both services are now commented out, each marked with the task that re-enables it
  (**backend → T54**, **ml → T32**). Also removed the obsolete top-level `version: "3.9"` key, and left
  the already-decided settings inline so nobody relitigates them: `--workers 1`, no `--reload`,
  `deploy.resources.reservations.devices` (not `runtime: nvidia`, which is dead on WSL2), bounded
  json-file logging.

- [x] **T7. Add `LICENSE` (Apache-2.0) + the EULA firewall note.** ✅ Added `LICENSE` (Apache-2.0 +
  an explicit scope section), `NOTICE` (the full firewall: dataset provenance, why weights inherit the
  restriction, third-party licences, intended use), and rewrote README's License section + added an
  honest project-status table.
  README references a LICENSE that doesn't exist. **This constrains deployment later, so decide now:**
  FF++ and Celeb-DF are signed research-only, no-redistribution agreements. Never commit frames/crops;
  never bake trained weights into a public image (that's redistribution of derived work). Apache-2.0
  covers **code only** — say so explicitly in both `LICENSE` and `README.md`.
  *Done when:* LICENSE exists and README states the code/weights/data split.

- [x] **T8. Put `git` on your PowerShell PATH** ✅ — appended `C:\Program Files\Git\cmd` to the user
  PATH. git 2.55.0. **Open a new terminal to pick it up.** Incidentally confirmed the branch is
  **`main`**, not `master`.

- [x] **T9. Set up `tests/` + pytest config.** ✅
  Created `pyproject.toml` (pytest + ruff + coverage config; **dependencies deliberately left in
  `requirements.txt`** — migrating to uv is T82's job and would mean solving the torch CUDA-index
  problem before we even have a GPU), `tests/conftest.py`, `tests/unit/test_scaffold.py`,
  `tests/integration/`.
  Confirmed working: `-m 'not gpu'` deselects by default · fixtures **generate** into `tmp_path`
  (never read the gitignored `data/dummy/`) · `pretrained=False` keeps the suite offline · implicit
  namespace packages resolve `ml.models.classifier` without needing `ml/__init__.py`.
  *Done when:* `pytest` runs green. → **4 passed, 1 deselected.**

---

## Milestone 1 — Data integrity ✅ COMPLETE

> Landed 2026-07-15. **86 passed · 2 xfailed (T21 handoff) · 1 gpu deselected · ~21 s.**
> Every bug in this milestone was **reproduced live before being fixed** — none were taken on faith
> from the audit. Two of them turned out to be worse than the audit described, and one guard I wrote
> had a false positive I only found by testing it.
>
> Verified end-to-end: `python data/audit_splits.py --images ... --ffpp ... --celebdf ...` → **AUDIT
> PASSED 3/3, exit 0**, with FF++ showing train balanced 4/4 and val/test at their natural 4-real/8-fake
> skew — T16's policy working in the open.
>
> See PRODUCTION_CHECKLIST Phase 1 for the original findings.

- [x] **T10. Fix `default_identity_fn`** (`data/dataset_manager.py`). ✅
  **Bug reproduced live before fixing** — `test_all_splits_are_non_empty` failed with
  *"split 'val' is empty -- identity split collapsed"*, and `test_identity_fn_distinguishes_dummy_subjects`
  got `1` identity (`{'person'}`) instead of 10. A third test caught a symptom I hadn't predicted:
  **the seed had no effect**, because shuffling a single identity is a no-op.
  Fix: strip an explicit `_frame_<n>` suffix and keep the rest, so `person_001_frame_001` → `person_001`
  **and** `person_001` (the video) → `person_001` — a subject's frames and its video now share an id.
  The docstring no longer claims to handle FF++/Celeb-DF: it *can't*, and pretending to was the root
  cause. Those need T15's pair-aware functions.

- [x] **T11. Make `DeepfakeDataset` raise instead of returning empty.** ✅
  Added `_validate_identities` (runs at construction, before splitting) + `_require_non_empty_split`.
  Catches **both** silent directions: *collapse* (one id for everything → val/test empty) and
  *explosion* (a unique id per file → identity split degenerates into a random one). Errors name the
  problem and show the offending mapping.
  ⚠️ **Writing the tests surfaced a false positive in my own guard**: a dataset with genuinely one image
  per subject would trip the explosion check even though it's harmless (a subject with one sample cannot
  leak). Added `validate_identities=False` as the escape hatch, and the error message now says so.
  Guards that fire on correct input get disabled, which is worse than no guard.

- [x] **T12. Fix `create_dummy_dataset.py` to emit real-shaped filenames.** ✅
  Went further than a rename: it now also generates **miniature FF++ and Celeb-DF trees** with the real
  directory layouts (`original_sequences/youtube/c23/videos/`, `manipulated_sequences/<Method>/c23/videos/`,
  `Celeb-real/`, `Celeb-synthesis/`, `YouTube-real/`), the official `splits/{train,val,test}.json`, and
  `List_of_testing_videos.txt`. **Crucially it emits reciprocal swap pairs (`000_001` AND `001_000`)** —
  without those, a test of T15's fix would prove nothing. This makes the entire two-identity leak
  testable offline, instead of after a two-week EULA wait and a 24 GB download.
  Also fixed `print_summary`, which under-reported (claimed 100 files when it had written 170).

- [x] **T13. Write `tests/unit/test_dataset_splits.py::test_no_identity_leakage`.** ✅
  **It failed first, exactly as predicted** — but the instructive part is which test *passed*:
  `test_no_identity_leakage` was **green while val and test were empty**, because empty sets don't
  intersect. That is the same false-green as `data_stats.py:123` (T18), reproduced live. A leakage
  assertion is worthless without a non-empty assertion beside it.
  `tests/unit/test_dummy_dataset_shapes.py::test_ffpp_naive_leading_token_split_provably_leaks` now
  demonstrates the T15 bug executably, on real-shaped names.
  → **28 passed, 1 deselected.**

- [x] **T14. Write the rest of the unit tests** ✅ — and **T4's canary is now live**, so T4 is closed too.
  `tests/unit/test_transforms.py`: the albumentations canary constructs `GaussNoise(var_limit=…)` and
  `ImageCompression(quality_lower=…)` directly, plus an explicit `A.__version__ == "1.4.3"` assert, so a
  dependabot bump **fails the build with a message naming the cause** instead of quietly breaking
  training augmentation. Also: output contract, PIL+ndarray inputs, val determinism, train actually
  augments, and ImageNet stats verified by inverting the normalization on a constant image.
  `tests/unit/test_model_shapes.py`: lifted the `__main__` asserts into pytest (they were real checks
  nobody ever ran — they only fired if a human executed the module by hand). Covers all three branches,
  both fusions, both classifiers, FFT gradient finiteness incl. the constant-image degenerate case, and
  `test_image_and_video_state_dicts_differ_only_by_temporal` — which pins the T33 two-stage transfer, so
  if the module paths ever drift, "transfer" silently becoming "train from scratch" gets caught.
  🔴 **`test_fusion_survives_batch_of_one_in_train_mode` is `xfail(strict=True)`** — it reproduced T21
  live (`ValueError: Expected more than 1 value per channel when training, got input size [1, 512]`).
  `strict=True` is deliberate: the moment T21 lands, this XPASSes and **fails the suite**, forcing the
  marker off rather than letting it rot.

- [x] **T15. Fix the FF++ two-identity leak** ✅ — *the leak is closed, and proved closed.*
  `ff_identities()` / `celebdf_identities()` now keep **both** ids (`033_097` → `("033","097")`), and
  `group_identities()` does **union-find over identity co-occurrence** so swap chains bind
  transitively (`000_001` + `001_002` ⇒ 000/001/002 all travel together — a per-video rule misses this).
  Splitting is now over *groups*, never raw identities.
  `FFPlusPlusLoader.load_official_splits()` parses FF++'s own `splits/*.json` and uses them when
  present, falling back to the grouped splitter with a loud warning. Inconsistent official splits raise;
  identities absent from them are warned about rather than silently dropped.
  Celeb-DF got the same treatment (`id0_id1_0000` → both ids).
  Tests: `test_ffpp_no_identity_leak_across_splits`, `test_ffpp_reciprocal_pairs_land_together`
  (000_001 and 001_000 must never separate), `test_group_identities_is_transitive`, plus
  non-empty + determinism guards. **23 tests, all passing.**
  <details><summary>Original finding</summary>

  (`ml/preprocessing/video_processor.py:177,246`)</details>
  `stem.split("_")[0]` on `033_097` returns the *target*, but the swapped-in face is the *source* (`097`).
  **Simulated on a realistic FF++ layout: 828/4000 fakes land in train while the source identity's real
  video sits in val/test.** Fix by using FF++'s **official `splits/*.json`** (720/140/140 identity pairs) —
  it already guarantees both members of a pair stay together, and it makes your numbers comparable to
  published baselines. For Celeb-DF, key on `frozenset{id3, id5}` and split by connected components of
  the co-occurrence graph (swap pairs chain identities together transitively).
  *Done when:* T17 passes on realistic filenames.

- [x] **T16. Fix balancing: train only, never test.** ✅
  `FFPlusPlusLoader.get_split(..., balance=None)` now defaults to **`True` for train, `False` for
  val/test** — balancing an eval split discards 75% of the fakes, destroys the per-manipulation
  breakdown, and buys nothing (AUC is prevalence-insensitive).
  `CelebDFLoader.load_testing_list()` parses the **official `List_of_testing_videos.txt`**, and
  `get_video_paths(official_only=True)` is now the default — so the cross-dataset headline is measured
  on the published 518-video benchmark instead of the full 86.4%-fake corpus where an always-"fake"
  model scores 86.4%.
  ⚠️ **Landmine found while writing this**: Celeb-DF's list labels **real=1, fake=0** — the exact
  inverse of this project's `REAL=0/FAKE=1`. Getting it backwards silently inverts every metric.
  Encoded as an explicit `_CELEBDF_LABEL_TO_OURS` map and pinned by
  `test_celebdf_label_polarity_is_not_inverted`. Malformed lines raise rather than being skipped.
  ⬜ Still to do: `WeightedRandomSampler` + class-weighted loss (lands with the training loop, T30).
  `_balance_5050` (`video_processor.py:91`) downsamples FF++ fakes 4000→1000 — **discarding 75% of your
  fake data** — and does it to test too, destroying the per-method breakdown. And `prepare_datasets.py:89`
  skips it for Celeb-DF entirely, leaving the cross-dataset test set **86.4% fake (890 real / 5639 fake)**
  — an always-"fake" model scores 86.4% on your headline metric. Use `WeightedRandomSampler` +
  class-weighted loss on train; leave val/test at the natural prior (AUC is prevalence-insensitive).
  Also filter Celeb-DF to the **official `List_of_testing_videos.txt` (518 videos)** or your number is
  incomparable to every published result.
  *Done when:* train is balanced by sampler, test reflects the real prior, Celeb-DF uses the official list.

- [x] **T17. Write `data/audit_splits.py`** ✅ — a standalone CLI, **exits non-zero on any finding** so a
  leak breaks CI instead of quietly inflating a number. Six audits: non-empty splits (checked **first**,
  since every other check is vacuous over empty sets), identity disjointness **counting both ids of every
  fake**, path disjointness, **group straddle** (a swap chain that got cut), balance policy, and a
  **split fingerprint** — a stable hash logged per run, so "did the split change?" is answerable after
  the fact rather than a shrug.
  Its own tests prove it **fails when it should**: an auditor that always passes is worse than none,
  because it launders a broken split as a verified one. `test_detects_identity_leak_via_second_id` is
  the key one — it's built so an auditor checking `video["identity"]` (the group key) instead of
  `video["identities"]` would miss the leak entirely.
  → `AUDIT PASSED: 3/3` on images + FF++ + Celeb-DF fixtures.

- [x] **T18. Fix the vacuous check in `data/data_stats.py`.** ✅ **Proved dead, not just patched.**
  Forcing the original condition (`split_ratios=(1.0, 0.0, 0.0)`) now prints:
  ```
  val      FAILED       -       -      -
  FAIL: split 'val' could not be built -- split 'val' is empty: 10 identities split by (1.0, 0.0, 0.0)...
  Separation is UNVERIFIED: fix the above before trusting any metric.
  ```
  where it previously printed *"OK — no identity appears in more than one split."*
  The success path now states its **evidence**, not just a verdict — `OK — 10 identities (train=7,
  val=2, test=1), none shared` — so "OK" is falsifiable at a glance. Also: catches T11's new exceptions
  and reports them legibly rather than dying with a traceback (it's a *diagnostic tool*; crashing is
  the one thing it must not do), fixed the bare `from dataset_manager import` that only resolved
  because `sys.path[0]` happened to be `data/`, and stopped building every split twice.

- [x] **T19. Return `manipulation` + `identity` from dataset items.** ✅
  Found the exact drop point: `process_dataset` carried `manipulation` all the way through, and
  `DeepfakeVideoDataset.__getitem__` **threw it away on the last line**. Now returned (plus
  `video_path`), so the per-method breakdown is finally possible — and that breakdown is where the real
  finding lives: Deepfakes/Face2Face/FaceSwap all score ~0.98–0.99 while **NeuralTextures sits at
  ~0.90–0.95** because it only edits the mouth. A single averaged AUC hides that entirely.
  `DeepfakeDataset.__getitem__` deliberately **keeps** its `(tensor, label)` tuple — that's the
  torchvision contract every training loop unpacks — and gains a `metadata(index)` method instead, so
  eval can group by identity without decoding the image.

---

## Milestone 2 — Model fixes + the one decision ✅ COMPLETE

> Landed 2026-07-15. **115 passed · 1 gpu deselected · ~37 s.** `tests/` and `ml/checkpoint.py` lint clean.
>
> 🔬 **Discovery that will matter later:** `SpatialBranch(pretrained=False).eval()` outputs
> **effectively zero** — measured std `7.4e-15`, versus `8.0e-02` in `train()` and `8.7e-02` with real
> ImageNet weights. EfficientNet's BatchNorms hold their *initial* running stats (mean=0, var=1), so in
> eval mode they act as the identity, nothing rescales between layers, and the signal collapses.
> **1536 of the fusion MLP's 2176 inputs are dead in every test using the `image_model` fixture.**
> Shape tests are unaffected, but any test asserting the spatial branch *influences* an output would
> pass vacuously — including T51's ablation tests, where ablating an already-zero branch yields a delta
> of exactly 0.0 that reads like a working measurement. Found it by noticing an ablation demo print
> `delta +0.0000` and refusing to wave it through. Now pinned by
> `test_untrained_spatial_branch_is_dead_in_eval_mode` and a `synthetic_branch_features` fixture.

- [x] **T20. DECISION: fusion mode.** ✅ **DECIDED: `concat`, with attribution from ablation.**
  → **[docs/adr/0001-fusion-mode.md](adr/0001-fusion-mode.md)**
  The plan's earlier recommendation (attention) was **wrong**, and rested on an unchecked assumption:
  that branch attribution requires `AttentionFusion`. It doesn't. **Leave-one-out ablation gives
  per-sample attribution on concat** — and gives *better* attribution, because it's causal
  ("removing frequency moves p_fake by −0.53") rather than a gate score, and it isn't zero-sum.
  Attention's softmax forces the three weights to sum to 1, so it **cannot distinguish "all three
  branches agree" from "none are confident"** — a poor instrument for a project about honest explanation.
  Measured before deciding: `score_temporal` gradient in stage 1 = **exactly 0.0 (dead)**; the
  stage1→stage2 shift is a per-sample uniform rescale of the (spatial,frequency) block, factor
  **0.41–0.89**; ablation works on concat; attention costs **+0.175% params**.
  No code change needed — `classifier.py` already defaults to `"concat"`, and this matches the spec's
  own recommendation. `AttentionFusion` stays in the tree, tested, one arg away.

- [x] **T21. Replace `nn.BatchNorm1d` → `nn.LayerNorm`** ✅
  **The `xfail(strict=True)` handoff worked exactly as designed**: the moment LayerNorm landed, the two
  known-bug tests XPASSed and *failed the suite*, forcing the marker off instead of letting it rot.
  They're now plain regression tests.
  Two bonuses: parameter count is **identical** (1,247,488 — LayerNorm(512) has the same weight+bias as
  BatchNorm1d(512)), and the **running-stat buffers are gone**, so checkpoints have one less category
  of state to save, restore and get subtly wrong.
  Added two tests that guard the *reason*, not just the symptom: `test_fusion_has_no_batch_dependent_norm`
  (someone could "fix" the crash with `drop_last=True` and reintroduce BN, and every other test would
  still pass) and `test_fusion_output_is_batch_size_independent`.
  ⚠️ Writing that second one taught me something: it must pass `dropout=0.0`, because dropout is *also*
  active in `train()` mode and random per call — with it enabled the test fails under LayerNorm too and
  tells you nothing. And it must use `train()`, because `eval()` is exactly where BatchNorm switches to
  running stats and would look batch-independent, hiding the bug being tested for.

- [x] **T22. Thread `dropout` through fusion.** ✅ Now reaches the fusion MLP, not just the final head —
  so the spec's "dropout 0.3–0.5" is reachable from config for the first time. It matters most exactly
  where it was missing: the fusion MLP holds **1.2M of the model's parameters**, which is where the
  overfitting risk lives. A config knob that silently does nothing is worse than no knob — you tune it,
  see no effect, and conclude dropout doesn't help.

- [x] **T23. Expose the explainability hooks.** ✅
  `forward_explain(x) -> (logits, aux)` on both classifiers, sharing **one** `_forward_impl` with
  `forward` — deliberately, because if the explain path ran different code from the training path, the
  explanation would describe a model that never ran.
  Verified reachable end-to-end: image aux carries `spatial (2,1536)` + `frequency (2,128)`; video aux
  adds `temporal`, `spatial_seq (1,16,1536)`, `frequency_seq`, and `temporal_attn (1,16)` summing to 1.0.
  **Plus `fuse_and_classify(spatial, frequency, temporal)`** — the workhorse ADR 0001 needs: it re-runs
  *only* fusion+head on cached branch features, so a 3-branch ablation costs ~3 tiny MLP passes instead
  of 3 full forwards through EfficientNet.

- [x] **T24. The checkpoint contract** (`ml/checkpoint.py`). ✅
  **`weights_only=True` safety is the point.** torch ≥2.6 made it the default, so one `Path` or
  dataclass in the payload yields a checkpoint that *saves fine and can never be loaded by anyone* —
  discovered after the run ends, when the file is the only artifact left. `save()` therefore validates
  the payload recursively **before** writing, auto-converting the common offenders (`Path`→str,
  dataclass→`asdict()`) and raising on the rest with a message that names both the location
  (`config['a']['b'][0]['c']`) and the fix.
  Also: **atomic writes** (temp + `os.replace`, since `best.pt` is rewritten every improvement and a
  truncated write is indistinguishable from a good one until you try to load it); git SHA + dirty flag,
  so "which code made this?" is answerable from the file alone; `load_for_inference()` rebuilds the arch
  from stored primitives and imports **nothing but `ml.models`**, so the backend never depends on
  training code; `calibrated` defaults to **False** because no calibration exists yet (T78) and the
  honest default is "not calibrated"; and `transfer_image_to_video()` **raises** unless the missing keys
  are exactly `temporal.*` — without that, a renamed module makes `strict=False` match nothing and
  stage 2 trains from scratch while reporting success, a failure that is completely invisible: the run
  completes, the loss goes down, the number is just quietly worse.

---

## Milestone 3 — Training pipeline ✅ COMPLETE

> Landed 2026-07-16. **186 passed · 1 gpu deselected · ruff clean across `ml/`, `data/`, `tests/`.**
> `train.py --smoke` → exit 0. `evaluate.py` → exit 0, writes `results.json`.
>
> ## 🎉 T34 PASSES — the training loop is verified, on CPU, before any GPU time is spent
> ```
> epoch 30: 0.0041   epoch 40: 0.0027   epoch 49: 0.0025
> train loss: 0.7033 -> 0.0025 over 50 epochs (best 0.0025)
> PASS -- the training loop works. Safe to spend GPU time.
> ```
> 20× below the 0.05 threshold and **stays** there — converged, not a lucky minimum. Reproduces exactly.
>
> **It failed four times first, and every failure was a real finding.** The smoke test paid for itself
> before the GPU even arrived:
>
> 1. **`backbone_lr` crippled 88.9% of the model.** `backbone_lr=1e-5` vs `lr=1e-3` is *correct* for
>    fine-tuning ImageNet features and *catastrophic* from scratch — and smoke sets `pretrained: false`.
>    Measured: 10,588,856 of 12,037,930 params crawling at 1/100th the head's rate. **Nothing errors.**
>    `build_optimizer` now warns on this combination.
> 2. **My smoke *assertion* was wrong.** It checked eval accuracy, but eval accuracy is unmeasurable on
>    10 samples — EfficientNet's BN running stats need far more data. Measured: **train loss 0.02 while
>    eval assigned all 10 samples an identical p_fake of 0.996.** "Do gradients flow?" is a train-mode
>    question; conflating it with BN convergence made the test report a broken loop that wasn't.
> 3. **`shuffle=True` made the train loss meaningless.** BN normalizes per batch, so re-partitioning 10
>    samples each epoch means a sample's output depends on its batch-mates. Loss swung **0.0045 → 0.89
>    at lr=1e-6**, where weights physically cannot move.
> 4. **`shuffle=False` was worse, and quietly so.** Samples are path-sorted, so batches became
>    **single-class** (`fake/`×5, then `real/`×5) and BN leaked the label through the batch statistics:
>    train loss fell while **val AUC dropped 1.00 → 0.60**. Full batch (10) fixes all of it.
>
> **Two real bugs fixed in the engine along the way:**
> * **Partial accumulation cycles were never flushed.** With 5 batches and accum=2, batch 5's gradient
>   sat in `.grad` — never stepped, never zeroed — and leaked into the *next* epoch's first step, mixed
>   with different data. Any dataset whose batch count isn't a multiple of `accum_steps` hits this.
> * The FFT now runs in **explicit** fp32 rather than relying on autocast's implicit allowlist.

> Layout: `ml/config.py` · `ml/engine.py` · `ml/checkpoint.py` · `ml/train.py` · `ml/evaluate.py` ·
> `ml/utils/{seed,logging,metrics,tensorboard}.py` · `ml/configs/{image,video,smoke}.yaml` ·
> artifacts → `runs/<name>/{best.pt,last.pt,config.yaml,train.log,tb/}`

- [ ] **T25. `ml/config.py` — frozen dataclasses + YAML overlay.** Not Hydra (its multirun/CLI buys
  nothing for two runs and hijacks output dirs; dataclasses give typed defaults and an `asdict()` that
  drops straight into the checkpoint). Groups: `DataCfg`, `ModelCfg`, `OptimCfg`, `TrainCfg`.
  **Every number from the spec lives here, nowhere else.**

- [ ] **T26. `ml/utils/seed.py`.** Seed random/numpy/torch/cuda; `cudnn.benchmark=True`.
  **Do not default `use_deterministic_algorithms(True)`** — cuDNN's BiLSTM backward has no deterministic
  kernel and raises. Expose it as a flag for smoke runs only.

- [ ] **T27. `ml/utils/logging.py`** — stdout + `runs/<name>/train.log`; dump resolved config +
  `git rev-parse HEAD` + dirty flag at startup.

- [ ] **T28. `ml/utils/metrics.py`.** **Accuracy is the wrong headline** — val is 50:50 only because of
  downsampling (not the field prior), argmax freezes the threshold at 0.5 when the product needs a
  tunable one, and one number hides that NeuralTextures sits near chance while Deepfakes hits ~99%.
  Compute **AUC-ROC** (primary), **AP**, **EER + its threshold** (ships in checkpoint meta),
  **confusion matrix**, **per-manipulation recall @ EER threshold** (needs T19), **video-level**
  (mean of frame probabilities) beside frame-level.

- [ ] **T29. `ml/utils/tensorboard.py`** — TensorBoard, not W&B. Built into torch, offline/Docker-safe,
  no account, no network call. A solo project needs no hosted sweeps.

- [ ] **T30. `ml/engine.py`** — `train_one_epoch` / `evaluate`. No CLI, no file IO.
  - **AMP**: bf16 when `torch.cuda.is_bf16_supported()`, else fp16 + GradScaler. bf16 specifically
    because `frequency_branch.py:37` (`log(mag + 1e-8)`) and `fusion.py:113` (`-inf` mask) are
    exponent-range hazards, and bf16 needs no scaler. Wrap `log_magnitude_spectrum` in
    `autocast(enabled=False)` to make the fp32 FFT explicit rather than incidental. *(No-op on CPU now;
    correct when the server arrives.)*
  - **Accumulation**: `accum_steps` → image effective batch 32 (16×2). Scale loss `1/accum`; under fp16
    `scaler.unscale_()` **before** `clip_grad_norm_(1.0)`.
  - **Scheduler**: 1-epoch linear warmup → cosine to 1e-6, **stepped per-iteration**. Not
    ReduceLROnPlateau — with 15–30 epochs and early-stop patience 5, its patience must be ≤2 to react at
    all, and it muddies resume state.
  - **Freeze schedule**: epochs 0–2 freeze `model.spatial.features` (train freq+fusion+head @1e-3), then
    unfreeze with param groups backbone 1e-5 / rest 1e-4. **Also call `model.spatial.eval()` while
    frozen** — `requires_grad=False` does *not* stop EfficientNet's BN from updating running stats, so a
    "frozen" backbone silently drifts.

- [ ] **T31. `ml/checkpoint.py`.**
  Payload: model/optimizer/scheduler/scaler state, epoch, global_step, config **as a plain dict**, git
  SHA + dirty, metrics, RNG states, class_names, image_size, norm stats, arch, fusion, EER threshold.
  **torch ≥2.6 defaults `torch.load(weights_only=True)`** — a dataclass or `Path` in the payload makes
  every load raise. `load_for_inference(path, map_location) -> (model, meta)` rebuilds the arch from
  stored config so `backend/` imports **zero** training code. Write to tmp + `os.replace` so a crash
  can't corrupt a checkpoint.
  *Done when:* a checkpoint round-trips under `weights_only=True` and yields identical logits.

- [ ] **T32. `ml/train.py`** — CLI, epoch loop, early stopping (patience 5, monitor **val loss** per spec,
  restore best before the final test pass), `--resume last.pt`, `--init-from`, `--smoke`.
  **Windows `spawn` re-imports `__main__`, so this file MUST guard `if __name__ == "__main__": main()`**
  or DataLoader workers fork-bomb. Use `num_workers=4, persistent_workers=True, pin_memory=True,
  drop_last=True`.

- [ ] **T33. Wire up two-stage transfer.**
  `VideoClassifier` subclasses `DeepfakeClassifier`, so `spatial.*`, `frequency.*`, `fusion.*`,
  `classifier.*` are **identical module paths** — `video.load_state_dict(img_sd, strict=False)` moves all
  four. **Assert `missing == temporal.*` and `unexpected == []`** — a silent typo here means you trained
  from scratch and never found out. **Caveat:** stage 1 runs fusion with temporal **zeros**, so MLP input
  dims 1664:2176 never saw a gradient; feeding real temporal features there shifts the fused distribution
  hard. Mitigate: stage 2 epochs 0–1 freeze spatial+frequency, train temporal+fusion+head @1e-4.

- [ ] **T34. `ml/configs/smoke.yaml` + the smoke test — RUN IT ON CPU.**
  10 samples (5/5), **`val_transform` on train** (the random crop/blur/noise/JPEG makes 10 samples
  unmemorizable), 100 steps, `pretrained=false`, deterministic. **Assert train loss <0.05 and acc 100%.**
  If it can't overfit 10 samples the loop is broken — you want to know that now, on CPU, in two minutes,
  not at epoch 12 on a shared college GPU.
  Also assert: val loader non-empty (catches T10), one accum cycle changes weights, no NaN through the
  FFT `log`, batch-of-1 survives the fusion norm (catches T21), checkpoint round-trips.
  *Done when:* **`python ml/train.py --smoke` passes on CPU. This is the milestone that proves the
  training pipeline actually works.**

- [x] **T35. `ml/evaluate.py`** ✅ Verified end-to-end on the real smoke checkpoint: threshold **chosen
  on val (0.4272) and frozen** through to test, metric table printed, `results.json` written, exit 0.
  **The threshold rule is structural, not documentary.** It routes through
  `select_threshold(split="val")` rather than reading `val.eer_threshold` — because *every* split's
  `Metrics` carries an `eer_threshold` computed from its own data, so reading one off `test` would be a
  one-character mistake that nothing catches and that only makes the number look better.
  `test_threshold_guard_is_in_the_real_path` asserts the production path actually uses the guard, not
  just the tests.
  **`sanity_commentary()` — the report is suspicious on your behalf.** Every failure mode in this
  project makes numbers look BETTER, so a leak never announces itself; it just hands you 0.98 and lets
  you write it up. The report now flags: cross-dataset AUC >0.95 (*"a leak, not a breakthrough"* — and
  names `audit_splits.py` as the thing to run), AUC <0.55 (*"check label polarity — Celeb-DF encodes
  real=1/fake=0, the inverse of ours"*, and prints 1−AUC as the likely true score), a generalization gap
  under 0.05, an unexpected worst-method, and uncalibrated checkpoints. It also **reassures** on a
  0.65–0.75 result — without that, the expected ~30-point drop reads as failure and someone goes hunting
  for a bug that doesn't exist.
  Loader preprocessing comes from the **checkpoint's** `meta`, not a config file that may have changed
  since — evaluating with different preprocessing than training used is a silent way to destroy a
  model's numbers.

---

## Milestone 4 — Preprocessing fixes ✅ COMPLETE

> Landed 2026-07-16. **227 passed · ruff clean · smoke still green (best 0.0036).**
> Each fix was **measured before and after**, not assumed.
>
> **Extraction verified end-to-end on the dummy FF++/Celeb-DF trees, including a simulated crash:**
> ```
> RUN 1: 56 manifest rows, 56 .npy written
> RUN 2: truncated to 53 (as if killed mid-run) -> "Resuming: 53 already in the manifest, 3 to go" -> 56
> RUN 3: "Nothing to do."
> ```
> The summary shows T16's policy working in the open: `ffpp/train 4 real / 16 fake` (natural 4:1, never
> downsampled) while `celebdf/test 6/6` (the official benchmark list).

- [x] **T36. Fix the train/val geometry mismatch.** ✅ val is now `Resize(256)→CenterCrop(224)`,
  matching train's field of view **and its resample ratio**. The second half is the subtle one:
  resampling leaves its own signature in the frequency domain, and the frequency branch is built to
  read frequency-domain signatures — so a val path that resamples 300→224 while train resamples 300→256
  shifts the exact distribution that branch learned.

- [x] **T37. Fix the rotation border.** ✅ **Measured before: 41% of training images carried hard black
  wedges** (mean 1.7% of image area); val images never did. **After: 0/200.**
  A hard edge to black is a step function, and a step function is broadband energy in the Fourier
  domain — precisely the part of the spectrum the frequency branch reads. So the model was being handed
  a strong, label-independent spectral cue present in 41% of train images and 0% of eval images.
  Fixed both halves: `BORDER_REFLECT_101` (fills from real neighbouring pixels — continuous, no step)
  **and** rotate-before-crop at 256, so residual corner artifacts get cropped away.

- [x] **T38. Fix per-clip augmentation.** ✅ `A.ReplayCompose` — transform frame 0, capture the
  parameters actually drawn, replay them on the other 15. The clip is augmented as a unit, so the only
  motion in it is motion that was filmed.
  ⚠️ **My first test was too weak and I nearly shipped it.** It fed *identical* frames, so it couldn't
  tell "replayed correctly" from "same input, same output" — it would have passed even if
  ReplayCompose did nothing. And albumentations explicitly warns that *"Rotate could work incorrectly
  in ReplayMode because its params depend on targets"*, so this needed proving, not assuming.
  Rewrote it with **differing** frames sharing a corner marker: if flip is replayed, the marker stays
  on one side for all 16 frames. **Measured: 8/8 clips internally consistent, and both L and R seen
  across clips** — so it replays correctly *and* augmentation is still alive (a "fix" that froze the
  transform globally would pass the naive test while silently disabling augmentation).

- [x] **T39. Make the FaceDetector import lazy** ✅ **— done early (out of order), because it was
  actively hurting.** Surfaced itself: T15's tests took **110 s**, but `--durations` showed the tests
  themselves totalled ~3 s. The rest was import time.
  Measured with `python -X importtime`:
  ```
  tensorflow                        11.0 s
  ml.preprocessing.video_processor  17.9 s   (before)
  ml.preprocessing.video_processor   4.45 s  (after)   -> 4x faster, 13.4 s saved
  ```
  17.9 s to import a module that lists directories. Fixed via PEP 562 module `__getattr__` in
  `ml/preprocessing/__init__.py` + a lazy `_import_face_detector()` in `video_processor.py`, so only
  code that actually detects faces pays. `from ml.preprocessing import FaceDetector` still works
  unchanged. Verified: TF is **not** in `sys.modules` after `import ml.preprocessing`, and loads on
  first attribute access.
  **Full suite: 110 s → 12.9 s.** This cost was previously paid *per DataLoader worker* (Windows
  spawns) and by the backend at startup, where TF would also sit on the GPU competing with PyTorch.

- [x] **T40. Thread FPS + source indices through.** ✅ New `sample_frames() -> FrameSample` carrying
  `{frames, source_indices, fps, total_frames, duration_s, n_padded}` + a `timestamps()` helper.
  Everything needed was **already being computed and thrown away** — `extract_frames` built
  `np.linspace(...)` and discarded it, and never read `CAP_PROP_FPS` at all. `extract_frames()` stays
  as a thin wrapper for callers that genuinely only want pixels.
  Also added `FaceSequence`, which carries **`interpolated: list[bool]`**. A crop can be in a sequence
  for three different reasons — detected, copied from a neighbour because detection failed, or
  duplicated as padding — and **only the first is an observation**. The old code collapsed all three
  into one list of arrays, so the timeline could not tell evidence from filler and would plot a copied
  frame as a measured point (T50). `process_dataset` now emits all of it, plus `face_rate`.
  Promoted `_build_face_sequence` → `build_face_sequence` (public): the backend needs the per-video
  face stats to answer `insufficient_faces` honestly, and reaching into a private method beats nothing
  but exporting one beats both.

- [x] **T41. Replace pickle with per-video `.npy` + a manifest.** ✅
  ✏️ **JSONL, not Parquet** — a deliberate departure from the audit. Measured the row at 613 bytes, so
  the manifest tops out at **~8 MB for 13k videos**. Parquet's wins (columnar reads, compression,
  predicate pushdown) all arrive north of a million rows, and it would cost pandas+pyarrow (~100 MB) to
  get them. JSONL adds **zero dependencies**, is greppable and diffable, and is **append-only** — which
  is what makes T43's resumability nearly free. Same reasoning that picked `.npy` over LMDB: match the
  format to the scale.
  `data/manifest.py` (schema, atomic write, per-row append, validation, `summarize`) + streaming
  extraction. Crops are stored **uint8, never JPEG re-encoded** — re-encoding would overwrite the exact
  compression artifacts the frequency branch is trained to read.
  🔴 **Trap caught while writing it:** FF++ **reuses stems across all four methods** — `033_097.mp4`
  exists under Deepfakes, Face2Face, FaceSwap *and* NeuralTextures. Keying the output file on the stem
  would have four videos overwrite one `.npy`: **3/4 of the fakes silently lost**, with four manifest
  rows pointing at a single array. `_npy_path_for` includes the manipulation, and
  `test_npy_paths_do_not_collide_across_manipulations` pins it.

- [x] **T41b. Stage-1 data path.** ✅ `data/processed_dataset.py` — `DeepfakeFrameDataset` (frame view,
  stage 1) and `DeepfakeClipDataset` (clip view, stage 2) read the **same manifest**. One extraction,
  two views. `.npy` opened with `mmap_mode="r"`: a clip is 2.4 MB and the frame view wants ~150 KB of
  it, so mmap lets the OS page cache be shared across DataLoader workers instead of each materialising
  every clip.
  **Wired into `train.py`**: `build_loaders` auto-detects `<root>/manifest.jsonl` → processed path,
  else → image folders (dummy/smoke). Auto-detection beats a config flag — the flag is one more thing
  to set wrong, and the filesystem already knows the answer.
  Splits come **from the manifest**, never re-derived — re-deriving would reintroduce the exact
  two-identity leak T15 exists to prevent.

- [ ] **T41b. 🔴 GAP — stage 1 has no data path from the real dataset. Found 2026-07-16.**
  Traced the pipeline end to end and it does not connect:
  * `DeepfakeDataset` (what `image.yaml` and stage 1 train on) wants an **image-folder** layout:
    `root/real/*.jpg` + `root/fake/*.jpg`.
  * `prepare_datasets.py` produces **16-frame video sequences** — `{frames, label, identity,
    manipulation, video_path}` — not image folders.
  * **Nothing bridges them.** Today `image.yaml` points at `data/dummy/images`, which only exists
    because `create_dummy_dataset.py` writes that layout by hand. There is no route from FF++ videos
    to anything stage 1 can read.
  This is invisible until the data lands and stage 1 has nowhere to read from — which is exactly when
  it is most expensive.
  **Fix (folds into T41): one extraction, two views.** `prepare_datasets.py` writes per-video `.npy`
  (16 crops) + a Parquet manifest; then add a `DeepfakeFrameDataset` that reads the *same* manifest and
  treats each **frame** as a sample for stage 1, while `DeepfakeVideoDataset` reads it and treats each
  **clip** as a sample for stage 2.
  Why this over dumping JPEGs into `real/`/`fake/` folders: one face-detection pass instead of two
  (that pass is the 1.5–3 h bottleneck), no re-encoding (T41 — re-encoding overwrites the compression
  artifacts the frequency branch trains on), and `identity`/`manipulation` come from the manifest
  rather than being re-parsed out of filenames, so the T15 leak fix and the per-method breakdown apply
  to both stages for free.
  Keep `DeepfakeDataset` for the dummy set and tests — it is not wasted, it is just not the real path.

- [x] **T42. Kill the random seeks.** ✅ `_read_sequential` decodes forward and keeps the wanted
  indices. `test_sampled_indices_are_exact` now asserts the returned frames are *exactly* the requested
  `np.linspace` indices.
  The correctness half mattered more than the speed half: seeking on H.264 lands on the nearest
  **keyframe**, so "uniform sampling" was silently sampling whatever keyframes sat near the requested
  indices — **differently per video**, depending on each file's GOP structure. Nobody would ever have
  noticed.
  Also capped the `_read_all` fallback at `MAX_DECODE_FRAMES` (18k). That path runs whenever
  `CAP_PROP_FRAME_COUNT <= 0`, which a crafted or VFR file can arrange at will, and it decodes into a
  Python list at ~6 MB/frame — a direct OOM. The backend gates uploads with ffprobe too (T55), but this
  function is reachable from the offline pipeline and must not depend on a caller it cannot see.

- [x] **T44. Pass `confidence_threshold` through to RetinaFace.** ✅ Confirmed the bug by reading
  `face_detector.py:80`: `detect_faces()` was called with **no `threshold=`**, so RetinaFace applied its
  internal default of 0.9 and the score filter below could only ever *tighten*.
  `FaceDetector(confidence_threshold=0.5)` silently still returned 0.9 — the argument was a no-op for
  every value under 0.9, so lowering it to recover faces from hard frames did nothing, with no error.

- [x] **T43. Multiprocessing + resumability + a failure log.** ✅ `ProcessPoolExecutor` with a
  per-worker detector (built once per process — the TF import is ~600 MB, so once per *video* would be
  absurd). Resume is manifest-driven: `done_videos()` → skip. **A crash now costs one video, not 4,900.**
  Verified by truncating the manifest mid-run and re-running (see the milestone header).
  `failures.csv` **names each lost video with a reason** instead of incrementing a `skipped` counter.
  The counter hid the question that matters: were the failures disproportionately real or fake? A
  detector that fails more on one class silently shifts the label prior and every downstream metric
  inherits it — while the run cheerfully reports "skipped: 400". The writer now flags a >2:1 class skew
  explicitly.

- [x] **T45. ADR: replace RetinaFace/TF with SCRFD on onnxruntime.** ✅
  → **[docs/adr/0002-replace-retinaface.md](adr/0002-replace-retinaface.md)** — **Proposed, deliberately
  deferred until after the first real training run.**
  RetinaFace was the root cause in *five separate milestones* (11.0 s of the 17.9 s import in T39;
  ~600 MB/worker in T43; the single-image API that IS the 1.5–3 h extraction bottleneck; TF-vs-torch
  VRAM contention in T54; the `numpy<2` pin in T82; 0.5–2 s/frame CPU inference in T86). Swapping it
  looks like an obvious win.
  **The reason to wait:** changing the face detector **changes the data**. Different crops, boxes and
  alignment mean a model trained on RetinaFace crops and evaluated on SCRFD crops is being tested on a
  distribution it never saw — and that drop would be indistinguishable from a modelling failure. Doing
  it now means changing two things at once with no baseline to compare against. Get one real
  cross-dataset number first, then swap, then re-measure. The ADR records the exit criteria and a
  crop-shift measurement to run before committing to a retrain.

---

## Milestone 5 — Explainability ✅ COMPLETE

> Landed 2026-07-16. **296 passed · ruff clean.** Verified end-to-end against the **real**
> `runs/smoke/best.pt`:
> ```
> verdict     : real  (p_fake 0.381)      calibrated: False
> degenerate  : {'heatmap': True}
>   "The heatmap carried no signal for this input and has been omitted
>    rather than shown as a map of noise."
> attribution : spatial -0.2334 | frequency +0.1650  -> sum -0.0685 (deliberately not 1.0)
> json 2167 bytes; spectrum.png alone is 304 KB (~405 KB as base64 -> a 200x bloat, if inlined)
> ```
> **The T53 guard fired on a real artifact.** The smoke model trained on random noise, so its CAM
> genuinely carries no signal — and the system omitted the heatmap instead of min-max normalising noise
> into a convincing picture. That is exactly the failure it exists to catch, caught in the wild rather
> than in a fixture.
>
> `ml/explainability/`: `contracts.py` · `render.py` · `gradcam.py` · `attribution.py` ·
> `frequency_viz.py` · `temporal_viz.py` · `feature_evolution.py` · `explainer.py` (facade).

> The plumbing is verifiable without training — shapes and gradients are correct even with random weights.
> Layout: `ml/explainability/{gradcam,frequency_viz,temporal_viz,feature_evolution,render,contracts,explainer}.py`

- [ ] **T46. `contracts.py` + `render.py` first** (everything else depends on them).
  `Explanation` frozen dataclass → `to_dict()` returning **JSON-safe scalars only** — never numpy arrays
  or `Figure` objects across the boundary. **`render.py` must call `matplotlib.use("Agg")` BEFORE any
  pyplot import** — the default backend in a FastAPI worker attempts a GUI and will crash or leak
  figures. `plt.close(fig)` in `finally`, always.

- [ ] **T47. `gradcam.py` — hand-rolled (~15 lines). Drop `grad-cam` from requirements.**
  Verified in the installed library source: `BaseCAM.get_target_width_height` reads a 5D input as a *3D
  conv volume* → garbage for video; and it assumes activation-batch == input-batch, which
  `VideoClassifier`'s `(B,T,…)→(B*T,…)` flatten violates (16 CAMs returned for 1 input). It also silently
  mutates your model (`self.model = model.eval()`). **The video path is structurally incompatible** —
  maintaining two CAM paths is worse than owning 15 lines.
  - **Target layer: `model.spatial.features[-1]`** — verified as `features[8]`, the final
    `Conv2dNormActivation(384→1536, k=1, SiLU)`, output `(B,1536,7,7)`. Hook the **whole block**
    (post-BN/SiLU), not `features[8][0]`. 7×7 upsampled to 224 is inherently coarse — say so in the UI,
    don't oversell pixel precision.
  - **Exploit the flatten:** one backward on a `(1,16,…)` clip yields all 16 frame CAMs in a single
    `(16,1536,7,7)` capture. No per-frame loop.
  - **Skip frequency GradCAM** — the frequency CNN's H/W axes are FFT coordinates, not image
    coordinates; overlaying that CAM on a face is actively misleading.

- [ ] **T48. Solve the `@torch.no_grad()` problem.** Keep `predict()` decorated as the fast path; add an
  **undecorated** `predict_with_explanation(x)` wrapping `with torch.enable_grad():` (nesting *does*
  re-enable inside a `no_grad` caller). Set `model.eval()` **and** grad-on — independent concerns.
  **Silent-failure guard:** if the backbone is frozen and the input doesn't require grad, activations
  won't require grad, the hook never fires, and you get an *empty* gradient list — **not an error**.
  Call `input.requires_grad_(True)` and assert gradients were captured. A frozen-backbone fine-tune is
  in the plan (T30), so this will bite.

- [ ] **T49. `frequency_viz.py`.** Ship the **high-frequency energy ratio** first — one number, one
  sentence ("87% more high-frequency energy than a typical real face"). That's the convincing artifact.
  Then the **radial power profile** against real/fake reference bands (compute those means once over the
  training set, ship as `.npz` — *the curve alone is meaningless*). The raw spectrum heatmap is
  **decorative**: keep it as the small panel, not the headline.

- [x] **T50. `temporal_viz.py`.** ✅ **The spec's 0.6 threshold is unreachable — measured, not argued.**
  Over 200 random clips the *largest* attention weight in any clip was **0.0662**; a raw `>= 0.6` test
  flagged **0/200**. It would flag zero frames forever and report that as *"no manipulation detected"*
  rather than *"this threshold is unreachable"*. Now thresholded on `w / w.max()`, with `w * T`
  ("3.2× more attention than average") as the human-readable form.
  Per-frame scores reuse `spatial_seq`/`frequency_seq` + fusion+head — ~16 tiny MLP passes, backbone
  doesn't re-run. A frame needs **both** high p_fake **and** high normalized attention, and an
  **interpolated frame can never be suspicious** (it's a copy, not an observation). Spans need ≥2
  *consecutive* samples, and every description states the sampling caveat — *"2–5s suspicious"* implies
  continuous analysis when it was 16 samples ~19s apart.

- [ ] **T51. Branch attribution — via ablation** (per **[ADR 0001](adr/0001-fusion-mode.md)**; this task
  got *simpler* and *better* as a result).
  Leave-one-out on the fused vector: re-run **fusion + head only** with one branch replaced, and report
  the Δ in the fake logit. The backbone features are already computed, so this is ~3 cheap passes, not
  3 full forwards.
  - **Ablate to the training-set MEAN branch feature, not to zeros.** Zeroing feeds the MLP a vector it
    never saw in training (off-manifold), making the measured Δ partly an artifact of the ablation.
    Compute the means once at the end of training; ship them in the checkpoint (T31).
  - Works on **every** checkpoint, not just attention-trained ones — so no `isinstance` branching and no
    "degrade to None" path in the backend.
  - **Not zero-sum**: report three independent Δs. The frontend must **not** render them as a pie chart
    or normalised stacked bar (T62) — they're independent effects, not shares.
  - For `ImageClassifier` there is no temporal branch: emit **2 entries, not 3 with a zero**. "0%
    temporal" reads as a measurement when it's a structural absence.
  - Sanity test: ablating a branch the model ignores should move the logit ~0; ablating all three should
    collapse the prediction toward the prior.

- [ ] **T52. `feature_evolution.py` — the honest version.** Original/frequency/attention/prediction are
  real. **`cv2.Sobel`+`Canny` panels labelled "what the model sees" are pure theater** — the model never
  computes them. Salvage it: source **edge** from `spatial.features[0]` (the stem, 112×112 — genuinely an
  edge/color-blob detector) and **texture** from a mid stage (`features[2]`/`features[3]`), rendered as
  mean/top-k channel activation. Same visual, actually true.

- [ ] **T53. `tests/unit/test_explainability.py`.**
  - **Degenerate-map guard**: min-max normalizing turns a dead CAM into **amplified float noise that
    looks exactly like a real explanation.** Assert `cam.max() - cam.min() > 1e-6`, else return `None` +
    `degenerate=True`. **A confident-looking fake heatmap is worse than no heatmap.**
  - **Class sensitivity**: assert `corr(cam(x, target=0), cam(x, target=1)) < 0.99` — 2-class heads
    produce near-mirror maps that trivially pass a `!=` check.
  - **Weight randomization (Adebayo et al. 2018)**: progressively randomize `spatial.features`; the CAM
    must degrade toward noise. If it doesn't, your CAM is an edge detector, not an explanation.
    **The only test that actually validates the headline feature.**

---

## Milestone 6 — Backend ✅ COMPLETE

> Landed 2026-07-16. **312 passed (296 ML + 16 backend) · ruff clean.**
> **`backend.main:app` imports and boots** — the docker-compose entrypoint that failed at the very
> start of this project (M0) now works, and docker-compose's `backend` service is re-enabled.
> Every test uses `with TestClient(app)` (or lifespan never runs) and a fake model + fake detector, so
> the suite needs no weights, no TensorFlow, and no GPU.
>
> The security and honest-failure surface is all tested green: **no-face → 422 `no_face_detected`**
> (never a guess), magic-byte rejection → 415, streaming oversize abort → 413, artifact path-traversal
> blocked, health-green-while-not-ready, and the uncalibrated disclaimer in-band on every response.

> Layout: `backend/{__init__,main}.py` · `core/{config,logging,errors,metrics}.py` ·
> `api/routes/{health,predict,jobs,artifacts,model}.py` · `services/{registry,inference,explain,uploads,jobs}.py` ·
> `schemas/` · `tests/`. Must match compose's `backend.main:app`.

- [ ] **T54. Skeleton + config + lifespan + `/health` + `/ready`.**
  **`/health` must not touch the GPU, the model, or disk** — else orchestrators kill the pod mid-load.
  `/ready` 503s until weights load. **Never serve an untrained head**: `pretrained=True` only loads
  ImageNet into `SpatialBranch`; fusion/classifier/temporal are random, so the API would emit random
  verdicts at ~50% confidence. Gate on `SEETHRU_ALLOW_UNTRAINED=false` (flip it true for local dev only).
  **Never call `classifier.predict()` from the server** — it does `self.eval()`/`self.train()`, mutating
  *shared* state without restoring on exception. Call `.eval()` once at startup, then `forward()` only.
  Load **two** models: `VideoClassifier` for the verdict, `ImageClassifier` for per-frame timeline scores.
  Warm up both **plus a dummy `RetinaFace.detect_faces`** — it lazily builds and globally caches its TF
  model on first call (multi-second stall, not thread-safe).
  ✏️ **Superseded by [ADR 0001](adr/0001-fusion-mode.md):** build with `fusion="concat"` (the default).
  An earlier draft said "must be `attention` or attribution is fabricated" — no longer true, attribution
  comes from ablation (T51) and works on any checkpoint. Read `fusion` from the checkpoint, don't
  hardcode it.

- [ ] **T55. `services/uploads.py` — the security layer. Do this before the endpoints, not after.**
  - **Size cap BEFORE buffering**: `Content-Length` first (cheap, but spoofable/absent under chunked
    encoding), then **stream `file.read(chunk)` and abort at cumulative cap** — that's the authoritative
    check. Starlette spools to disk past 1 MB, so a 10 GB body silently fills the container. Image 15 MB,
    video 200 MB.
  - **Magic bytes** via `filetype` against an allowlist. Extension and client `Content-Type` are **not
    evidence**.
  - **`ffprobe` gate every video** (subprocess, `timeout=10s`) — because
    **`VideoProcessor.extract_frames` calls `_read_all` whenever `CAP_PROP_FRAME_COUNT <= 0`**, so a
    crafted/VFR file with a broken header decodes the *entire* video into a RAM list. **Direct OOM DoS.**
    Reject if duration missing or >60 s, no video stream, >1 video stream, or dims over cap.
  - **Decompression bombs**: set `Image.MAX_IMAGE_PIXELS` explicitly (PIL's default only *warns*),
    `LOAD_TRUNCATED_IMAGES=False`, `verify()` then reopen. **`cv2.imread` has no bomb guard at all.**
  - **Path traversal**: never use the client filename for a path. Write to
    `{TMP}/{uuid4().hex}{validated_ext}`; keep the sanitized original as a display string only.

- [ ] **T56. `POST /v1/predict/image`** (200 sync, ~1–2 s) + the error taxonomy.
  **`no_face_detected` is the common case → `422`, not a real/fake guess** — the model is only meaningful
  on aligned crops. **Multiple faces → analyze all**, return a `faces[]` array each with its own
  verdict/heatmap/bbox; never silently pick face 0. Also: `insufficient_faces` (>8/16 frames lack a face),
  `unreadable_media` (422), `gpu_busy` (503 + `Retry-After`, catch `torch.cuda.OutOfMemoryError` →
  `empty_cache()`), `model_not_ready` (503). One error envelope everywhere:
  `{error_code, message, request_id, details}`.

- [x] **T54–T59 all done.** ✅ File layout as planned: `main.py` (lifespan, load-once, worker) ·
  `core/{config,errors,logging}.py` · `services/{registry,uploads,inference,jobs}.py` ·
  `api/routes/{health,model,predict,jobs}.py` · `schemas/responses.py` · `tests/`.
  Key decisions honoured: **`/health` touches nothing** (green during model load, so orchestrators
  don't kill it mid-startup) · **never `classifier.predict()`** (registry calls `.eval()` once, then
  `forward()`) · **`allow_untrained=false`** default (an untrained head is confident noise) ·
  **one GPU lock** for torch + RetinaFace · **RetinaFace warmed at startup** · **artifacts as URLs**
  (measured: 304 KB spectrum PNG would be ~405 KB base64 in a 2 KB JSON) · **CORS never `*`**
  (validator rejects it). The details below are kept for reference.

- [x] **T57. Video jobs: in-process `asyncio.Queue` + one worker + `SqliteJobStore`. Not Celery.** ✅
  SQLite is the source of truth (survives restart), the asyncio.Queue is the in-memory hand-off.
  **Orphaned `running` jobs reconciled to `failed` on startup** (a crash means the process that ran
  them is gone), TTL sweep on startup too (the container may have been down past a TTL), `DELETE`
  endpoint so the frontend's Cancel isn't a lie. The `--workers 1` / no-`--reload` constraint is
  written into docker-compose with the reason inline.
  On a single GPU, a Celery worker is a *second* CUDA context plus a second TF/RetinaFace allocation
  (~2–3 GB idle VRAM) on the same card — for a queue whose steady-state depth is ~1. Concurrency must be
  1 regardless. Keep it behind a `JobStore` protocol so swapping to Celery is one implementation when a
  2nd GPU appears. **This mandates `--workers 1`** (job submitted to worker A, polled from B → spurious
  404) **and no `--reload`** (the reloader kills in-flight jobs) — fix `docker-compose.yml` accordingly.
  States `QUEUED → RUNNING → SUCCEEDED|FAILED|CANCELLED|EXPIRED`; reconcile orphaned `RUNNING` rows to
  `FAILED{interrupted}` on startup; `JOB_TTL_HOURS=24` + sweeper.
  Endpoints: `POST /v1/predict/video` → **202 + job_id** · `GET /v1/jobs/{id}` (+ `progress`) ·
  `GET /v1/jobs/{id}/result` · **`DELETE /v1/jobs/{id}`** (the frontend needs this or "Cancel" is a lie
  that leaves the GPU busy) · `GET /v1/artifacts/{job_id}/{name}` · `GET /v1/model/info`.

- [ ] **T58. Response contract + the honesty surface.**
  **Artifact URLs, not base64** — a video result carries 16 timeline heatmaps + spectrum + CAM ≈ 1.6 MB
  of base64 in one JSON blob: fully buffered, JSON-parsed, uncacheable. Return relative paths; JSON stays
  <10 KB and PNGs drop straight into `<img src>` with `Cache-Control: immutable`.
  **Do not ship an uncalibrated softmax labelled "confidence"** — either temperature-scale on val and set
  `calibrated: true`, or name the field `raw_score` and set `calibrated: false`. **There is no calibration
  code in `ml/` at all, so `false` is the default state, not the edge case.**
  **Three-way verdict — `real | fake | uncertain`** with an explicit margin band. A binary forced verdict
  on a near-0.5 margin is the single most misleading thing this API could do.
  Non-strippable `disclaimer` on **every** response: `not_forensic_evidence: true`, model_version,
  trained_on, known_limitations. FF++-trained detectors generalize poorly out-of-distribution — the API
  must say so **in-band**, not in docs the caller won't read.

- [ ] **T59. Backend tests.** **Use `with TestClient(app) as c:`** — bare `TestClient(app)` does **not**
  run lifespan, so the registry is never populated and every test 503s. Fixture a tiny fake model
  (`nn.Module` returning fixed logits + a fake `FaceDetector` returning a synthetic array) so tests never
  import TF, need weights, or need a GPU.

---

## Milestone 7 — Frontend ✅ COMPLETE

> Landed 2026-07-17. **`npm run build` clean (TS strict, 85 KB gzipped) · 19 Vitest tests pass.**
> Node v24 installed and on PATH. `frontend/` scaffolded by hand (Vite + React + TS + TanStack Query +
> react-router + axios + Tailwind v3) — 235 packages, no interactive `create vite`.
> Frontend service added to docker-compose (nginx :5173 → proxies /v1 → backend).
>
> The load-bearing correctness pieces are tested:
> - **`lib/coords.ts`** — the bbox mapping bug solved *structurally* (SVG `viewBox` + `preserveAspectRatio`,
>   no client-side scale math), tested at 4 container sizes incl. the letterbox-padding case the naive
>   version forgets.
> - **`ConfidenceBand`** — proven to **suppress the percentage** when `calibrated: false`; the raw score
>   is only reachable inside a collapsed `<details>`. A "FAKE 97%" from an uncalibrated model is exactly
>   what the test forbids.
> - **`pollInterval`** — the 1s→2s→5s backoff, tested monotonic.
> - **`AttributionBar`** — signed-Δ bars, not a pie chart (ADR 0001), with causal "what the model leaned
>   on" framing.
> - **`ErrorState`** — keyed on `error_code`; `no_face_detected` is a gentle empty state, 503s are
>   recoverable, request-id always shown.

- [x] **T60–T66 all done** (see the milestone header above for the summary). Two deviations from the
  plan, both toward less: **plain Tailwind, no shadcn/ui** (its copy-in components were more surface than
  this app needs), and **no `d3-scale`** — the timeline and attribution bar are hand-rolled SVG with
  arithmetic simple enough not to warrant a dependency. Everything else landed as specified.
  Original scaffold note kept for reference:
  Vite + React + TS, Node 20. TanStack Query v5 (job polling makes it near-essential), react-router v6,
  **axios everywhere** (`fetch` cannot report upload progress). **No chart library** — the timeline
  (16 pts), radial spectrum (polar bins), and attribution bar (3 segments) are all unconventional;
  hand-rolled SVG costs less than bending Recharts.

- [ ] **T61. Upload + job polling.** On 202 → `navigate('/analyze/' + job_id, {replace:true})` **before**
  rendering progress — job id in URL = refresh survival + deep-linkable + resumes polling cold.
  Poll **1 s for the first 10 s → 2 s to 60 s → 5 s after**; stop on terminal state; cap at 10 min.
  Pre-validate client-side mirroring the server caps. **The 5-minute wait is the hardest UX problem
  here**: determinate bar **only if** the backend sends `progress`, else indeterminate — **never a fake
  ETA**. Show their own poster frame, a stage list lit by progress band, monotonic elapsed time, and
  update `document.title` so a backgrounded tab shows state.

- [ ] **T62. The explainability screen.**
  **BLOCKER — the coordinate problem is worse than the usual scaled-`<img>` bug:** heatmaps are computed
  on the **224×224 aligned crop**; bboxes are in **original image space**; and alignment can *rotate*.
  No client-side stretching maps one to the other correctly. **Backend decision required (T58):** return
  original-space composited heatmaps, or ship the per-face 2×3 affine. Until then, overlay on the crop only.
  For the bbox layer, render SVG with `viewBox="0 0 naturalW naturalH"` + `preserveAspectRatio` matching
  the img's `object-fit` — then bbox coords need **zero** JS math and survive every resize.
  Opacity slider default 0.5, clamped 0.4–0.6 per spec, with a side-by-side ⇄ overlay toggle.
  **Timeline cannot be a continuous curve** — 16 uniform probes across the whole duration = one sample
  per ~19 s on a 5-min video. Render **discrete marks on a seconds axis**; shade a span only where ≥2
  consecutive samples clear threshold, labelled "sampled region", not "detected span".
  ✏️ **Attribution bar, revised per [ADR 0001](adr/0001-fusion-mode.md):** values are now **independent
  causal Δs** from ablation (T51), not gate scores. So the UI *can* say the strong thing —
  *"removing the frequency branch drops the fake score from 0.91 to 0.38"* — because that's a
  measured, re-runnable fact rather than an inference about a softmax.
  **But they no longer sum to 100%: do NOT render a pie chart or a normalised stacked bar.** Use
  independent horizontal bars on a signed Δ axis. Three branches can all be large (they agree) or all
  be ~0 (nothing is driving it) — and both of those are *informative* states the old attention weights
  literally could not express.

- [ ] **T63. The honest UI.** When `calibrated === false` (**the default**), **suppress the percentage
  entirely** — render a 3-band qualitative scale (weak/moderate/strong signal) + "this score is
  uncalibrated — it is not a probability". Raw score only in a collapsed "Technical detail" drawer.
  **`uncertain` is a first-class, visually equal state** — slate/neutral, not a grey footnote under a red
  one. **No red "FAKE 97%" hero, ever.** Render `disclaimer` inline and non-collapsible next to the
  verdict, not in a footer.

- [ ] **T64. Error/empty states + a11y.** `no_face_detected` is a **first-class empty state**, not an
  error toast. `gpu_busy`/`model_not_ready` → auto-retry with backoff, "still queued" — recoverable, not
  failure. Network loss mid-poll → **keep last known state on screen** with a "reconnecting…" chip; the
  job is still running server-side, so never wipe progress to an error page.
  A11y: verdict never color-alone (icon + word + pattern); `aria-live="polite"` on progress/verdict;
  timeline marks are keyboard-navigable `<button>`s; **generated** alt text on heatmaps —
  `alt="heatmap"` on the product's core artifact is a fail.

- [ ] **T65. Docker + nginx.** **Vite inlines `VITE_*` at BUILD time**, so a compose `environment:` on the
  frontend does *nothing*. Fix by design: **relative `/v1` baseURL + nginx `proxy_pass
  http://backend:8000`** → same-origin, no env var in prod, no CORS, one portable image.
  `client_max_body_size 200m` (**must match the video cap or uploads die at the proxy with an opaque
  413**), `proxy_read_timeout 300s`, SPA fallback, assets immutable. Multi-stage
  `node:20-alpine → nginx:alpine`; add a `frontend` service to compose.

- [ ] **T66. Frontend tests.** MSW handlers: 202→QUEUED→RUNNING→SUCCEEDED, `no_face_detected` 422,
  `gpu_busy` 503, mid-poll network drop. Fixtures for calibrated **and uncalibrated**. Vitest priority:
  coords mapping at ≥3 container sizes · `ConfidenceBand` suppresses % when uncalibrated · polling
  backoff + terminal stop. One Playwright happy path.

---

## Milestone 8 — Real data (unblocks when the EULA replies land — T2)

- [ ] **T67. Download FF++ c23.** Official size: **~10 GB** for c23 videos (c40 ~2 GB, raw ~500 GB —
  **never download raw**). Their server is slow and the script is serial and flaky, so wrap it in a
  retry/resume loop.
  ```
  python download-FaceForensics.py <output_path> -d all -c c23 -t videos
  ```
  **Also grab `dataset/splits/{train,val,test}.json` from the FF++ repo** — these are the official
  720/140/140 identity-*pair* splits that T15 depends on. Confirmed to exist at
  [github.com/ondyari/FaceForensics/tree/master/dataset/splits](https://github.com/ondyari/FaceForensics/tree/master/dataset).
- [ ] **T68. Download Celeb-DF v2** (~12–16 GB): 590 Celeb-real + 300 YouTube-real + 5,639 fake.
  **Also grab `List_of_testing_videos.txt`** — the official 518-video benchmark subset (178 real /
  340 fake) that every published cross-dataset number is reported on. Without it, your headline
  metric is incomparable to the literature (T16).
- [ ] **T69. Run `prepare_datasets.py`** (needs T41–T43). ~1.5–3 h for FF++, 1–2 h for Celeb-DF with the
  fixes; **20–40 h without them.** Disk: ~10 GB (FF++ c23) + ~14 GB (Celeb-DF) + ~28 GB processed
  crops → **budget 60 GB**.
- [ ] **T70. Run `audit_splits.py` (T17) on the real data and read the output carefully.** This is the
  moment the FF++ two-identity trap either bites or doesn't.
- [ ] **T71. [Optional] DFDC public test set** (~4–5 GB, Kaggle, instant access, no EULA wait) as a second
  cross-dataset point. **Skip the 470 GB train set.**

---

## Milestone 9 — Train for real (college GPU)

- [ ] **T72. Set up the college server.** Install CUDA torch, verify `torch.cuda.is_available()`,
  **re-run the smoke test (T34) on GPU before anything else.** Check bf16 support
  (`torch.cuda.is_bf16_supported()`) — it selects your AMP path.
- [ ] **T73. Watch VRAM on the video model.** `classifier.py:146` flattens `B*T` into the backbone —
  **measured ~150 MB/frame of fp32 activations → B=8, T=16 ≈ 19.3 GB for the spatial branch alone**,
  before the frequency branch (also on `B*T`), gradients, and optimizer state. **Start at
  `batch_size=2–4`** and raise only if headroom allows.
- [ ] **T74. Stage 1: train the image model** → `runs/image/best.pt`.
- [ ] **T75. Stage 2: train the video model** with `--init-from runs/image/best.pt` (T33).
- [ ] **T76. Evaluate — and read this before you panic.**
  - In-dataset (FF++ → FF++): expect **video-level AUC 0.97–0.995**. Easy; proves almost nothing.
  - **Cross-dataset (FF++ → Celeb-DF, official 518 list): expect AUC 0.65–0.75. This is the headline,
    and 0.70 is a good result.** Published baselines: Xception 0.653, EfficientNet-B4 0.64–0.69,
    RECCE 0.687. Only blending-artifact methods (SBI 0.93, Face X-ray 0.74, FTCN 0.87) break 0.80.
    **The ~30-point drop is expected physics, not failure. If you see >0.95, you have a leak, not a result.**
  - Per-manipulation: Deepfakes/Face2Face/FaceSwap ~0.98–0.99, **NeuralTextures ~0.90–0.95 — reliably
    your worst** (it only edits the mouth region). Report all four.
- [ ] **T77. Compression robustness: train c23 → test c40** (+8–10 GB download). **Expect 0.97–0.99 →
  0.86–0.92.** Cheapest credibility win available.
- [ ] **T78. Calibrate.** Temperature scaling fit **on val**. Then flip `calibrated: true` and let the
  frontend show real percentages (T63).
- [ ] **T79. [Optional] Negative control** — train on shuffled labels; AUC must be 0.50 ± 0.05. Anything
  higher means the pipeline leaks the label through a path you haven't found.
- [ ] **T80. Bootstrap 95% CIs** over videos (n=518 → CI ≈ ±0.04 AUC). Without them, 0.71 vs 0.73 is noise
  being reported as progress.

---

## Milestone 10 — Ship

- [ ] **T81. Weights → Hugging Face Hub, not Git LFS.** GitHub LFS gives 1 GB storage + **1 GB
  bandwidth/mo** free — CI pulling a 50 MB checkpoint burns that in ~20 runs. HF is free/unlimited for
  public models, git-lfs versioned, has a native model card, and `hf_hub_download(revision=<sha>)` pins +
  caches. **Keep it gated/private until the T7 EULA question is settled.**
  Runtime: **download-at-startup, not baked** — baking couples model version to image tag, forces a 7 GB
  rebuild per experiment, and **a public image containing FF++-derived weights is redistribution.**
  Naming: `seethru-{arch}-{dataset}-{date}-{gitsha7}-auc{x.xx}.pt`. Always
  `torch.load(..., weights_only=True)` — pickle is RCE.
- [ ] **T82. CI** (`.github/workflows/ci.yml`): `setup-uv` with cache → ruff → ruff format → mypy →
  pytest. py3.11 only. **CPU-only** — resolve torch from the cpu index (~200 MB vs ~5 GB), `pretrained=False`
  everywhere, `-m "not gpu"`, no network. mypy: `ignore_missing_imports=true`, `disallow_untyped_defs` on
  `ml/`+`backend/` only — **do not enable `--strict`**, torch/cv2/albumentations stubs will bury you.
- [ ] **T83. `.pre-commit-config.yaml`**: ruff, ruff-format, **`check-added-large-files --maxkb=1024`**
  (what actually stops a 50 MB `.pth` landing in git), detect-private-key, nbstripout.
  Branch protection on `main` + dependabot + Trivy + `pip-audit`. Note the remote is
  **`github.com/Harshittgupta/SeeThru`** — a shared repo, so add `CODEOWNERS`.
- [ ] **T84. `MODEL_CARD.md`** — intended use, training data **+ EULA provenance**, metrics *per dataset*,
  **the cross-dataset generalization gap published honestly**, failure modes, bias audit by skin
  tone/lighting, explicit **"not for forensic or legal use"**.
- [ ] **T85. Rewrite `README.md` honestly** — it currently advertises a React frontend and a
  `docker compose up` that both don't exist. Status table + real quickstart. Add `CONTRIBUTING.md`,
  `SECURITY.md`, `.env.example`, `data/README.md` (how to obtain FF++/Celeb-DF + EULA links + "we ship no
  data"), `docs/{architecture,api,runbook}.md`.
- [ ] **T86. Deploy.** CPU inference is **~0.6–2 s/image** (EfficientNet-B3 is 40–80 ms/frame, but
  **RetinaFace on CPU is 0.5–2 s per 1080p frame and dominates** — T45 cuts this ~10×); a 32-frame clip is
  20–60 s, **not viable over sync HTTP**.
  - **Portfolio/demo → local GPU + Cloudflare Tunnel, $0/mo.** Caveats: uptime = your PC's uptime, and
    the **free tier caps request bodies at 100 MB** → large videos need presigned direct-to-R2.
  - **Real service → Modal.** Scale-to-zero, per-second billing (A10G ≈ $1.10/hr *while running*),
    $30/mo free credits. At ~100 req/mo × 3 s ≈ **$0.09/mo**.
  - **Rejected:** a 24/7 rented GPU VM — RunPod 4090 **$245–500/mo**, Lambda A10 **$540/mo**,
    HF persistent T4 ≈ $290/mo.

---

## Critical path

```
T1 (env) ─┬─> T10–T19 (data integrity) ──> T20–T24 (models) ──> T25–T35 (training loop)
          │                                                            │
          │                                                            v
          │                                                     T34 SMOKE ON CPU  ← proves it works
          │                                                            │
          ├─> T36–T45 (preprocessing) ────────────────────────────────>│
          │                                                            v
          └─> T46–T53 (explainability) ──> T54–T59 (backend) ──> T60–T66 (frontend)
                                                                       │
T2 (EULA, day 0) ····· 2 days–2 weeks ····> T67–T71 (real data) ──────>│
                                                                       v
                                                        T72–T80 (train on college GPU)
                                                                       │
                                                                       v
                                                                 T81–T86 (ship)
```

**Do today:** T2 (ten minutes, two weeks of latency), then T1.
**The milestone that de-risks everything:** T34 — a smoke test passing on CPU means the entire training
pipeline is correct before you ever queue for the college GPU.
