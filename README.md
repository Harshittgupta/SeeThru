# SEETHRU

SEETHRU checks whether a face in a photo or video is **real or AI-manipulated**,
and — this is the point — **shows you why** it decided that, instead of just
handing back a number.

You upload an image or a video. It finds the face, analyses it three different
ways, and returns a verdict (**real / fake / uncertain**) alongside a plain
explanation: a heatmap of where it looked, which kind of evidence drove the call,
and (for video) which moments look suspicious.

---

## How it works

Deepfakes leave different kinds of traces, and no single check catches all of
them, so SEETHRU runs three in parallel and combines them.

| Branch | What it looks at | Catches |
|---|---|---|
| **Spatial** | the face pixels (EfficientNet-B3) | blending seams, warped texture, odd edges |
| **Frequency** | the image's Fourier spectrum (FFT + a small CNN) | AI-generation patterns invisible to the eye |
| **Temporal** *(video)* | how the face changes across frames (BiLSTM) | flicker, jitter, an identity that won't hold still |

Their outputs are fused and passed to a classifier that decides real vs fake.
Face detection and alignment (RetinaFace) run first so every analysis sees a
clean, centered 224×224 crop.

## What you get back

Not just a verdict — an explanation you can actually read:

- **A heatmap** over the face showing where the model focused.
- **Which branch mattered.** SEETHRU removes each branch in turn and measures how
  much the answer changes, so it can say *"this call was driven mostly by
  frequency evidence"* — a measured effect, not a guess.
- **A frequency profile** — how much high-frequency energy the face has compared
  to a normal one (where AI upsampling artifacts show up).
- **A timeline** (video) marking which sampled moments look manipulated.

### It's built to be honest about what it doesn't know

- It answers **"uncertain"** when the signal is weak, instead of forcing a
  real/fake guess.
- It does **not** show a confidence percentage, because the model isn't
  calibrated yet and a percentage would imply a precision it doesn't have. You
  see a weak / moderate / strong signal instead.
- Every result carries a disclaimer: this is a research tool, **not forensic
  evidence**.

---

## Status

The full stack is built and tested. It has **not been trained yet** — that needs
the datasets (being downloaded) and a GPU.

| Part | State |
|---|---|
| Models (spatial / frequency / temporal / fusion) | ✅ built + tested |
| Preprocessing & dataset handling | ✅ built + tested |
| Training + evaluation pipeline | ✅ built, passes a CPU smoke test |
| Explainability | ✅ built + tested |
| Backend API (FastAPI) | ✅ built + tested |
| Frontend (React) | ✅ built + tested |
| Trained model weights | ❌ not yet — needs data + GPU |

Roughly 330 automated tests pass across the Python and TypeScript code. What's
left is: get the data → train on a GPU → deploy. See
[docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) for the full task list.

---

## Running it

**Requirements:** Python 3.11 (TensorFlow, via RetinaFace, doesn't support newer),
and Node 20+ for the frontend.

```bash
# Backend
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

The backend won't serve predictions without trained weights (an untrained model
would return confident nonsense). To click through the UI end-to-end before
training exists, run it in wiring-only mode:

```bash
# backend, with the safety gate off (returns meaningless verdicts — plumbing only)
set SEETHRU_ALLOW_UNTRAINED=true
uvicorn backend.main:app --port 8000

# frontend, in another terminal
cd frontend && npm run dev        # http://localhost:5173
```

Or the whole stack in Docker: `docker compose up` (frontend on :5173, proxying
the API on :8000).

---

## Project layout

```
backend/    FastAPI service — accepts uploads, runs inference, serves results
frontend/   React + Vite web app
ml/         models, training, evaluation, and the explainability engine
data/       dataset loaders, splitting, and preprocessing scripts
docker/     Dockerfiles
docs/       build plan, design decisions (ADRs), and the original audit
```

## Limitations

Be clear-eyed about this. Deepfake detectors trained on one dataset **degrade a
lot** on manipulation methods, compression, and cameras they haven't seen — a
model that scores ~99% on its own test set typically drops to ~65–75% on a
different dataset. SEETHRU is a research and educational project. Its output must
not be treated as proof, and it is not validated for legal, journalistic, or
evidentiary use. Video analysis assumes a single subject.

## License & data

**Code** is under the [Apache License 2.0](LICENSE).

**Datasets and trained weights are not, and never will be in this repo.** SEETHRU
trains on FaceForensics++ and Celeb-DF v2, which come under signed, research-only
agreements that forbid redistribution. So:

- No dataset media is committed here, and none may be added — you obtain the
  datasets yourself under your own agreement.
- Trained weights are derived from that restricted data and inherit its terms.
  They aren't Apache-licensed and are never baked into a public image.

See [NOTICE](NOTICE) for the full terms and dataset citations.
