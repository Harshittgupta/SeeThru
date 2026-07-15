# SEETHRU

**SEETHRU** is a deepfake detection system that uses deep learning to identify manipulated faces and media. It analyzes images and video to flag synthetic or tampered content, combining computer-vision preprocessing with PyTorch-based neural network models served through a modern web stack. The goal is to provide fast, reliable, and explainable detection of deepfakes for end users and downstream applications.

## Repository Structure

```
SEETHRU/
├── backend/      FastAPI application serving the detection API
├── frontend/     React web application UI
├── ml/           PyTorch models, training scripts, and inference code
├── data/         Dataset management and preprocessing scripts
├── docker/       Dockerfiles for containerizing services
└── notebooks/    Jupyter notebooks for experiments and analysis
```

- **`backend/`** — FastAPI application serving the detection API. Exposes REST endpoints that accept media uploads and return detection results.
- **`frontend/`** — React web application UI. Provides the user-facing interface for submitting media and viewing detection results.
- **`ml/`** — PyTorch models, training scripts, and inference code. Houses model architectures, training pipelines, and the inference logic used by the backend.
- **`data/`** — Dataset management and preprocessing scripts. Handles dataset download, organization, cleaning, and preprocessing.
- **`docker/`** — Dockerfiles for containerizing services. Contains the container definitions used to build and deploy each service.
- **`notebooks/`** — Jupyter notebooks for experiments and analysis. Used for exploratory research, model evaluation, and visualization.

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/SEETHRU.git
   cd SEETHRU
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # On macOS/Linux
   source .venv/bin/activate
   # On Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the full stack with Docker**

   ```bash
   docker compose up
   ```

## Tech Stack

- **PyTorch** — deep learning framework for model training and inference
- **FastAPI** — high-performance Python web framework powering the detection API
- **React** — front-end library for the web user interface
- **OpenCV** — computer-vision library for image and video preprocessing
- **Docker** — containerization for building and deploying services

## License

This project is released under a license to be determined. See the `LICENSE` file for details (placeholder).
