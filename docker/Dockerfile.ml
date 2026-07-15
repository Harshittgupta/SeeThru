# SEETHRU - ML image
# Container for training / inference jobs.
#
# NOTE: For GPU-accelerated training, substitute a CUDA-enabled base image
# such as `pytorch/pytorch:2.x.x-cuda12.1-cudnn8-runtime` and run with the
# NVIDIA container runtime (e.g. `--gpus all`).
FROM python:3.11-slim

WORKDIR /app

# System dependencies required by OpenCV (cv2):
#   libgl1        -> provides libGL.so.1 for cv2 image operations
#   libglib2.0-0  -> provides libgthread / glib runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
COPY . .

# Default command: simple readiness message. Actual training/inference jobs
# are typically launched via `docker compose exec ml python ml/train.py`.
CMD ["python", "-c", "print('SEETHRU ML container ready')"]
