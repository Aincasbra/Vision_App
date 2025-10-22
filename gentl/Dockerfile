# syntax=docker/dockerfile:1
# JetPack 5.1.1 base (L4T r35.3.1, CUDA 11.4)
FROM nvcr.io/nvidia/l4t-jetpack:r35.3.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake git wget curl unzip \
    libgl1 libglib2.0-0 libgtk2.0-0 \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    gpiod \
    xauth \
    libgirepository1.0-dev python3-gi gir1.2-gtk-3.0 \
    && rm -rf /var/lib/apt/lists/*

# Optional Aravis (enable if available in apt)
# RUN apt-get update && apt-get install -y --no-install-recommends gir1.2-aravis-0.8 aravis-tools && rm -rf /var/lib/apt/lists/* || true

WORKDIR /app

# Python deps from captured requirements (if present)
COPY replication_files/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    if [ -s /tmp/requirements.txt ]; then \
      python3 -m pip install -r /tmp/requirements.txt; \
    else \
      python3 -m pip install ultralytics opencv-python psutil pyyaml; \
    fi

# App code
COPY . /app

# Default command (can be overridden in docker-compose)
CMD ["python3", "gentl/PruebaAravis.py"]
