# interactor-trellis2-image-to-textured-mesh -- vast.ai worker, RFD 0036.
#
# Target: RTX 4090, 24GB. Base image for 4 other catalog entries (RFD 0038) --
# a packaging mistake here costs five models, not one.
#
# Weights: 8.0 GB bf16, microsoft/TRELLIS.2-4B on Hugging Face. RFD 0038's own
# cog.yaml pointed weight URLs at weights.invalid (RFC 2606's reserved "known
# invalid" domain) -- a placeholder, never a real pin. Real source: HF, above,
# verified against the actual model card, not guessed.
#
# Build:
#   docker build -t weftspun/trellis2-base .
# Test the handler contract with no GPU and no weights:
#   docker build --target contract -t weftspun/trellis2-base:contract .
#   docker run --rm -p 8000:8000 weftspun/trellis2-base:contract

# ---------------------------------------------------------------------
# The contract stage. Handler + usd-core, no model.
# ---------------------------------------------------------------------
FROM python:3.11-slim AS contract

WORKDIR /app
RUN pip install --no-cache-dir usd-core==25.5 fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3
COPY server.py /app/server.py
COPY test_input.json /app/test_input.json
ENV WEFTSPUN_STUB=1 PORT=8000
EXPOSE 8000
CMD ["python", "/app/server.py"]

# ---------------------------------------------------------------------
# The worker. CUDA 12.4, matching upstream's own pinned torch/CUDA pairing.
# ---------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS worker

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRELLIS2_SRC=/src/TRELLIS.2 \
    TRELLIS2_WEIGHTS=microsoft/TRELLIS.2-4B

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3-pip git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      torch==2.5.1 numpy==2.1.3 safetensors==0.4.5 pillow==12.0.0 \
      transformers==4.46.3 usd-core==25.5 huggingface_hub==0.26.2 \
      trimesh==4.10.1 opencv-python-headless==4.12.0.88 \
      fastapi==0.115.5 uvicorn==0.32.1 pydantic==2.10.3

# Upstream itself, and o_voxel (its own postprocessing package, imported
# directly per the README's documented usage). Pin the commit: master moves,
# and a moved master changes the mesh with no build change to show for it.
ARG TRELLIS2_REF=main
RUN git clone https://github.com/microsoft/TRELLIS.2.git /src/TRELLIS.2 \
    && git -C /src/TRELLIS.2 checkout "${TRELLIS2_REF}" \
    && pip install --no-cache-dir -e /src/TRELLIS.2

WORKDIR /app
COPY server.py /app/server.py

# The pipeline calls from_pretrained("microsoft/TRELLIS.2-4B") at first use,
# not at build time -- huggingface_hub caches it in the image layer via a
# warm-up import, so a cold start pulls nothing.
RUN python3 -c "from trellis2.pipelines import Trellis2ImageTo3DPipeline; \
Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')"

ENV PORT=8000
EXPOSE 8000

CMD ["python3", "-u", "/app/server.py"]
