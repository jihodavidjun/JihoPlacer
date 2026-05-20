FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PINE_SUBMISSION_TUNED=1 \
    PINE_HEURISTIC_TIME=600

WORKDIR /workspace

# Install only lightweight Python dependencies not guaranteed by the base
# PyTorch image. The challenge package is supplied by the evaluator.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir numpy scipy tqdm absl-py matplotlib networkx pandas

COPY pine_place ./pine_place
COPY submissions ./submissions
COPY pyproject.toml README.md LICENSE.md ./

ENV PYTHONPATH=/workspace:${PYTHONPATH}

