# Live demo image for the donation-platform recommender (Slice 4).
#
# Loads the trained two-tower + demo bundle from app/artifacts/ (committed to
# git so we don't retrain at boot). Lean runtime — no sentence-transformers,
# no implicit, no matplotlib — those were training-only deps.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System deps: faiss-cpu wheels need libgomp1 at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so Docker layer cache is friendly.
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/app/requirements.txt

# Copy the recommender source (bench/models/two_tower.py is needed because the
# service imports its `Recommender`/`TwoTowerRecommender` classes at load time).
COPY bench/__init__.py /app/bench/__init__.py
COPY bench/models /app/bench/models
COPY app /app/app

EXPOSE 8000

CMD ["python", "-m", "app.main"]
