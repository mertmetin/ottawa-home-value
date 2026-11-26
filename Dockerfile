# Runs training / evaluation / the photo job / the API next to the database
# that holds the licensed data. CPU-only PyTorch keeps the image small enough
# for a 4 GB VPS. Mount the checkout at /app to run new code without a rebuild.
#
#   docker build -t ottawa-home-value .
#   docker run --rm --network <db network> -e DATABASE_URL=... \
#     -v $PWD:/app ottawa-home-value python -m src.evaluate --data db --out results/real
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/models/hf
# Dependencies first (cached), against a stub package; the code comes last.
COPY pyproject.toml README.md ./
RUN mkdir src && touch src/__init__.py \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[db,api,photos]"
# LightGBM's OpenMP runtime (torch used to bring its own copy along).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY . .
