# Runs training / evaluation / the API next to the database that holds the
# licensed data. CPU-only PyTorch keeps the image small enough for a 4 GB VPS.
#
#   docker build -t ottawa-home-value .
#   docker run --rm --network <db network> -e DATABASE_URL=... \
#     -v $PWD/results:/app/results ottawa-home-value \
#     python -m src.evaluate --data db --out results/real
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY src src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[db,api]"
COPY . .
