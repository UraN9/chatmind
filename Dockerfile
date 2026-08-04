FROM python:3.12-slim

# Tesseract binary for processing/ocr.py. psycopg[binary] and
# pgvector ship their own wheels, so no extra Postgres client
# libraries are needed here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so `pip install` is cached as its own layer
# and only re-runs when dependencies actually change.
COPY requirements.txt .

# CPU-only torch build: much smaller than the default CUDA-bundled
# wheel, and this bot has no GPU to use anyway. --extra-index-url
# makes pip prefer the CPU build for torch while still resolving
# everything else from PyPI as normal.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

# Without this, Python block-buffers stdout when it's not attached to
# a real terminal (which is always the case in a container) -- log
# lines sit in an internal buffer and may not show up in `docker logs`
# until the buffer fills or the process exits.
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "bot.main"]