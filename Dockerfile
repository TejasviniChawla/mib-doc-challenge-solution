FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY solution /app/solution
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp

ENTRYPOINT ["/app/run.sh"]
