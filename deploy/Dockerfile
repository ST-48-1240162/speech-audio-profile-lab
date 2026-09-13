FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
ENV PYTHONPATH=/app/src
ENV HF_HOME=/tmp/hf_cache

EXPOSE 8080
CMD ["uvicorn", "audio_ml_jd_lab.serve:app", "--host", "0.0.0.0", "--port", "8080"]
