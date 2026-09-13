#!/usr/bin/env bash
# GCP Cloud Run deploy (requires gcloud auth + Artifact Registry).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_ID="${GCP_PROJECT_ID:?set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-west1}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/audio-ml/audio-ml-jd-lab:latest"

docker build -f "${ROOT}/deploy/Dockerfile" -t "${IMAGE}" "${ROOT}"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker push "${IMAGE}"
gcloud run deploy audio-ml-jd-lab \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --port 8080
