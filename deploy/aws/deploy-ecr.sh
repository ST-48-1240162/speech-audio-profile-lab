#!/usr/bin/env bash
# Push container to Amazon ECR for AWS Batch / ECS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ACCOUNT_ID="${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID}"
REGION="${AWS_REGION:-us-west-2}"
REPO="audio-ml-jd-lab"
IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest"

aws ecr describe-repositories --repository-names "${REPO}" --region "${REGION}" 2>/dev/null \
  || aws ecr create-repository --repository-name "${REPO}" --region "${REGION}"

aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin \
  "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

docker build -f "${ROOT}/deploy/Dockerfile" -t "${IMAGE}" "${ROOT}"
docker push "${IMAGE}"
echo "Pushed ${IMAGE}"
