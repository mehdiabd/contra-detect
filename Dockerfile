ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

WORKDIR /app

ARG PIP_INDEX=https://nexus.repo.stinascloud.ir/repository/pypi/simple

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url ${PIP_INDEX} \
      torch==2.14.0 \
    && pip install --no-cache-dir -r requirements-runtime.txt --index-url ${PIP_INDEX}

COPY config.py ./config.py
COPY src ./src
COPY data/sample ./data/sample
COPY models/release.json /opt/contra-detect/models/release.json
COPY models/saved/best_model/inference_config.json /opt/contra-detect/models/saved/best_model/inference_config.json
COPY models/saved/targeted_v1 /opt/contra-detect/models/saved/targeted_v1
COPY models/saved/parsnlu_farstail /opt/contra-detect/models/saved/parsnlu_farstail
COPY models/saved/mdeberta_nli /opt/contra-detect/models/saved/mdeberta_nli

RUN python -c "from pathlib import Path; from src.models.release import verify_model_release; verify_model_release(Path('/opt/contra-detect/models/saved/best_model'))"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/opt/contra-detect/models/saved/best_model \
    REQUIRE_MODEL_RELEASE=true \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
