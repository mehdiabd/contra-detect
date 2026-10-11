ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

WORKDIR /app

ARG PIP_INDEX=https://pypi.org/simple

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
COPY models/saved/best_model ./models/saved/best_model
COPY models/saved/targeted_v1 ./models/saved/targeted_v1
COPY models/saved/parsnlu_farstail ./models/saved/parsnlu_farstail
COPY models/saved/mdeberta_nli ./models/saved/mdeberta_nli

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models/saved/best_model

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
