ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

FROM --platform=linux/amd64 ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WAKAREERU_SERVICE_CONFIG=configs/service_config.yaml

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-image.txt ./
RUN pip install -r requirements-image.txt

COPY . ./
RUN pip install --no-deps -e .

CMD ["python", "-m", "wakareeru_inference.handler"]
