FROM public.ecr.aws/docker/library/python:3.10-slim-bullseye

ARG USER=signal-aichat
ARG HOME_DIR=/home/$USER

ENV CMAKE_ARGS "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN adduser \
	--disabled-password \
	--uid 1000 \
	$USER

USER $USER
WORKDIR $HOME_DIR

COPY requirements.txt ai.py signal_aichat.py ./
COPY models/ /models/

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "signal_aichat.py"]
