# The sandbox image for container mode: Python with numpy, nothing else. The agent's code runs in it as
# nobody, with no network and a read-only root filesystem; only the sandbox folder is mounted.
#
# Build it with `python -m level1.sandbox build`, which passes this machine's Python version and the numpy
# version installed here (the pin in requirements.txt when numpy is not installed).
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

ARG NUMPY_VERSION
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir "numpy==${NUMPY_VERSION}" \
 && rm -rf /root/.cache

WORKDIR /box
