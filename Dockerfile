FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install pydantic rich pyyaml
RUN pip install lightning
RUN pip install torchvision
RUN pip install opencv-python einops
RUN pip install imageio scipy pytest

RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

CMD ["bash"]