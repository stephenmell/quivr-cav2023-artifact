FROM debian:bullseye

# Install base utilities
RUN apt-get update && apt-get install -y wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda

COPY conda.yaml /tmp
RUN /opt/conda/bin/conda env update --file /tmp/conda.yaml --prune
ENV PATH="$PATH:/opt/conda/bin"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility