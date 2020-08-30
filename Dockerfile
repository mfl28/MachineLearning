FROM jupyter/base-notebook
COPY requirements.txt /home/jovyan/work
WORKDIR /home/jovyan/work
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*
USER jovyan
RUN sed -i /jupyter/d requirements.txt && \ 
    pip install -r requirements.txt --no-cache-dir
