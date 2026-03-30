# Use python 3.9 base image
FROM python:3.9-slim

# Set working directory essentially pointing to app root
WORKDIR /app

# Install system GUI and display dependencies required by OpenCV and Matplotlib interactive windows
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    pkg-config \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install all python library requirements natively required by the codebase
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    scikit-image

# Move all project files into the container
COPY . /app/

# Set default execution location to the python code folder where scripts expect to be run from
WORKDIR /app/python

# Keep container alive by default so users can run scripts interactively via bash
CMD ["/bin/bash"]
