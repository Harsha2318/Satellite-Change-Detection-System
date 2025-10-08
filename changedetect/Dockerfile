FROM python:3.10-slim

LABEL maintainer="PS-10 Change Detection Team"
LABEL description="Satellite image change detection for the PS-10 challenge"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install GDAL python bindings
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN pip install --no-cache-dir GDAL==$(gdal-config --version)

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Set entrypoint
ENTRYPOINT ["python", "-m", "changedetect.src.main"]

# Default command
CMD ["--help"]