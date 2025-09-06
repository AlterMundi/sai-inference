# SAI Inference Service Docker Image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY run.py .

# Create models directory
RUN mkdir -p models

# Expose ports
EXPOSE 8888 9090

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SAI_HOST=0.0.0.0
ENV SAI_PORT=8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8888/api/v1/health'); exit(0 if r.status_code == 200 else 1)"

# Run the service
CMD ["python", "run.py"]