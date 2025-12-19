# Dockerfile
# Fair Credit Score Prediction - Training/Batch Pipeline

FROM python:3.11-slim

# Metadata
LABEL maintainer="Pierce Brookins & Aleya Banthavong"
LABEL description="Fair Credit Score Prediction Model - Training Pipeline"
LABEL version="0.1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config/ ./config/
COPY src/ ./src/
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/explanations /app/reports /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

# Default command
CMD ["python", "-m", "src.main", "--help"]
