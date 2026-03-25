FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright
RUN playwright install chromium && playwright install-deps chromium

# Copy application
COPY . .
RUN pip install -e .

# Create data directories
RUN mkdir -p data/raw data/markdown data/index

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-m", "vnull.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
