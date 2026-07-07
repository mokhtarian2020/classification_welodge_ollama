# Use official slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Startup script: wait for Ollama before accepting requests
RUN chmod +x app_entrypoint.sh

# Expose FastAPI port
EXPOSE 8003

# Health check — verifies API + Ollama model availability
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8003/health || exit 1

# Start via entrypoint (waits for Ollama, then uvicorn)
CMD ["./app_entrypoint.sh"]
