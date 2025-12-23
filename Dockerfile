# Multi-stage build Dockerfile
# Stage 1: Build frontend React application
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend dependency files
COPY frontend/package*.json ./

# Install frontend dependencies
RUN npm ci --legacy-peer-deps

# Copy frontend source code
COPY frontend/ ./

# Build frontend application
RUN npm run build

# Stage 2: Build backend Python application
FROM python:3.10-slim AS backend-builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend dependency files
COPY backend/requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies (including git for cloning NE301 project, Docker CLI for compiling NE301 models,
# and mosquitto/mosquitto-clients for managing and testing MQTT)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    git \
    ca-certificates \
    gnupg \
    lsb-release \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgthread-2.0-0 \
    mosquitto \
    mosquitto-clients \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Install Docker CLI (for executing docker commands inside container)
# Note: This installation requires access to Docker official repository, may need to configure mirror if network issues occur
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/* \
    && docker --version

# Copy Python dependencies from build stage
COPY --from=backend-builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy backend code
COPY backend/ ./backend/

# Copy frontend build artifacts from stage 1
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Create necessary directories
RUN mkdir -p datasets data backend/data

# Copy initialization script
COPY scripts/init-ne301.sh /usr/local/bin/init-ne301.sh
RUN chmod +x /usr/local/bin/init-ne301.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Startup command (initialize NE301 first, then start application)
CMD ["sh", "-c", "/usr/local/bin/init-ne301.sh && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"]
