#!/bin/bash
# NE301 project initialization script
# Automatically check and clone NE301 project to the host directory on container start (if empty)
# Also automatically pull NE301 Docker image for model compilation

set -e

NE301_HOST_DIR="/workspace/ne301"  # Host directory path mounted by Docker Compose
NE301_DOCKER_IMAGE="${NE301_DOCKER_IMAGE:-camthink/ne301-dev:latest}"  # NE301 Docker image for compilation

# Check if running inside a Docker container
if [ -f "/.dockerenv" ]; then
    # Inside container: check host directory (via mount)
    if [ -d "$NE301_HOST_DIR" ]; then
        # Host directory exists (via docker-compose volume mount)
        echo "[NE301 Init] Detected host directory mount: $NE301_HOST_DIR"
        
        # Check if directory is empty or missing key files
        if [ ! "$(ls -A $NE301_HOST_DIR 2>/dev/null)" ] || [ ! -d "$NE301_HOST_DIR/Model" ]; then
            echo "[NE301 Init] Host directory is empty or incomplete, cloning from GitHub..."
            # If directory not empty but missing files, clean up first
            if [ "$(ls -A $NE301_HOST_DIR 2>/dev/null)" ]; then
                rm -rf "$NE301_HOST_DIR"/*
            fi
            git clone https://github.com/camthink-ai/ne301.git "$NE301_HOST_DIR"
            echo "[NE301 Init] Clone completed"
        else
            echo "[NE301 Init] Complete NE301 project found in host directory, skipping clone"
        fi
    else
        echo "[NE301 Init] Warning: Host directory mount not detected ($NE301_HOST_DIR)"
        echo "[NE301 Init] Please make sure ./ne301:/workspace/ne301 is configured in docker-compose.yml"
        echo "[NE301 Init] Falling back to container-internal directory..."
        
        # Fallback: use container internal directory
        NE301_CONTAINER_DIR="/app/ne301"
        if [ ! -d "$NE301_CONTAINER_DIR" ]; then
            echo "[NE301 Init] Cloning to container internal directory..."
            git clone https://github.com/camthink-ai/ne301.git "$NE301_CONTAINER_DIR"
        fi
    fi
else
    # On host: check project root directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
    NE301_DIR="$PROJECT_ROOT/ne301"
    
    if [ ! -d "$NE301_DIR" ]; then
        echo "[NE301 Init] Cloning NE301 project to: $NE301_DIR"
        git clone https://github.com/camthink-ai/ne301.git "$NE301_DIR"
    else
        echo "[NE301 Init] NE301 project directory already exists: $NE301_DIR"
    fi
fi

# Pull NE301 Docker image for model compilation (if Docker is available)
# Note: This section uses set +e to prevent pull failures from stopping container startup
set +e
if command -v docker >/dev/null 2>&1; then
    echo "[NE301 Init] Checking NE301 Docker image: $NE301_DOCKER_IMAGE"
    
    # Check if image exists locally
    if docker images -q "$NE301_DOCKER_IMAGE" 2>/dev/null | grep -q .; then
        echo "[NE301 Init] Docker image $NE301_DOCKER_IMAGE already exists locally"
    else
        echo "[NE301 Init] Docker image $NE301_DOCKER_IMAGE not found, pulling..."
        
        # Detect system architecture for cross-platform support
        ARCH=$(uname -m)
        PLATFORM_FLAG=""
        if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
            # ARM64 architecture needs to pull AMD64 image (using --platform)
            PLATFORM_FLAG="--platform linux/amd64"
            echo "[NE301 Init] Detected ARM64 architecture, pulling AMD64 image for compatibility"
        fi
        
        # Pull Docker image (with timeout if available, don't fail if pull fails - image might be pulled later)
        PULL_CMD="docker pull $PLATFORM_FLAG $NE301_DOCKER_IMAGE"
        if command -v timeout >/dev/null 2>&1; then
            # Use timeout if available (5 minutes timeout)
            if timeout 300 $PULL_CMD 2>&1; then
                echo "[NE301 Init] Successfully pulled Docker image: $NE301_DOCKER_IMAGE"
            else
                echo "[NE301 Init] Warning: Failed to pull Docker image $NE301_DOCKER_IMAGE (timeout or error)"
                echo "[NE301 Init] This may cause NE301 model compilation to fail. Please check network connection and try again."
                echo "[NE301 Init] You can manually pull the image later with: docker pull $PLATFORM_FLAG $NE301_DOCKER_IMAGE"
            fi
        else
            # Fallback: pull without timeout (may hang if network is slow)
            if $PULL_CMD 2>&1; then
                echo "[NE301 Init] Successfully pulled Docker image: $NE301_DOCKER_IMAGE"
            else
                echo "[NE301 Init] Warning: Failed to pull Docker image $NE301_DOCKER_IMAGE"
                echo "[NE301 Init] This may cause NE301 model compilation to fail. Please check network connection and try again."
                echo "[NE301 Init] You can manually pull the image later with: docker pull $PLATFORM_FLAG $NE301_DOCKER_IMAGE"
            fi
        fi
    fi
else
    echo "[NE301 Init] Warning: Docker command not found, skipping image pull"
    echo "[NE301 Init] NE301 model compilation requires Docker. Please ensure Docker is installed and accessible."
fi
set -e  # Re-enable error exit

echo "[NE301 Init] Done"
