#!/bin/bash
# CalmCrypto MCP Server Deployment Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure mcp-shared network exists
docker network inspect mcp-shared >/dev/null 2>&1 || docker network create mcp-shared

# Ensure .env.prod exists
if [ ! -f .env.prod ]; then
    echo "Creating .env.prod from .env.local.example..."
    if [ -f .env.local.example ]; then
        cp .env.local.example .env.prod
        echo "Please edit .env.prod with your credentials before running again."
        exit 1
    else
        echo "Error: .env.local.example not found"
        exit 1
    fi
fi

# Ensure data directory exists with proper permissions
DATA_DIR="/home/ubuntu/mcp/data"
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory: $DATA_DIR"
    sudo mkdir -p "$DATA_DIR"
    sudo chown -R ubuntu:ubuntu "$DATA_DIR"
fi

# Stop existing containers
echo "Stopping existing containers..."
docker compose -f docker-compose.prod.yml down 2>/dev/null || true

# Remove old images to force rebuild
echo "Removing old images..."
docker rmi mcp-mcp-calmcrypto 2>/dev/null || true

# Build and start
echo "Building and starting CalmCrypto MCP server..."
docker compose -f docker-compose.prod.yml up --build -d

# Show status
echo ""
echo "CalmCrypto MCP server started!"
echo "Endpoint: http://localhost:8007/calmcrypto/"
echo ""
echo "View logs: docker logs -f mcp-calmcrypto"
