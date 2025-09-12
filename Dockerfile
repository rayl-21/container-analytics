# Multi-stage build for Container Analytics Production
# Stage 1: Base builder with system dependencies
FROM python:3.10-slim-bullseye as base-builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    pkg-config \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    # Tesseract OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # Chrome dependencies for Selenium
    wget \
    gnupg \
    unzip \
    # Additional system libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome or Chromium for Selenium based on architecture
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
        echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list && \
        apt-get update && \
        apt-get install -y google-chrome-stable && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && \
        apt-get install -y chromium chromium-driver && \
        rm -rf /var/lib/apt/lists/* && \
        ln -s /usr/bin/chromium /usr/bin/google-chrome-stable && \
        ln -s /usr/bin/chromedriver /usr/local/bin/chromedriver; \
    fi

# Install ChromeDriver for amd64 architecture only
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        CHROME_DRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE) && \
        wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip && \
        unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/ && \
        rm /tmp/chromedriver.zip && \
        chmod +x /usr/local/bin/chromedriver; \
    fi

# Stage 2: Python dependencies builder
FROM base-builder as python-builder

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Add Prometheus monitoring
RUN pip install prometheus_client==0.17.1

# Stage 3: Production runtime
FROM python:3.10-slim-bullseye as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV runtime dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # Tesseract OCR runtime
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # Chrome runtime dependencies
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgcc1 \
    libgconf-2-4 \
    libgdk-pixbuf2.0-0 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    lsb-release \
    xdg-utils \
    # Additional libraries
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Chrome or Chromium based on architecture
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
        echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list && \
        apt-get update && \
        apt-get install -y google-chrome-stable && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && \
        apt-get install -y chromium chromium-driver && \
        rm -rf /var/lib/apt/lists/* && \
        ln -s /usr/bin/chromium /usr/bin/google-chrome-stable; \
    fi

# Copy virtual environment from builder
COPY --from=python-builder /opt/venv /opt/venv

# Copy ChromeDriver from builder (skip if using system chromium-driver)
RUN if [ -f "/usr/local/bin/chromedriver" ]; then \
        echo "Using system chromedriver"; \
    else \
        echo "ChromeDriver will be from chromium-driver package"; \
    fi

# Create application user for security
RUN groupadd -r analytics && useradd -r -g analytics -u 1001 analytics \
    && mkdir -p /app /data /logs \
    && chown -R analytics:analytics /app /data /logs

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=analytics:analytics . .

# Create required directories and set permissions
RUN mkdir -p /app/data/images /app/data/models /app/logs \
    && chmod +x /app/entrypoint.sh || true \
    && chown -R analytics:analytics /app

# Download YOLOv12 model if not present (fallback)
RUN mkdir -p /app/data/models
# Note: The model should be copied from the host, but we provide a fallback download
RUN if [ ! -f /app/data/models/yolov8x.pt ]; then \
        python -c "from ultralytics import YOLO; YOLO('yolov8x.pt').save('/app/data/models/yolov8x.pt')" 2>/dev/null || \
        echo "Warning: Could not download YOLO model. Ensure yolov8x.pt is mounted or copied."; \
    fi

# Switch to non-root user
USER analytics

# Create health check script
COPY --chown=analytics:analytics <<'EOF' /app/health_check.py
#!/usr/bin/env python3
"""Health check script for Container Analytics services."""

import sys
import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

def check_database_health():
    """Check if database is accessible and has recent data."""
    try:
        db_path = Path("/data/database.db")
        if not db_path.exists():
            return False, "Database file does not exist"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if we have recent image downloads (within last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        cursor.execute(
            "SELECT COUNT(*) FROM images WHERE created_at > ?",
            (one_hour_ago.isoformat(),)
        )
        recent_count = cursor.fetchone()[0]
        conn.close()
        
        if recent_count > 0:
            return True, f"Found {recent_count} recent images"
        else:
            return False, "No recent images found"
            
    except Exception as e:
        return False, f"Database error: {str(e)}"

def check_scheduler_health():
    """Check scheduler health based on recent activity."""
    try:
        health_file = Path("/data/.scheduler_health")
        if not health_file.exists():
            return False, "Scheduler health file not found"
        
        # Check if health file was updated recently (within 15 minutes)
        mtime = datetime.fromtimestamp(health_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(minutes=15):
            return False, "Scheduler health file is stale"
        
        health_data = json.loads(health_file.read_text())
        status = health_data.get("status", "unknown")
        
        return status == "healthy", f"Scheduler status: {status}"
        
    except Exception as e:
        return False, f"Scheduler health check error: {str(e)}"

def check_detector_health():
    """Check if detector service is responsive."""
    try:
        # Check if detector process files exist
        detector_pid = Path("/data/.detector_pid")
        if detector_pid.exists():
            return True, "Detector process file found"
        else:
            return True, "Detector not running (optional service)"
            
    except Exception as e:
        return False, f"Detector health check error: {str(e)}"

def main():
    """Main health check function."""
    checks = [
        ("database", check_database_health),
        ("scheduler", check_scheduler_health),
        ("detector", check_detector_health),
    ]
    
    all_healthy = True
    results = {}
    
    for name, check_func in checks:
        healthy, message = check_func()
        results[name] = {"healthy": healthy, "message": message}
        if not healthy:
            all_healthy = False
    
    # Write health status
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "overall_healthy": all_healthy,
        "checks": results
    }
    
    health_file = Path("/data/.health_status")
    health_file.write_text(json.dumps(health_status, indent=2))
    
    if all_healthy:
        print("All health checks passed")
        sys.exit(0)
    else:
        print("Some health checks failed:")
        for name, result in results.items():
            if not result["healthy"]:
                print(f"  {name}: {result['message']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Create entrypoint script for different services
COPY --chown=analytics:analytics <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

# Set default service type
SERVICE_TYPE=${SERVICE_TYPE:-scheduler}

# Ensure data directories exist
mkdir -p /data/images /data/models /logs

# Initialize database if it doesn't exist or is empty
if [ ! -f /data/database.db ] || [ ! -s /data/database.db ]; then
    echo "Initializing database..."
    python -c "
import sys
sys.path.insert(0, '/app')
from modules.database.models import init_database
init_database()
print('Database initialized successfully')
" || echo "Warning: Database initialization failed, continuing..."
fi

echo "Starting Container Analytics service: $SERVICE_TYPE"

case "$SERVICE_TYPE" in
    "scheduler")
        echo "Starting image download scheduler..."
        exec python -m modules.downloader.scheduler \
            --streams ${STREAMS:-"in_gate out_gate"} \
            --interval ${DOWNLOAD_INTERVAL_MINUTES:-10}
        ;;
    "detector")
        echo "Starting YOLO detector with file watching..."
        exec python -m modules.detection.yolo_detector --watch
        ;;
    "dashboard")
        echo "Starting Streamlit dashboard..."
        exec streamlit run app.py \
            --server.port=${STREAMLIT_PORT:-8501} \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    "metrics")
        echo "Starting Prometheus metrics server..."
        exec python -m utils.metrics_server
        ;;
    *)
        echo "Unknown service type: $SERVICE_TYPE"
        echo "Available types: scheduler, detector, dashboard, metrics"
        exit 1
        ;;
esac
EOF

RUN chmod +x /app/entrypoint.sh /app/health_check.py

# Expose ports for different services
EXPOSE 8501 9090

# Health check
HEALTHCHECK --interval=2m --timeout=30s --start-period=1m --retries=3 \
    CMD python /app/health_check.py

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]