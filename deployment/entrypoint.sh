#!/bin/bash
set -e

SERVICE_TYPE=${SERVICE_TYPE:-scheduler}

# Ensure data directories exist
mkdir -p /data/images /data/models /logs

# Initialize database if needed
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
            --streams ${STREAMS:-"in_gate"} \
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