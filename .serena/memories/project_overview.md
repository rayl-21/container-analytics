# Container Analytics Project Overview

## Project Summary
Container Analytics is a Python-based MVP application that automatically downloads port gate camera images from Dray Dog and derives analytics using YOLOv12 computer vision with a Streamlit dashboard for visualization.

## Current Status (2025-09-12)

### Completed Features ✅
- **Database Module**: SQLAlchemy ORM with SQLite (89% test coverage)
- **Analytics Engine**: KPI calculations and aggregations
- **Test Infrastructure**: 234 tests across all modules
- **Real Data Pipeline**: Complete with database integration
- **Automated Scheduling**: Production-ready with retry logic and monitoring
- **Downloader Module**: Selenium-based with simplified date navigation (no timezone checking)
- **Detection Module**: YOLOv12 integration with watch mode and batch processing
- **Dashboard**: Streamlit multi-page application with real data
- **Container Tracking**: ByteTrack with OCR (EasyOCR + Tesseract)
- **Production Infrastructure**: Docker deployment stack with monitoring

### Recent Updates
- **2025-09-12**: Removed timezone validation logic from selenium downloader - now trusts History section images after date selection
- **2025-09-11**: Completed MVP operationalization with parallel development
- **2025-09-11**: Merged real data pipeline implementation from three branches

## Technical Architecture

### Module Structure
```
modules/
├── database/       # SQLAlchemy ORM with SQLite
├── analytics/      # KPI calculations and aggregations  
├── detection/      # YOLOv12 computer vision with tracking
├── downloader/     # Selenium-based image collection
```

### Key Components

#### 1. Image Downloader
- **URL Pattern**: `https://cdn.draydog.com/apm/[date]/[hour]/[timestamp]-[stream_name].jpeg`
- **Navigation**: Uses date picker at `https://app.draydog.com/terminal_cameras/apm?streamName={stream}`
- **Date Selection**: Clicks calendar to select date, trusts History section shows correct images
- **Verification**: Simply checks that History section has images after navigation
- **Database**: Saves metadata after successful download with file hash for deduplication

#### 2. Detection Pipeline
- **Model**: YOLOv12 (ultralytics) with yolov12x.pt weights
- **Watch Mode**: Continuous monitoring with watchdog library
- **Batch Processing**: Multi-threaded queue system
- **Tracking**: ByteTrack for multi-object tracking
- **OCR**: Dual-engine for container number recognition

#### 3. Database Schema
- **Image**: Camera metadata (id, path, timestamp, stream, file_hash)
- **Detection**: YOLO results (bbox, confidence, class)
- **Container**: Tracking data (container_id, timestamps, movements)
- **Metric**: Aggregated KPIs (dwell_time, throughput, efficiency)

#### 4. Dashboard Pages
- **Analytics**: KPI cards and trend charts
- **Live Feed**: Real-time camera view with detections
- **Historical**: Time-series analysis and patterns
- **Settings**: Configuration management

## Deployment Configuration

### Running Components
```bash
# Activate environment
source venv/bin/activate

# Start Streamlit dashboard
streamlit run app.py

# Start image downloader scheduler
python -m modules.downloader.scheduler --streams in_gate

# Start detection with watch mode
python -m modules.detection.yolo_detector --watch

# Docker deployment
docker-compose -f deployment/docker/docker-compose.yml up -d
```

### Production Settings
```python
config = DownloadConfig(
    stream_names=["in_gate"],
    download_interval_minutes=10,
    cleanup_interval_hours=24,
    retention_days=30,
    max_retries=3,
    exponential_backoff=True,
    enable_health_check=True
)
```

## Performance Metrics
- Detection Speed: <2 seconds per image
- Detection Accuracy: 95%+ target
- Dashboard Load: <3 seconds
- Database Query: <100ms for aggregations
- Download Success: 95%+ with retry logic

## Testing
- **Framework**: pytest with fixtures
- **Coverage Goals**: 80%+ for production code
- **Test Count**: 234 total tests
- **Categories**: Unit, Integration, E2E, Performance

## Development Workflow
1. Feature branches from `develop`
2. Frequent commits with descriptive messages
3. PR reviews before merging
4. Never push directly to `main`
5. Run tests before committing: `pytest tests/`

## Common Commands
```bash
# Run tests with coverage
pytest tests/ --cov=modules --cov-report=term-missing

# Format code
black . && flake8

# Type checking
mypy modules/

# Initialize database
python -m modules.database.models --init
```

## Next Priority Tasks
1. Obtain/train YOLOv12 model weights
2. Performance optimization for scale
3. Enhanced monitoring and alerting
4. API development for external integrations
5. Advanced analytics features