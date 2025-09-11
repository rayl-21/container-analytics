# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Container Analytics - A Python-based MVP application that automatically downloads port gate camera images from Dray Dog (no login required - direct public access) and derives analytics using YOLOv8 computer vision with a Streamlit dashboard for visualization.

## Current Status: MVP Development Phase
**Last Updated**: 2025-09-07

### Completed Milestones ✅
- **Database Module**: 89% test coverage with SQLAlchemy ORM, 20+ query functions
- **Core Architecture**: Modular structure with clean separation of concerns
- **Test Infrastructure**: 165+ tests across all modules with pytest
- **Analytics Engine**: KPI calculations for dwell time, throughput, efficiency
- **E2E Pipeline**: End-to-end testing framework with mock data support

### In Progress 🔄
- **Detection Module**: YOLOv8 integration with container tracking
- **Downloader Module**: Selenium-based automated image collection
- **Streamlit Dashboard**: Multi-page application with real-time updates

## Technology Stack

### Core Technologies
- **Python 3.10+** - Primary language
- **Streamlit 1.28.0** - Dashboard framework (10x faster development than Flask)
- **YOLOv8 (ultralytics)** - Object detection with real-time processing (30+ FPS)
- **Selenium 4.15+** - Automated image downloading from Dray Dog
- **SQLite + SQLAlchemy 2.0** - Data persistence with ORM
- **Pandas 1.5+** - Data analysis and aggregation
- **Plotly 5.17+** - Interactive visualizations

### Supporting Libraries
- **APScheduler 3.10+** - Background task scheduling
- **Loguru 0.7+** - Structured logging
- **Pydantic 2.4+** - Configuration management
- **pytest 7.4+** - Testing framework with coverage tools
- **Black/flake8/mypy** - Code quality tools

## Project Architecture

```
container-analytics/
├── app.py                      # Main Streamlit dashboard
├── pages/                      # Streamlit multi-page apps
│   ├── 1_📊_Analytics.py      # Analytics dashboard with KPIs
│   ├── 2_🖼️_Live_Feed.py      # Live camera view with detections
│   ├── 3_📈_Historical.py      # Historical trends analysis
│   └── 4_⚙️_Settings.py        # Configuration management
├── modules/
│   ├── downloader/             # Image collection from Dray Dog
│   │   ├── selenium_client.py  # Selenium WebDriver automation
│   │   └── scheduler.py        # APScheduler for automated downloads
│   ├── detection/              # Computer vision module
│   │   ├── yolo_detector.py    # YOLOv8 implementation
│   │   ├── tracker.py          # Multi-object tracking
│   │   └── ocr.py              # Container number OCR
│   ├── analytics/              # Analytics engine
│   │   ├── metrics.py          # KPI calculations
│   │   ├── aggregator.py       # Data aggregation
│   │   └── alerts.py           # Anomaly detection
│   └── database/               # Data persistence layer
│       ├── models.py           # SQLAlchemy models (86% tested)
│       └── queries.py          # Database queries (93% tested)
├── components/                 # Reusable Streamlit components
│   ├── charts.py              # Plotly chart generators
│   ├── image_viewer.py        # Image display with annotations
│   └── metrics.py             # KPI cards and displays
├── utils/
│   ├── config.py              # Configuration management
│   ├── logging_config.py      # Logging setup
│   └── cache.py               # Caching utilities
├── tests/                     # Comprehensive test suite (165+ tests)
│   ├── test_database.py       # Database tests (41 tests)
│   ├── test_analytics.py      # Analytics tests (20+ tests)
│   ├── test_detection.py      # Detection tests
│   ├── test_downloader.py     # Downloader tests (25 tests)
│   └── test_e2e_pipeline.py   # End-to-end pipeline tests
├── data/                      # Data storage
│   ├── images/                # Downloaded camera images
│   ├── models/                # YOLO model weights (yolov8n.pt)
│   └── database.db            # SQLite database
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── .streamlit/
    └── config.toml           # Streamlit configuration

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start Streamlit dashboard (port 8501)
streamlit run app.py

# Start image downloader service (scheduled downloads)
python -m modules.downloader.scheduler --streams in_gate out_gate

# Run YOLO detector with file watching (continuous detection)
python -m modules.detection.yolo_detector --watch
```

### Testing
```bash
# Run all tests with coverage
pytest tests/ --cov=modules --cov-report=term-missing

# Run specific test module
pytest tests/test_database.py -v

# Run end-to-end pipeline tests
pytest tests/test_e2e_pipeline.py -v
```

### Code Quality
```bash
# Format code with Black
black .

# Check code style
flake8

# Type checking
mypy modules/

# Run all quality checks
black . && flake8 && mypy modules/
```

### Database Management
```bash
# Initialize database
python -m modules.database.models --init

# Run migrations
python -m modules.database.models --migrate
```

## Key Implementation Details

### Image Download Strategy
- Download images every 10 minutes (matching Dray Dog's capture interval)
- Direct access to public camera feeds - no authentication required
- Filter out thumbnail images, only download full-resolution
- URL pattern: `https://cdn.draydog.com/apm/[date]/[hour]/[timestamp]-[stream_name].jpeg`
- Selenium WebDriver navigates to camera history pages
- Automatic retry logic with exponential backoff
- Store metadata in database for quick retrieval

### YOLO Detection Pipeline
```python
from ultralytics import YOLO
import supervision as sv

class ContainerDetector:
    def __init__(self):
        self.model = YOLO('data/models/yolov8n.pt')  # Use model from data/models/
        self.tracker = sv.ByteTrack()
        
    def process_image(self, image_path):
        results = self.model(image_path)
        detections = sv.Detections.from_yolov8(results[0])
        tracked = self.tracker.update_with_detections(detections)
        return tracked
```

### Database Models
- **Image**: Camera image metadata with timestamps
- **Detection**: YOLO detection results with bounding boxes
- **Container**: Container tracking data with IDs
- **Metric**: Aggregated analytics and KPIs

All models use proper indexes for performance and foreign key constraints for data integrity.

### Performance Targets
- Detection Speed: <2 seconds per image
- Detection Accuracy: 95%+ for containers
- Dashboard Load Time: <3 seconds
- Real-time Update Latency: <1 second
- Database Query Response: <100ms for aggregations

## Code Style Guidelines

- **Python Version**: 3.10+ with type hints
- **Formatting**: Black (88 char line length)
- **Linting**: flake8 with standard rules
- **Imports**: isort for organization
- **Docstrings**: Google style for all functions/classes
- **Testing**: pytest with mock isolation
- **Git Workflow**: Feature branches with descriptive commits

## Testing Strategy

### Coverage Goals
- Database Module: ✅ 89% (achieved)
- Analytics Module: ✅ Good coverage
- Detection Module: 🔄 Target 80%
- Downloader Module: 🔄 Target 80%
- Overall Target: 80%+ for all production code

### Test Categories
- **Unit Tests**: Isolated component testing with mocks
- **Integration Tests**: Module interaction testing
- **E2E Tests**: Full pipeline validation
- **Performance Tests**: Speed and efficiency benchmarks

## Monitoring & Alerts

- Track detection accuracy over time
- Monitor system performance metrics
- Alert on anomalies (unusual traffic patterns)
- Daily summary reports via email
- Real-time dashboard updates via WebSocket

## Important Notes

- Always use virtual environment for development
- Test file paths must use absolute paths: `Path(__file__).parent.parent / "data"`
- YOLO models are stored in `data/models/` only
- Never commit `.env` files or credentials
- Run tests before committing: `pytest tests/`
- Update this file when architecture changes significantly