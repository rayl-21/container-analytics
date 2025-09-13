# Container Analytics

Production-ready Python application that downloads port gate camera images from Dray Dog and performs analytics using YOLOv12 computer vision with a Streamlit dashboard.

## Quick Start

```bash
# Clone and setup
git clone [repository-url]
cd container-analytics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize database
python -m modules.database.models --init

# Run dashboard
streamlit run app.py
```

## Architecture

```
container-analytics/
├── app.py                      # Main Streamlit dashboard
├── pages/                      # Multi-page dashboard
├── modules/
│   ├── downloader/            # Selenium-based image collection
│   ├── detection/             # YOLOv12 object detection pipeline
│   ├── analytics/             # Metrics and KPI calculations
│   └── database/              # SQLAlchemy ORM layer
├── deployment/                # Docker, systemd, monitoring configs
├── tests/                     # 250+ comprehensive tests
└── data/                      # Storage directory
```

## Features

### Core Functionality
- **Image Downloader**: Automated Selenium scraper with retry logic
- **Scheduler**: APScheduler with 10-minute intervals
- **Database**: SQLAlchemy ORM with optimized queries (800+ lines)
- **Detection Pipeline**: YOLOv12 integration with batch processing
- **Analytics Engine**: KPI calculations (dwell time, throughput, efficiency)
- **Dashboard**: 4-page Streamlit interface with real-time metrics
- **Deployment**: Docker, systemd, Nginx, Grafana monitoring
- **Testing**: 250+ tests across 7,000+ lines

### Configuration Required
- YOLO model weights (`data/models/yolov12x.pt`)
- Dray Dog camera access credentials
- Email server for alerts (optional)

## Running Components

```bash
# Automated image downloads (every 10 minutes)
python -m modules.downloader.scheduler --streams in_gate

# Run detection on images
python -m modules.detection.yolo_detector --watch

# Start dashboard (port 8501)
streamlit run app.py
```

## Deployment

### Docker
```bash
docker-compose -f deployment/docker/docker-compose.yml up -d
```

### systemd
```bash
sudo cp deployment/systemd/container-analytics-scheduler.service /etc/systemd/system/
sudo systemctl enable --now container-analytics-scheduler
```

## Testing

```bash
# Run all tests with coverage
pytest tests/ --cov=modules --cov-report=term-missing

# Specific modules
pytest tests/test_database.py -v    # 41 tests
pytest tests/test_detection.py -v   # 40 tests
pytest tests/test_analytics.py -v   # 60 tests
pytest tests/test_downloader.py -v  # 61 tests
```

## Key Metrics

- Container dwell time at gates
- Terminal throughput (containers/hour)
- Gate utilization and efficiency
- Peak operational periods
- Container type distribution

## Configuration

Edit `.env`:
- Stream names and detection thresholds
- Database and scheduler settings
- Alert email configuration
- Data retention policies

## Development

```bash
# Code formatting
black .

# Run quality checks
flake8 && mypy modules/

# Install pre-commit hooks
pre-commit install
```

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

## License

[Your License Here]