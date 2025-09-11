# Container Analytics

A Python-based MVP application that automatically downloads port gate camera images from Dray Dog and derives analytics using YOLOv12 computer vision with a Streamlit dashboard for visualization.

**Last Updated**: 2025-09-11 | **Status**: Production-Ready Pipeline with Automated Scheduling

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone [repository-url]
cd container-analytics

# 2. Set up Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your Dray Dog credentials and settings

# 5. Initialize database
python -m modules.database.models --init

# 6. Run the dashboard
streamlit run app.py
```

## 📁 Project Structure

```
container-analytics/
├── app.py                      # Main Streamlit dashboard
├── pages/                      # Multi-page Streamlit apps
│   ├── 1_📊_Analytics.py      # Analytics dashboard
│   ├── 2_🖼️_Live_Feed.py      # Live camera feed
│   ├── 3_📈_Historical.py      # Historical trends
│   └── 4_⚙️_Settings.py        # Configuration
├── modules/
│   ├── downloader/             # Selenium-based image collection (70% coverage)
│   ├── detection/              # YOLOv12 computer vision
│   ├── analytics/              # Analytics engine
│   └── database/               # SQLAlchemy ORM (89% coverage)
├── deployment/                 # Production deployment configs
│   ├── systemd/               # Linux service configuration
│   └── docker/                # Docker compose setup
├── components/                 # Reusable UI components
├── utils/                      # Utility functions
├── tests/                      # Test suite (165+ tests)
└── data/                       # Data storage
```

## ✅ Features Implemented

### Core Modules
- ✅ **Image Downloader**: Selenium-based with database integration and retry logic
- ✅ **Automated Scheduler**: APScheduler with 10-minute intervals and health monitoring
- ✅ **Database Layer**: SQLAlchemy ORM with SQLite (89% test coverage)
- ✅ **Analytics Engine**: KPI calculations for dwell time, throughput, efficiency
- ✅ **Real Data Pipeline**: Complete with database persistence and monitoring
- 🔄 **YOLO Detection**: Container and vehicle detection with YOLOv12 (in progress)
- 🔄 **Container Tracking**: Object tracking across frames with ByteTrack
- 🔄 **OCR Module**: Container number extraction
- 🔄 **Alert System**: Anomaly detection and notifications
- 🔄 **Streamlit Dashboard**: Real-time visualization

### Dashboard Pages
- 🔄 **Main Overview**: Key metrics and system status
- 🔄 **Analytics**: Detailed metrics with charts
- 🔄 **Live Feed**: Real-time camera view with detections
- 🔄 **Historical**: Long-term trends and patterns
- 🔄 **Settings**: Configuration management

### Supporting Components
- ✅ **Configuration Management**: Pydantic-based settings
- ✅ **Logging System**: Structured logging with loguru
- ✅ **Caching Layer**: Memory and Redis caching
- ✅ **Chart Components**: Plotly visualizations
- ✅ **Test Suite**: Comprehensive pytest coverage

## 🚢 Deployment Options

### Production Deployment with systemd
```bash
# Copy service file
sudo cp deployment/systemd/container-analytics-scheduler.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable container-analytics-scheduler
sudo systemctl start container-analytics-scheduler

# Check status
sudo systemctl status container-analytics-scheduler
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose -f deployment/docker/docker-compose.yml up -d

# Check logs
docker-compose -f deployment/docker/docker-compose.yml logs -f

# Stop services
docker-compose -f deployment/docker/docker-compose.yml down
```

## 🏃 Running Components

### Start Image Downloader
```bash
# Automated scheduling (recommended - runs every 10 minutes)
python -m modules.downloader.scheduler --streams in_gate out_gate

# One-time download (specific date)
python -m modules.downloader.selenium_client --date 2025-09-07

# Download date range
python -m modules.downloader.selenium_client --date-range 2025-09-01 2025-09-07

# Note: Dray Dog images are publicly accessible - no authentication required
```

### Run YOLO Detection
```bash
# Process single image
python -m modules.detection.yolo_detector --image path/to/image.jpg

# Process batch of images
python -m modules.detection.yolo_detector --batch data/images

# Watch directory for new images (continuous detection)
python -m modules.detection.yolo_detector --watch
```

### Dashboard
```bash
# Start Streamlit dashboard (default port 8501)
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8080
```

## 🧪 Testing

```bash
# Run all tests with coverage report
pytest tests/ --cov=modules --cov-report=term-missing

# Run specific test modules
pytest tests/test_database.py -v        # Database tests (41 tests)
pytest tests/test_downloader.py -v      # Downloader tests (25 tests)
pytest tests/test_scheduler_automation.py -v  # Scheduler tests (29 tests)
pytest tests/test_analytics.py -v       # Analytics tests (20+ tests)
pytest tests/test_e2e_pipeline.py -v    # End-to-end tests

# Current coverage:
# - Database: 89%
# - Downloader: 70%
# - Scheduler: 64%
# - Overall: 165+ tests
```

## 📊 Key Metrics Tracked

- **Container Dwell Time**: Time containers spend at the gate
- **Terminal Throughput**: Containers processed per hour/day
- **Gate Efficiency**: Processing time and utilization
- **Peak Hours**: Busiest operational periods
- **Container Types**: Distribution by size (20ft, 40ft, 45ft)

## 🔧 Configuration

Edit `.env` file for:
- Stream names (in_gate, out_gate)
- Detection thresholds
- Alert settings and email notifications
- Database configuration
- Scheduler intervals and retry settings
- Retention policies (default: 30 days)

## 📈 Performance Targets

- Detection Speed: <2 seconds per image
- Detection Accuracy: 95%+ for containers
- Dashboard Load Time: <3 seconds
- Real-time Update Latency: <1 second
- Download Success Rate: 95%+ with retry logic
- Database Query Response: <100ms for aggregations

## 🚀 Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Initialize database**: `python -m modules.database.models --init`
3. **Start scheduler**: `python -m modules.downloader.scheduler --streams in_gate out_gate`
4. **Deploy to production**: Use systemd or Docker configurations
5. **Monitor system**: Check logs and health status regularly

## 📝 Development

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines and architecture documentation.

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Run tests
4. Submit pull request

## 📄 License

[Your License Here]