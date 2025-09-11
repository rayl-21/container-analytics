# Container Analytics

A Python-based MVP application that automatically downloads port gate camera images from Dray Dog and derives analytics using YOLOv8 computer vision with a Streamlit dashboard for visualization.

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
│   ├── downloader/             # Dray Dog image collection
│   ├── detection/              # YOLOv8 computer vision
│   ├── analytics/              # Analytics engine
│   └── database/               # Data persistence
├── components/                 # Reusable UI components
├── utils/                      # Utility functions
├── tests/                      # Test suite
└── data/                       # Data storage
```

## ✅ Features Implemented

### Core Modules
- ✅ **Image Downloader**: Selenium-based automated downloading from Dray Dog
- ✅ **YOLO Detection**: Container and vehicle detection with YOLOv8
- ✅ **Container Tracking**: Object tracking across frames with ByteTrack
- ✅ **OCR Module**: Container number extraction
- ✅ **Database Layer**: SQLAlchemy models with SQLite
- ✅ **Analytics Engine**: KPI calculations and aggregations
- ✅ **Alert System**: Anomaly detection and notifications
- ✅ **Streamlit Dashboard**: Real-time visualization

### Dashboard Pages
- ✅ **Main Overview**: Key metrics and system status
- ✅ **Analytics**: Detailed metrics with charts
- ✅ **Live Feed**: Real-time camera view with detections
- ✅ **Historical**: Long-term trends and patterns
- ✅ **Settings**: Configuration management

### Supporting Components
- ✅ **Configuration Management**: Pydantic-based settings
- ✅ **Logging System**: Structured logging with loguru
- ✅ **Caching Layer**: Memory and Redis caching
- ✅ **Chart Components**: Plotly visualizations
- ✅ **Test Suite**: Comprehensive pytest coverage

## 🏃 Running Components

### Start Image Downloader
```bash
# One-time download (today's images)
python -m modules.downloader.selenium_client --username USER --password PASS

# Download specific date
python -m modules.downloader.selenium_client --username USER --password PASS --date 2025-09-07

# Download date range
python -m modules.downloader.selenium_client --username USER --password PASS --date-range 2025-09-01 2025-09-07

# Scheduled downloads (every 10 minutes)
python -m modules.downloader.scheduler --streams in_gate out_gate
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
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=modules tests/

# Run specific test module
pytest tests/test_detection.py -v
```

## 📊 Key Metrics Tracked

- **Container Dwell Time**: Time containers spend at the gate
- **Terminal Throughput**: Containers processed per hour/day
- **Gate Efficiency**: Processing time and utilization
- **Peak Hours**: Busiest operational periods
- **Container Types**: Distribution by size (20ft, 40ft, 45ft)

## 🔧 Configuration

Edit `.env` file for:
- Dray Dog credentials
- Detection thresholds
- Alert settings
- Database configuration
- Email notifications

## 📈 Performance Targets

- Detection Speed: <2 seconds per image
- Detection Accuracy: 95%+ for containers
- Dashboard Load Time: <3 seconds
- Real-time Update Latency: <1 second

## 🚀 Next Steps

1. **Install dependencies** and configure environment
2. **Run validation**: `python validate_modules.py`
3. **Start dashboard**: `streamlit run app.py`
4. **Configure Dray Dog** credentials in `.env`
5. **Begin image collection** with scheduler

## 📝 Development

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines and architecture documentation.

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Run tests
4. Submit pull request

## 📄 License

[Your License Here]