# Container Analytics Project Knowledge Base

## Project Overview
Container Analytics is a Python-based MVP application that automatically downloads port gate camera images from Dray Dog and derives analytics using YOLOv8 computer vision with a Streamlit dashboard for visualization.

## Current Status (2025-09-11)
### Completed Features ✅
- **Database Module**: Complete with 89% test coverage, SQLAlchemy ORM
- **Analytics Engine**: Implemented with KPI calculations
- **Test Infrastructure**: 165+ tests across all modules
- **Real Data Pipeline**: Fully implemented with database integration
- **Automated Scheduling**: Production-ready with retry logic and monitoring
- **Downloader Module**: Selenium-based with database persistence

### In Progress 🔄
- **Detection Module**: YOLOv8 integration for container detection
- **Dashboard**: Streamlit multi-page application
- **Container OCR**: Number recognition system

## Technical Architecture

### Module Structure
```
modules/
├── database/       # SQLAlchemy ORM with SQLite (89% tested)
├── analytics/      # KPI calculations and aggregations
├── detection/      # YOLOv8 computer vision
├── downloader/     # Selenium-based image collection with DB integration
```

### Recent Implementation Highlights

#### Real Data Pipeline (Completed)
- **Downloader Database Integration**: Modified `selenium_client.py` to save image metadata
- **Scheduler Persistence**: Enhanced `scheduler.py` with database queries for stats
- **Automated Scheduling**: APScheduler with 10-minute intervals, retry logic
- **Production Deployment**: systemd service and Docker compose configurations

#### Key Technical Features
1. **Database Integration**
   - Seamless SQLAlchemy ORM integration
   - Transaction management with `session_scope()`
   - Duplicate prevention logic
   - Graceful fallback when database unavailable

2. **Scheduling & Automation**
   - APScheduler with configurable intervals
   - Exponential backoff retry (3 retries, 2x multiplier)
   - Health monitoring and status reporting
   - Disk space monitoring with alerts

3. **Error Handling**
   - Database failures don't break downloads
   - Comprehensive logging at all levels
   - Graceful degradation strategies

### Database Schema
- **Image**: Camera metadata (id, path, timestamp, stream, file_hash)
- **Detection**: YOLO results (bbox, confidence, class)
- **Container**: Tracking data (container_id, timestamps)
- **Metric**: Aggregated KPIs (dwell_time, throughput)

## Deployment Configuration

### Running the Scheduler
```bash
# Direct execution
python -m modules.downloader.scheduler --streams in_gate out_gate

# Using systemd
sudo systemctl start container-analytics-scheduler

# Using Docker
docker-compose -f deployment/docker/docker-compose.yml up -d
```

### Configuration Example
```python
config = DownloadConfig(
    stream_names=["in_gate", "out_gate"],
    download_interval_minutes=10,
    cleanup_interval_hours=24,
    retention_days=30,
    max_retries=3,
    exponential_backoff=True,
    enable_health_check=True,
    alert_email="admin@example.com"
)
```

## Testing Infrastructure
- **Framework**: pytest with fixtures in conftest.py
- **Coverage**: Database 89%, Downloader 70%, Scheduler 64%
- **Test Count**: 165+ total tests
- **Mock Strategy**: Isolated external dependencies
- **E2E Tests**: Full pipeline validation

## Performance Metrics
- Detection Speed: <2 seconds per image
- Detection Accuracy: 95%+ target for containers
- Dashboard Load: <3 seconds
- Database Query: <100ms for aggregations
- Download Success Rate: 95%+ with retry logic

## Image Download Implementation
- URL Pattern: `https://cdn.draydog.com/apm/[date]/[hour]/[timestamp]-[stream_name].jpeg`
- Selenium WebDriver for navigation
- Filters thumbnails, downloads full-resolution only
- 10-minute intervals matching camera rate
- Database persistence after successful download
- File hash calculation for duplicate detection

## Development Workflow
1. Virtual environment: `source venv/bin/activate`
2. Run tests: `pytest tests/ --cov=modules`
3. Format code: `black . && flake8`
4. Feature branches from `develop`
5. Never push to `main` directly

## Git History Summary
- Latest merge: Real Data Pipeline implementation
- Three parallel branches successfully merged:
  - feature/downloader-db-integration
  - feature/scheduler-db-persistence  
  - feature/automated-scheduling
- Initial E2E pipeline with truck detection complete

## Common Issues & Solutions
- **YOLO model location**: Must be in `data/models/yolov8n.pt`
- **Test paths**: Use absolute paths `Path(__file__).parent.parent`
- **Selenium timeouts**: Increase waits for dynamic content
- **Database locks**: Proper session management required
- **Memory management**: Clear image cache periodically

## Next Priority Tasks
1. Complete YOLOv8 detection integration
2. Implement container number OCR
3. Build real-time dashboard with WebSocket
4. Add production monitoring and alerting
5. Performance optimization for scale