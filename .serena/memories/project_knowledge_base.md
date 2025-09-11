# Container Analytics Project Knowledge Base

## Project Overview
Container Analytics is a Python-based MVP application that automatically downloads port gate camera images from Dray Dog and derives analytics using YOLOv8 computer vision with a Streamlit dashboard for visualization.

## Current Status (2025-09-07)
- **Database Module**: ✅ Complete (89% test coverage)
- **Analytics Engine**: ✅ Implemented with KPI calculations
- **Test Infrastructure**: ✅ 165+ tests across all modules
- **Detection Module**: 🔄 YOLOv8 integration in progress
- **Downloader Module**: 🔄 Selenium automation in progress
- **Dashboard**: 🔄 Streamlit multi-page app in progress

## Technical Architecture

### Module Structure
```
modules/
├── database/       # SQLAlchemy ORM with SQLite (89% tested)
├── analytics/      # KPI calculations and aggregations
├── detection/      # YOLOv8 computer vision
├── downloader/     # Selenium-based image collection
```

### Key Technical Decisions
1. **Streamlit over Flask**: 10x faster development for MVP
2. **YOLOv8**: Pre-trained models with real-time processing
3. **SQLite**: Lightweight for MVP, easy PostgreSQL migration
4. **Selenium**: Direct camera access without authentication

### Database Schema
- **Image**: Camera metadata (id, path, timestamp, stream)
- **Detection**: YOLO results (bbox, confidence, class)
- **Container**: Tracking data (container_id, timestamps)
- **Metric**: Aggregated KPIs (dwell_time, throughput)

## Testing Infrastructure
- **Framework**: pytest with fixtures in conftest.py
- **Coverage**: Database 89%, Analytics good, others improving
- **Mock Strategy**: Isolate external dependencies
- **E2E Tests**: Full pipeline validation with mock data

## Image Download Implementation
- URL Pattern: `https://cdn.draydog.com/apm/[date]/[hour]/[timestamp]-[stream_name].jpeg`
- Selenium navigates to camera history pages
- Filters thumbnails, downloads only full-resolution
- 10-minute intervals matching camera capture rate
- Retry logic with exponential backoff

## Performance Requirements
- Detection: <2 seconds per image
- Accuracy: 95%+ for containers
- Dashboard: <3 seconds load time
- Database: <100ms query response

## Development Workflow
1. Use virtual environment: `source venv/bin/activate`
2. Run tests before commits: `pytest tests/`
3. Format code: `black . && flake8`
4. Use absolute paths in tests: `Path(__file__).parent.parent`
5. YOLO models in `data/models/` only

## Common Issues & Solutions
- **YOLO model in wrong location**: Must be in `data/models/yolov8n.pt`
- **Test artifacts**: Use absolute paths from project root
- **Selenium timeouts**: Increase wait times for dynamic content
- **Database locks**: Use proper session management

## Next Priority Tasks
1. Complete YOLOv8 integration testing
2. Finalize Selenium downloader reliability
3. Implement WebSocket for real-time updates
4. Add container number OCR
5. Deploy MVP to production