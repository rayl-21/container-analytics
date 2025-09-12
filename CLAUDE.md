# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Container Analytics - A Python-based MVP application that automatically downloads port gate camera images from Dray Dog (no login required - direct public access) and derives analytics using YOLOv12 computer vision with a Streamlit dashboard for visualization.

## Current Status: MVP Development Phase
**Last Updated**: 2025-09-11

### Completed Milestones ✅
- **Database Module**: 89% test coverage with SQLAlchemy ORM, 20+ query functions
- **Core Architecture**: Modular structure with clean separation of concerns
- **Test Infrastructure**: 165+ tests across all modules with pytest
- **Analytics Engine**: KPI calculations for dwell time, throughput, efficiency
- **E2E Pipeline**: End-to-end testing framework with mock data support
- **Real Data Pipeline**: Complete with database integration and automated scheduling
- **Downloader Module**: Selenium-based with database persistence (70% coverage)
- **Scheduler Module**: Production-ready with retry logic and monitoring (64% coverage)
- **Deployment Configs**: systemd service and Docker compose for production

### In Progress 🔄
- **Detection Module**: YOLOv12 integration with container tracking
- **Streamlit Dashboard**: Multi-page application with real-time updates
- **Container OCR**: Number recognition system for container IDs

## Technology Stack

### Core Technologies
- **Python 3.10+** - Primary language
- **Streamlit 1.28.0** - Dashboard framework (10x faster development than Flask)
- **YOLOv12 (ultralytics)** - Attention-centric object detection with real-time processing (30+ FPS)
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
│   │   ├── yolo_detector.py    # YOLOv12 implementation
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
│   ├── test_scheduler_automation.py  # Scheduler automation tests (29 tests)
│   └── test_e2e_pipeline.py   # End-to-end pipeline tests
├── data/                      # Data storage
│   ├── images/                # Downloaded camera images
│   ├── models/                # YOLO model weights (yolov12x.pt)
│   └── database.db            # SQLite database
├── deployment/                # Production deployment configs
│   ├── systemd/              # Linux service configuration
│   │   └── container-analytics-scheduler.service
│   └── docker/               # Docker deployment
│       └── docker-compose.yml
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
python -m modules.downloader.scheduler --streams in_gate

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
        self.model = YOLO('data/models/yolov12x.pt')  # Use model from data/models/
        self.tracker = sv.ByteTrack()
        
    def process_image(self, image_path):
        results = self.model(image_path)
        detections = sv.Detections.from_ultralytics(results[0])
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
- Downloader Module: ✅ 70% (achieved)
- Scheduler Module: ✅ 64% (achieved)
- Detection Module: 🔄 Target 80%
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

## Git Workflow & Parallel Development Strategy

### Branch Strategy (GitFlow Model)
- **main** - Production-ready code only
- **develop** - Integration branch for feature merging
- **feature/** - Individual feature branches (e.g., `feature/detection-yolo`, `feature/dashboard-websocket`)
- **hotfix/** - Emergency production fixes
- **release/** - Release preparation and testing

### Parallel Development Workflow

#### 1. Initial Setup
```bash
# Configure Git for better collaboration
git config pull.rebase true
git config branch.autosetuprebase always

# Create develop branch if not exists
git checkout -b develop
git push -u origin develop
```

#### 2. Feature Development Process
```bash
# Start new feature (always branch from develop)
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# Work on feature with frequent commits
git add -A
git commit -m "feat(module): descriptive message"

# Before pushing, ensure code quality
pytest tests/                    # Run tests
black . && flake8                # Format and lint
git push -u origin feature/your-feature-name
```

#### 3. Merging Strategy
```bash
# Update feature branch with latest develop
git checkout feature/your-feature
git fetch origin
git rebase origin/develop        # Preferred over merge for cleaner history

# After PR approval, merge to develop
git checkout develop
git merge --no-ff feature/your-feature  # Preserve feature history
git push origin develop
```

### Parallel Session Organization

#### Module-Based Sessions
- **Session 1 (Main)**: Coordination, integration, conflict resolution
- **Session 2 (Backend)**: `modules/detection/` and `modules/analytics/`
- **Session 3 (Frontend)**: `app.py`, `pages/`, `components/`
- **Session 4 (Infrastructure)**: `modules/database/`, `utils/`, `tests/`

#### Task Parallelization Example
```bash
# Terminal 1: Detection Module
git checkout -b feature/detection-improvements
# Focus: modules/detection/, tests/test_detection.py

# Terminal 2: Dashboard Development
git checkout -b feature/dashboard-realtime
# Focus: app.py, pages/, components/

# Terminal 3: Testing & Quality
git checkout -b feature/increase-coverage
# Focus: tests/, quality checks
```

### Conflict Prevention

#### Best Practices
- **Module Ownership**: Assign clear ownership per module
- **Interface Stability**: Keep module interfaces stable during parallel work
- **File Separation**: Create new files rather than heavily modifying shared ones
- **Communication**: Document changes in PR descriptions

#### Daily Workflow
1. **Morning**: Pull latest from `develop`, create/update feature branches
2. **During Day**: Work in parallel sessions, commit frequently (every 1-2 hours)
3. **Evening**: Push changes, create PRs, review team's work
4. **Before Merge**: Run full test suite, ensure 80%+ coverage

### Commit Message Convention
```
feat(module): Add new feature
fix(module): Fix bug description
refactor(module): Restructure code
test(module): Add/update tests
docs: Update documentation
style: Format code (Black, flake8)
perf: Performance improvements
```

### Pre-commit Hooks Setup
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/ --fail-under=70
        language: system
        pass_filenames: false
EOF

# Install hooks
pre-commit install
```

### CI/CD with GitHub Actions
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=modules --cov-report=term-missing
      - run: black --check .
      - run: flake8
```

### Pull Request Guidelines
- Keep PRs small and focused (< 400 lines changed)
- Include tests for new features
- Update documentation if needed
- Link to relevant issues
- Request review from module owner

## Important Notes

- Always use virtual environment for development
- Test file paths must use absolute paths: `Path(__file__).parent.parent / "data"`
- YOLO models are stored in `data/models/` only
- Never commit `.env` files or credentials
- Run tests before committing: `pytest tests/`
- Database integration is complete - use `session_scope()` for transactions
- Scheduler runs every 10 minutes - matches Dray Dog camera interval
- Production deployment configs available in `deployment/` directory
- Update this file when architecture changes significantly
- Use feature branches for ALL development work
- Never push directly to `main` or `develop`
- Resolve conflicts locally before pushing
- Keep commit history clean with meaningful messages