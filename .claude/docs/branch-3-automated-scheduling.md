# Branch 3: Automated Scheduling Configuration

## Branch Name: `feature/automated-scheduling`

## Estimated Time: 2-3 hours

## Objective
Configure production-ready automated scheduling with proper intervals, retry logic, and monitoring.

## Original Requirements

### 2.3 Set Up Automated Scheduling

**File**: modules/downloader/scheduler.py

**Required Changes**:
1. Add APScheduler job configuration:
   - Set interval to 10 minutes (matching camera capture rate)
   - Add job for in_gate and out_gate cameras
   - Implement error recovery and retry logic

## Implementation Plan

### Step 1: Update DownloadConfig Class (30 min)

**Location**: Lines 22-40

```python
@dataclass
class DownloadConfig:
    """Enhanced configuration for image download scheduler."""
    stream_names: List[str] = field(default_factory=lambda: ["in_gate", "out_gate"])
    download_dir: str = "data/images"
    
    # Scheduling configuration
    download_interval_minutes: int = 10  # Match camera capture rate
    cleanup_interval_hours: int = 24
    retention_days: int = 30
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_retry_delay_seconds: int = 600  # 10 minutes max
    
    # Performance configuration
    batch_size: int = 50
    concurrent_downloads: int = 3
    
    # Monitoring
    enable_health_check: bool = True
    health_check_port: int = 8080
    alert_email: Optional[str] = None
    
    # Browser configuration
    headless: bool = True
    timeout: int = 30
    
    def validate(self):
        """Validate configuration values."""
        if self.download_interval_minutes < 5:
            raise ValueError("Download interval must be at least 5 minutes")
        if self.retention_days < 1:
            raise ValueError("Retention days must be at least 1")
        if self.batch_size < 1 or self.batch_size > 1000:
            raise ValueError("Batch size must be between 1 and 1000")
```

### Step 2: Implement Retry Logic with Exponential Backoff (45 min)

**Location**: Add new method after line 148

```python
def _download_with_retry(self, download_func, *args, **kwargs):
    """
    Execute download function with retry logic and exponential backoff.
    
    Args:
        download_func: Function to execute
        *args, **kwargs: Arguments for the function
    
    Returns:
        Result of the download function
    """
    last_exception = None
    retry_delay = self.config.retry_delay_seconds
    
    for attempt in range(self.config.max_retries):
        try:
            self.logger.info(f"Download attempt {attempt + 1}/{self.config.max_retries}")
            result = download_func(*args, **kwargs)
            
            if attempt > 0:
                self.logger.info(f"Download succeeded after {attempt + 1} attempts")
            
            return result
            
        except Exception as e:
            last_exception = e
            self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            
            if attempt < self.config.max_retries - 1:
                # Calculate next retry delay
                if self.config.exponential_backoff:
                    retry_delay = min(
                        retry_delay * self.config.backoff_multiplier,
                        self.config.max_retry_delay_seconds
                    )
                
                self.logger.info(f"Retrying in {retry_delay} seconds...")
                
                # Check for shutdown during wait
                if self._shutdown_event.wait(retry_delay):
                    self.logger.info("Shutdown requested during retry wait")
                    break
            
            # Reset delay for next retry cycle
            if retry_delay >= self.config.max_retry_delay_seconds:
                retry_delay = self.config.retry_delay_seconds
    
    # All retries failed
    self.logger.error(f"All {self.config.max_retries} download attempts failed")
    self.stats.failed_downloads += 1
    
    if last_exception:
        raise last_exception
```

### Step 3: Configure APScheduler Jobs (45 min)

**Location**: Modify start() method, lines 292-353

```python
def start(self):
    """Start the scheduler with configured jobs."""
    if self.is_running:
        self.logger.warning("Scheduler is already running")
        return
    
    try:
        # Validate configuration
        self.config.validate()
        
        # Add download job with interval trigger
        self.scheduler.add_job(
            func=self._download_images_job_with_retry,
            trigger="interval",
            minutes=self.config.download_interval_minutes,
            id="download_images",
            name="Download Camera Images",
            misfire_grace_time=60,  # Allow 1 minute grace period
            coalesce=True,  # Coalesce missed jobs
            max_instances=1,  # Only one instance at a time
            next_run_time=datetime.now()  # Run immediately on start
        )
        
        # Add cleanup job
        self.scheduler.add_job(
            func=self._cleanup_old_images,
            trigger="interval",
            hours=self.config.cleanup_interval_hours,
            id="cleanup_images",
            name="Cleanup Old Images",
            misfire_grace_time=3600,  # 1 hour grace period
            coalesce=True,
            max_instances=1
        )
        
        # Add health check job if enabled
        if self.config.enable_health_check:
            self.scheduler.add_job(
                func=self._health_check,
                trigger="interval",
                minutes=5,
                id="health_check",
                name="Health Check",
                misfire_grace_time=30,
                coalesce=True,
                max_instances=1
            )
        
        # Add listener for job events
        self.scheduler.add_listener(
            self._job_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
        )
        
        # Start scheduler
        self.scheduler.start()
        self.is_running = True
        
        self.logger.info(f"Scheduler started with {len(self.scheduler.get_jobs())} jobs")
        self.logger.info(f"Download interval: {self.config.download_interval_minutes} minutes")
        self.logger.info(f"Monitoring streams: {', '.join(self.config.stream_names)}")
        
        # Print job schedule
        self._print_schedule()
        
    except Exception as e:
        self.logger.error(f"Failed to start scheduler: {e}")
        raise

def _download_images_job_with_retry(self):
    """Wrapper for download job with retry logic."""
    try:
        self._download_with_retry(self._download_images_job)
    except Exception as e:
        self.logger.error(f"Download job failed after all retries: {e}")
        self._send_alert(f"Download job failed: {e}")
```

### Step 4: Add Health Check and Monitoring (30 min)

**Location**: Add new methods after line 420

```python
def _health_check(self):
    """Perform health check and report status."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'jobs': []
        }
        
        # Check each job
        for job in self.scheduler.get_jobs():
            job_info = {
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'pending': job.pending
            }
            health_status['jobs'].append(job_info)
        
        # Check database connection
        try:
            with session_scope() as session:
                session.execute("SELECT 1")
            health_status['database'] = 'connected'
        except:
            health_status['database'] = 'disconnected'
            health_status['status'] = 'degraded'
        
        # Check disk space
        disk_usage = self._check_disk_space()
        health_status['disk_usage_percent'] = disk_usage
        
        if disk_usage > 90:
            health_status['status'] = 'critical'
            self._send_alert(f"Disk usage critical: {disk_usage}%")
        elif disk_usage > 80:
            health_status['status'] = 'warning'
        
        # Write health status to file for external monitoring
        health_file = os.path.join(self.config.download_dir, '.health')
        with open(health_file, 'w') as f:
            json.dump(health_status, f, indent=2)
        
        self.logger.debug(f"Health check: {health_status['status']}")
        
        return health_status
        
    except Exception as e:
        self.logger.error(f"Health check failed: {e}")
        return {'status': 'error', 'error': str(e)}

def _check_disk_space(self) -> float:
    """Check disk space usage percentage."""
    import shutil
    
    usage = shutil.disk_usage(self.config.download_dir)
    percent_used = (usage.used / usage.total) * 100
    return round(percent_used, 2)

def _send_alert(self, message: str):
    """Send alert notification."""
    if self.config.alert_email:
        # Implement email alerting here
        self.logger.warning(f"ALERT: {message}")
        # For now, just log
    else:
        self.logger.warning(f"ALERT (no email configured): {message}")

def _print_schedule(self):
    """Print the current job schedule."""
    print("\n" + "="*60)
    print("SCHEDULED JOBS")
    print("="*60)
    
    for job in self.scheduler.get_jobs():
        print(f"\nJob: {job.name} (ID: {job.id})")
        print(f"  Next run: {job.next_run_time}")
        print(f"  Trigger: {job.trigger}")
    
    print("="*60 + "\n")
```

### Step 5: Create Systemd Service File (30 min)

**Location**: New file `deployment/systemd/container-analytics-scheduler.service`

```ini
[Unit]
Description=Container Analytics Image Download Scheduler
After=network.target

[Service]
Type=simple
User=container-analytics
Group=container-analytics
WorkingDirectory=/opt/container-analytics
Environment="PATH=/opt/container-analytics/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/opt/container-analytics/venv/bin/python -m modules.downloader.scheduler --config /etc/container-analytics/scheduler.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
MemoryLimit=2G
CPUQuota=50%

# Security
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

### Step 6: Create Docker Compose Configuration (30 min)

**Location**: New file `deployment/docker/docker-compose.yml`

```yaml
version: '3.8'

services:
  scheduler:
    build: .
    container_name: container-analytics-scheduler
    restart: always
    environment:
      - PYTHONUNBUFFERED=1
      - DOWNLOAD_INTERVAL_MINUTES=10
      - RETENTION_DAYS=30
      - DATABASE_URL=sqlite:////data/database.db
    volumes:
      - ./data:/data
      - ./logs:/logs
    healthcheck:
      test: ["CMD", "python", "-c", "import json; health=json.load(open('/data/images/.health')); exit(0 if health['status']!='critical' else 1)"]
      interval: 5m
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 2G
        reservations:
          cpus: '0.1'
          memory: 256M
    networks:
      - container-analytics

  database:
    image: postgres:14-alpine
    container_name: container-analytics-db
    restart: always
    environment:
      - POSTGRES_DB=container_analytics
      - POSTGRES_USER=analytics
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - container-analytics

volumes:
  postgres_data:

networks:
  container-analytics:
    driver: bridge
```

### Step 7: Update Tests (30 min)

**File**: tests/test_scheduler_automation.py (new file)

```python
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from modules.downloader.scheduler import ImageDownloadScheduler, DownloadConfig

def test_scheduler_retry_logic():
    """Test retry with exponential backoff."""
    config = DownloadConfig(
        max_retries=3,
        retry_delay_seconds=1,
        exponential_backoff=True,
        backoff_multiplier=2
    )
    
    scheduler = ImageDownloadScheduler(config)
    
    # Mock function that fails twice then succeeds
    mock_func = Mock(side_effect=[Exception("Error 1"), Exception("Error 2"), "Success"])
    
    with patch('time.sleep'):  # Don't actually sleep in tests
        result = scheduler._download_with_retry(mock_func)
    
    assert result == "Success"
    assert mock_func.call_count == 3

def test_scheduler_health_check():
    """Test health check functionality."""
    config = DownloadConfig(enable_health_check=True)
    scheduler = ImageDownloadScheduler(config)
    
    health = scheduler._health_check()
    
    assert 'status' in health
    assert 'timestamp' in health
    assert health['status'] in ['healthy', 'degraded', 'warning', 'critical']

def test_scheduler_job_configuration():
    """Test that jobs are configured correctly."""
    config = DownloadConfig(
        download_interval_minutes=10,
        cleanup_interval_hours=24
    )
    
    scheduler = ImageDownloadScheduler(config)
    scheduler.start()
    
    jobs = scheduler.scheduler.get_jobs()
    job_ids = [job.id for job in jobs]
    
    assert 'download_images' in job_ids
    assert 'cleanup_images' in job_ids
    
    # Check intervals
    download_job = next(j for j in jobs if j.id == 'download_images')
    assert download_job.trigger.interval == timedelta(minutes=10)
```

## Dependencies
- Branch 2 (scheduler-db-persistence) - Should be merged first
- APScheduler - Already installed
- Database module - Already complete

## Testing Requirements
- Test retry logic with various failure scenarios
- Test health check functionality
- Test job scheduling intervals
- Test alert mechanisms
- Ensure 80%+ code coverage

## Success Criteria
- [ ] Jobs run at correct intervals (10 minutes for downloads)
- [ ] Retry logic works with exponential backoff
- [ ] Health checks report accurate status
- [ ] Systemd service file works on Linux
- [ ] Docker compose runs successfully
- [ ] Tests pass with 80%+ coverage

## Potential Issues & Solutions

### Issue 1: Missed Jobs During Downtime
**Solution**: Use coalesce=True to merge missed jobs

### Issue 2: Long-Running Jobs Overlap
**Solution**: Set max_instances=1 per job

### Issue 3: Memory Leaks in Long-Running Process
**Solution**: Implement periodic restart in systemd

## Commands to Run

```bash
# Checkout branch
git checkout feature/automated-scheduling

# Run tests
pytest tests/test_scheduler_automation.py -v

# Test systemd service (on Linux)
sudo systemctl daemon-reload
sudo systemctl start container-analytics-scheduler
sudo systemctl status container-analytics-scheduler

# Test Docker compose
docker-compose -f deployment/docker/docker-compose.yml up -d
docker-compose logs -f scheduler

# Check health status
cat data/images/.health | jq .
```

## Production Deployment Checklist
- [ ] Configure environment variables
- [ ] Set up log rotation
- [ ] Configure monitoring alerts
- [ ] Set up backup strategy
- [ ] Test failover scenarios
- [ ] Document runbook for operators

## Merge Checklist
- [ ] All tests passing
- [ ] Code coverage >= 80%
- [ ] Code formatted with Black
- [ ] Deployment files tested
- [ ] No merge conflicts with develop
- [ ] PR description updated
- [ ] Code reviewed