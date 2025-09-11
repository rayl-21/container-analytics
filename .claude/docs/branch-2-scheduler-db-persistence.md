# Branch 2: Scheduler Database Persistence

## Branch Name: `feature/scheduler-db-persistence`

## Estimated Time: 3-4 hours

## Objective
Integrate the ImageDownloadScheduler with the database to persist download statistics and image metadata.

## Original Requirements

### 2.2 Fix Scheduler Database Integration

**File**: modules/downloader/scheduler.py

**Current Issues**:
- ImageDownloadScheduler doesn't connect to database
- Stats tracked in memory only, not persisted

**Required Changes**:
1. Line 69-91 (init method): Add database session initialization
2. Add new method: save_to_database() to persist downloaded images
3. Modify download job: Call database queries after each download batch

## Implementation Plan

### Step 1: Add Database Imports (15 min)
```python
# Add at the top of scheduler.py
from ..database import queries
from ..database.models import session_scope, Image
from sqlalchemy import func
```

### Step 2: Modify __init__ Method (30 min)

**Location**: Lines 69-91

**Changes Required**:
```python
def __init__(self, config: DownloadConfig):
    """Initialize scheduler with database connection."""
    self.config = config
    self.scheduler = BackgroundScheduler()
    self.stats = DownloadStats()
    self.is_running = False
    self._shutdown_event = threading.Event()
    
    # Add database connection check
    self._validate_database_connection()
    
    # Initialize stats from database instead of JSON
    self._load_stats_from_database()
    
    # Setup logging and signal handlers
    self._setup_logging()
    self._setup_signal_handlers()
    
def _validate_database_connection(self):
    """Verify database is accessible."""
    try:
        with session_scope() as session:
            # Simple query to test connection
            session.execute("SELECT 1")
        self.logger.info("Database connection verified")
    except Exception as e:
        self.logger.error(f"Database connection failed: {e}")
        raise
```

### Step 3: Add save_to_database() Method (1 hour)

**Location**: After line 167 (after _save_stats method)

```python
def save_to_database(self, downloaded_files: List[Dict[str, Any]]):
    """
    Save downloaded file information to database.
    
    Args:
        downloaded_files: List of file info dictionaries
    """
    if not downloaded_files:
        return
    
    saved_count = 0
    failed_count = 0
    
    try:
        with session_scope() as session:
            for file_info in downloaded_files:
                try:
                    image_id = queries.insert_image(
                        filepath=file_info['filepath'],
                        camera_id=file_info['camera_id'],
                        timestamp=file_info.get('timestamp'),
                        file_size=file_info.get('file_size')
                    )
                    saved_count += 1
                    self.logger.debug(f"Saved image {image_id}: {file_info['filepath']}")
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Failed to save {file_info['filepath']}: {e}")
        
        self.logger.info(f"Database save complete: {saved_count} saved, {failed_count} failed")
        
        # Update stats
        self.stats.total_downloaded += saved_count
        self.stats.last_download = datetime.now()
        
    except Exception as e:
        self.logger.error(f"Database batch save failed: {e}")
        raise
```

### Step 4: Modify _download_images_job() Method (1 hour)

**Location**: Lines 169-244

**Changes Required**:
```python
def _download_images_job(self):
    """Download images and save to database."""
    self.logger.info(f"Starting download job for streams: {self.config.stream_names}")
    
    try:
        with DrayDogDownloader(
            download_dir=self.config.download_dir,
            headless=self.config.headless
        ) as downloader:
            
            for stream_name in self.config.stream_names:
                if self._shutdown_event.is_set():
                    break
                
                self.logger.info(f"Processing stream: {stream_name}")
                
                # Get timestamps to download
                timestamps = self._get_pending_timestamps(stream_name)
                
                if not timestamps:
                    self.logger.info(f"No new images for {stream_name}")
                    continue
                
                # Download images
                results = downloader.download_images_direct(
                    stream_name=stream_name,
                    timestamps=timestamps,
                    date_str=datetime.now().strftime("%Y-%m-%d"),
                    base_dir=self.config.download_dir
                )
                
                # Prepare data for database
                downloaded_files = []
                for filepath in results['downloaded']:
                    if os.path.exists(filepath):
                        downloaded_files.append({
                            'filepath': filepath,
                            'camera_id': stream_name,
                            'timestamp': self._extract_timestamp_from_filepath(filepath),
                            'file_size': os.path.getsize(filepath)
                        })
                
                # Save to database
                if downloaded_files:
                    self.save_to_database(downloaded_files)
                
                # Update stats
                self.stats.downloads_by_stream[stream_name] += len(downloaded_files)
                
    except Exception as e:
        self.logger.error(f"Download job failed: {e}")
        self.stats.failed_downloads += 1
```

### Step 5: Replace JSON Stats with Database Queries (45 min)

**Location**: Lines 122-145 (_load_stats) and 372-396 (get_stats)

```python
def _load_stats_from_database(self):
    """Load statistics from database instead of JSON file."""
    try:
        with session_scope() as session:
            # Get total downloaded images
            total = session.query(func.count(Image.id)).scalar() or 0
            self.stats.total_downloaded = total
            
            # Get downloads by stream
            stream_counts = session.query(
                Image.camera_id,
                func.count(Image.id)
            ).group_by(Image.camera_id).all()
            
            for camera_id, count in stream_counts:
                self.stats.downloads_by_stream[camera_id] = count
            
            # Get last download time
            last_image = session.query(Image).order_by(
                Image.created_at.desc()
            ).first()
            
            if last_image:
                self.stats.last_download = last_image.created_at
            
            self.logger.info(f"Loaded stats from database: {total} total images")
            
    except Exception as e:
        self.logger.error(f"Failed to load stats from database: {e}")
        # Fall back to empty stats
        self.stats = DownloadStats()

def get_stats(self) -> Dict[str, Any]:
    """Get current statistics from database."""
    try:
        with session_scope() as session:
            stats = {
                'total_downloaded': session.query(func.count(Image.id)).scalar(),
                'downloads_by_stream': {},
                'last_24h': 0,
                'last_hour': 0
            }
            
            # Get counts by stream
            stream_counts = session.query(
                Image.camera_id,
                func.count(Image.id)
            ).group_by(Image.camera_id).all()
            
            stats['downloads_by_stream'] = dict(stream_counts)
            
            # Get last 24 hours count
            day_ago = datetime.now() - timedelta(days=1)
            stats['last_24h'] = session.query(func.count(Image.id)).filter(
                Image.created_at >= day_ago
            ).scalar()
            
            # Get last hour count
            hour_ago = datetime.now() - timedelta(hours=1)
            stats['last_hour'] = session.query(func.count(Image.id)).filter(
                Image.created_at >= hour_ago
            ).scalar()
            
            return stats
            
    except Exception as e:
        self.logger.error(f"Failed to get stats from database: {e}")
        return {}
```

### Step 6: Update Tests (30 min)

**File**: tests/test_downloader.py

Add tests for database persistence:
```python
def test_scheduler_saves_to_database(mock_scheduler, mock_database):
    """Test that scheduler saves downloaded images to database."""
    config = DownloadConfig(
        stream_names=["in_gate"],
        download_dir="/tmp/test"
    )
    
    scheduler = ImageDownloadScheduler(config)
    
    downloaded_files = [
        {'filepath': '/tmp/img1.jpg', 'camera_id': 'in_gate', 'file_size': 1024},
        {'filepath': '/tmp/img2.jpg', 'camera_id': 'in_gate', 'file_size': 2048}
    ]
    
    with patch('modules.database.queries.insert_image') as mock_insert:
        mock_insert.return_value = 1
        scheduler.save_to_database(downloaded_files)
        
        assert mock_insert.call_count == 2
        assert scheduler.stats.total_downloaded == 2
```

## Dependencies
- Database module (modules/database/) - Already complete ✅
- DrayDogDownloader class - From Branch 1 (can work in parallel)
- APScheduler - Already configured

## Testing Requirements
- Test database connection validation
- Test stats loading from database
- Test batch save operations
- Mock database for unit tests
- Ensure 80%+ code coverage

## Success Criteria
- [ ] All stats persisted to database
- [ ] JSON stats file deprecated
- [ ] Real-time stats from database queries
- [ ] Graceful handling of database failures
- [ ] Tests pass with 80%+ coverage

## Potential Issues & Solutions

### Issue 1: Database Performance with Large Stats Queries
**Solution**: Add indexes on created_at and camera_id columns

### Issue 2: Concurrent Access from Multiple Schedulers
**Solution**: Use database transactions and row-level locking

### Issue 3: Memory Usage with Large Batch Saves
**Solution**: Process in chunks of 100 images

## Commands to Run

```bash
# Checkout branch
git checkout feature/scheduler-db-persistence

# Run tests
pytest tests/test_downloader.py::TestScheduler -v

# Check coverage
pytest tests/test_downloader.py --cov=modules.downloader.scheduler --cov-report=term-missing

# Format code
black modules/downloader/scheduler.py

# Type checking
mypy modules/downloader/scheduler.py
```

## Merge Checklist
- [ ] All tests passing
- [ ] Code coverage >= 80%
- [ ] Code formatted with Black
- [ ] Type hints added
- [ ] No merge conflicts with develop
- [ ] PR description updated
- [ ] Code reviewed