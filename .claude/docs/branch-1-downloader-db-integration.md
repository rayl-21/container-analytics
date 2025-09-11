# Branch 1: Downloader Database Integration

## Branch Name: `feature/downloader-db-integration`

## Estimated Time: 4-5 hours

## Objective
Integrate the DrayDogDownloader class with the database module to persist downloaded image metadata.

## Original Requirements

### 2.1 Fix Selenium Downloader Integration

**File**: modules/downloader/selenium_client.py

**Current Issues**:
- DrayDogDownloader class exists but not integrated with database
- Missing database session management in download methods

**Required Changes**:
1. Line 319-387 (download_image method): Add database integration
   - After successful download, call queries.insert_image()
   - Store metadata: timestamp, filepath, camera_id, file_size
2. Line 572-710 (download_images_direct method): Add batch database insertion
   - Use database session for batch inserts
   - Track download statistics in database

## Implementation Plan

### Step 1: Add Database Imports (15 min)
```python
# Add at the top of selenium_client.py
from ..database import queries
from ..database.models import session_scope
from datetime import datetime
```

### Step 2: Modify download_image() Method (1.5 hours)

**Location**: Lines 319-387

**Changes Required**:
```python
def download_image(self, url: str, save_dir: str, stream_name: str = "unknown"):
    # ... existing download logic ...
    
    # After successful download (around line 380)
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        
        # Extract timestamp from filename or URL
        timestamp = self._extract_timestamp_from_url(url)
        
        # Save to database
        try:
            image_id = queries.insert_image(
                filepath=save_path,
                camera_id=stream_name,
                timestamp=timestamp,
                file_size=file_size
            )
            self.logger.info(f"Saved to database with ID: {image_id}")
        except Exception as e:
            self.logger.error(f"Failed to save to database: {e}")
            # Don't fail the download if DB insert fails
    
    return save_path
```

### Step 3: Modify download_images_direct() Method (2 hours)

**Location**: Lines 572-710

**Changes Required**:
```python
def download_images_direct(self, stream_name: str, timestamps: List[str], ...):
    # ... existing logic ...
    
    downloaded_files = []
    
    for timestamp in timestamps:
        # ... existing download logic ...
        
        if filepath:
            downloaded_files.append({
                'filepath': filepath,
                'camera_id': stream_name,
                'timestamp': parsed_timestamp,
                'file_size': os.path.getsize(filepath)
            })
    
    # Batch insert to database
    if downloaded_files:
        try:
            with session_scope() as session:
                for file_info in downloaded_files:
                    image_id = queries.insert_image(**file_info)
                    self.logger.debug(f"Inserted image {image_id}")
            self.logger.info(f"Batch inserted {len(downloaded_files)} images")
        except Exception as e:
            self.logger.error(f"Batch database insert failed: {e}")
    
    return results
```

### Step 4: Add Helper Method (30 min)
```python
def _extract_timestamp_from_url(self, url: str) -> datetime:
    """Extract timestamp from DrayDog URL format."""
    # URL format: https://cdn.draydog.com/apm/[date]/[hour]/[timestamp]-[stream].jpeg
    import re
    from datetime import datetime
    
    pattern = r'/(\d{4}-\d{2}-\d{2})/(\d{2})/(\d{10,13})-'
    match = re.search(pattern, url)
    
    if match:
        date_str = match.group(1)
        hour_str = match.group(2)
        timestamp_ms = int(match.group(3))
        
        # Convert millisecond timestamp to datetime
        return datetime.fromtimestamp(timestamp_ms / 1000)
    
    return datetime.utcnow()
```

### Step 5: Update Tests (45 min)

**File**: tests/test_downloader.py

Add tests for database integration:
```python
def test_download_image_saves_to_database(mock_downloader, mock_database):
    """Test that downloaded images are saved to database."""
    downloader = DrayDogDownloader()
    
    with patch('modules.database.queries.insert_image') as mock_insert:
        mock_insert.return_value = 123
        
        filepath = downloader.download_image(
            url="https://cdn.draydog.com/apm/2024-01-15/10/1705316400000-in_gate.jpeg",
            save_dir="/tmp",
            stream_name="in_gate"
        )
        
        assert mock_insert.called
        assert mock_insert.call_args[1]['camera_id'] == 'in_gate'
        assert mock_insert.call_args[1]['filepath'] == filepath
```

## Dependencies
- Database module (modules/database/) - Already complete ✅
- queries.insert_image() function - Available
- session_scope context manager - Available

## Testing Requirements
- Mock database calls in unit tests
- Test error handling when database is unavailable
- Test batch insertion performance
- Ensure 80%+ code coverage

## Success Criteria
- [ ] All downloaded images are persisted to database
- [ ] Batch operations use transactions properly
- [ ] Database failures don't break downloads
- [ ] Tests pass with 80%+ coverage
- [ ] No performance degradation

## Potential Issues & Solutions

### Issue 1: Database Connection Failures
**Solution**: Implement retry logic with exponential backoff

### Issue 2: Large Batch Inserts
**Solution**: Chunk inserts into batches of 100-500 images

### Issue 3: Duplicate Images
**Solution**: Use filepath as unique constraint, handle IntegrityError

## Commands to Run

```bash
# Checkout branch
git checkout feature/downloader-db-integration

# Run tests
pytest tests/test_downloader.py -v

# Check coverage
pytest tests/test_downloader.py --cov=modules.downloader --cov-report=term-missing

# Format code
black modules/downloader/selenium_client.py

# Type checking
mypy modules/downloader/selenium_client.py
```

## Merge Checklist
- [ ] All tests passing
- [ ] Code coverage >= 80%
- [ ] Code formatted with Black
- [ ] Type hints added
- [ ] No merge conflicts with develop
- [ ] PR description updated
- [ ] Code reviewed