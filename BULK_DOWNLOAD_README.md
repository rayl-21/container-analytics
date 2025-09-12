# Bulk Image Downloader

This document describes the BulkImageDownloader implementation for downloading large batches of images from Dray Dog terminal cameras.

## Overview

The `BulkImageDownloader` class extends the existing `DrayDogDownloader` functionality to support:

- **Date Range Downloads**: Download images across multiple dates
- **Multiple Streams**: Support for both `in_gate` and `out_gate` streams
- **Database Integration**: Automatic metadata storage using existing Image model
- **File Organization**: Structured directory layout by date and stream
- **Progress Tracking**: Comprehensive statistics and reporting
- **Error Handling**: Robust error recovery and retry logic

## Quick Start

### Basic Usage

```python
from utils.bulk_download import BulkImageDownloader

# Download images for a date range
with BulkImageDownloader() as downloader:
    results = downloader.download_date_range(
        start_date="2025-09-01",
        end_date="2025-09-07", 
        streams=["in_gate", "out_gate"]
    )
    
    # Generate report
    report = downloader.generate_download_report()
    downloader.save_report(report)
```

### Command Line Usage

```bash
# Run the built-in bulk download script
cd .worktrees/bulk-download
python -m utils.bulk_download

# Or run with custom parameters (modify main() function)
python utils/bulk_download.py
```

## Features

### 1. Date Range Downloading

Downloads images across multiple dates with support for:
- Flexible date range specification (YYYY-MM-DD format)
- Multiple camera streams per date
- Optional limits on images per date
- Automatic retry logic for failed downloads

```python
results = downloader.download_date_range(
    start_date="2025-09-01",
    end_date="2025-09-07",
    streams=["in_gate", "out_gate"],
    max_images_per_date=100  # Optional limit
)
```

### 2. Database Integration

Automatically saves image metadata to the database using the existing Image model:

```python
metadata = [
    {
        'filepath': '/path/to/image.jpg',
        'camera_id': 'in_gate',
        'timestamp': datetime(2025, 9, 1, 10, 0, 0),
        'file_size': 1024
    }
]

saved_count = downloader.save_to_database(metadata)
```

### 3. File Organization

Organizes downloaded files into a structured directory layout:

```
data/images/
├── 2025-09-01/
│   ├── in_gate/
│   │   ├── 20250901100000_in_gate.jpg
│   │   └── 20250901101000_in_gate.jpg
│   └── out_gate/
│       ├── 20250901100500_out_gate.jpg
│       └── 20250901101500_out_gate.jpg
└── 2025-09-02/
    ├── in_gate/
    └── out_gate/
```

### 4. Progress Tracking and Reporting

Comprehensive statistics tracking with detailed reports:

```python
# Access statistics
print(f"Success rate: {downloader.stats.success_rate:.1f}%")
print(f"Total downloaded: {downloader.stats.successful_downloads}")
print(f"Total size: {downloader.stats.total_file_size / (1024*1024):.2f} MB")

# Generate detailed report
report = downloader.generate_download_report()
report_path = downloader.save_report(report)
```

## Configuration Options

### Initialization Parameters

```python
downloader = BulkImageDownloader(
    download_dir="data/images",      # Base download directory
    headless=True,                   # Run browser in headless mode
    max_retries=3,                   # Maximum retry attempts
    retry_delay=1.0,                 # Base delay between retries
    timeout=30,                      # Timeout for web operations
    use_direct_download=True         # Use direct URL construction
)
```

### Download Methods

The downloader supports two approaches:

1. **Direct Download** (Recommended): Constructs URLs directly without browser automation
2. **Selenium-based**: Uses browser automation for complex scenarios

## Error Handling

The downloader includes robust error handling:

- **Individual Image Failures**: Continue downloading other images if one fails
- **Stream Failures**: Continue with other streams if one stream fails
- **Date Failures**: Continue with other dates if one date fails
- **Retry Logic**: Automatic retries with exponential backoff
- **Graceful Degradation**: Partial success reporting

## Performance Considerations

- **Direct Download**: 5-10x faster than Selenium-based approach
- **Parallel Processing**: Downloads images concurrently where possible
- **Memory Efficiency**: Streams large files to avoid memory issues
- **Network Optimization**: Reuses connections and implements throttling

## Testing

Comprehensive unit tests with >80% coverage:

```bash
# Run tests with coverage
pytest tests/test_bulk_download.py --cov=utils.bulk_download --cov-report=term-missing

# Run specific test categories
pytest tests/test_bulk_download.py::TestDownloadStats -v
pytest tests/test_bulk_download.py::TestBulkImageDownloader -v
```

## Example Use Cases

### 1. Historical Data Collection

```python
# Download all images from September 2025
with BulkImageDownloader() as downloader:
    results = downloader.download_date_range(
        start_date="2025-09-01",
        end_date="2025-09-30",
        streams=["in_gate", "out_gate"]
    )
```

### 2. Specific Date Analysis

```python
# Download images for specific analysis dates
analysis_dates = ["2025-09-01", "2025-09-03", "2025-09-05"]

with BulkImageDownloader() as downloader:
    for date in analysis_dates:
        results = downloader.download_date_range(
            start_date=date,
            end_date=date,
            streams=["in_gate", "out_gate"],
            max_images_per_date=50
        )
```

### 3. Data Migration

```python
# Organize existing downloaded files
with BulkImageDownloader() as downloader:
    success = downloader.organize_files(
        source_path="old_downloads",
        target_path="organized_data"
    )
```

## Integration with Existing Code

The BulkImageDownloader follows existing patterns from:

- `modules/downloader/selenium_client.py` - Download logic and patterns
- `modules/database/models.py` - Database model integration  
- `modules/database/queries.py` - Database query functions
- `tests/test_downloader.py` - Testing patterns and fixtures

## Monitoring and Logging

All operations are logged using the project's standard logging configuration:

- **INFO**: Download progress and summary statistics
- **DEBUG**: Detailed operation information
- **WARNING**: Non-fatal errors and retries
- **ERROR**: Fatal errors and exceptions

## Future Enhancements

Potential improvements for future versions:

1. **Parallel Processing**: Download multiple streams simultaneously
2. **Resume Capability**: Resume interrupted downloads
3. **Bandwidth Throttling**: Configurable download speed limits
4. **Cloud Storage**: Direct upload to S3/cloud storage
5. **Real-time Monitoring**: Web interface for monitoring progress
6. **Scheduled Downloads**: Integration with APScheduler for automatic downloads

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory with proper Python path
2. **Database Errors**: Verify database is initialized and accessible
3. **Network Timeouts**: Increase timeout values or check network connectivity
4. **File Permissions**: Ensure write permissions for download directories
5. **Memory Issues**: Use direct download mode for better memory efficiency

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run downloader with debug logging
with BulkImageDownloader() as downloader:
    # ... your code here
```

## License and Contribution

This module is part of the Container Analytics project and follows the same contribution guidelines and license terms.