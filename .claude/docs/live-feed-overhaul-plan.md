# Live Feed Overhaul Plan - Parallel Development Strategy

## Overview
Complete overhaul of the Live Feed section with simplified UI and robust backend integration. The implementation will follow the parallel development guide using git worktrees and sub-agents.

## Current State Analysis
- **Live Feed UI**: Complex with many controls, auto-refresh, camera selection, detection filters
- **Backend**: Partial integration - downloader works but detection/database integration is incomplete
- **Data Flow**: Images fetched on-demand, detection runs in memory, limited database persistence

## Target State
- **Simplified UI**: Clean display of 7 days of images with detection toggle
- **Backend**: Full integration with database persistence and batch processing
- **Data Flow**: One-time bulk download → Detection pipeline → Database storage → UI display

## Parallel Development Workstreams

### **Workstream 1: Data Ingestion Script**
**Branch**: `feature/bulk-image-downloader`
**Location**: `utils/bulk_download.py`
**Tasks**:
1. Create bulk download script for 2025-09-01 to 2025-09-07
2. Download images from both in_gate and out_gate streams
3. Organize images in `data/images/` by date and stream
4. Save metadata to database (Image table)
5. Add progress tracking and error handling
6. Create test script for verification

### **Workstream 2: Detection Pipeline Integration**
**Branch**: `feature/detection-pipeline`
**Location**: `modules/detection/` modifications
**Tasks**:
1. Enhance YOLODetector for batch processing
2. Add database persistence for all detections
3. Create container tracking records
4. Implement detection result caching
5. Add detection statistics aggregation
6. Create integration tests

### **Workstream 3: Live Feed UI Simplification**
**Branch**: `feature/live-feed-redesign`
**Location**: `pages/2_🖼️_Live_Feed.py`
**Tasks**:
1. Strip down complex UI controls
2. Implement simple 7-day image gallery
3. Add detection overlay toggle
4. Display truck count per image
5. Add sidebar toggle for new data pulling
6. Implement efficient image loading from database

## Implementation Details

### Workstream 1 Details - Bulk Download Script
```python
# utils/bulk_download.py structure:
- BulkImageDownloader class
  - download_date_range(start_date, end_date, streams)
  - save_to_database(image_metadata)
  - organize_files(source_path, target_path)
  - generate_download_report()
```

### Workstream 2 Details - Detection Pipeline
```python
# Enhanced detection flow:
1. Load unprocessed images from database
2. Run YOLO detection in batches
3. Save Detection records with bbox, confidence
4. Update Container tracking data
5. Mark images as processed
6. Generate detection metrics
```

### Workstream 3 Details - UI Redesign
```python
# Simplified Live Feed structure:
- Header: Simple title with live indicator
- Main Area: 
  - 7-day image grid (sorted by date)
  - Detection overlay (toggleable)
  - Truck count badge per image
- Sidebar:
  - Toggle: "Pull New Data" (on/off)
  - Current status indicator
  - Last update timestamp
```

## Database Schema Usage
- **Image Table**: Store all downloaded images with metadata
- **Detection Table**: Store YOLO detection results
- **Container Table**: Track unique containers
- **Metric Table**: Aggregate statistics

## Git Worktree Setup
```bash
# Create worktrees for parallel development
git worktree add .worktrees/bulk-download -b feature/bulk-image-downloader
git worktree add .worktrees/detection -b feature/detection-pipeline  
git worktree add .worktrees/ui-redesign -b feature/live-feed-redesign
```

## Sub-Agent Task Distribution

### Agent 1: Data Ingestion Specialist
- Focus: `utils/bulk_download.py`
- Deliverables:
  - Bulk download script with date range support
  - Database integration for image metadata
  - Progress tracking and error handling
  - Test coverage > 80%

### Agent 2: Detection Pipeline Engineer
- Focus: `modules/detection/`
- Deliverables:
  - Batch processing capability
  - Full database persistence
  - Container tracking
  - Performance optimization
  - Integration tests

### Agent 3: UI/UX Developer
- Focus: `pages/2_🖼️_Live_Feed.py`
- Deliverables:
  - Simplified, clean UI
  - Efficient image loading
  - Detection overlay feature
  - Real-time toggle controls
  - Responsive design

## Integration Strategy
1. All agents work independently in their worktrees
2. Database models remain stable (no schema changes)
3. Use existing query functions from `modules/database/queries.py`
4. Coordinate through git commits and structured messages
5. Final integration in main branch after testing

## Success Criteria
- ✅ 7 days of images downloaded and stored
- ✅ All images processed through detection pipeline
- ✅ Database fully populated with detection data
- ✅ UI loads quickly with all features working
- ✅ Toggle controls function properly
- ✅ Test coverage > 80% for new code
- ✅ No regression in existing functionality

## Timeline
- Parallel Development: 3-4 hours
- Integration & Testing: 1 hour
- Total: ~5 hours

This plan ensures clean separation of concerns, parallel execution, and minimal conflicts while delivering a robust Live Feed overhaul.