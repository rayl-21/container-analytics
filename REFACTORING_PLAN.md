# YOLO Detector Module Refactoring Plan

## Current Structure Analysis
- **yolo_detector.py**: 1374 lines (too large for maintainability)
- Contains multiple responsibilities:
  1. YOLOv12 patching logic (lines 44-149)
  2. Main detector class (lines 150-938)
  3. Processing queue management (lines 940-984)
  4. File system watching (lines 986-1013)
  5. Watch mode orchestration (lines 1015-1262)
  6. CLI interface (lines 1264-1375)

## Proposed Refactored Structure

### Core Detection Module
```
modules/detection/
├── __init__.py                    # Public API exports
├── detector.py                     # Main YOLODetector class (300 lines)
├── patches/                        # Model compatibility patches
│   ├── __init__.py
│   ├── yolov12.py                 # YOLOv12 AAttn patch
│   └── base.py                    # Base patch interface
├── processing/                     # Batch and queue processing
│   ├── __init__.py
│   ├── queue.py                   # ImageProcessingQueue class
│   ├── batch.py                   # Batch processing logic
│   └── pipeline.py                # Full pipeline integration
├── watch/                          # Watch mode functionality
│   ├── __init__.py
│   ├── handler.py                 # File system event handler
│   ├── monitor.py                 # YOLOWatchMode class
│   └── worker.py                  # Processing worker threads
├── utils/                          # Detection utilities
│   ├── __init__.py
│   ├── annotations.py             # Image annotation helpers
│   ├── metrics.py                 # Performance tracking
│   └── database.py                # Database save operations
└── cli.py                          # CLI interface

```

## Refactoring Tasks

### Task 1: Extract Patching Logic
- **Files**: patches/yolov12.py, patches/base.py
- **Agent**: Sub-agent 1
- **Scope**: Lines 44-149 from original
- **Dependencies**: None

### Task 2: Split Core Detector
- **Files**: detector.py, utils/annotations.py
- **Agent**: Sub-agent 2
- **Scope**: Lines 150-780 from original
- **Dependencies**: Task 1

### Task 3: Extract Processing Components
- **Files**: processing/queue.py, processing/batch.py
- **Agent**: Sub-agent 3
- **Scope**: Lines 327-638, 940-984 from original
- **Dependencies**: None

### Task 4: Separate Watch Mode
- **Files**: watch/handler.py, watch/monitor.py, watch/worker.py
- **Agent**: Sub-agent 4
- **Scope**: Lines 986-1262 from original
- **Dependencies**: Task 3

### Task 5: Database & Metrics Utilities
- **Files**: utils/database.py, utils/metrics.py
- **Agent**: Sub-agent 5
- **Scope**: Lines 640-876 from original
- **Dependencies**: None

### Task 6: Integration & Testing
- **Files**: __init__.py files, cli.py
- **Agent**: Main agent
- **Scope**: Module integration and API design
- **Dependencies**: All tasks

## Benefits of Refactoring

1. **Maintainability**: Each file focuses on a single responsibility
2. **Testability**: Smaller units are easier to test in isolation
3. **Reusability**: Components can be used independently
4. **Parallel Development**: Multiple developers can work on different components
5. **Performance**: Lazy imports reduce memory footprint
6. **Extensibility**: Easy to add new model patches or processing strategies

## Migration Strategy

1. Create new structure in worktree
2. Move code with minimal modifications
3. Update imports and fix dependencies
4. Run existing tests to ensure compatibility
5. Add new unit tests for each module
6. Update documentation
7. Merge back to main branch

## Success Criteria

- [ ] All existing tests pass
- [ ] No file exceeds 400 lines
- [ ] Each module has clear single responsibility
- [ ] Public API remains backward compatible
- [ ] Performance metrics remain same or better
- [ ] Code coverage maintained or improved