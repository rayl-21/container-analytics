# Parallel Development Guide for Real Data Pipeline

## Overview
This guide outlines how to implement the Real Data Pipeline feature across three parallel development branches.

## Task Context

### Original Requirements: 2. Implement Real Data Pipeline

#### 2.1 Fix Selenium Downloader Integration
**File**: modules/downloader/selenium_client.py
- **Current Issues**:
  - DrayDogDownloader class exists but not integrated with database
  - Missing database session management in download methods
- **Required Changes**:
  1. Line 319-387 (download_image method): Add database integration
     - After successful download, call queries.insert_image()
     - Store metadata: timestamp, filepath, camera_id, file_size
  2. Line 572-710 (download_images_direct method): Add batch database insertion
     - Use database session for batch inserts
     - Track download statistics in database

#### 2.2 Fix Scheduler Database Integration
**File**: modules/downloader/scheduler.py
- **Current Issues**:
  - ImageDownloadScheduler doesn't connect to database
  - Stats tracked in memory only, not persisted
- **Required Changes**:
  1. Line 69-91 (init method): Add database session initialization
  2. Add new method: save_to_database() to persist downloaded images
  3. Modify download job: Call database queries after each download batch

#### 2.3 Set Up Automated Scheduling
**File**: modules/downloader/scheduler.py
- **Required Changes**:
  1. Add APScheduler job configuration:
     - Set interval to 10 minutes (matching camera capture rate)
     - Add job for in_gate and out_gate cameras
     - Implement error recovery and retry logic

## Branch Structure

```
develop
├── feature/downloader-db-integration (Branch 1)
├── feature/scheduler-db-persistence (Branch 2)
└── feature/automated-scheduling (Branch 3)
```

## Task Sizing

| Branch | Task | Time Estimate | Complexity |
|--------|------|--------------|------------|
| Branch 1 | Downloader DB Integration | 4-5 hours | Medium |
| Branch 2 | Scheduler DB Persistence | 3-4 hours | Medium |
| Branch 3 | Automated Scheduling | 2-3 hours | Low-Medium |
| **Total** | **Full Pipeline Implementation** | **9-12 hours** | **Medium** |

## Dependencies

### Independent (Can Start Immediately)
- **Branch 1**: Depends only on existing database module
- **Branch 2**: Depends only on existing database module

### Sequential Dependencies
- **Branch 3**: Should wait for Branch 2 to complete (uses enhanced scheduler)

### Shared Dependencies (Already Complete)
- Database module (models.py, queries.py)
- Base DrayDogDownloader class
- Base ImageDownloadScheduler class

## Parallel Development Workflow

### Session Distribution

#### Session 1: Branch 1 - Downloader DB Integration
```bash
# Setup
git checkout develop
git pull origin develop
git checkout -b feature/downloader-db-integration

# Focus files
- modules/downloader/selenium_client.py
- tests/test_downloader.py

# Work items
1. Add database imports
2. Modify download_image() method
3. Modify download_images_direct() method
4. Add timestamp extraction helper
5. Update tests
```

#### Session 2: Branch 2 - Scheduler DB Persistence
```bash
# Setup
git checkout develop
git pull origin develop
git checkout -b feature/scheduler-db-persistence

# Focus files
- modules/downloader/scheduler.py
- tests/test_downloader.py

# Work items
1. Add database imports
2. Modify __init__ for DB connection
3. Create save_to_database() method
4. Update _download_images_job()
5. Replace JSON stats with DB queries
```

#### Session 3: Branch 3 - Automated Scheduling
```bash
# Setup (wait for Branch 2 to merge)
git checkout develop
git pull origin develop
git checkout -b feature/automated-scheduling

# Focus files
- modules/downloader/scheduler.py
- deployment/systemd/
- deployment/docker/

# Work items
1. Update DownloadConfig class
2. Implement retry logic
3. Configure APScheduler jobs
4. Add health monitoring
5. Create deployment configs
```

## Communication Points

### Critical Interfaces to Maintain

1. **Database Interface** (Used by Branch 1 & 2)
   ```python
   queries.insert_image(filepath, camera_id, timestamp, file_size)
   session_scope()  # Context manager
   ```

2. **Downloaded Files Format** (Branch 1 → Branch 2)
   ```python
   {
       'filepath': str,
       'camera_id': str,
       'timestamp': datetime,
       'file_size': int
   }
   ```

3. **Scheduler Config** (Branch 2 → Branch 3)
   ```python
   DownloadConfig class attributes
   ```

## Merge Strategy

### Phase 1: Independent Branches (Day 1)
1. Branch 1 and Branch 2 can be developed and tested independently
2. Both can create PRs to develop branch
3. Review and merge can happen in any order

### Phase 2: Dependent Branch (Day 1-2)
1. After Branch 2 merges, pull latest develop
2. Start Branch 3 development
3. Complete and create PR

### Phase 3: Integration Testing (Day 2)
1. Merge all branches to develop
2. Run full integration tests
3. Fix any integration issues
4. Prepare for merge to main

## Conflict Prevention

### File Ownership
- **Branch 1**: Owns `selenium_client.py`
- **Branch 2**: Owns `scheduler.py` lines 1-430
- **Branch 3**: Owns `scheduler.py` lines 430+ and deployment files

### Best Practices
1. Don't modify files outside your branch's scope
2. Use consistent naming for new methods
3. Add new imports at the top of existing import sections
4. Create new test files rather than heavily modifying existing ones

## Testing Strategy

### Unit Tests (Each Branch)
```bash
# Branch 1
pytest tests/test_downloader.py::TestDrayDogDownloader -v

# Branch 2
pytest tests/test_downloader.py::TestScheduler -v

# Branch 3
pytest tests/test_scheduler_automation.py -v
```

### Integration Tests (After Merge)
```bash
# Full pipeline test
pytest tests/test_e2e_pipeline.py -v

# Coverage check
pytest tests/ --cov=modules.downloader --cov-report=term-missing
```

## Success Metrics

### Individual Branch Metrics
- [ ] Code coverage >= 80%
- [ ] All unit tests passing
- [ ] No linting errors
- [ ] Type hints added

### Integration Metrics
- [ ] Full pipeline executes without errors
- [ ] Images saved to database
- [ ] Scheduler runs at correct intervals
- [ ] Health checks report correct status

## Quick Reference Commands

### For All Branches
```bash
# Check branch status
git status

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=modules --cov-report=term-missing

# Format code
black modules/

# Push changes
git add -A
git commit -m "feat(module): description"
git push -u origin feature/branch-name
```

### Create PR
```bash
# After pushing, create PR on GitHub
gh pr create --title "Feature: Branch Description" \
  --body "## Summary\n- Implementation details\n\n## Testing\n- Test coverage: X%" \
  --base develop
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure virtual environment is activated
   - Check PYTHONPATH includes project root

2. **Database Connection Issues**
   - Verify database file exists: `data/database.db`
   - Check permissions on data directory

3. **Merge Conflicts**
   - Pull latest develop before starting work
   - Rebase feature branch if needed

## Documentation

Each branch has detailed documentation in:
- `.claude/docs/branch-1-downloader-db-integration.md`
- `.claude/docs/branch-2-scheduler-db-persistence.md`
- `.claude/docs/branch-3-automated-scheduling.md`

Refer to these files for detailed implementation steps and code examples.