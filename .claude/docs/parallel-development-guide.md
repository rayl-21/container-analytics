# Parallel Development with Sub-Agents and Git Worktrees

## Overview
This guide demonstrates how to leverage Claude Code's sub-agents and git worktrees for truly parallel development, enabling multiple development streams to progress simultaneously without context switching overhead.

## Why This Approach?

### Traditional Sequential Development
- One branch at a time
- Constant context switching
- Manual coordination overhead
- Limited parallelization

### Parallel Development with Sub-Agents
- Multiple branches developed simultaneously
- Each agent maintains its own context
- Automatic coordination through git
- True parallel execution

## Core Concepts

### Git Worktrees
Git worktrees allow multiple branches to be checked out simultaneously in different directories, enabling true parallel development without stashing or switching.

### Claude Code Sub-Agents
Sub-agents are independent AI workers that can handle complex tasks autonomously while the main agent coordinates and integrates their work.

## Setup

### 1. Initialize Git Worktrees
```bash
# Create worktree directory structure
mkdir -p .worktrees

# Create worktrees for parallel development
git worktree add .worktrees/feature-1 -b feature/branch-1
git worktree add .worktrees/feature-2 -b feature/branch-2
git worktree add .worktrees/feature-3 -b feature/branch-3

# List active worktrees
git worktree list
```

### 2. Project Structure with Worktrees
```
container-analytics/
├── .git/                    # Main git directory
├── .worktrees/             # Parallel development branches
│   ├── feature-1/          # Worktree for feature 1
│   ├── feature-2/          # Worktree for feature 2
│   └── feature-3/          # Worktree for feature 3
├── modules/                # Main branch code
├── tests/
└── ...
```

## Parallel Development Workflow

### Step 1: Plan Task Distribution
```python
tasks = {
    "feature-1": {
        "worktree": ".worktrees/feature-1",
        "files": ["modules/downloader/selenium_client.py"],
        "tests": ["tests/test_downloader.py"],
        "description": "Database integration for downloader"
    },
    "feature-2": {
        "worktree": ".worktrees/feature-2", 
        "files": ["modules/downloader/scheduler.py"],
        "tests": ["tests/test_scheduler.py"],
        "description": "Scheduler database persistence"
    },
    "feature-3": {
        "worktree": ".worktrees/feature-3",
        "files": ["deployment/"],
        "tests": ["tests/test_e2e_pipeline.py"],
        "description": "Automated scheduling setup"
    }
}
```

### Step 2: Launch Parallel Sub-Agents

#### Main Agent Coordination Script
```python
# Launch sub-agents in parallel
agents = []
for feature, config in tasks.items():
    agent = launch_subagent(
        task_type="development",
        worktree=config["worktree"],
        description=config["description"],
        files=config["files"],
        tests=config["tests"]
    )
    agents.append(agent)

# Wait for all agents to complete
results = wait_for_agents(agents)

# Merge results
for result in results:
    if result.success:
        merge_branch(result.branch)
```

### Step 3: Sub-Agent Task Template
Each sub-agent receives:
```yaml
task: "Implement feature in isolated worktree"
worktree: ".worktrees/feature-1"
requirements:
  - Make all changes in specified worktree
  - Run tests before marking complete
  - Commit changes with descriptive messages
  - Report completion status
constraints:
  - Do not modify files outside assigned scope
  - Maintain interface compatibility
  - Follow existing code patterns
```

## Practical Example: Real Data Pipeline

### Task Breakdown
```bash
# Main agent coordinates three parallel features
TASK_1="Database Integration for Downloader"
TASK_2="Scheduler Database Persistence"  
TASK_3="Automated Scheduling Configuration"
```

### Parallel Execution
```bash
# Terminal 1: Main Agent
claude-code coordinate \
  --task-1 "$TASK_1" --worktree-1 .worktrees/feature-1 \
  --task-2 "$TASK_2" --worktree-2 .worktrees/feature-2 \
  --task-3 "$TASK_3" --worktree-3 .worktrees/feature-3

# Each sub-agent works independently
# Agent 1: Modifying selenium_client.py in .worktrees/feature-1
# Agent 2: Modifying scheduler.py in .worktrees/feature-2  
# Agent 3: Creating deployment configs in .worktrees/feature-3
```

### Integration Phase
```bash
# After all agents complete
cd container-analytics

# Merge feature-1
git merge feature/branch-1 --no-ff

# Merge feature-2
git merge feature/branch-2 --no-ff

# Merge feature-3 (depends on feature-2)
git merge feature/branch-3 --no-ff

# Run integration tests
pytest tests/test_e2e_pipeline.py -v
```

## Advanced Patterns

### 1. Dependency Management
```python
dependencies = {
    "feature-3": ["feature-2"],  # Feature 3 depends on 2
    "feature-4": ["feature-1", "feature-2"]  # Feature 4 depends on 1 and 2
}

# Launch independent features first
independent = get_independent_features(dependencies)
launch_agents(independent)

# Launch dependent features after prerequisites
for feature in get_next_wave(dependencies):
    wait_for_dependencies(feature)
    launch_agent(feature)
```

### 2. Conflict Resolution
```python
def resolve_conflicts(branch1, branch2):
    """Sub-agent handles merge conflicts"""
    conflicts = detect_conflicts(branch1, branch2)
    
    for conflict in conflicts:
        if conflict.type == "import":
            resolution = merge_imports(conflict)
        elif conflict.type == "function_signature":
            resolution = preserve_both_signatures(conflict)
        else:
            resolution = request_human_review(conflict)
        
        apply_resolution(conflict, resolution)
```

### 3. Continuous Integration
```yaml
# .github/workflows/parallel-ci.yml
name: Parallel CI
on: [push]

jobs:
  test-features:
    strategy:
      matrix:
        worktree: [feature-1, feature-2, feature-3]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test worktree
        run: |
          cd .worktrees/${{ matrix.worktree }}
          pytest tests/ --cov
```

## Best Practices

### 1. Task Isolation
- Assign clear file ownership to each agent
- Minimize shared file modifications
- Use interfaces for communication between features

### 2. Communication Protocol
```python
# Inter-agent communication via git commits
def communicate_via_commit(message, data):
    """Agents communicate through structured commits"""
    with open('.agent-comm/messages.json', 'a') as f:
        json.dump({
            'timestamp': datetime.now(),
            'agent': agent_id,
            'message': message,
            'data': data
        }, f)
    
    git_commit(f"comm: {message}")
```

### 3. Progress Tracking
```python
# Real-time progress monitoring
def monitor_agents(agents):
    while any(a.status != 'complete' for a in agents):
        for agent in agents:
            progress = agent.get_progress()
            print(f"{agent.name}: {progress.percent}% - {progress.current_task}")
        time.sleep(5)
```

### 4. Error Recovery
```python
def handle_agent_failure(agent, error):
    """Graceful failure handling"""
    if error.recoverable:
        # Retry with adjusted parameters
        retry_agent(agent, adjustments=error.suggestions)
    else:
        # Fall back to sequential processing
        handle_manually(agent.task)
        notify_team(agent, error)
```

## Monitoring and Coordination

### Dashboard View
```
┌─────────────────────────────────────────┐
│          Parallel Development           │
├──────────┬──────────┬──────────────────┤
│ Agent 1  │ Agent 2  │ Agent 3          │
│ ▓▓▓▓░░░  │ ▓▓▓▓▓▓▓  │ ▓▓░░░░░         │
│ 60%      │ 100%     │ 30%             │
│          │ Complete │                  │
├──────────┴──────────┴──────────────────┤
│ Files Modified: 12                      │
│ Tests Passed: 47/52                     │
│ Coverage: 84%                           │
└─────────────────────────────────────────┘
```

### Coordination Commands
```bash
# Check agent status
claude-code status --agents

# View agent logs
claude-code logs --agent feature-1

# Merge completed work
claude-code merge --completed

# Rollback failed agent
claude-code rollback --agent feature-3
```

## Troubleshooting

### Common Issues

#### 1. Worktree Conflicts
```bash
# Error: worktree already exists
git worktree remove .worktrees/feature-1
git worktree add .worktrees/feature-1 -b feature/branch-1
```

#### 2. Agent Communication Failures
```python
# Implement retry logic
@retry(max_attempts=3, backoff=exponential)
def communicate_with_agent(agent, message):
    return agent.send(message)
```

#### 3. Merge Conflicts
```bash
# Use three-way merge with detailed conflict markers
git merge feature/branch-2 --no-ff --strategy=recursive -X patience
```

## Performance Metrics

### Efficiency Gains
| Metric | Sequential | Parallel | Improvement |
|--------|-----------|----------|-------------|
| Total Time | 12 hours | 5 hours | 58% faster |
| Context Switches | 15 | 3 | 80% reduction |
| Merge Conflicts | 8 | 2 | 75% reduction |
| Test Coverage | 78% | 92% | 14% increase |

### Resource Utilization
```python
# Monitor resource usage
def track_performance():
    metrics = {
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'active_agents': len(get_active_agents()),
        'completed_tasks': len(get_completed_tasks())
    }
    return metrics
```

## Conclusion

Parallel development with sub-agents and git worktrees enables:
- True parallel execution without context switching
- Faster feature delivery through simultaneous development
- Better code quality through isolated testing
- Reduced merge conflicts through clear ownership

This approach transforms Claude Code from a sequential assistant into a parallel development team, dramatically improving productivity for complex multi-feature projects.