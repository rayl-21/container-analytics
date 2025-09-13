#!/bin/bash

# PreToolUse hook for injecting context to sub-agents
# This hook runs before the Task tool is executed

cat << 'EOF'

==============================================================
IMPORTANT SUB-AGENT CONTEXT AND GUIDELINES
==============================================================

REMEMBER:
1. Respect project structure and save files according to the project structure.
2. Use serena whenever possible.
3. Use playwright to understand browser and webpage, and convert the logic to Python.
4. Use virtual environment in venv/ for development.
5. Always prioritize executing code inline.
6. You can write your plans to .claude/docs/ folder if needed.
7. Use context7 for most up-to-date documentation.

==============================================================

EOF