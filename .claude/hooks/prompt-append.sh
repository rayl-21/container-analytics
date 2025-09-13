#!/bin/bash

# Read the input JSON from stdin
input=$(cat)

# Extract the original prompt
prompt=$(echo "$input" | python3 -c "
import sys
import json
data = json.load(sys.stdin)
print(data.get('prompt', ''))
")

# Define your custom message to append
APPEND_MESSAGE="
REMEMBER: 
1. Respect project structure and save files according to the project structure. 
2. Use serena whenever possible. 
3. Use playwright to understand browser and webpage, and convert the logic to Python. 
4. Use virtual environment in venv/ for development. 
5. Always prioritize executing code inline. 
6. You can write your plans to .claude/docs/ folder."

# Create the modified prompt
modified_prompt="${prompt}${APPEND_MESSAGE}"

# Output the modified JSON
python3 -c "
import json
output = {
    'prompt': '''${modified_prompt}'''
}
print(json.dumps(output))
"