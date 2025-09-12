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

ALWAYS RESPECT PROJECT STRUCTURE. ALWAYS STRIVE TO USE AVAILABLE MCP SERVERS AND THEIR TOOLS WHENEVER APPLICABLE."

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