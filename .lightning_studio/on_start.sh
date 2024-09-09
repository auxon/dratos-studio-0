#!/bin/bash

# This script runs every time your Studio starts, from your home directory.

# List files under fast_load that need to load quickly on start (e.g. model checkpoints).
#
# ! fast_load
# <your file here>

# Add your startup commands below.
#
# Example: streamlit run my_app.py
# Example: gradio my_app.py
sudo systemctl start redis-stack-server

# Create a new tmux session named "firecrawl" and start the workers
tmux new-session -d -s firecrawl 'cd ~/firecrawl/apps/api && pnpm run workers'

# Create a new window in the "firecrawl" session and start the API
tmux new-window -t firecrawl 'cd ~/firecrawl/apps/api && pnpm run start'

tmux new-session -d -s chatbot 'fastapi run ~/chatbot/server/server.py --host 0.0.0.0 --port 8888'

tmux new-window -t chatbot 'cd ~/chatbot/chatbot-ui && npm run dev ----port 5173'
