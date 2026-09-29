#!/bin/sh

set -eu

host="${LAYA_MCP_REMOTE_HOST:-magnusknutas@192.168.1.26}"
exec ssh -T "$host" \
  "USE_TF=0 HIP_VISIBLE_DEVICES=1 LAYA_MCP_BACKEND=torch LAYA_MCP_MODEL=/home/magnusknutas/laya-mcp-training/models/mcp-router-reviewed LAYA_MCP_MCP_SERVERS='{\"jcodemunch\":\"Indexed source code structure, symbols, references, and change impact.\",\"docs-mcp-server\":\"Current third-party library documentation and API guidance.\",\"gitea\":\"Repository issues, pull requests, reviews, and project work tracking.\"}' /home/magnusknutas/laya-mcp-training/venv/bin/laya-mcp"
