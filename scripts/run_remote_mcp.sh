#!/bin/sh

set -eu

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
# Load project-local launch settings when the MCP host does not inherit direnv.
. "$script_dir/../.envrc"

: "${LAYA_MCP_REMOTE_HOST:?Set LAYA_MCP_REMOTE_HOST}"
: "${LAYA_MCP_REMOTE_GPU:?Set LAYA_MCP_REMOTE_GPU}"
: "${LAYA_MCP_REMOTE_MODEL:?Set LAYA_MCP_REMOTE_MODEL}"
: "${LAYA_MCP_REMOTE_EXECUTABLE:?Set LAYA_MCP_REMOTE_EXECUTABLE}"
: "${LAYA_MCP_MCP_SERVERS:?Set LAYA_MCP_MCP_SERVERS}"

remote_quote() {
  printf "'%s'" "$(printf %s "$1" | command sed "s/'/'\\\\''/g")"
}

remote_command="env"
for setting in \
  "USE_TF=${LAYA_MCP_REMOTE_USE_TF:-0}" \
  "HIP_VISIBLE_DEVICES=$LAYA_MCP_REMOTE_GPU" \
  "LAYA_MCP_BACKEND=torch" \
  "LAYA_MCP_MODEL=$LAYA_MCP_REMOTE_MODEL" \
  "LAYA_MCP_MCP_SERVERS=$LAYA_MCP_MCP_SERVERS"; do
  remote_command="$remote_command $(remote_quote "$setting")"
done

exec ssh -T "$LAYA_MCP_REMOTE_HOST" \
  "$remote_command $(remote_quote "$LAYA_MCP_REMOTE_EXECUTABLE")"
