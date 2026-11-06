#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for Qwen3-8B-AWQ with thinking disabled
# in the chat template (mem0 sends its own requests and cannot pass template kwargs).
ROOT=${ROOT:-/root/autodl-tmp}
V=$ROOT/venv/bin
M=$ROOT/models/Qwen3-8B-AWQ
T=$ROOT/qwen3_nothink.jinja
$V/python - "$M" "$T" <<'EOF'
import json, sys
from pathlib import Path
model, target = Path(sys.argv[1]), Path(sys.argv[2])
jinja = model / "chat_template.jinja"
template = jinja.read_text() if jinja.exists() else json.loads((model / "tokenizer_config.json").read_text())["chat_template"]
target.write_text("{%- set enable_thinking = false %}\n" + template)
EOF
nohup $V/vllm serve $M --served-model-name qwen3-8b-awq --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --enable-prefix-caching --chat-template $T \
  --port 8000 --seed 0 > $ROOT/vllm_server.log 2>&1 &
for i in $(seq 1 120); do
  if curl -s -m 2 http://127.0.0.1:8000/health >/dev/null; then echo SERVER_UP; exit 0; fi
  sleep 5
done
echo SERVER_TIMEOUT; tail -30 $ROOT/vllm_server.log; exit 1
