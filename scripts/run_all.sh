#!/usr/bin/env bash
# Full revision pipeline on the GPU box. Resumable: finished runs are skipped.
#   bash scripts/run_all.sh check   # 4-user dry run of one config per backend
#   bash scripts/run_all.sh full    # everything
ROOT=${ROOT:-/root/autodl-tmp}
cd $ROOT/agentic_popularity_lockin
export PYTHONPATH=. MEM0_TELEMETRY=False TOKENIZERS_PARALLELISM=false
# no nvcc on the box, so FlashInfer's JIT sampler cannot build
export PATH=$ROOT/venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
V=$ROOT/venv/bin/python
LOG=$ROOT/suite.log
$V scripts/make_revision_configs.py --model-name $ROOT/models/Qwen3-8B-AWQ >> $LOG 2>&1

if [ "$1" = "check" ]; then
  $V scripts/run_revision_suite.py --backend vllm --only main_ml1m_seed42 --n-users 4 >> $LOG 2>&1
  bash scripts/start_vllm_server.sh >> $LOG 2>&1 && \
    $V scripts/run_revision_suite.py --backend openai --only mem0_ml1m_seed42 --n-users 2 >> $LOG 2>&1
  pkill -f "vllm serve"
  echo CHECK_FINISHED >> $LOG
  exit 0
fi

$V scripts/run_revision_suite.py --backend vllm >> $LOG 2>&1
bash scripts/start_vllm_server.sh >> $LOG 2>&1 && \
  $V scripts/run_revision_suite.py --backend openai >> $LOG 2>&1
pkill -f "vllm serve"
echo SUITE_FINISHED >> $LOG
