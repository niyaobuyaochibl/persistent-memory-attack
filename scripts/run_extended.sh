#!/usr/bin/env bash
# Run the three extended experiment sets on the GPU box.
#   bash scripts/run_extended.sh kuairec   # only KuaiRec observed-feedback runs
#   bash scripts/run_extended.sh llama     # only Llama-3.1-8B main reproduction
#   bash scripts/run_extended.sh qwen14b   # only Qwen3-14B main reproduction
#   bash scripts/run_extended.sh all       # everything (default)
ROOT=${ROOT:-/root/autodl-tmp}
cd $ROOT/agentic_popularity_lockin
export PYTHONPATH=. MEM0_TELEMETRY=False TOKENIZERS_PARALLELISM=false
export PATH=$ROOT/venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
V=$ROOT/venv/bin/python
LOG=$ROOT/extended.log
PHASE=${1:-all}

run_set() {
  local tag="$1"; local model="$2"; local out="$3"
  $V scripts/make_revision_configs.py --set model --tag "$tag" \
    --model-name "$model" --out "configs/$tag" >> $LOG 2>&1
  $V scripts/run_revision_suite.py --backend vllm --configs "configs/$tag" >> $LOG 2>&1
  echo "DONE $tag $(date)" >> $LOG
}

run_kuairec() {
  $V scripts/make_revision_configs.py --set kuairec \
    --model-name $ROOT/models/Qwen3-8B-AWQ \
    --out configs/kuairec >> $LOG 2>&1
  $V scripts/run_revision_suite.py --backend vllm --configs configs/kuairec >> $LOG 2>&1
  echo "DONE kuairec $(date)" >> $LOG
}

if [ "$PHASE" = "kuairec" ] || [ "$PHASE" = "all" ]; then run_kuairec; fi
if [ "$PHASE" = "llama" ] || [ "$PHASE" = "all" ]; then
  run_set llama_v1 $ROOT/models/Llama-3.1-8B-Instruct configs/llama_v1
fi
if [ "$PHASE" = "qwen14b" ] || [ "$PHASE" = "all" ]; then
  run_set qwen14b_v1 $ROOT/models/Qwen3-14B-AWQ configs/qwen14b_v1
fi
echo EXTENDED_FINISHED >> $LOG
