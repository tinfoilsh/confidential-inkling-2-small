#!/bin/sh
# Runs INSIDE the engine container: concurrency curve via vllm bench serve.
# Pattern from deepseek-flash-cc-lab/benchmarks/natural/run_natural.py:
# warmup trial per point, fixed seeds, percentile metrics, saved result JSONs.
set -eu
OUT=${1:?output dir}
MODEL=inkling-2-small
TOK=/tinfoil/mpk/mpk-2eb1ad6339674a38f3e9179c2933c16fc52f66d8e8b48047007665d4d48b477f
ILEN=${ILEN:-1024}
OLEN=${OLEN:-256}
SEED_BASE=20261113
mkdir -p "$OUT"

run() { # concurrency num_prompts tag seed
  vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --host 127.0.0.1 --port 18001 \
    --model "$MODEL" --tokenizer "$TOK" --served-model-name "$MODEL" \
    --dataset-name random --random-input-len "$ILEN" --random-output-len "$OLEN" \
    --num-prompts "$2" --max-concurrency "$1" --request-rate inf --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --seed "$4" \
    --save-result --result-dir "$OUT" --result-filename "$3.json" \
    > "$OUT/$3.log" 2>&1
  echo "done $3"
}

for C in 1 8 16 32; do
  WARM=$(( C * 2 )); [ "$WARM" -lt 4 ] && WARM=4
  N=$(( C * 8 )); [ "$N" -lt 16 ] && N=16
  run "$C" "$WARM" "c$C-warmup" $(( SEED_BASE + C ))
  run "$C" "$N"    "c$C-r1"     $(( SEED_BASE + C + 1000 ))
done
echo "curve complete"
