#!/bin/sh
# Runs ON THE GUEST TOOLBOX. Kimi-lab engine replacement discipline:
# graceful stop (60s), retain old container, wait for GPU contexts to retire
# (120s gate), then start the experimental engine with restart=no.
# Usage: switch-engine.sh <new-name> <extra-vllm-args...>
set -eu
NEW=${1:?new engine container name}
shift
CURRENT=""
for c in inkling-2-small $(docker ps --format '{{.Names}}' | grep -v toolbox); do
  if docker inspect "$c" >/dev/null 2>&1 && [ "$(docker inspect -f '{{.State.Running}}' "$c")" = "true" ]; then
    case "$c" in *inkling*) CURRENT=$c; break;; esac
  fi
done
IMG=$(docker inspect inkling-2-small --format '{{.Config.Image}}')
echo "current=$CURRENT image=$IMG"

if [ -n "$CURRENT" ]; then
  echo "== graceful stop $CURRENT (60s)"
  docker stop --time 60 "$CURRENT"
  [ "$(docker inspect -f '{{.State.Running}}' "$CURRENT")" = "false" ] || { echo "engine did not stop"; exit 1; }
  docker rename "$CURRENT" "$CURRENT-retained-$(date +%s)" 2>/dev/null || true
fi

echo "== GPU retirement gate (120s)"
DEADLINE=$(( $(date +%s) + 120 ))
while :; do
  ACTIVE=$(docker run --rm --runtime nvidia --gpus all --network none \
    --entrypoint nvidia-smi "$IMG" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || echo probe-failed)
  [ -z "$ACTIVE" ] && break
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "GPU contexts remain: $ACTIVE; no replacement started"; exit 1; }
  sleep 5
done
echo "GPUs clear"

echo "== starting $NEW (restart=no)"
docker inspect inkling-2-small --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -v '^$' > /tmp/env.txt
ENVFLAGS=""
while IFS= read -r e; do ENVFLAGS="$ENVFLAGS -e \"$e\""; done < /tmp/env.txt
eval docker run -d --name "$NEW" \
  --network bridge \
  --runtime nvidia --gpus all \
  --ipc private --shm-size 412316860416 \
  --restart no \
  -v /mnt/ramdisk/public:/tinfoil:ro \
  $ENVFLAGS \
  --entrypoint /opt/tinfoil/inference-sidecar "$IMG" vllm serve \
  --model /tinfoil/mpk/mpk-2eb1ad6339674a38f3e9179c2933c16fc52f66d8e8b48047007665d4d48b477f \
  --tensor-parallel-size 4 --max-model-len 524288 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --reasoning-parser inkling --tool-call-parser inkling \
  --enable-auto-tool-choice --enable-prompt-tokens-details --enable-force-include-usage \
  --served-model-name inkling-2-small --scheduling-policy priority \
  --load-format runai_streamer --disable-custom-all-reduce \
  --no-enable-flashinfer-autotune --port 8001 "$@"
echo "started $NEW"
