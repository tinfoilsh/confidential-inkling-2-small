#!/bin/bash
# Workstation driver for the natural-workload sweep (exact JSON oracles).
# Usage: scripts/run-natural.sh <ssh-port> <variant> [run-name] [corpus]
set -euo pipefail
PORT=${1:?ssh port}
VARIANT=${2:?dspark-off|dspark-1|dspark-3|dspark-5}
RUN=${3:-natural-$VARIANT-$(date +%Y%m%d-%H%M%S)}
CORPUS=${4:-mtp-natural-v1.jsonl}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
SSH="ssh -o BatchMode=yes -o ClearAllForwardings=yes -o StrictHostKeyChecking=accept-new -p $PORT root@inf29.tinfoil.sh"
TOK=/tinfoil/mpk/mpk-2eb1ad6339674a38f3e9179c2933c16fc52f66d8e8b48047007665d4d48b477f

COPYFILE_DISABLE=1 tar -C "$ROOT/benchmarks" --exclude '._*' -czf /tmp/inkling-natural.tgz natural
scp -q -o ClearAllForwardings=yes -P "$PORT" /tmp/inkling-natural.tgz root@inf29.tinfoil.sh:/tmp/
$SSH "cat /tmp/inkling-natural.tgz | docker exec -i inkling-2-small sh -c 'mkdir -p /root/lab && tar xzf - -C /root/lab --no-same-owner'"
$SSH "docker exec inkling-2-small python3 /root/lab/natural/run_natural.py \
      --base-url http://127.0.0.1:18001 \
      --model inkling-2-small --tokenizer $TOK \
      --variant $VARIANT --natural-kind oracle \
      --server-id inkling-debug-inf29 --cc-mode on \
      --dataset-path /root/lab/natural/$CORPUS \
      --out-dir /root/lab/$RUN"
$SSH "docker exec inkling-2-small tar czf - -C /root/lab $RUN" > /tmp/$RUN.tgz
mkdir -p "$ROOT/evidence"
tar -C "$ROOT/evidence" -xzf /tmp/$RUN.tgz
echo "evidence in evidence/$RUN"
python3 -c "
import json, pathlib
d = pathlib.Path('$ROOT/evidence/$RUN')
r = json.loads((d/'receipt.json').read_text())
print('status:', r.get('status'))
for row in r.get('records', r.get('phases', [])):
    print(json.dumps(row)[:200])
"
