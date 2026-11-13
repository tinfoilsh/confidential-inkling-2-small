#!/bin/bash
# Workstation driver: push bench-curve.sh into the engine container, run the
# c1/c8/c16/c32 curve on loopback, pull evidence back into the repo.
# Usage: scripts/run-bench-curve.sh <ssh-port> [run-name]
set -euo pipefail
PORT=${1:?ssh port}
ENGINE=${ENGINE:-inkling-2-small}
RUN=${2:-perf-$(date +%Y%m%d-%H%M%S)}
HERE=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -o BatchMode=yes -o ClearAllForwardings=yes -o StrictHostKeyChecking=accept-new -p $PORT root@inf29.tinfoil.sh"

COPYFILE_DISABLE=1 tar -C "$HERE" --exclude '._*' -czf /tmp/inkling-bench.tgz bench-curve.sh
scp -q -o ClearAllForwardings=yes -P "$PORT" /tmp/inkling-bench.tgz root@inf29.tinfoil.sh:/tmp/
$SSH "cat /tmp/inkling-bench.tgz | docker exec -i $ENGINE sh -c 'mkdir -p /root/lab && tar xzf - -C /root/lab --no-same-owner'"
$SSH "docker exec $ENGINE sh /root/lab/bench-curve.sh /root/lab/$RUN"
$SSH "docker exec $ENGINE tar czf - -C /root/lab $RUN" > /tmp/$RUN.tgz
mkdir -p "$HERE/../evidence"
tar -C "$HERE/../evidence" -xzf /tmp/$RUN.tgz
python3 "$HERE/summarize-bench.py" "$HERE/../evidence/$RUN"
