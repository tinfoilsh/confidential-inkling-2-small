#!/bin/bash
# Push the qualification harness into the live engine container and run it.
# Usage: scripts/run-qualification.sh <ssh-port> [run-name]
set -euo pipefail
PORT=${1:?ssh port}
RUN=${2:-run-$(date +%Y%m%d-%H%M%S)}
HERE=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -o BatchMode=yes -o ClearAllForwardings=yes -o StrictHostKeyChecking=accept-new -p $PORT root@inf29.tinfoil.sh"

tar -C "$HERE" -czf /tmp/inkling-lab-scripts.tgz qualify-api.py capture-baseline.py fixtures
scp -q -o ClearAllForwardings=yes -P "$PORT" /tmp/inkling-lab-scripts.tgz root@inf29.tinfoil.sh:/tmp/
$SSH "docker exec inkling-2-small mkdir -p /root/lab && \
      docker cp /tmp/inkling-lab-scripts.tgz inkling-2-small:/root/lab/ && \
      docker exec inkling-2-small sh -c 'cd /root/lab && tar xzf inkling-lab-scripts.tgz'"

$SSH "docker logs inkling-2-small 2>&1 | grep -v 'proxy error' > /tmp/engine.log && \
      docker cp /tmp/engine.log inkling-2-small:/root/lab/engine.log && \
      docker exec inkling-2-small python3 /root/lab/capture-baseline.py /root/lab/$RUN-baseline.json /root/lab/engine.log"

$SSH "docker exec inkling-2-small python3 /root/lab/qualify-api.py \
      --base-url http://localhost:18001 --fixtures /root/lab/fixtures \
      --output /root/lab/$RUN" || true

$SSH "docker cp inkling-2-small:/root/lab/$RUN /tmp/$RUN && \
      docker cp inkling-2-small:/root/lab/$RUN-baseline.json /tmp/$RUN/baseline.json && \
      tar -C /tmp -czf /tmp/$RUN.tgz $RUN"
scp -q -o ClearAllForwardings=yes -P "$PORT" root@inf29.tinfoil.sh:/tmp/$RUN.tgz /tmp/
mkdir -p "$HERE/../evidence"
tar -C "$HERE/../evidence" -xzf /tmp/$RUN.tgz
echo "evidence in evidence/$RUN"
python3 -c "import json;r=json.load(open('$HERE/../evidence/$RUN/summary.json'));print('passed:',r['passed']);[print(' ',c['name'],'PASS' if c.get('passed') else 'FAIL',c.get('error','')) for c in r['cases']]"
