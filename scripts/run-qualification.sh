#!/bin/bash
# Push the qualification harness into the live engine container and run it.
# Usage: scripts/run-qualification.sh <ssh-port> [run-name]
set -euo pipefail
PORT=${1:?ssh port}
ENGINE=${ENGINE:-inkling-2-small}
RUN=${2:-run-$(date +%Y%m%d-%H%M%S)}
HERE=$(cd "$(dirname "$0")" && pwd)
SSH="ssh -o BatchMode=yes -o ClearAllForwardings=yes -o StrictHostKeyChecking=accept-new -p $PORT root@inf29.tinfoil.sh"

COPYFILE_DISABLE=1 tar -C "$HERE" --exclude '._*' -czf /tmp/inkling-lab-scripts.tgz qualify-api.py capture-baseline.py fixtures
scp -q -o ClearAllForwardings=yes -P "$PORT" /tmp/inkling-lab-scripts.tgz root@inf29.tinfoil.sh:/tmp/
# /root is tmpfs: docker cp cannot see it, so stream tars through docker exec -i
$SSH "cat /tmp/inkling-lab-scripts.tgz | docker exec -i $ENGINE sh -c 'mkdir -p /root/lab && tar xzf - -C /root/lab --no-same-owner'"

$SSH "docker logs inkling-2-small 2>&1 | grep -v 'proxy error' | \
      docker exec -i $ENGINE sh -c 'cat > /root/lab/engine.log' && \
      docker exec $ENGINE python3 /root/lab/capture-baseline.py /root/lab/$RUN-baseline.json /root/lab/engine.log"

$SSH "docker exec $ENGINE python3 /root/lab/qualify-api.py \
      --base-url http://localhost:18001 --fixtures /root/lab/fixtures \
      --output /root/lab/$RUN" || true

$SSH "docker exec $ENGINE sh -c 'cp /root/lab/$RUN-baseline.json /root/lab/$RUN/baseline.json; tar czf - -C /root/lab $RUN' > /tmp/$RUN.tgz"
scp -q -o ClearAllForwardings=yes -P "$PORT" root@inf29.tinfoil.sh:/tmp/$RUN.tgz /tmp/
mkdir -p "$HERE/../evidence"
tar -C "$HERE/../evidence" -xzf /tmp/$RUN.tgz
echo "evidence in evidence/$RUN"
python3 -c "import json;r=json.load(open('$HERE/../evidence/$RUN/summary.json'));print('passed:',r['passed']);[print(' ',c['name'],'PASS' if c.get('passed') else 'FAIL',c.get('error','')) for c in r['cases']]"
