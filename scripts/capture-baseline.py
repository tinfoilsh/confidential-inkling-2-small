"""Capture the frozen baseline identity of the running Inkling engine.

Run inside the engine container (docker exec). Records model identity, engine
configuration, KV capacity, CUDA graph inventory, GPU memory, and the sidecar
view. Follows kimi-k31-cc-lab's step 0: document everything before changing it.
"""
import json
import re
import subprocess
import sys
import time
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else '/root/lab/baseline.json'
LOG = sys.argv[2] if len(sys.argv) > 2 else None  # engine log text file (optional)

base = {}
base['captured_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

models = json.loads(urllib.request.urlopen('http://localhost:18001/v1/models', timeout=30).read())
base['models'] = models['data']

try:
    smi = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
         '--format=csv,noheader'], text=True, timeout=60)
    base['gpu_memory'] = smi.strip().splitlines()
except Exception as exc:  # driver access may be mediated
    base['gpu_memory_error'] = str(exc)

if LOG:
    text = open(LOG, errors='replace').read()
    picks = {
        'kv_cache': r'.*GPU KV cache size.*|.*Available KV cache memory.*',
        'graphs': r'.*[Cc]aptur.*graph.*|.*cudagraph.*sizes.*',
        'weights': r'.*[Ww]eights.*t(?:ook|ake).*GiB.*|.*Model loading took.*',
        'init': r'.*init engine.*|.*Engine.*took.*seconds.*|.*startup complete.*',
    }
    base['log_extracts'] = {}
    for key, pat in picks.items():
        rows = [line.strip()[:300] for line in text.splitlines() if re.match(pat, line)]
        base['log_extracts'][key] = rows[-8:]

with open(OUT, 'w') as f:
    json.dump(base, f, indent=2)
print('baseline written to', OUT)
