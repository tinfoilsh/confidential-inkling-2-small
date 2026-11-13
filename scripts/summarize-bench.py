"""Summarize a bench-curve evidence directory into a table + JSON."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
rows = []
for f in sorted(d.glob('c*-r*.json')):
    r = json.loads(f.read_text())
    rows.append({
        'point': f.stem,
        'concurrency': r.get('max_concurrency'),
        'completed': r.get('completed'),
        'req_per_s': round(r.get('request_throughput', 0), 2),
        'output_tok_s': round(r.get('output_throughput', 0), 1),
        'total_tok_s': round(r.get('total_token_throughput', 0), 1),
        'ttft_ms_p50': round(r.get('median_ttft_ms', 0), 1),
        'ttft_ms_p95': round(r.get('p95_ttft_ms', 0), 1),
        'tpot_ms_p50': round(r.get('median_tpot_ms', 0), 2),
        'tpot_ms_p95': round(r.get('p95_tpot_ms', 0), 2),
        'decode_tok_s_p50': round(1000 / r['median_tpot_ms'], 1) if r.get('median_tpot_ms') else None,
        'decode_tok_s_p95_tail': round(1000 / r['p95_tpot_ms'], 1) if r.get('p95_tpot_ms') else None,
        'e2el_ms_p50': round(r.get('median_e2el_ms', 0), 1),
    })
(d / 'curve-summary.json').write_text(json.dumps(rows, indent=2) + '\n')
cols = list(rows[0].keys()) if rows else []
widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
print(' | '.join(c.ljust(widths[c]) for c in cols))
print('-|-'.join('-' * widths[c] for c in cols))
for r in rows:
    print(' | '.join(str(r[c]).ljust(widths[c]) for c in cols))
