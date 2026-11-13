# Natural engineering workloads

These are unchanged, synthetic GLM-lab workloads at source commit
`75a5ef14ce1a4e47ae40a48e96e3edb2460a63f1`, with no customer prompts.

- `mtp-natural-v1.jsonl`: 256 invoice, inventory, Python-summary, and
  normalization cases with exact JSON oracles. SHA256:
  `d18a48e1017183464143262c74941e0b03f1aa2f4281f04e8ba60f05266b9a4d`.
- `mtp-long-natural-v1.jsonl`: 24 design, code-generation, and review prompts.
  SHA256: `225737f87d95829251e4d984b8b54856dd15622595a6e19a897db8f72812cc63`.

Preserve natural termination and retain caps, empty final answers, repetition,
and invalid JSON as failures. Measure final content separately from reasoning.
Input lengths must be measured with the pinned DeepSeek tokenizer. These
realistic synthetic prompts supplement the captured coding sessions and actual
repository tasks; they do not represent a production traffic distribution.

`run_natural.py` reuses the GLM `scripts/benchmark-sweep.py` at that commit.
The adaptation selects the DeepSeek tokenizer, DSpark labels, and concurrency
1/8/32. The driver retains its source warmups, seeds, request counts, exact
JSON checks, cap rejection, and final-content token measurement. Its DSpark
label does not configure the engine. Run it in the pinned natural client
image using `/opt/natural-venv/bin/python`; this environment inherits the
frozen release image, never an upgraded engine's packages.

`inspect_latency.py` compares the retained client and native timing records
using `prometheus-client` in the local analysis venv. It checks matching corpus
hashes, workload arguments, seeds, input lengths, and histogram counts. It adds
no serving requests. Reproduce the final investigation with:

```bash
.venv/bin/python benchmarks/natural/inspect_latency.py \
  --baseline-archive runs/baseline-natural-context-inputs.tar.gz \
  --candidate-archive runs/final-static-staging-output-capacity-measurements.tar.gz \
  --candidate-label final-static-staging-output-capacity \
  --output runs/natural-latency-review.json
```
