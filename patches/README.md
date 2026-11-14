# vLLM patches

Unified-diff patches applied on top of the base image in `../Dockerfile`.
Applied in filename order at build time.

## Rules

- Diffs are `-p1` rooted at `/`, so paths start with
  `usr/local/lib/python3.12/dist-packages/...`
- Each patch is one reviewable change with a prefixed number
  (`NNNN-short-slug.patch`)
- Each patch includes an in-code comment explaining the upstream bug so a
  future reader knows when it can be retired after a base-image bump

## Patches

| patch | what / why |
| --- | --- |
| `0001-runai-local-ignore-patterns.patch` | RunaiModelStreamerLoader only honors `--ignore-patterns` in its HF-download branch; local model directories glob every `*.safetensors`, so checkpoints that bundle loose draft weights (`mtp.safetensors`) crash target-only loads with `Unexpected checkpoint tensor: fuse_weight`. Filters the local file list by `load_config.ignore_patterns`. Retire when upstream applies ignore_patterns to local paths. |

## Adding a patch

```bash
F=vllm/model_executor/model_loader/runai_streamer_loader.py
BASE=$(sed -n 's/^ARG VLLM_BASE_IMAGE=//p' ../Dockerfile)

docker run --rm --entrypoint cat "$BASE" "/usr/local/lib/python3.12/dist-packages/$F" > orig
cp orig patched && $EDITOR patched
diff -u --label "a/usr/local/lib/python3.12/dist-packages/$F" \
        --label "b/usr/local/lib/python3.12/dist-packages/$F" \
        orig patched > NNNN-slug.patch
docker build ..
```
