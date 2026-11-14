"""Make vLLM honor --ignore-patterns for local model paths.

Upstream bug: RunaiModelStreamerLoader only applies load_config.ignore_patterns
in the HF-download branch; local directories glob every *.safetensors, so the
bundled DSpark draft (mtp.safetensors) crashes loading with
"Unexpected checkpoint tensor: fuse_weight".

Applied at image build. Idempotent; fails the build loudly if the anchor moves.
"""
import os

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/runai_streamer_loader.py"
ANCHOR = "        hf_weights_files = list_safetensors(path=hf_folder)"
MARK = "TINFOIL PATCH: local ignore_patterns"
EXTRA = f"""
        # {MARK}
        import fnmatch, os as _os
        _ig = self.load_config.ignore_patterns or []
        if isinstance(_ig, str): _ig = [_ig]
        hf_weights_files = [f for f in hf_weights_files
                            if not any(fnmatch.fnmatch(_os.path.basename(f), pat) for pat in _ig)]"""

src = open(PATH).read()
if MARK in src:
    print("already patched")
    raise SystemExit(0)
assert src.count(ANCHOR) == 1, f"anchor count={src.count(ANCHOR)}; vllm changed, update patch"
tmp = PATH + ".tmp"
with open(tmp, "w") as f:
    f.write(src.replace(ANCHOR, ANCHOR + EXTRA))
os.replace(tmp, PATH)
print("patched")
