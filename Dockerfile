# syntax=docker/dockerfile:1.6
#
# vLLM image for Inkling 2 Small (NVFP4-native MoE, 594B/17B active) on
# 4x NVIDIA B300. Base is the upstream Inkling support image, digest-pinned
# for attestation (amd64 manifest of vllm/vllm-openai:inkling-2-small; use
# this tag until Inkling 2 Small support lands in a vLLM release).
#
# First pass: no vLLM patches and no baked FlashInfer artifacts. CC-specific
# patches get added here one controlled change at a time as the debug-mode
# lab on inf29 surfaces what breaks (see confidential-glm5-3-nvfp4 and
# kimi-k31-cc-lab for the prior patch sets this will likely draw from).
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:inkling-2-small@sha256:f4f3176fe12a5d77cb7e01399be01e906350ce900ae3642dbddc031c93fa7210
ARG SIDECAR_IMAGE=ghcr.io/tinfoilsh/inference-sidecar@sha256:65ce23d6560c46a1e8614ede187fcbf9798b267aa33878905b4872404787f47d
FROM ${SIDECAR_IMAGE} AS sidecar

# CUDA library headers for flashinfer's %globaltimer JIT (CC-reliable timing;
# the base image ships the CUDA toolkit without cusparse/cusolver/etc headers).
# Extract headers only — never install cu12 runtime libs into this image.
FROM ${VLLM_BASE_IMAGE} AS cuda-headers
WORKDIR /hdr
RUN pip3 download --no-deps --only-binary=:all: -d /hdr/wheels \
        nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-cufft-cu12 \
        nvidia-curand-cu12 nvidia-cublas-cu12 \
    && python3 - <<'PYEOF'
import glob, pathlib, zipfile
out = pathlib.Path('/hdr/include'); out.mkdir(parents=True)
for w in glob.glob('/hdr/wheels/*.whl'):
    with zipfile.ZipFile(w) as z:
        for n in z.namelist():
            p = pathlib.PurePosixPath(n)
            if '/include/' in n and p.suffix in ('.h', '.hpp'):
                (out / p.name).write_bytes(z.read(n))
PYEOF

FROM ${VLLM_BASE_IMAGE}

COPY --from=sidecar /inference-sidecar /opt/tinfoil/inference-sidecar

# vLLM patches (see patches/README.md), applied in filename order
COPY patches/*.patch /opt/tinfoil/patches/
RUN for p in /opt/tinfoil/patches/*.patch; do patch -p1 -d / --forward < "$p"; done

# CUDA lib headers (globaltimer JIT) + pandas (in-enclave bench harness)
COPY --from=cuda-headers /hdr/include/ /usr/local/cuda/include/
RUN pip3 install --no-cache-dir pandas==2.3.2

ENTRYPOINT ["/opt/tinfoil/inference-sidecar", "vllm", "serve"]
