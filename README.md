# confidential-inkling-2-small

vLLM image for [Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(Thinking Machines Lab) on 4x NVIDIA B300 with confidential computing.

Inkling 2 Small is a multimodal MoE (text and image in, text out) with 594B
total / 17B active parameters. The checkpoint's native precision is NVFP4 —
the model was pretrained, midtrained, and post-trained in NVFP4, so this is
the canonical model, not a quantization. Weights are ~473GB in VRAM; each
enclave serves one TP4 replica with a 524,288-token context window.

The current configuration is deliberately minimal: target model only, no
speculative decoding (the checkpoint bundles a DSpark draft — enabling it is
a later controlled change), stock attention backend, and the fleet-standard
CC environment hygiene carried over from
[confidential-glm5-3-nvfp4](https://github.com/tinfoilsh/confidential-glm5-3-nvfp4).
See [tinfoil-config.yml](tinfoil-config.yml) for the pinned model revision,
CVM version, resource allocation, and serving flags.

## Build

The [Dockerfile](Dockerfile) packages the inference sidecar on a digest-pinned
`vllm/vllm-openai:inkling-2-small` base (the upstream image carrying Inkling 2
Small support until it lands in a vLLM release).

```sh
docker build -t confidential-inkling-2-small .
```

## References

- [Official vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small)
- [Model card](https://huggingface.co/thinkingmachines/Inkling-2-Small)
