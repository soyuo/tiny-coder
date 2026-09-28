# TinyCode working specification

## Product intent

TinyCode is a CPU-only code-generation LLM runtime that can execute a model
larger than available RAM by keeping model weights on disk and loading only the
layer needed for the current computation.

## Confirmed system logic

- GPU is not used.
- Layer weights are disk-backed and are loaded on demand.
- A completed layer is unloaded before the next layer is processed.
- The runtime should support mmap-based weight access where possible.
- Layer cache capacity is controlled by a memory budget or explicit cache size.
- KV cache paging and repository context retrieval are implemented as separate
  runtime components.

## Implementation state

- `disk-backed-runtime`: complete — opaque layer mmap, bounded LRU cache,
  explicit unload, and a CPU-only layer lifecycle boundary.
- model manifest: complete — `model.json` provides layer count and filename width.
- tensor header: complete — dtype, shape, and payload size are validated before decoding.
- tensor-aware loading: complete — mapped layers can be validated before execution.
- async prefetch: complete — a single worker can warm the next layer without blocking the caller.
- disk-backed KV cache: complete — hot entries are bounded and older entries remain on disk.
- repository context search: complete — only ranked, limited files enter context.
- CLI configuration: complete — `run` validates model, memory, layer-cache, and KV-cache settings.
- CPU inference core: in progress — mapped tensors can be passed to CPU executors.
- tokenizer baseline: complete — UTF-8 text has a reversible byte-token representation.
- attention core: in progress — CPU scaled dot-product attention is available for 2D sequences.
- decoder-only block: in progress — multi-head causal attention, RMSNorm, SwiGLU, and residual paths are connected.
- packed layer format: in progress — one `layer_XX.bin` can contain named tensor payloads.
- disk decoder loading: in progress — decoder layers are materialized from packed files on demand.
- greedy generation: in progress — deterministic next-token generation is available.
