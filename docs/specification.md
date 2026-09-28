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
- KV cache paging and repository context retrieval are planned subsystems, not
  part of the first implementation unit.

## Implementation state

- `disk-backed-runtime`: complete — opaque layer mmap, bounded LRU cache,
  explicit unload, and a CPU-only layer lifecycle boundary.
- tensor/model format: pending
- asynchronous prefetch worker: pending
- KV cache storage: pending
- repository context retrieval: pending
- CLI and automatic memory budgeting: pending
