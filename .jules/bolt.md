## 2025-05-23 - Huge Data Transfer Bottleneck in OrthosGPU
**Learning:** The `OrthosGPU` class blindly transfers the entire allocated Trie buffer (27.5M integers per array, x3 arrays) to the GPU on every dictionary processing step. This happens inside nested loops in `generate_level`.
**Action:** Implement intelligent slicing based on `trie_max`. This tracks the actual used size of the Trie. Since typical Tries are <1% of the reserved buffer, this will reduce PCIe bandwidth usage by ~50-100x.

## 2025-05-23 - Memory Coalescing for Candidate Extraction
**Learning:** The `kernel_extract_candidates` GPU kernel was writing to global memory with a stride of `MAX_LEN * pat_len` (approx 1000 bytes) between adjacent threads. This uncoalesced access pattern severely limits memory bandwidth utilization.
**Action:** Reorganized `out_patterns`, `out_weights`, and `out_mask` arrays to use `order='F'` (Column-Major) with shape `(n_words, MAX_LEN, pat_len)`. The kernel now indexes as `[idx, dpos, k]`, ensuring that adjacent threads (adjacent `idx`) access adjacent memory addresses. This should significantly improve write throughput on GPU.
