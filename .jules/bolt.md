# Bolt's Journal

## 2025-05-23 - Huge Data Transfer Bottleneck in OrthosGPU
**Learning:** The `OrthosGPU` class blindly transfers the entire allocated Trie buffer (27.5M integers per array, x3 arrays) to the GPU on every dictionary processing step. This happens inside nested loops in `generate_level`.
**Action:** Implement intelligent slicing based on `trie_max`. This tracks the actual used size of the Trie. Since typical Tries are <1% of the reserved buffer, this will reduce PCIe bandwidth usage by ~50-100x.
