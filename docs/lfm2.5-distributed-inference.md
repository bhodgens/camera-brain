# Rock Cluster: LFM2.5-8B Distributed Inference Project

## Status: Paused - CPU-only inference not viable for distributed setup

## Cluster Inventory

### Nodes
| Node | SoC | CPU | Clock | RAM | Swap | Storage | OS | Kernel |
|------|-----|-----|-------|-----|------|---------|----|--------|
| rock0 | RPi5 | Cortex-A76 (4-core) | ~2.4GHz | 8GB | 200MB | 469GB NVMe (426GB free) | Debian 13 (trixie) | 6.6.47+rpt-rpi-2712 |
| rock1-5 | RK3568 | Cortex-A55 (4-core) | ~1.4-1.8GHz | 2GB each | 2GB each (on NVMe) | 6.8GB eMMC + 117GB NVMe at /home | Ubuntu 20.04 (focal) | 4.19.193-rockchip |

### Network
- All on gigabit ethernet, sub-millisecond latency (~1ms RTT)
- rock0: wlan0 (10.9.8.x, internet) + eth0 (10.0.0.x, cluster LAN)
- rock1-5: 10.9.8.x network
- **rock0 routing fix applied**: removed Gateway from systemd-networkd eth0 config to let wlan0 handle internet. Edit: `/etc/systemd/network/00-eth0.network` (removed Gateway=10.0.0.254)

### Toolchain
- rock0: gcc 14.2, cmake 3.31.6, Python 3.13.5 (upgraded to trixie)
- rock1-5: gcc 9.4, cmake 3.16, Python 3.8.10

### GPU/NPU Status
- **All NPU/GPU completely dark** on rock1-5
- rock0: V3D + Mali (not useful for compute)
- rock1-5: Mali GPU device nodes exist (/dev/dri/, /dev/mali0) but no NPU driver, no RKNN libs
- No kernel module loaded (`lsmod` shows nothing for rknpu/rknn)

## Model: LFM2.5-8B-A1B

### Architecture
- Type: MoE (Mixture of Experts)
- Blocks: 24 (2 leading dense + 22 MoE)
- Experts: 32 per MoE block, 4 active per token
- Active params: ~1B (A1B designation)
- Embedding dim: 2048, Vocab: 128K
- Context length: 128K
- GGUF tensors: 256
- Architecture key: `lfm2moe`

### Available Quantizations
| File | Size | Path |
|------|------|------|
| F16 | 16GB | `/Volumes/LLMs/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-F16.gguf` |
| Q4_K_M | 4.9GB | Originally `/Volumes/files/llms/`, now on rock0 at `/home/rock/` |

## Benchmark Results

### rock0 Solo (RPi5, Cortex-A76, 4-core, Q4_K_M)

**llama-bench (4 threads):**
| Test | tok/s |
|------|-------|
| Prompt processing (pp512) | 24.74 |
| Generation tg128 | 8.17 |
| Generation tg512 | 8.07 |

**llama-bench (1 thread - per-core reference):**
| Test | tok/s |
|------|-------|
| Prompt processing (pp256) | 7.31 |
| Generation tg64 | 4.67 |

**llama-server with concurrency (128 tokens each, continuous batching):**
| Concurrency | tok/s per req | Aggregate tok/s | Wall time |
|-------------|---------------|-----------------|-----------|
| 1 | 7.73 | 7.73 | 16.6s |
| 2 | 4.15 | 8.29 | 30.9s |
| 3 | 2.93 | 8.76 | 43.8s |
| 4 | 2.81 | 11.21 | 45.7s |

### rock1 Solo (RK3568, Cortex-A55, 4-core, Q4_K_M)

**llama-bench (4 threads, GGML_CPU_REPACK=OFF, mmap enabled):**
| Test | tok/s |
|------|-------|
| Prompt processing (pp512) | 4.45 |
| Generation (tg128) | 1.12 |

### Performance Ratio
| Metric | rock0 (A76) | rock1 (A55) | Ratio |
|--------|-------------|-------------|-------|
| pp512 | 24.74 | 4.45 | **5.6x** |
| tg128 | 8.17 | 1.12 | **7.3x** |

**Conclusion: A55 workers are too slow for beneficial layer offload in Q4_K_M. Offloading layers to workers makes inference slower, not faster.**

## llama.cpp Build Details

### rock0
- Location: `/home/rock/llama.cpp/`
- Build: `cmake -B build -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release`
- Binaries: `build/bin/llama-server`, `build/bin/llama-cli`, `build/bin/llama-bench`, `build/bin/ggml-rpc-server`
- Model: `/home/rock/LFM2.5-8B-A1B-Q4_K_M.gguf`
- Commit: 4f31eedb0 (build 9850)

### rock1
- Location: `/home/llama.cpp-build/llama.cpp/`
- Build: `cmake -B build -DGGML_RPC=ON -DGGML_CPU_REPACK=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-mcpu=native" -DCMAKE_CXX_FLAGS="-mcpu=native"`
- **Note**: GGML_CPU_REPACK=OFF required because rock1 doesn't have enough contiguous RAM for weight repacking (4.9GB model vs 1.6GB free). Uses mmap instead.
- Model: `/home/llama.cpp-build/LFM2.5-8B-A1B-Q4_K_M.gguf`
- Commit: 4f31eed (build 1)

### RPC Issues Discovered
1. **Tensor split doesn't work as expected with RPC-only devices**: When using `--rpc`, llama-bench assigns ALL layers to the RPC device because the CPU isn't registered as a "device" in CPU-only builds. `-ts` flag has no effect.
2. **CPU_REPACK requires full model in contiguous memory**: Workers with 2GB RAM can't load the Q4_K_M model (4.9GB) even with mmap, because CPU_REPACK tries to allocate a contiguous buffer. Fix: rebuild with `-DGGML_CPU_REPACK=OFF`.
3. **rpc-server needs persistent launch**: `setsid` or `nohup` with `< /dev/null` redirection required to survive SSH session disconnect.

## Benchmark Script
- Location: `/Users/caimlas/git/rock-cluster/benchmark.py`
- Deployed to rock0: `/home/rock/benchmark.py`

## What Would Need to Change for Distributed Inference to Work

1. **NPU acceleration on workers** - If the RKNPU (0.8 TOPS INT8) works, it could dramatically speed up matmul on workers, potentially making layer offload worthwhile
2. **F16 model distribution** - The 16GB F16 model can't fit on any single node and genuinely requires the cluster. Even slow, it enables higher quality inference.
3. **Expert-parallel approach** - Current llama.cpp RPC only supports layer-level pipeline parallelism. True expert parallelism (different experts on different nodes) would change the performance equation entirely.

## Next Steps
- Pivoting to RKNPU bring-up: see `docs/rknpu-bringup-plan.md`
