# RKNPU Bring-up Plan for Rock Cluster

## Status: Phases 1-2 Complete, Phase 3 Concluded (NPU not viable for LLM inference)

**Bottom line**: The RK3568 NPU works perfectly for vision tasks but its ~4 GFLOPS (FP16) / ~8 GFLOPS (INT8) throughput is insufficient to accelerate LLM inference. The NPU's 0.8 TOPS spec is INT8 peak theoretical - actual throughput is ~10x lower due to overhead, driver overhead, and memory bandwidth.

## Discovery

The RK3568 NPU is **already working** on rock1-5. Initial assessment was wrong - we assumed the `/dev/dri/renderD129` device was Mali GPU, but it's actually the NPU:

- **Driver**: `RKNPU` (built into kernel, `CONFIG_ROCKCHIP_RKNPU=y`)
- **Platform device**: `fde40000.npu`, status `okay`
- **DRM device**: `/dev/dri/card1` and `/dev/dri/renderD129`
- **Compatible**: `rockchip,rk3568-rknpu`
- **Clock**: active
- **IOMMU**: configured in device tree
- **Kernel**: 4.19.193-16-rockchip (vendor BSP from Nov 2021)

### Device Layout (per worker node)
| Device | Purpose |
|--------|---------|
| `/dev/dri/card0` + `renderD128` | Mali GPU + display (rockchip-drm) |
| `/dev/dri/card1` + `renderD129` | **RKNPU** (RKNPU driver) |
| `/dev/mali0` | Mali GPU (legacy interface) |

### NPU Specs (RK3568)
- 0.8 TOPS INT8 (0.5 TOPS effective with overhead)
- 3 NPU cores
- Supports INT8, INT16, FP16 compute
- DMA with IOMMU
- Clocks up to 900 MHz (device tree has OPP table)

## Phase 1: Install RKNN Userspace Runtime - COMPLETE

### What was installed on rock1

| Component | Path | Version |
|-----------|------|---------|
| `librknnrt.so` (v2 runtime) | `/usr/lib/librknnrt.so` | 2.3.2 (429f97ae6b@2025-04-09) |
| `librknn_runtime.so` (v1 runtime, backup) | `/usr/lib/librknn_runtime_v1.so.bak` | 1.7.5 |
| NPU userspace blobs (libOpenVX etc.) | `/usr/lib/` | From rknpu repo |
| `rknn_api.h` header | `/tmp/rknn_api.h` | v1 API header |
| `rknn-toolkit2` (Python) | `~/.local/lib/python3.8/` | 2.3.2 (full conversion tool) |
| `rknn-toolkit-lite2` (Python) | `~/.local/lib/python3.8/` | 2.3.2 |
| Model conversion tool | `rknn.api.RKNN` | Works on aarch64! |

### Key Discovery: v2 Runtime Works with v0.4.2 Driver

**The v2 userspace runtime (librknnrt.so v2.3.2) is backward-compatible with the v0.4.2 kernel driver.**

- The Python rknnlite wrappers (both v1 and v2) fail to load models
- The C API directly using `librknnrt.so` works perfectly
- Model conversion via rknn-toolkit2 works on aarch64 (no x86 needed)
- The v0.4.2 driver + v2 runtime combination successfully runs v2-format .rknn models

### What Also Works
- ONNX to RKNN conversion **on-device** (aarch64, no x86 needed)
- NPU clock ramps to 600MHz under load (100% utilization confirmed via devfreq)
- Input format must be NHWC (not NCHW) for the v2 runtime

### Model Conversion Steps (on any aarch64 node)
```python
from rknn.api import RKNN
rknn = RKNN()
rknn.config(mean_values=[[0,0,0]], std_values=[[1,1,1]], target_platform="rk3568")
rknn.load_onnx(model="model.onnx")
rknn.build(do_quantization=False)
rknn.export_rknn("model.rknn")
rknn.release()
```

## Phase 2: NPU Microbenchmark - DONE

### ResNet50 Results (RK3568 NPU)

| Metric | Value |
|--------|-------|
| Avg inference time | 90.1 ms |
| Min / Max | 89.6 / 90.8 ms |
| Throughput | 11.1 inferences/sec |
| NPU clock | 600 MHz (100% load) |
| Model | ResNet-50 v2 (FP32, no quantization) |
| Model size | 52 MB (.rknn) |

### NPU Stats During Inference
- `/sys/class/devfreq/fde40000.npu/cur_freq`: 600000000 (600 MHz)
- `/sys/class/devfreq/fde40000.npu/load`: 100@600000000Hz (100% utilization)

### MatMul Microbenchmark (FP32)

| Size | Avg Time | Throughput |
|------|----------|------------|
| 512x512 | 0.30 ms | 1.7 GFLOPS |
| 1024x1024 | 0.67 ms | 3.1 GFLOPS |
| 2048x2048 | 2.11 ms | 4.0 GFLOPS |
| 4096x4096 | 7.76 ms | 4.3 GFLOPS |

**NPU FP32 throughput: ~4 GFLOPS** (vs ~2-3 GFLOPS for A55 CPU)

### Cluster NPU Status (all 5 workers verified)

| Node | ResNet50 inf/sec | NPU Clock |
|------|-----------------|-----------|
| rock1 | 11.1 | 600 MHz |
| rock2 | 11.1 | 600 MHz |
| rock3 | 11.1 | 600 MHz |
| rock4 | 11.0 | 600 MHz |
| rock5 | 11.1 | 600 MHz |

**Aggregate cluster: ~55 inf/sec for ResNet50**

### INT8 Quantization Results

| Model | Format | Avg Time | Throughput | Size |
|-------|--------|----------|------------|------|
| ResNet50 | FP16 | 90.4 ms | 11.1 inf/sec | 49.6 MB |
| ResNet50 | INT8 | 35.2 ms | 28.4 inf/sec | 25.1 MB |
| **Speedup** | | **2.56x** | **2.56x** | **0.51x** |

### LLM MatMul Benchmark (LFM2.5 Dimensions)

| Operation | FP16 | INT8 | INT8 Speedup | NPU INT8 vs Naive CPU |
|-----------|------|------|-------------|----------------------|
| FFN up (2048x7168) | 6.8 ms | 3.6 ms | 1.9x | 771x faster |
| FFN down (7168x2048) | 6.9 ms | 3.7 ms | 1.8x | 737x faster |
| Attn Q (2048x2048) | 2.0 ms | 1.2 ms | 1.7x | 637x faster |
| **NPU throughput** | **4.2 GFLOPS** | **7.8 GFLOPS** | **1.9x** | |

### Per-Token LLM Inference Estimate (NPU INT8)

For LFM2.5-8B-A1B single-token generation:
- 4 active experts per MoE layer
- Per expert: FFN up (3.6ms) + FFN down (3.7ms) = 7.3ms
- Per layer (4 experts): 29.2ms
- 22 MoE layers: **~640ms for FFN compute on NPU**
- Attention projections (5 attention layers): ~50ms total
- **Estimated NPU total: ~690ms per token**

For comparison:
- rock1 A55 CPU full inference: ~890ms per token (all compute on CPU)
- rock0 A76 CPU full inference: ~122ms per token

### Conclusion: NPU Can Accelerate Worker Inference

The NPU at INT8 delivers ~7.8 GFLOPS for matmul, which is significantly faster than
what the A55 CPU achieves even with optimized SIMD. The NPU can compute FFN layers
while the CPU handles attention/routing/sampling, potentially halving per-token latency
on worker nodes.

Key advantage: NPU runs independently from CPU, enabling **CPU+NPU parallel execution**.

## Phase 3: LLM Integration - CONCLUDED (Not Viable)

### NPU Performance Summary
- **FP16 matmul**: ~4.2 GFLOPS (consistent across sizes)
- **INT8 matmul**: ~7.8 GFLOPS (1.9x faster than FP16)
- **NPU spec**: 0.8 TOPS INT8 (theoretical peak, ~10x higher than achieved)

### Per-Token LLM Inference Estimate

| Configuration | FFN Time (22 MoE layers) | vs A55 CPU Full Inference |
|---------------|--------------------------|---------------------------|
| NPU FP16 | 1,270 ms | Slower than CPU (890 ms full) |
| NPU INT8 | ~669 ms | Marginal, doesn't include attention/overhead |
| NPU INT8 FFN only | ~669 ms | Must add CPU attention + model swap overhead |

### Why NPU Doesn't Help for LLMs
1. The 0.8 TOPS spec is INT8 peak - actual matmul throughput is ~8 GFLOPS (100x lower)
2. RKNN model loading/destroy has overhead per inference call
3. Data transfer between CPU memory and NPU adds latency
4. The A55 CPU with llama.cpp's optimized quantized matmul is already efficient
5. NPU doesn't support Gather/Reshape ops - embedding must stay on CPU

### What the NPU IS Good At
- Vision model inference: ResNet50 INT8 at 28.4 inf/sec per node (142 inf/sec cluster)
- Pre-compiled RKNN model execution (no per-call compilation needed)
- INT8 quantized inference for CNN architectures
- Running independently from CPU (could overlap with CPU workloads)

## Phase 4: Distributed NPU Cluster

If Phase 2-3 prove viable:
1. **Deploy RKNN runtime** to all 5 workers via Ansible
2. **Distribute expert weights** across NPU nodes (each NPU handles different experts)
3. **Orchestrate inference** from rock0 with expert-parallel dispatch
4. **Benchmark** full cluster vs rock0 solo

## Ansible Automation Plan

Since Ansible isn't installed yet on the Mac:
1. Install Ansible on Mac: `pip3 install ansible`
2. Create inventory file for rock0-5
3. Playbooks for:
   - `rknpu-runtime.yml`: Install librknnrt + rknnlite on all workers
   - `rknpu-verify.yml`: Verify NPU is accessible and can run inference
   - `rknpu-bench.yml`: Deploy and run NPU benchmarks

## Key Files
- Cluster inventory and LFM2.5 work: `docs/lfm2.5-distributed-inference.md`
- This plan: `docs/rknpu-bringup-plan.md`
- Benchmark script: `benchmark.py`

## Critical Fix: swiotlb Memory Allocation (2026-07-09)

### Problem
NPU initialization was failing with "swiotlb: coherent allocation failed, size=11993088" errors.
The RKNN driver requires ~12MB of contiguous DMA-coherent memory, but the default swiotlb
(Software I/O Translation Lookaside Buffer) was too small.

### Root Cause
The kernel boot parameters didn't include `swiotlb=262144`, so the default swiotlb size
(~64MB total, with limited contiguous allocation size) was insufficient for NPU buffer allocation.

### Solution
Added `extraargs=swiotlb=262144` to `/boot/uEnv.txt` on all worker nodes.

**Steps:**
```bash
# On each worker node (rock1-5)
echo "extraargs=swiotlb=262144" | sudo tee -a /boot/uEnv.txt
sudo reboot
```

**Verification:**
```bash
cat /proc/cmdline  # Should show swiotlb=262144
dmesg | grep -i rknpu  # Should show "assigned reserved memory node rknpu"
systemctl is-active camera-brain-worker  # Should be "active"
```

### Current Status
| Node | swiotlb Applied | NPU Status |
|------|----------------|------------|
| rock1 | Yes (but offline - boot issue) | Unknown |
| rock2 | Yes | Working - detecting objects |
| rock3 | Yes | Working - detecting objects |
| rock4 | Yes | Working - detecting objects |
| rock5 | Yes | Working - detecting objects |

**Note**: rock1 failed to boot after the swiotlb change and may need physical recovery
(power cycle or console access to fix boot configuration).

## Risks (Updated)
- ~~**RKNN SDK version mismatch**: RESOLVED - v2 runtime works with v0.4.2 driver~~
- ~~**swiotlb allocation failure**: RESOLVED - added swiotlb=262144 to kernel cmdline~~
- **Memory constraints**: Workers have 2GB RAM. NPU shares system memory (no dedicated VRAM). Model + NPU working memory must fit.
- **INT8 quantization quality**: NPU is optimized for INT8. F16 compute may be limited/slow.
- **SDK closed-source**: librknnrt.so is a binary blob from Rockchip.
- **Python wrappers broken**: rknnlite (both v1 and v2) has compatibility issues. Must use C API directly.
- **No kernel upgrade needed**: The v0.4.2 driver works fine with v2.3.2 runtime. No need to reinstall OS or upgrade kernel.
- **rock1 boot failure**: One worker node failed to boot after swiotlb change - may need physical recovery.

## Software Stack Summary (Updated)

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| RKNPU kernel driver | Built into 4.19 kernel | NPU hardware interface | Working (v0.4.2) |
| `librknnrt.so` | `/usr/lib/` | Userspace NPU runtime (v2 C API) | Working (v2.3.2) |
| `rknn-toolkit2` | On-device (rock1) | Model conversion (ONNX -> RKNN) | Working |
| `rknn-toolkit-lite2` | On-device | Python inference wrapper | Broken (use C API) |
| NPU userspace blobs | `/usr/lib/` | libOpenVX etc. for v1 runtime | Installed (v1 compat only) |
- **RKNN SDK version mismatch**: The SDK version must match the kernel driver. Kernel is from Nov 2021 (4.19 BSP). Need to find the matching RKNPU2 SDK release.
- **Memory constraints**: Workers have 2GB RAM. NPU shares system memory (no dedicated VRAM). Model + NPU working memory must fit.
- **INT8 quantization quality**: NPU is optimized for INT8. F16 compute may be limited/slow.
- **SDK closed-source**: librknnrt.so is a binary blob from Rockchip.
