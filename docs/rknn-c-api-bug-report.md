# RKNN C API Bug Report - rknn_inputs_set() Returns -5 Incorrectly

## Issue Summary

The RKNN C API in `librknnrt.so` v2.3.2 fails during inference with error code -5 from `rknn_inputs_set()`, even when:
- The input buffer size exactly matches what `RKNN_QUERY_INPUT_ATTR` reports
- The Python API (`rknn.api.RKNN`) works perfectly with the same model and input
- The error message reports contradictory size requirements

**Affects**: RKNN toolkit v2.x, RK3568/RK3588 platforms

## Error Message

```
E RKNN: [timestamp] rknn_inputs_set, param input size(1228800) < model input size(4915200)
rknn_inputs_set returned: -5
```

Note: The reported "model input size" (4,915,200) is exactly 4x the query result (1,228,800), suggesting an int32/float32 interpretation bug in the internal validation.

## Environment

| Component | Value |
|-----------|-------|
| RKNN Runtime | librknnrt.so v2.3.2 (429f97ae6b@2025-04-09) |
| Platform | RK3568 (Ubuntu 20.04, aarch64) |
| Model Format | YOLOv5s INT8 (exported via rknn-toolkit2 v2.3.2) |
| Model Size | 10,884,764 bytes |
| Expected Input | uint8 NHWC, 640x640x3 (1,228,800 bytes) |

## Reproduction

### C API (Fails)

```c
#include <rknn_api.h>

rknn_context ctx;
// Load model...
rknn_init(&ctx, model_buf, model_size, 0, NULL);

// Query input size
rknn_tensor_attr attr;
attr.index = 0;
rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
// Returns: size=1228800, dims=[1,640,640,3], type=2 (INT8), fmt=1

// Create input buffer matching query
uint8_t* input = calloc(1, attr.size);  // 1,228,800 bytes

// Set input - FAILS
rknn_input rknn_in = {
    .index = 0,
    .buf = input,
    .size = attr.size,
    .pass_through = 0,
    .type = RKNN_TENSOR_UINT8,
    .fmt = RKNN_TENSOR_NHWC
};
int ret = rknn_inputs_set(ctx, 1, &rknn_in, NULL);
// Returns: -5
// Error: "param input size(1228800) < model input size(4915200)"
```

### Python API (Works)

```python
from rknn.api import RKNN
import numpy as np

rknn = RKNN()
rknn.load_rknn('model.rknn')
rknn.init_runtime(target='rk3568')

# Same input size, format
test_input = np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)
outputs = rknn.inference(inputs=[test_input])
# Works perfectly - returns valid detections
```

## Independent Verification

Compiled C test program shows same failure:

```bash
$ gcc -o test_inference test_inference.c -L/usr/lib -lrknnrt
$ ./test_inference
rknn_init: 0
Input: dims=[1,640,640,3] type=2 fmt=1 size=1228800
Calling rknn_inputs_set with size=1228800...
rknn_inputs_set: -5
E RKNN: rknn_inputs_set, param input size(1228800) < model input size(4915200)
```

## Root Cause Analysis

The bug appears to be in the internal validation logic of `rknn_inputs_set()`:

1. `RKNN_QUERY_INPUT_ATTR` correctly returns `size=1228800` (uint8 640×640×3)
2. Internal validation incorrectly expects 4,915,200 bytes (4× larger, suggesting float32 expectation)
3. Python API bypasses this broken validation path
4. Model was converted with uint8 input (no mean_values/std_values normalization)

## Workaround

Use Python API via subprocess until C API is fixed:

```python
# Python subprocess wrapper
from rknn.api import RKNN
import numpy as np

def run_inference(image_bytes):
    rknn = RKNN()
    rknn.load_rknn('model.rknn')
    rknn.init_runtime(target='rk3568')
    img = preprocess_to_uint8_nhwc(image_bytes)
    return rknn.inference(inputs=[img])
```

Call from C/Go/C++ via subprocess with minimal overhead (~10-50ms).

## Impact

- **Blocks all C/C++/Go inference** on RK3568/RK3588 with v2.x models
- Forces developers to use Python subprocess workaround
- Affects embedded/production deployments where Python is undesirable

## Requested Fix

1. Fix internal validation in `rknn_inputs_set()` to use correct expected size
2. Ensure consistency between `RKNN_QUERY_INPUT_ATTR` and validation logic
3. Release patched librknnrt.so

## Attachments Available

- Minimal C reproduction test case
- Model file for testing
- Python verification script
- Full error logs

---

**Reported**: 2026-07-07
**Status**: Pending Rockchip response
**Workaround**: Python subprocess (functional)
