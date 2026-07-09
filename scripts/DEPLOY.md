# YOLOv5s RKNN Model Deployment Guide

## Summary

The YOLOv5s ONNX model has been exported with **verified detection heads**:
- Output shape: `[1, 84, 8400]` = 4 box + 1 confidence + 80 classes per anchor
- File: `models/yolov5s.onnx` (verified working)
- Next step: Convert to RKNN INT8 format on rock1 and deploy to workers

## Quick Deploy (If SSH Access Works)

```bash
cd /Users/caimlas/git/rock-cluster
./scripts/deploy-yolov5-rknn.sh
```

This script will:
1. Transfer ONNX model to rock1
2. Run RKNN conversion with INT8 quantization
3. Deploy to all workers (rock1-4)
4. Restart workers and verify detections

## Manual Deploy (If SSH Key Required)

### Step 1: Set Up SSH Access

Option A: Copy SSH key to workers (if you know the password)
```bash
ssh-copy-id caimlas@rock1.local
# Repeat for rock2, rock3, rock4
```

Option B: Use password auth
```bash
# Edit ~/.ssh/config and add:
Host rock1 rock1.local
    HostName 10.9.8.171
    User caimlas
    PasswordAuthentication yes
```

### Step 2: Transfer ONNX Model to rock1

```bash
# From your Mac
scp models/yolov5s.onnx models/yolov5s.onnx.data caimlas@rock1.local:/tmp/
scp scripts/convert-yolov5-rknn.py caimlas@rock1.local:/tmp/
```

### Step 3: Run Conversion on rock1

```bash
ssh caimlas@rock1.local
cd /tmp

# Run conversion (creates INT8 quantized model)
python3 convert-yolov5-rknn.py \
    --model yolov5s.onnx \
    --output yolov5s_int8.rknn \
    --calib-size 20 \
    --verify
```

Expected output:
```
=== Verifying RKNN output structure ===
Output shape: (1, 84, 8400)
Confidence non-zero: 850/8400 (10.1%)
Class scores non-zero: 850/8400 (10.1%)
OK: Detection heads appear intact!
```

### Step 4: Deploy RKNN Model to Workers

```bash
# From rock1, copy to all workers
scp /tmp/yolov5s_int8.rknn caimlas@rock1.local:/home/caimlas/models/
scp /tmp/yolov5s_int8.rknn caimlas@rock2.local:/home/caimlas/models/
scp /tmp/yolov5s_int8.rknn caimlas@rock3.local:/home/caimlas/models/
scp /tmp/yolov5s_int8.rknn caimlas@rock4.local:/home/caimlas/models/

# Or from Mac, copy to all
scp models/yolov5s_int8.rknn caimlas@rock{1,2,3,4}.local:/home/caimlas/models/
```

### Step 5: Restart Workers

```bash
# On each worker (rock1-4)
ssh caimlas@rock1.local
pkill -f camera-worker
cd /home/caimlas/bin
./camera-worker > /tmp/worker.log 2>&1 &
```

### Step 6: Verify Detections

```bash
# Check worker logs on any worker
ssh caimlas@rock1.local 'tail -30 /tmp/worker.log | grep -E "detected|NPU"'
```

Expected output:
```
[17:23:45.123] [NPU] stride 8: maxConf=0.87 dets=12
[17:23:45.145] [NPU] stride 16: maxConf=0.72 dets=8
[17:23:45.167] [NPU] stride 32: maxConf=0.65 dets=3
[17:23:45.201] [NPU] final detections: 15
```

## Troubleshooting

### SSH Connection Refused
```
ssh: connect to host rock1.local port 22: Connection refused
```
- Check if rock1 is powered on and connected to network
- Try IP address: `ping 10.9.8.171`
- SSH may not be installed: `sudo apt install openssh-server`

### Authentication Failed
```
Permission denied (publickey,password)
```
- SSH key not authorized: `ssh-copy-id caimlas@rock1.local`
- Wrong password: ask user for actual credentials
- Check `/etc/ssh/sshd_config` allows password auth

### RKNN Conversion Fails
```
ERROR: rknn.load_onnx failed
```
- ONNX model corrupted: re-run export script
- rknn-toolkit2 not installed: `pip3 install rknn_toolkit2`

### Zero Detections After Deploy
1. Check worker logs: `ssh rock1 'tail /tmp/worker.log'`
2. Verify model path: `ls -la /home/caimlas/models/yolov5s_int8.rknn`
3. Check confidence threshold in worker config (should be 0.25)
4. Re-run RKNN verification to confirm model is valid

## Files Reference

| File | Purpose | Location |
|------|---------|----------|
| `yolov5s.onnx` | Exported ONNX model | `models/` |
| `yolov5s.onnx.data` | Model weights (37MB) | `models/` |
| `export-yolov5.py` | ONNX export script | `scripts/` |
| `convert-yolov5-rknn.py` | RKNN conversion | `scripts/` |
| `deploy-yolov5-rknn.sh` | Deployment automation | `scripts/` |
| `yolov5s_int8.rknn` | INT8 quantized model | Deploy to workers |

## Verification Checklist

- [ ] ONNX model exported successfully
- [ ] ONNX verification shows non-zero confidence values
- [ ] RKNN conversion completes without errors
- [ ] RKNN verification shows confidence > 0 for some anchors
- [ ] RKNN model deployed to all workers
- [ ] Workers restarted successfully
- [ ] Worker logs show detections (5-15 objects per frame)
- [ ] Crop images being created

## Success Criteria

**Before fix:**
```
[NPU] final detections: 0
```

**After fix:**
```
[NPU] final detections: 12
```
