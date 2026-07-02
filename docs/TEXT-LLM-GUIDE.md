# Text LLM Guide - LFM2.5-1.2B-Instruct

Camera Brain uses a text-only LLM for natural language query interpretation and answer generation. This guide covers setup, configuration, and troubleshooting.

## Why Text-Only?

Camera Brain previously used the vision-language model (LFM2.5-VL-1.6B) for all text tasks -- including simple query interpretation and answer generation. That meant every text query carried the full cost of image encoding and multi-modal processing. The text-only LLM eliminates that overhead:

| Factor | VLM (fallback) | Text-only LLM |
|--------|---------------|---------------|
| Model size | ~1.2GB | ~700MB |
| Image preprocessing | Required (224x224 RGB + mmproj) | Not needed |
| Prompt structure | Image + text tokens | Text only |
| RAM pressure | Higher (model + image buffers + mmproj ~850MB) | Lower (model only ~700MB) |
| Throughput on RPi5 | 10-20 tokens/sec | 40-60 tokens/sec |
| Latency per query | 2-5 seconds | <1 second |

The text-only path gives **2-3x faster inference** and **lower memory usage**, making it the default choice whenever the system needs to generate answers, parse queries, or write descriptions -- tasks that have nothing to do with vision.

## Model Download

### Where to Get It

The model is available on Hugging Face. Search for "LFM2.5-1.2B-Instruct" in the GGUF format (Q4_K_M quantization recommended).

**Recommended file:** `LFM2.5-1.2B-Instruct.Q4_K_M.gguf` (~700MB)

### Storage Location

Place the model on the head node (rock0 / rPi5):

```
/var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf
```

Create the directory if it does not exist:

```bash
sudo mkdir -p /var/lib/camera-brain/models
sudo cp LFM2.5-1.2B-Instruct.Q4_K_M.gguf /var/lib/camera-brain/models/
sudo chown root:root /var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf
```

### Architecture Note

The model uses `qwen2` architecture in llama.cpp. Verify compatibility:

```bash
llama-run --verbose -m /var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf "hello"
```

If it prints a coherent response, the model file is valid.

## Configuration

### Method 1: Environment Variables

Set these environment variables before starting the query engine:

```bash
# Path to the GGUF model file
export TEXT_MODEL_PATH=/var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf

# URL of llama-server running the text model
export LLAMA_TEXT_SERVER_URL=http://localhost:8889
```

### Method 2: YAML Configuration

In your `camera-brain.env` or YAML config file, add a `text_analysis` section:

```yaml
text_analysis:
  endpoint: http://localhost:8889
  model_path: /var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf
  max_tokens: 512
  temperature: 0.3
  timeout_sec: 60
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TEXT_MODEL_PATH` | `/var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf` | Path to the GGUF model |
| `LLAMA_TEXT_SERVER_URL` | `http://localhost:8889` | llama-server HTTP endpoint |
| `LLAMA_TEXT_SERVER_PORT` | (not used directly) | llama-server listens on `--port` flag |

## Starting llama-server

Launch llama-server with the text model on port 8889:

```bash
llama-server \
  -m /var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf \
  --port 8889 \
  --host 0.0.0.0 \
  --ctx-size 4096 \
  -ngl 33
```

### Recommended Flags

| Flag | Purpose | Recommended Value |
|------|---------|-------------------|
| `-m` | Model path | Path to Q4_K_M GGUF file |
| `--port` | HTTP listen port | `8889` |
| `--host` | Bind address | `0.0.0.0` (accessible from cbrain CLI and other services) |
| `--ctx-size` | Context window | `4096` (sufficient for query context) |
| `-ngl` | GPU layers offloaded | `33` (all layers on rPi5 Metal/NPU if available; use `0` for CPU-only) |

### Running as a Background Service

Use `systemd` for persistent operation:

```bash
sudo tee /etc/systemd/system/llama-text.service > /dev/null <<EOF
[Unit]
Description=Camera Brain Text LLM Server
After=network-online.target

[Service]
Type=simple
User=camera-brain
ExecStart=/usr/local/bin/llama-server -m /var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf --port 8889 --host 0.0.0.0 --ctx-size 4096
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now llama-text.service
```

### Port Separation

The text LLM uses port **8889** to avoid collision with the VLM llama-server which typically runs on port **8888**.

## Performance Expectations

### On RPi5 (Cortex-A76, 8GB RAM)

| Metric | Text LLM (LFM2.5-1.2B Q4) | VLM (LFM2.5-VL-1.6B Q8) |
|--------|---------------------------|-------------------------|
| Prompt processing (512 tokens) | ~40-60 tok/s | ~8-12 tok/s |
| Token generation (128 tokens) | ~35-50 tok/s | ~10-20 tok/s |
| Average query latency | <1 second | 2-5 seconds |
| Memory footprint | ~1.2GB (model + server) | ~3.5GB (model + image buffers + mmproj) |
| Concurrent users (tok/s each) | ~30+ @ concurrency 2 | ~5 @ concurrency 1 |

### On Workers (RK3568, Cortex-A55, 2GB RAM)

Text-only inference is not recommended on workers due to limited RAM and slower CPU. The RPi5 head node should handle all text LLM inference.

### Fallback Behavior

If the text LLM endpoint is unreachable or returns an error, the query engine gracefully falls back to the VLM analyzer. This means:

- The system still works correctly even without the text LLM
- Query latency increases but answers are still generated
- Logs will show `Text LLM initialization failed, falling back to VLM`

## When to Skip Text LLM

Skip the text LLM in these scenarios:

1. **Single-node setups with <8GB RAM** -- The VLM alone uses ~3.5GB. Adding the text LLM (~1.2GB) pushes total memory above 8GB, risking OOM kills.

2. **If sub-second query latency is not required** -- The VLM produces correct answers, just slower. For dashboards or manual use, 2-5 seconds is acceptable.

3. **If the VLM is already the bottleneck** -- If frame analysis (VLM) dominates the per-frame pipeline, the text LLM adds negligible additional load. The text LLM is only invoked during query responses, not per-frame.

Even when skipped, the system fully degrades: the `llamacpp-text` plugin is optional, and `generateAnswer` in the query engine will use the VLM analyzer as a drop-in replacement.

## Testing

### Step 1: Verify llama-server is Running

```bash
curl -s http://localhost:8889/health | python3 -m json.tool
```

Expected response: `{"status":"ok"}`

### Step 2: Quick Text Test

```bash
curl -s http://localhost:8889/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me about the weather today."}],
    "max_tokens": 64,
    "temperature": 0.3
  }' | python3 -m json.tool
```

The model should return a coherent English response.

### Step 3: Start the Query Engine

```bash
export LLAMA_TEXT_SERVER_URL=http://localhost:8889
export TEXT_MODEL_PATH=/var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf

cd cmd/queryengine
go run .
```

Check the startup logs for:

```
INFO Text LLM initialized endpoint=http://localhost:8889 model=/var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf
```

If you see `WARN Text LLM initialization failed, falling back to VLM`, check:
- llama-server is running and reachable at the configured endpoint
- The model path exists and is a valid GGUF file
- Enough free memory (>1.5GB)

### Step 4: Test a Query

```bash
curl -s http://localhost:8082/query \
  -H "Content-Type: application/json" \
  -d '{"query": "When did the mail carrier arrive today?"}' \
  | python3 -m json.tool
```

Expected response:

```json
{
  "success": true,
  "answer": "The mail carrier arrived at 9:14 AM today. I saw them carrying a padded envelope.",
  "parsed_query": {
    "sql": "SELECT * FROM observations WHERE entity_type='person' AND time >= '2026-07-02 00:00:00'",
    ...
  },
  "result_count": 3,
  "processing_ms": 450
}
```

### Step 5: Verify Text LLM Was Used

Check the query engine logs. A text-only path will show:

```
INFO Text LLM generated answer duration=380ms tokens_generated=47
```

Without text LLM (VLM fallback), expect:

```
INFO VLM generated answer duration=2.1s tokens_generated=52 image_processed=true
```

The difference in `duration` shows the performance advantage.

## Integration with cbrain CLI

The cbrain CLI tool sends queries to the query engine at `http://localhost:8082`. The text LLM integration is transparent to the CLI -- the query engine handles routing to the text or VLM analyzer internally.

### Query Command

```bash
cbrain query "Show me all visitors from yesterday"
cbrain query "Who was at the front door this morning?"
cbrain query "How many delivery trucks visited this week?"
```

### Infer Command

The `infer` command uses the query engine's backend to analyze new frames. When the text LLM is available, inferred descriptions are generated faster and without image processing delay.

```bash
cbrain infer --latest
```

### Correlate Command

The `correlate` command cross-references observations over time windows. The text LLM generates correlation summaries more quickly when the text-only path is used.

```bash
cbrain correlate --entity=pers --window=24h
```

### Output Formats

All cbrain commands support multiple output formats:

```bash
cbrain query "What happened this afternoon?" -o json    # JSON response
cbrain query "What happened this afternoon?" -o plain   # Answer only
cbrain query "What happened this afternoon?" -o table   # Answer + metadata (default)
```

### Configuration for cbrain

cbrain reads its config from `/etc/camera-brain/camera-brain.env` or a custom path:

```bash
cbrain query "Test query" --config /path/to/config.env
```

The relevant config key for the query engine URL is `CB_QUERY_URL`:

```
CB_QUERY_URL=http://localhost:8082
```

This must point to the query engine, which in turn connects to the text LLM via the `LLAMA_TEXT_SERVER_URL` environment variable (set on the query engine side, not the CLI side).

## Troubleshooting

### "Text LLM initialization failed"

1. Check llama-server is running: `curl http://localhost:8889/health`
2. Verify model path: `ls -la $TEXT_MODEL_PATH`
3. Check memory: `free -h` (need >1.5GB free)
4. Verify GGUF validity: `llama-run --verbose -m $TEXT_MODEL_PATH "test"`

### Slow query responses (>3s)

1. Check if text LLM is actually being used (look for "Text LLM initialized" in logs)
2. If falling back to VLM, verify `LLAMA_TEXT_SERVER_URL` is set
3. Check disk I/O -- mmap loading the GGUF from a slow disk adds latency

### Out of Memory

If the system swaps or kills processes:
1. Stop the VLM llama-server if not needed (uses ~3.5GB)
2. Reduce context size (`--ctx-size 2048`)
3. Consider using Q4_K_M quantization instead of higher-bit variants

### Port Conflicts

Text LLM uses port 8889. If already in use:

```bash
lsof -i :8889
# Change the --port flag and update LLAMA_TEXT_SERVER_URL accordingly
```
