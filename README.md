# Camera Brain

Distributed video surveillance system with edge AI inference and natural language querying.

This is a small project I created to make use of local cameras via RTSP stream and  unused hardware I had sitting around. Time invested: a couple hours worth of ideation and implementation.  

It's a proof of concept of what is trivially straightforward, with minimal resources, for a company like Flock: what kind of inferences and relationships can be built using very little resources? Everything this project does could easily be implemented within each camera installation, transporting only the metrics and relevant frame data to fairly minimalist could infrastructure: no massive datacenter required. 

## Example

### Web Chat Interface

Access the web interface at `http://rock0:8080` to chat with your camera observations:

- "Show me all people detected today"
- "How many cars were detected yesterday?"
- "What detections occurred on camera cam4?"
- "Show me high confidence detections above 0.8"

The chat interface uses LFM2.5-1.2B-Instruct to convert natural language to SQL queries.

### CLI Queries with cbrain

```bash
# Natural language queries
cbrain query "Who was at the front door this morning?"
cbrain query "Show me all vehicles in the driveway last week" -o json

# Direct SQL (read-only)
cbrain sql "SELECT count(*) FROM observations"
cbrain sql "SELECT class_name, count(*) FROM observations GROUP BY class_name"

# Pattern analysis
cbrain infer routines      # Detect daily routines
cbrain infer anomalies     # Find unusual activity
cbrain infer vehicles      # Vehicle usage patterns

# Cross-camera correlation
cbrain correlate timeline  # Chronological event timeline
cbrain correlate track person  # Track person movement
```

### Inferences and Observations:

| Category | Example Inference | Data Sources |
|----------|-------------------|--------------|
| Family Routines | "Kids leave for school between 7:45-8:15 AM, return 2:45-4:30 PM" | Timestamped detections at front door + backpack attribute |
| Visitor Patterns | "Mail carrier arrives 11:00 AM-12:00 PM, never weekends" | Daily person detection at mailbox + temporal aggregation |
| Vehicle Usage | "Work truck only used on weekdays, SUV used weekends" | Vehicle classification + license plate recognition + day-of-week |
| Security Alerts | "Unknown vehicle in driveway 2:00-4:00 AM (not family/friends)" | Unfamiliar plate + unusual time + no indoor motion correlation |
| Package Delivery | "FedEx arrives Tue/Thu 1-3 PM, packages left at front door" | Uniform color + package attribute + drop location tracking |
| Pet Monitoring | "Dog escapes yard 3x this week through loose gate" | Animal classification + yard boundary crossing + gate state |
| Service Provider Verification | "Landscaper arrived 8 AM, 4 workers, stayed 3 hours" | Headcount + vehicle count + duration tracking |
| Anomaly Detection | "Motion at back door at 3 AM - raccoon, not intruder" | Size classification + gait analysis + time context |

### Cross-Camera Correlations:

| Camera | Observation |
|--------|-------------|
| Camera 1 (Front Door) | Person detected 7:52 AM, red jacket, heading east |
| Camera 2 (Driveway) | Vehicle departure 7:53 AM, black SUV |
| Camera 3 (Back Yard) | No activity |

**Inference:** Family member left for work via driveway (normal pattern)


## Implementation 

This was developed on and for a rPi5 w/ 8GB of memory as the primary brain and 5 rock3a boards with 2GB for NPU frame processing. With the NPUs it can process about 130 frames/second.

**Model architecture:**
- **LFM2.5-VL-1.6B**: Vision-language model for image crop analysis
- **LFM2.5-1.2B-Instruct**: Text-only LLM for natural language query interpretation and answer generation
- **YOLOv5s**: Detection model (NPU-accelerated)

## Quick Start

### Option 1: Docker (Recommended for Testing)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/camera-brain.git
cd camera-brain/docker

# 2. Configure
cp .env.example .env
# Edit .env: set MODEL_DIR to your models directory

# 3. Download models (manual, large files)
# - LFM2.5-VL-1.6B.Q8_0.gguf (~1.2GB)
# - LFM2.5-VL-1.6B.mmproj-f16.gguf (~850MB)
# - yolov5s_int8.rknn (~8MB, for NPU detection)

# 4. Start
./start.sh

# 5. Access services
# - Gateway: http://localhost:8080
# - VLM API: http://localhost:8081
# - Query API: http://localhost:8082
# - Grafana: http://localhost:3000 (admin/admin)
```

### Option 2: Native Install (ARM64 Linux)

```bash
# 1. Clone and run installer
git clone https://github.com/your-org/camera-brain.git
cd camera-brain
sudo ./deploy/install.sh

# 2. Configure
sudo nano /etc/camera-brain/camera-brain.env

# 3. Download models to /var/lib/camera-brain/models/

# 4. Services auto-start
systemctl status camera-brain-*
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Camera Brain Cluster                     │
├─────────────────────────────────────────────────────────────┤
│                                              ┌──────────────┐│
│  ┌─────────────┐    ┌─────────────┐         │  PostgreSQL  ││
│  │   Workers   │───▶│   Gateway   │◀────────│  TimescaleDB ││
│  │  (rock1-5)  │    │  (rock0)    │         │    + NATS    ││
│  │  NPU YOLOv5 │    │  HTTP + NATS│         └──────────────┘│
│  └─────────────┘    └──────┬──────┘              │
│                            │                      ▼
│                     ┌──────┴──────┐       ┌──────────────┐
│                     │   VLM       │       │   Query      │
│                     │  Processor  │       │   Engine     │
│                     │  (8081)     │       │   (8082)     │
│                     └──────┬──────┘       └──────┬───────┘
│                            │                      │
│                            ▼                      ▼
│                     ┌─────────────────────────────────┐
│                     │      llama-server (8888)        │
│                     │   LFM2.5-VL-1.6B (Vision LM)    │
│                     ├─────────────────────────────────┤
│                     │      llama-server (8889)        │
│                     │   LFM2.5-1.2B-Instruct (Text)   │
│                     └─────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

## Components

| Service | Port | Purpose |
|---------|------|---------|
| Gateway | 8080 | Worker coordination, HTTP API |
| Chat Web UI | 8080 | Natural language chat interface (Flask) |
| VLM Processor | 8081 | Image analysis via VLM |
| Query Engine | 8082 | Natural language queries |
| llama-server | 8888 | VLM inference (LFM2.5-VL-1.6B) |
| llama-server | 8889 | Text LLM inference (LFM2.5-1.2B-Instruct) |
| PostgreSQL | 5432 | Time-series storage |
| NATS | 4222 | Message bus |
| Grafana | 3000 | Dashboards |

## Configuration

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=camera_brain
DB_USER=camera_brain
DB_PASSWORD=generate_random_password

# Models
MODEL_DIR=/path/to/models
LLAMA_MODEL=LFM2.5-VL-1.6B.Q8_0.gguf
LLAMA_MMPROJ=LFM2.5-VL-1.6B.mmproj-f16.gguf

# Service Ports
VLM_PORT=8081
QUERY_PORT=8082
GATEWAY_PORT=8080

# Text LLM (optional, for faster query responses)
TEXT_MODEL_PATH=/var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf
LLAMA_TEXT_SERVER_URL=http://localhost:8889
```

### Plugin Configuration

Camera Brain supports pluggable backends:

```yaml
# config.yaml
detection:
  plugin: rknn        # or "onnx", "api"
  config:
    model_path: /models/yolov5s_int8.rknn

analysis:
  plugin: llamacpp    # or "anthropic", "openai"
  config:
    endpoint: http://localhost:8888
```

See [docs/PLUGIN-GUIDE.md](docs/PLUGIN-GUIDE.md) for creating custom plugins.

## Chat Interface

The web-based chat interface allows natural language querying of camera observations:

```
┌─────────────────────────────────────────────────────────┐
│                    Browser                               │
│              http://rock0:8080                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │  "Show me all cars detected today"                  │ │
│  │                                                     │ │
│  │  Found 23 car detections today.                     │ │
│  │  SELECT * FROM observations WHERE class_name=...    │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   Flask App (8080)      │
              │   - Natural language    │
              │     → SQL (LFM 1.2B)    │
              │   - SQL validation      │
              │   - Query execution     │
              └───────────┬─────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  llama-server│  │   SQLite     │  │   Results    │
│  (8889)      │  │   Database   │  │   Display    │
│  LFM2.5-1.2B │  │   camera-    │  │              │
│              │  │   brain.db   │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Deployment

```bash
# Deploy to rock0
cd /Users/caimlas/git/rock-cluster
./deploy/chat-service.sh
```

The deployment script:
1. Creates Python virtual environment
2. Installs Flask and dependencies
3. Copies application to rock0
4. Installs systemd service
5. Starts the chat service

**Access:** `http://rock0:8080`

**Logs:** `ssh rock0 'sudo journalctl -u camera-brain-chat -f'`

### Configuration

Environment variables (set in systemd service):

```bash
DB_PATH=/home/camera-brain/camera-brain.db
LLAMA_SERVER_URL=http://localhost:8889
MODEL_PATH=/home/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf
```

### Example Queries

```
- "Show me all people detected today"
- "How many trucks were detected this week?"
- "What detections occurred on camera cam4?"
- "Show high confidence detections above 0.8"
- "Find all bicycles detected between 6am and 8am"
```

## Model Downloads

| Model | Size | Source |
|-------|------|--------|
| LFM2.5-VL-1.6B.Q8_0.gguf | ~1.2GB | Hugging Face |
| LFM2.5-VL-1.6B.mmproj-f16.gguf | ~850MB | Hugging Face |
| LFM2.5-1.2B-Instruct.Q4_K_M.gguf | ~700MB | Hugging Face |
| yolov5s_int8.rknn | ~8MB | Export from ONNX |

## Hardware Requirements

### Minimum (Single Node)
- ARM64 or x86_64 CPU
- 8GB RAM
- 50GB storage
- Gigabit ethernet

### Recommended (Cluster)
- **Head node (rock0)**: RPi5 8GB or similar
- **Worker nodes (rock1-5)**: RK3568 with 2GB+ RAM
- NPU support for workers (optional but recommended)

## Development

```bash
# Build all services
make build

# Run tests
make test

# Docker workflow
make build-docker
make run-docker
make stop-docker

# Code quality
make lint
```

## Documentation

- [Plugin Guide](docs/PLUGIN-GUIDE.md) - Create custom detection/analysis plugins
- [Distribution Options](docs/distribution/) - Installation and deployment
- [API Reference](docs/api.md) - HTTP API documentation

## License

MIT License - see LICENSE for details.
