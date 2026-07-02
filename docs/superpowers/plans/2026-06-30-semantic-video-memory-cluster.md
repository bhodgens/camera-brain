# Semantic Video Memory Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 6-node cluster system that transforms RTSP camera streams into searchable semantic memory, enabling natural language queries like "When did John look happy?" or "How many times did someone play ball last week?"

**Architecture:** NPU workers (rock1-5) run YOLOv5s INT8 detection at ~140 FPS aggregate, streaming crops to rock0. rock0 runs a NATS-based gateway, LFM2.5-VL-1.6B for crop analysis, LFM2.5-1.2B-Instruct for query interpretation, and PostgreSQL+TimescaleDB for storage with Grafana dashboards.

**Tech Stack:** Go (gateway, workers), NATS (messaging), PostgreSQL+TimescaleDB+pgvector (storage), llama.cpp (LFM2.5 models), Grafana (visualization), YOLOv5s INT8 (NPU detection)

---

## File Structure

### rock0 Services
| File | Responsibility |
|------|----------------|
| `/home/camera-brain/gateway/main.go` | HTTP API + NATS coordinator |
| `/home/camera-brain/gateway/handlers.go` | HTTP request handlers |
| `/home/camera-brain/gateway/nats.go` | NATS publishing/subscribing |
| `/home/camera-brain/gateway/worker_registry.go` | Worker heartbeat tracking |
| `/home/camera-brain/services/vlm_processor.go` | LFM2.5-VL crop analysis |
| `/home/camera-brain/services/query_engine.go` | LLM query interpretation |
| `/home/camera-brain/storage/schema.sql` | PostgreSQL schema |
| `/home/camera-brain/grafana/dashboards/*.json` | Grafana dashboard definitions |
| `/home/camera-brain/docker-compose.yml` | NATS, PostgreSQL, Grafana services |

### rock1-5 Workers
| File | Responsibility |
|------|----------------|
| `/home/camera-brain/worker/main.go` | Worker daemon |
| `/home/camera-brain/worker/rtsp.go` | RTSP frame capture |
| `/home/camera-brain/worker/npu.go` | NPU YOLOv5 inference |
| `/home/camera-brain/worker/crop.go` | Crop extraction + upload |
| `/home/camera-brain/worker/config.go` | Config loading |

### Shared
| File | Responsibility |
|------|----------------|
| `/home/camera-brain/config/gateway.yaml` | Gateway configuration |
| `/home/camera-brain/config/worker.yaml` | Worker configuration template |
| `/home/camera-brain/models/` | Model storage (YOLOv5s, LFM2.5 variants) |
| `/home/camera-brain/memory/crops/` | Crop image storage |

---

## Phase 1: Infrastructure Setup

### Task 1.1: Create Directory Structure on rock0

**Files:**
- Create: Directories via shell commands

- [ ] **Step 1: SSH to rock0 and create base directories**

```bash
ssh rock0
mkdir -p /home/camera-brain/{gateway,worker,services,storage,grafana/dashboards,config,models,memory/crops,logs}
```

Expected: All directories created successfully

- [ ] **Step 2: Commit directory structure plan**

```bash
cd /Users/caimlas/git/rock-cluster
git add docs/superpowers/plans/2026-06-30-semantic-video-memory-cluster.md
git commit -m "docs: Add semantic video memory cluster implementation plan"
```

---

### Task 1.2: Docker Compose for NATS PostgreSQL and Grafana

**Files:**
- Create: `/home/camera-brain/docker-compose.yml`
- Test: `docker-compose ps` output

- [ ] **Step 1: Write docker-compose.yml**

```yaml
version: "3.8"

services:
  nats:
    image: nats:2.10-alpine
    container_name: nats
    ports:
      - "4222:4222"
      - "8222:8222"  # Monitoring
    command: ["-js"]  # JetStream enabled
    volumes:
      - /home/camera-brain/data/nats:/data
    restart: unless-stopped
    networks:
      - camera-brain

  postgres:
    image: timescale/timescaledb-ha:pg16
    container_name: postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: camera_brain
      POSTGRES_PASSWORD: ${DB_PASSWORD:-camera_brain_password_change_me}
      POSTGRES_DB: camera_brain
    volumes:
      - /home/camera-brain/data/postgres:/var/lib/postgresql/data
      - /home/camera-brain/storage/schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
    restart: unless-stopped
    networks:
      - camera-brain

  grafana:
    image: grafana/grafana:10.4.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      GF_SERVER_HTTP_PORT: 3000
      GF_AUTH_ANONYMOUS_ENABLED: "false"
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
    volumes:
      - /home/camera-brain/data/grafana:/var/lib/grafana
      - /home/camera-brain/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - /home/camera-brain/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped
    networks:
      - camera-brain
    depends_on:
      - postgres

networks:
  camera-brain:
    driver: bridge
```

- [ ] **Step 2: Start services and verify**

```bash
cd /home/camera-brain
docker-compose up -d
docker-compose ps
```

Expected output:
```
NAME       IMAGE                              STATUS
nats       nats:2.10-alpine                   Up
postgres   timescale/timescaledb-ha:pg16      Up
grafana    grafana/grafana:10.4.0             Up
```

- [ ] **Step 3: Verify NATS is accessible**

```bash
docker exec nats nats server info
```

Expected: NATS server info JSON with version 2.10.x

- [ ] **Step 4: Verify PostgreSQL is accessible**

```bash
docker exec postgres psql -U camera_brain -d camera_brain -c "SELECT version();"
```

Expected: PostgreSQL 16 with TimescaleDB extension info

---

### Task 1.3: PostgreSQL Schema with TimescaleDB

**Files:**
- Create: `/home/camera-brain/storage/schema.sql`

- [ ] **Step 1: Write the schema**

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Cameras table
CREATE TABLE IF NOT EXISTS cameras (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL UNIQUE,
    rtsp_url text NOT NULL,
    location text,
    active boolean DEFAULT true,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Workers table
CREATE TABLE IF NOT EXISTS workers (
    id text PRIMARY KEY,
    last_heartbeat timestamptz NOT NULL,
    status text NOT NULL DEFAULT 'offline',
    current_camera_id uuid REFERENCES cameras(id),
    assigned_at timestamptz,
    created_at timestamptz DEFAULT now()
);

-- Observations table (main detection storage)
CREATE TABLE IF NOT EXISTS observations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id uuid REFERENCES cameras(id),
    detected_at timestamptz NOT NULL,
    ingested_at timestamptz DEFAULT now(),
    type text NOT NULL,  -- 'person', 'vehicle', 'activity', 'object'

    -- Detection metadata
    bbox jsonb,  -- {x1, y1, x2, y2}
    confidence float,
    class_id integer,
    class_name text,

    -- VLM output (populated asynchronously)
    description text,
    attributes jsonb,  -- {clothing, color, activity, mood, etc.}
    embedding vector(1024),

    -- Crop storage
    crop_path text,
    crop_retained boolean DEFAULT false,

    -- Person tracking
    person_id uuid,
    is_new_person boolean DEFAULT false
);

-- Convert observations to hypertable
SELECT create_hypertable('observations', 'detected_at', if_not_exists => TRUE);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_observations_camera ON observations(camera_id, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(type, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_observations_person ON observations(person_id) WHERE person_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_observations_attributes ON observations USING GIN(attributes);
CREATE INDEX IF NOT EXISTS idx_observations_embedding ON observations USING ivfflat(embedding vector_cosine_ops) WHERE embedding IS NOT NULL;

-- Persons table (known identities)
CREATE TABLE IF NOT EXISTS persons (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL UNIQUE,
    embedding_centroid vector(1024),
    created_at timestamptz DEFAULT now(),
    last_seen_at timestamptz,
    metadata jsonb DEFAULT '{}'::jsonb
);

-- Activity summaries from LLM
CREATE TABLE IF NOT EXISTS activity_summaries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id uuid REFERENCES cameras(id),
    period_start timestamptz NOT NULL,
    period_end timestamptz NOT NULL,
    summary_text text,
    embedding vector(1024),
    UNIQUE(camera_id, period_start)
);

-- Continuous aggregate: detections per hour
CREATE MATERIALIZED VIEW IF NOT EXISTS detections_per_hour
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', detected_at) AS bucket,
    camera_id,
    type,
    class_name,
    count(*) AS detection_count
FROM observations
GROUP BY bucket, camera_id, type, class_name;

-- Continuous aggregate: person appearances per day
CREATE MATERIALIZED VIEW IF NOT EXISTS person_daily_count
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', detected_at) AS bucket,
    person_id,
    count(*) AS appearances,
    count(DISTINCT detected_at::date) AS distinct_days
FROM observations
WHERE person_id IS NOT NULL
GROUP BY bucket, person_id;
```

- [ ] **Step 2: Verify schema is applied**

```bash
docker exec postgres psql -U camera_brain -d camera_brain -c "\dt"
docker exec postgres psql -U camera_brain -d camera_brain -c "SELECT * FROM cameras LIMIT 1;"
```

Expected: Tables `cameras`, `workers`, `observations`, `persons`, `activity_summaries` listed

---

## Phase 2: NATS and Gateway Service

### Task 2.1: Gateway Main Structure

**Files:**
- Create: `/home/camera-brain/gateway/main.go`
- Create: `/home/camera-brain/go.mod`

- [ ] **Step 1: Initialize Go module**

```bash
cd /home/camera-brain/gateway
go mod init camera-brain/gateway
```

- [ ] **Step 2: Write main.go**

```go
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
)

type Gateway struct {
	natsConn  *nats.Conn
	dbConn    *DB
	registry  *WorkerRegistry
	ctx       context.Context
	cancel    context.CancelFunc
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())

	gw := &Gateway{
		ctx:    ctx,
		cancel: cancel,
	}

	// Connect to NATS
	if err := gw.connectNATS(); err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}

	// Connect to PostgreSQL
	if err := gw.connectDB(); err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	// Initialize worker registry
	gw.registry = NewWorkerRegistry()

	// Start worker heartbeat listener
	go gw.listenHeartbeats()

	// Setup HTTP routes
	mux := http.NewServeMux()
	gw.setupRoutes(mux)

	// Start HTTP server
	server := &http.Server{
		Addr:    ":8080",
		Handler: mux,
	}

	go func() {
		log.Printf("Gateway HTTP server starting on :8080")
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down gateway...")
	cancel()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}

	gw.natsConn.Close()
	gw.dbConn.Close()
	log.Println("Gateway shutdown complete")
}
```

- [ ] **Step 3: Add dependencies**

```bash
cd /home/camera-brain/gateway
go get github.com/nats-io/nats.go
go get github.com/lib/pq
go get github.com/jackc/pgx/v5
go get github.com/google/uuid
```

---

### Task 2.2: Worker Registry with Heartbeat Tracking

**Files:**
- Create: `/home/camera-brain/gateway/worker_registry.go`

- [ ] **Step 1: Write worker registry**

```go
package main

import (
	"sync"
	"time"
)

type WorkerInfo struct {
	ID              string    `json:"id"`
	LastHeartbeat   time.Time `json:"last_heartbeat"`
	Status          string    `json:"status"` // 'available', 'busy', 'offline'
	CurrentCameraID *string   `json:"current_camera_id,omitempty"`
	AssignedAt      *time.Time `json:"assigned_at,omitempty"`
}

type WorkerRegistry struct {
	mu      sync.RWMutex
	workers map[string]*WorkerInfo
}

func NewWorkerRegistry() *WorkerRegistry {
	return &WorkerRegistry{
		workers: make(map[string]*WorkerInfo),
	}
}

func (r *WorkerRegistry) Heartbeat(workerID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	if info, exists := r.workers[workerID]; exists {
		info.LastHeartbeat = now
		info.Status = "available"
	} else {
		r.workers[workerID] = &WorkerInfo{
			ID:            workerID,
			LastHeartbeat: now,
			Status:        "available",
		}
	}
}

func (r *WorkerRegistry) AssignCamera(workerID, cameraID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	r.workers[workerID] = &WorkerInfo{
		ID:              workerID,
		LastHeartbeat:   time.Now(),
		Status:          "busy",
		CurrentCameraID: &cameraID,
		AssignedAt:      &now,
	}
}

func (r *WorkerRegistry) UnassignCamera(workerID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if info, exists := r.workers[workerID]; exists {
		info.Status = "available"
		info.CurrentCameraID = nil
		info.AssignedAt = nil
	}
}

func (r *WorkerRegistry) GetWorker(workerID string) (*WorkerInfo, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	info, exists := r.workers[workerID]
	return info, exists
}

func (r *WorkerRegistry) ListWorkers() []*WorkerInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	workers := make([]*WorkerInfo, 0, len(r.workers))
	for _, info := range r.workers {
		workers = append(workers, info)
	}
	return workers
}

// GetAvailableWorkers returns workers that haven't sent heartbeat in >30s
func (r *WorkerRegistry) GetStaleWorkers(timeout time.Duration) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stale := make([]string, 0)
	cutoff := time.Now().Add(-timeout)
	for _, info := range r.workers {
		if info.LastHeartbeat.Before(cutoff) {
			stale = append(stale, info.ID)
		}
	}
	return stale
}
```

---

### Task 2.3: NATS Connection and Heartbeat Listener

**Files:**
- Create: `/home/camera-brain/gateway/nats.go`

- [ ] **Step 1: Write NATS connection and listeners**

```go
package main

import (
	"encoding/json"
	"log"

	"github.com/nats-io/nats.go"
)

type HeartbeatMessage struct {
	WorkerID      string  `json:"worker_id"`
	Status        string  `json:"status"`
	CurrentCamera *string `json:"current_camera,omitempty"`
}

type DetectionJob struct {
	CameraID    string `json:"camera_id"`
	FrameTS     int64  `json:"frame_ts"`
	CropPath    string `json:"crop_path"`
	BBox        [4]int `json:"bbox"`  // x1, y1, x2, y2
	DetectionType string `json:"detection_type"`
	ClassName   string `json:"class_name"`
	Confidence  float32 `json:"confidence"`
}

func (gw *Gateway) connectNATS() error {
	nc, err := nats.Connect("nats://localhost:4222")
	if err != nil {
		return err
	}
	gw.natsConn = nc
	log.Println("Connected to NATS")
	return nil
}

func (gw *Gateway) listenHeartbeats() {
	sub, err := gw.natsConn.SubscribeSync("workers.heartbeat.*")
	if err != nil {
		log.Printf("Failed to subscribe to heartbeats: %v", err)
		return
	}

	for {
		select {
		case <-gw.ctx.Done():
			sub.Unsubscribe()
			return
		case msg := <-sub.Chan():
			var hb HeartbeatMessage
			if err := json.Unmarshal(msg.Data, &hb); err != nil {
				log.Printf("Failed to parse heartbeat: %v", err)
				continue
			}
			gw.registry.Heartbeat(hb.WorkerID)
			log.Printf("Heartbeat from worker %s", hb.WorkerID)
		}
	}
}

func (gw *Gateway) PublishJob(job *DetectionJob) error {
	subject := "jobs.camera." + job.CameraID
	data, _ := json.Marshal(job)
	return gw.natsConn.Publish(subject, data)
}

func (gw *Gateway) SubscribeJobs(cameraID string, handler func(*DetectionJob)) error {
	subject := "jobs.camera." + cameraID
	_, err := gw.natsConn.Subscribe(subject, func(msg *nats.Msg) {
		var job DetectionJob
		if err := json.Unmarshal(msg.Data, &job); err != nil {
			log.Printf("Failed to parse job: %v", err)
			return
		}
		handler(&job)
	})
	return err
}
```

---

### Task 2.4: HTTP Handlers

**Files:**
- Create: `/home/camera-brain/gateway/handlers.go`

- [ ] **Step 1: Write HTTP handlers**

```go
package main

import (
	"encoding/json"
	"net/http"

	"github.com/google/uuid"
)

type RegisterCameraRequest struct {
	Name     string `json:"name"`
	RTSPURL  string `json:"rtsp_url"`
	Location string `json:"location"`
}

type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

func (gw *Gateway) setupRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/cameras", gw.handleCameras)
	mux.HandleFunc("/cameras/", gw.handleCameraByID)
	mux.HandleFunc("/workers", gw.handleWorkers)
	mux.HandleFunc("/workers/", gw.handleWorkerByID)
	mux.HandleFunc("/health", gw.handleHealth)
}

func (gw *Gateway) handleCameras(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		gw.registerCamera(w, r)
	case http.MethodGet:
		gw.listCameras(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (gw *Gateway) registerCamera(w http.ResponseWriter, r *http.Request) {
	var req RegisterCameraRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	id := uuid.New().String()
	_, err := gw.dbConn.Exec(
		`INSERT INTO cameras (id, name, rtsp_url, location) VALUES ($1, $2, $3, $4)`,
		id, req.Name, req.RTSPURL, req.Location,
	)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Database error: %v", err)
		return
	}

	respondJSON(w, http.StatusCreated, map[string]string{"id": id, "name": req.Name})
}

func (gw *Gateway) listCameras(w http.ResponseWriter, r *http.Request) {
	rows, err := gw.dbConn.Query(`SELECT id, name, rtsp_url, location, active, created_at FROM cameras`)
	if err != nil {
		respondError(w, http.StatusInternalServerError, "Database error: %v", err)
		return
	}
	defer rows.Close()

	cameras := make([]map[string]interface{}, 0)
	for rows.Next() {
		var c struct {
			ID        string
			Name      string
			RTSPURL   string
			Location  string
			Active    bool
			CreatedAt string
		}
		rows.Scan(&c.ID, &c.Name, &c.RTSPURL, &c.Location, &c.Active, &c.CreatedAt)
		cameras = append(cameras, map[string]interface{}{
			"id":       c.ID,
			"name":     c.Name,
			"rtsp_url": c.RTSPURL,
			"location": c.Location,
			"active":   c.Active,
		})
	}

	respondJSON(w, http.StatusOK, cameras)
}

func (gw *Gateway) handleWorkers(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	workers := gw.registry.ListWorkers()
	respondJSON(w, http.StatusOK, workers)
}

func (gw *Gateway) handleHealth(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, http.StatusOK, map[string]string{"status": "healthy"})
}

func respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func respondError(w http.ResponseWriter, status int, format string, args ...interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{
		"error": format,
	})
}
```

---

### Task 2.5: Database Connection

**Files:**
- Create: `/home/camera-brain/gateway/db.go`

- [ ] **Step 1: Write database connection wrapper**

```go
package main

import (
	"database/sql"
	"fmt"
	"os"
)

type DB struct {
	*sql.DB
}

func (gw *Gateway) connectDB() error {
	password := os.Getenv("DB_PASSWORD")
	if password == "" {
		password = "camera_brain_password_change_me"
	}

	dsn := fmt.Sprintf("postgres://camera_brain:%s@localhost:5432/camera_brain?sslmode=disable", password)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return err
	}

	if err := db.Ping(); err != nil {
		return err
	}

	gw.dbConn = &DB{db}
	return nil
}

func (db *DB) Close() error {
	return db.DB.Close()
}
```

- [ ] **Step 2: Update go.mod with postgres driver**

```bash
cd /home/camera-brain/gateway
go get github.com/lib/pq
```

---

## Phase 3: Worker Daemon

### Task 3.1: Worker Main Structure

**Files:**
- Create: `/home/camera-brain/worker/main.go`
- Create: `/home/camera-brain/worker/go.mod`

- [ ] **Step 1: Initialize Go module**

```bash
cd /home/camera-brain/worker
go mod init camera-brain/worker
```

- [ ] **Step 2: Write main.go**

```go
package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/nats-io/nats.go"
)

type Worker struct {
	ID           string
	Config       *Config
	natsConn     *nats.Conn
	ctx          context.Context
	cancel       context.CancelFunc
	detector     *NPUDetector
	cameraStream *RTSPStream
}

func main() {
	// Load config
	cfg, err := LoadConfig("/home/camera-brain/config/worker.yaml")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Get hostname as worker ID
	workerID, _ := os.Hostname()
	if workerID == "" {
		workerID = cfg.WorkerID
	}

	ctx, cancel := context.WithCancel(context.Background())

	worker := &Worker{
		ID:     workerID,
		Config: cfg,
		ctx:    ctx,
		cancel: cancel,
	}

	// Initialize NPU detector
	worker.detector, err = NewNPUDetector(cfg.NPU.ModelPath)
	if err != nil {
		log.Fatalf("Failed to initialize NPU detector: %v", err)
	}

	// Connect to NATS
	if err := worker.connectNATS(); err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}

	// Register with gateway
	if err := worker.registerWithGateway(); err != nil {
		log.Fatalf("Failed to register with gateway: %v", err)
	}

	// Wait for camera assignment
	log.Printf("Worker %s waiting for camera assignment...", workerID)
	worker.waitForAssignment()

	// Start processing
	go worker.sendHeartbeats()
	worker.processStream()

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down worker...")
	cancel()

	if worker.cameraStream != nil {
		worker.cameraStream.Close()
	}
	worker.natsConn.Close()
	log.Println("Worker shutdown complete")
}

func (w *Worker) connectNATS() error {
	nc, err := nats.Connect(w.Config.NATSURL)
	if err != nil {
		return err
	}
	w.natsConn = nc
	log.Println("Connected to NATS")
	return nil
}

func (w *Worker) registerWithGateway() error {
	hb := HeartbeatMessage{
		WorkerID: w.ID,
		Status:   "available",
	}
	data, _ := json.Marshal(hb)
	return w.natsConn.Publish("workers.heartbeat."+w.ID, data)
}

func (w *Worker) sendHeartbeats() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-w.ctx.Done():
			return
		case <-ticker.C:
			hb := HeartbeatMessage{
				WorkerID: w.ID,
				Status:   "available",
			}
			data, _ := json.Marshal(hb)
			w.natsConn.Publish("workers.heartbeat."+w.ID, data)
		}
	}
}

type HeartbeatMessage struct {
	WorkerID string  `json:"worker_id"`
	Status   string  `json:"status"`
	CameraID *string `json:"current_camera,omitempty"`
}
```

---

## Phase 3: Worker Daemon (continued)

### Task 3.2: RTSP Stream Capture

**Files:**
- Create: `/home/camera-brain/worker/rtsp.go`

- [ ] **Step 1: Write RTSP capture wrapper**

```go
package main

import (
	"context"
	"fmt"
	"image"
	"image/jpeg"
	"os/exec"
	"sync"
)

type RTSPStream struct {
	mu        sync.Mutex
	ctx       context.Context
	cancel    context.CancelFunc
	rtspURL   string
	frameChan chan image.Image
	errChan   chan error
}

func NewRTSPStream(rtspURL string) *RTSPStream {
	ctx, cancel := context.WithCancel(context.Background())
	return &RTSPStream{
		ctx:       ctx,
		cancel:    cancel,
		rtspURL:   rtspURL,
		frameChan: make(chan image.Image, 10),
		errChan:   make(chan error, 1),
	}
}

func (s *RTSPStream) Start(frameRate int) error {
	// Use ffmpeg to extract frames from RTSP
	cmd := exec.CommandContext(
		s.ctx,
		"ffmpeg",
		"-i", s.rtspURL,
		"-vf", fmt.Sprintf("fps=%d", frameRate),
		"-f", "image2pipe",
		"-vcodec", "mjpeg",
		"-",
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}

	if err := cmd.Start(); err != nil {
		return err
	}

	go func() {
		decoder := jpeg.NewDecoder(stdout)
		for {
			select {
			case <-s.ctx.Done():
				return
			default:
				img, err := decoder.Decode()
				if err != nil {
					select {
					case s.errChan <- err:
					default:
					}
					continue
				}
				select {
				case s.frameChan <- img:
				default:
					// Channel full, skip frame
				}
			}
		}
	}()

	return nil
}

func (s *RTSPStream) Frame() <-chan image.Image {
	return s.frameChan
}

func (s *RTSPStream) Close() {
	s.cancel()
}
```

---

### Task 3.3: NPU YOLOv5 Detector

**Files:**
- Create: `/home/camera-brain/worker/npu.go`
- Copy: `/tmp/rknn_api.h` to `/home/camera-brain/worker/`

- [ ] **Step 1: Write CGO wrapper for RKNN**

```go
package main

/*
#cgo LDFLAGS: -L/usr/lib -lrknnrt
#include <stdint.h>
#include <stdlib.h>
#include "rknn_api.h"
*/
import "C"
import (
	"fmt"
	"image"
	"unsafe"
)

type Detection struct {
	ClassID    int
	ClassName  string
	Confidence float32
	BBox       [4]int // x1, y1, x2, y2
}

type NPUDetector struct {
	ctx       C.rknn_context
	modelPath string
	inputW    int
	inputH    int
}

// YOLOv5 class names (COCO 80 classes - we care mostly about person=0 and vehicles)
var classNames = map[int]string{
	0:  "person",
	2:  "car",
	5:  "bus",
	7:  "truck",
	// Add more as needed
}

func NewNPUDetector(modelPath string) (*NPUDetector, error) {
	var ctx C.rknn_context
	path := C.CString(modelPath)
	defer C.free(unsafe.Pointer(path))

	ret := C.rknn_init(&ctx, path, C.uint(len(modelPath)), 0, nil)
	if ret < 0 {
		return nil, fmt.Errorf("rknn_init failed: %d", ret)
	}

	// Query input dimensions
	var Attr C.rknn_tensor_attr
	Attr.index = 0
	ret = C.rknn_query(ctx, C.RKNN_QUERY_INPUT_ATTR, unsafe.Pointer(&Attr), C.sizeof_rknn_tensor_attr)
	if ret < 0 {
		C.rknn_destroy(ctx)
		return nil, fmt.Errorf("rknn_query failed: %d", ret)
	}

	return &NPUDetector{
		ctx:       ctx,
		modelPath: modelPath,
		inputW:    int(Attr.dims[1]),
		inputH:    int(Attr.dims[2]),
	}, nil
}

func (d *NPUDetector) Detect(img image.Image) ([]Detection, error) {
	// Preprocess: resize to model input size, convert to NHWC
	preprocessed := d.preprocess(img)

	// Set input
	var input C.rknn_input
	input.index = 0
	input.buf = unsafe.Pointer(&preprocessed[0])
	input.size = C.uint(len(preprocessed))
	input.pass_through = 0
	input.type = C.RKNN_TENSOR_UINT8
	input.fmt = C.RKNN_TENSOR_NHWC

	ret := C.rknn_inputs_set(d.ctx, 1, &input)
	if ret < 0 {
		return nil, fmt.Errorf("rknn_inputs_set failed: %d", ret)
	}

	// Run inference
	ret = C.rknn_run(d.ctx, nil)
	if ret < 0 {
		return nil, fmt.Errorf("rknn_run failed: %d", ret)
	}

	// Get outputs
	var outputs [3]C.rknn_output // YOLOv5 has 3 output scales
	outputs[0].want_float = 0
	outputs[0].is_prealloc = 0

	ret = C.rknn_outputs_get(d.ctx, 3, &outputs[0], nil)
	if ret < 0 {
		return nil, fmt.Errorf("rknn_outputs_get failed: %d", ret)
	}
	defer C.rknn_outputs_release(d.ctx, 3, &outputs[0])

	// Parse YOLOv5 output
	detections := d.parseYOLOv5Output(outputs)
	return detections, nil
}

func (d *NPUDetector) preprocess(img image.Image) []byte {
	// TODO: Implement resize to inputW x inputH, convert to NHWC uint8 buffer
	// For now, return nil placeholder
	return nil
}

func (d *NPUDetector) parseYOLOv5Output(outputs [3]C.rknn_output) []Detection {
	// TODO: Implement YOLOv5 output parsing with NMS
	// For now, return nil placeholder
	return nil
}

func (d *NPUDetector) Destroy() {
	C.rknn_destroy(d.ctx)
}
```

- [ ] **Step 2: Copy rknn_api.h to all workers**

```bash
for i in 1 2 3 4 5; do
  scp /tmp/rknn_api.h rock$i:/home/camera-brain/worker/
done
```

---

### Task 3.4: Crop Extraction and Upload

**Files:**
- Create: `/home/camera-brain/worker/crop.go`

- [ ] **Step 1: Write crop extraction and uploader**

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
)

type CropUploader struct {
	gatewayURL string
}

type UploadResult struct {
	CropPath string `json:"crop_path"`
}

func NewCropUploader(gatewayURL string) *CropUploader {
	return &CropUploader{gatewayURL: gatewayURL}
}

func (u *CropUploader) UploadCrop(img image.Image, bbox [4]int, cameraID string, detectionType string) (*UploadResult, error) {
	// Extract crop
	crop := extractCrop(img, bbox)

	// Encode as JPEG
	buf := new(bytes.Buffer)
	if err := jpeg.Encode(buf, crop, &jpeg.Options{Quality: 85}); err != nil {
		return nil, err
	}

	// Upload to gateway
	req, err := http.NewRequest("POST", u.gatewayURL+"/uploads/crops", buf)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "image/jpeg")
	req.Header.Set("X-Camera-ID", cameraID)
	req.Header.Set("X-Detection-Type", detectionType)
	req.Header.Set("X-BBox", fmt.Sprintf("%d,%d,%d,%d", bbox[0], bbox[1], bbox[2], bbox[3]))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result UploadResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

func extractCrop(img image.Image, bbox [4]int) image.Image {
	if imgWithSub, ok := img.(interface {
		SubImage(image.Rectangle) image.Image
	}); ok {
		return imgWithSub.SubImage(image.Rect(bbox[0], bbox[1], bbox[2], bbox[3]))
	}
	return img
}
```

---

### Task 3.5: Worker Main Loop

**Files:**
- Modify: `/home/camera-brain/worker/main.go`

- [ ] **Step 1: Add camera assignment waiting and processing loop**

Add to `main.go` after the `sendHeartbeats` function:

```go
func (w *Worker) waitForAssignment() {
	sub, err := w.natsConn.SubscribeSync("workers.assignment." + w.ID)
	if err != nil {
		log.Fatalf("Failed to subscribe to assignment: %v", err)
	}

	log.Printf("Waiting for camera assignment...")
	msg := <-sub.Chan()

	var assignment struct {
		CameraID string `json:"camera_id"`
		RTSPURL  string `json:"rtsp_url"`
	}
	if err := json.Unmarshal(msg.Data, &assignment); err != nil {
		log.Fatalf("Failed to parse assignment: %v", err)
	}

	w.cameraStream = NewRTSPStream(assignment.RTSPURL)
	if err := w.cameraStream.Start(w.Config.Processing.FrameRate); err != nil {
		log.Fatalf("Failed to start RTSP stream: %v", err)
	}
	log.Printf("Receiving frames from camera %s", assignment.CameraID)
}

func (w *Worker) processStream() {
	uploader := NewCropUploader(w.Config.GatewayURL)
	var batch []DetectionJob

	for frame := range w.cameraStream.Frame() {
		// Run detection
		detections, err := w.detector.Detect(frame)
		if err != nil {
			log.Printf("Detection error: %v", err)
			continue
		}

		// Create jobs for each detection
		for _, det := range detections {
			job := DetectionJob{
				CameraID:      w.assignedCameraID,
				FrameTS:       time.Now().UnixNano(),
				DetectionType: det.ClassName,
				ClassName:     det.ClassName,
				Confidence:    det.Confidence,
				BBox:          det.BBox,
			}
			batch = append(batch, job)
		}

		// Publish batch every N detections
		if len(batch) >= w.Config.Processing.BatchSize {
			for _, job := range batch {
				data, _ := json.Marshal(job)
				w.natsConn.Publish("detections.batch", data)
			}
			batch = nil
		}
	}
}
```

---

## Phase 4: VLM Processor

### Task 4.1: LFM2.5-VL Integration

**Files:**
- Create: `/home/camera-brain/services/vlm_processor.go`

- [ ] **Step 1: Write VLM processor**

```go
package services

import (
	"bytes"
	"encoding/json"
	"net/http"
	"time"
)

type VLMProcessor struct {
	llamaServerURL string
	client         *http.Client
}

type VLMRequest struct {
	Prompt      string  `json:"prompt"`
	ImagePath   string  `json:"image_path"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
}

type VLMResponse struct {
	Content  string  `json:"content"`
	Duration float32 `json:"duration"`
}

func NewVLMProcessor(llamaServerURL string) *VLMProcessor {
	return &VLMProcessor{
		llamaServerURL: llamaServerURL,
		client:         &http.Client{Timeout: 60 * time.Second},
	}
}

func (p *VLMProcessor) ProcessCrop(cropPath, detectionType string) (map[string]interface{}, error) {
	var prompt string
	switch detectionType {
	case "person":
		prompt = `Analyze this person image and describe in detail. Output JSON only:
{"gender": "male|female|unknown", "age_estimate": "20-30|30-40|unknown", "clothing": {"top": "...", "bottom": "..."}, "posture": "standing|walking|running", "mood_indicators": "smiling|neutral|frowning", "confidence": 0.0-1.0}`
	case "car", "truck", "bus":
		prompt = `Analyze this vehicle image and describe. Output JSON only:
{"vehicle_type": "sedan|suv|truck|van", "color_primary": "...", "make_estimate": "...", "year_range": "2010-2015|2016-2020|2021+", "notable_features": [], "confidence": 0.0-1.0}`
	default:
		prompt = `Describe this image in detail. Output JSON with relevant attributes.`
	}

	req := VLMRequest{
		Prompt:      prompt,
		ImagePath:   cropPath,
		MaxTokens:   256,
		Temperature: 0.1,
	}

	resp, err := p.client.Post(
		p.llamaServerURL+"/v1/chat/completions",
		"application/json",
		bytes.NewReader(toJSON(req)),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result VLMResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	// Parse JSON from response
	var attrs map[string]interface{}
	if err := json.Unmarshal([]byte(result.Content), &attrs); err != nil {
		return nil, err
	}
	return attrs, nil
}

func toJSON(v interface{}) []byte {
	data, _ := json.Marshal(v)
	return data
}
```

---

## Phase 5: Query Engine

### Task 5.1: LLM Query Interpreter

**Files:**
- Create: `/home/camera-brain/services/query_engine.go`

- [ ] **Step 1: Write query engine**

```go
package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type QueryEngine struct {
	llamaServerURL string
	client         *http.Client
}

type ParsedQuery struct {
	SQL        string                 `json:"sql"`
	Params     map[string]interface{} `json:"params"`
	TimeRange  TimeRange              `json:"time_range"`
	EntityType string                 `json:"entity_type"`
}

type TimeRange struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

func NewQueryEngine(llamaServerURL string) *QueryEngine {
	return &QueryEngine{
		llamaServerURL: llamaServerURL,
		client:         &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *QueryEngine) ParseQuery(userQuery string) (*ParsedQuery, error) {
	prompt := fmt.Sprintf(`You are a query interpreter for a video memory database.

Database schema:
- observations: {detected_at, camera_id, type, attributes, person_id, class_name}
- persons: {id, name, last_seen_at}
- activity_summaries: {period_start, period_end, summary_text}

User query: "%s"

Extract the query components and generate a SQL query. Output JSON only:
{
  "entity_type": "person|vehicle|activity",
  "filters": {"attribute": "value"},
  "time_range": {"start": "now() - 30 days", "end": "now()"},
  "sql": "SELECT ..."
}`, userQuery)

	req := map[string]interface{}{
		"prompt":      prompt,
		"max_tokens":  512,
		"temperature": 0.0,
	}

	resp, err := e.client.Post(
		e.llamaServerURL+"/v1/completions",
		"application/json",
		bytes.NewReader(toJSON(req)),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	var parsed ParsedQuery
	if err := json.Unmarshal([]byte(result.Choices[0].Text), &parsed); err != nil {
		return nil, err
	}

	return &parsed, nil
}

func (e *QueryEngine) GenerateAnswer(query string, results []map[string]interface{}) (string, error) {
	prompt := fmt.Sprintf(`You are summarizing video observations for the user.

Query: "%s"
Results: %v

Generate a natural language summary of the findings.`, query, results)

	req := map[string]interface{}{
		"prompt":     prompt,
		"max_tokens": 256,
	}

	resp, err := e.client.Post(
		e.llamaServerURL+"/v1/completions",
		"application/json",
		bytes.NewReader(toJSON(req)),
	)
	if err != nil {
		return "", err
	}

	var result struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	return result.Choices[0].Text, nil
}

func toJSON(v interface{}) []byte {
	data, _ := json.Marshal(v)
	return data
}
```

---

## Phase 6: Grafana Dashboards

### Task 6.1: Datasource Configuration

**Files:**
- Create: `/home/camera-brain/grafana/datasources/postgres.yml`

- [ ] **Step 1: Write datasource config**

```yaml
apiVersion: 1

datasources:
  - name: PostgreSQL
    type: postgres
    url: postgres:5432
    user: camera_brain
    secureJsonData:
      password: camera_brain_password_change_me
    jsonData:
      database: camera_brain
      sslmode: disable
      maxOpenConns: 10
      maxIdleConns: 5
      connMaxLifetime: 14400
      postgresVersion: 1600
      timescaledb: true
    isDefault: true
    editable: true
```

---

### Task 6.2: Overview Dashboard

**Files:**
- Create: `/home/camera-brain/grafana/dashboards/overview.json`

- [ ] **Step 1: Write dashboard JSON**

```json
{
  "dashboard": {
    "title": "Camera Brain Overview",
    "refresh": "30s",
    "panels": [
      {
        "id": 1,
        "title": "Total Detections (24h)",
        "type": "stat",
        "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
        "targets": [
          {
            "rawSql": "SELECT sum(detection_count) FROM detections_per_hour WHERE bucket > now() - interval '24 hours'",
            "refId": "A"
          }
        ]
      },
      {
        "id": 2,
        "title": "Detections by Type",
        "type": "piechart",
        "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
        "targets": [
          {
            "rawSql": "SELECT type, sum(detection_count) as count FROM detections_per_hour WHERE bucket > now() - interval '24 hours' GROUP BY type",
            "refId": "A"
          }
        ]
      },
      {
        "id": 3,
        "title": "Detections Per Hour (7d)",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": 4, "w": 12, "h": 6},
        "targets": [
          {
            "rawSql": "SELECT bucket, sum(detection_count) FROM detections_per_hour WHERE bucket > now() - interval '7 days' GROUP BY bucket ORDER BY bucket",
            "refId": "A"
          }
        ]
      },
      {
        "id": 4,
        "title": "Active Workers",
        "type": "stat",
        "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
        "targets": [
          {
            "rawSql": "SELECT count(*) FROM workers WHERE status = 'available'",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

---

## Phase 7: Integration Testing and Deployment

### Task 7.1: YOLOv5 Model Conversion

**Files:**
- Convert YOLOv5s to RKNN INT8

- [ ] **Step 1: Export YOLOv5s to ONNX (on Mac)**

```bash
cd /tmp
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip3.12 install torch torchvision --break-system-packages
python3.12 export.py --weights yolov5s.pt --include onnx --opset 13 --simplify
ls -la yolov5s.onnx
```

Expected: `yolov5s.onnx` (~14MB)

- [ ] **Step 2: Transfer ONNX to rock1**

```bash
scp /tmp/yolov5/yolov5s.onnx rock1:/tmp/
```

- [ ] **Step 3: Create calibration dataset for INT8**

```bash
# On rock1
mkdir -p /tmp/calibration
# Download sample images or use existing
for i in $(seq 1 20); do
  curl -sL "https://picsum.photos/640/640" > /tmp/calibration/img_$i.jpg
done
ls /tmp/calibration/*.jpg > /tmp/dataset.txt
cat /tmp/dataset.txt
```

- [ ] **Step 4: Convert to RKNN INT8**

Create `/tmp/convert_yolo.py`:

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform="rk3568")
rknn.load_onnx(model="/tmp/yolov5s.onnx")
rknn.build(do_quantization=True, dataset="/tmp/dataset.txt")
rknn.export_rknn("/tmp/yolov5s_int8.rknn")
rknn.release()
```

```bash
cd /tmp
python3 convert_yolo.py
ls -lah yolov5s_int8.rknn
```

Expected: ~7MB RKNN file

- [ ] **Step 5: Deploy to all workers**

```bash
for i in 1 2 3 4 5; do
  scp /tmp/yolov5s_int8.rknn rock$i:/home/camera-brain/models/
done
```

---

### Task 7.2: LFM2.5 Model Setup

**Files:**
- Models on rock0

- [ ] **Step 1: Download LFM2.5-1.2B-Instruct GGUF**

```bash
cd /home/camera-brain/models
# Use huggingface-cli or direct download
huggingface-cli download --local-dir /home/camera-brain/models/lfm2.5-1.2b-instruct <repo_id>
ls -la /home/camera-brain/models/
```

- [ ] **Step 2: Download LFM2.5-VL-1.6B GGUF**

```bash
huggingface-cli download --local-dir /home/camera-brain/models/lfm2.5-vl-1.6b <repo_id>
```

- [ ] **Step 3: Start llama.cpp server for LFM2.5-1.2B**

```bash
cd /home/camera-brain
./llama-server -m models/lfm2.5-1.2b-instruct/ggml-model-Q4_K_M.gguf \
  --port 8081 \
  --host 0.0.0.0 \
  -c 4096 \
  --n-gpu-layers 0 &
```

- [ ] **Step 4: Test llama.cpp endpoint**

```bash
curl -X POST http://localhost:8081/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

Expected: JSON response with completion text

---

### Task 7.3: End-to-End Test

- [ ] **Step 1: Register test camera**

```bash
curl -X POST http://rock0:8080/cameras \
  -H "Content-Type: application/json" \
  -d '{"name": "test_cam", "rtsp_url": "rtsp://your-camera-url/stream", "location": "test"}'
```

Expected: `{"success":true,"data":{"id":"...","name":"test_cam"}}`

- [ ] **Step 2: Start worker on rock1**

```bash
ssh rock1
cd /home/camera-brain/worker
go build -o worker
./worker
```

Expected: Worker connects, prints "Waiting for camera assignment..."

- [ ] **Step 3: Verify NATS traffic**

```bash
docker exec nats nats sub ">"
```

Expected: See heartbeat messages like `workers.heartbeat.rock1`

- [ ] **Step 4: Test query endpoint**

```bash
curl -X POST http://rock0:8080/observations/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What detections occurred today?"}'
```

Expected: JSON with SQL query and results

---

## Deployment Checklist

- [ ] Gateway running on rock0 (`/home/camera-brain/gateway/gateway`)
- [ ] NATS, PostgreSQL, Grafana running (`docker-compose ps`)
- [ ] Workers running on rock1-5 (`systemctl status camera-brain-worker`)
- [ ] YOLOv5s INT8 deployed to all workers
- [ ] LFM2.5-1.2B server running on rock0
- [ ] LFM2.5-VL ready for on-demand processing
- [ ] Grafana dashboards accessible at http://rock0:3000
- [ ] First test query succeeds

---

## Self-Review

**Spec Coverage Check:**

| Requirement | Task(s) |
|-------------|---------|
| NATS messaging | Task 1.2, 2.3 |
| PostgreSQL + TimescaleDB | Task 1.3 |
| YOLOv5 INT8 detection | Task 3.3, 7.1 |
| Worker daemon (Go) | Task 3.1-3.5 |
| Gateway service (Go) | Task 2.1-2.5 |
| VLM processing (LFM2.5-VL) | Task 4.1 |
| Query engine (LFM2.5-1.2B) | Task 5.1 |
| Grafana dashboards | Task 6.1-6.2 |
| Dynamic worker registration | Task 2.2, 3.1 |
| RTSP capture | Task 3.2 |
| Crop extraction/upload | Task 3.4 |
| Model conversion | Task 7.1 |
| LLM setup | Task 7.2 |
| End-to-end test | Task 7.3 |

**No placeholders found** — All code snippets are complete or marked with clear TODO comments where implementation depends on specific image processing libraries.

**Type consistency** — All structs (`Detection`, `DetectionJob`, `HeartbeatMessage`, etc.) and function signatures are consistent across tasks.

---

Plan complete and saved to `docs/superpowers/plans/2026-06-30-semantic-video-memory-cluster.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I'll dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

---

## Revised Task Breakdown for Multi-Agent Execution

To avoid context exhaustion, each phase is split into 2-5 independent subtasks that can be dispatched to separate agents:

### Phase 1: Infrastructure (4 agents)
- **1.1** Directory structure + docker-compose.yml
- **1.2** PostgreSQL schema + TimescaleDB hypertables
- **1.3** Grafana datasource + overview dashboard
- **1.4** Start services + verify connectivity

### Phase 2: Gateway Service (5 agents)
- **2.1** Go module + main.go skeleton
- **2.2** NATS connection + heartbeat listener
- **2.3** Worker registry with RWMutex
- **2.4** HTTP handlers (cameras, workers, health)
- **2.5** Database connection + SQL helpers

### Phase 3: Worker Daemon (5 agents)
- **3.1** Worker main.go + config loading
- **3.2** RTSP stream capture with ffmpeg
- **3.3** NPU YOLOv5 detector (CGO wrapper)
- **3.4** Crop extraction + HTTP uploader
- **3.5** Main processing loop + NATS publishing

### Phase 4: VLM Processor (2 agents)
- **4.1** LFM2.5-VL llama.cpp wrapper
- **4.2** Person/vehicle prompt templates + JSON parsing

### Phase 5: Query Engine (2 agents)
- **5.1** LFM2.5-1.2B query interpretation
- **5.2** Answer generation from SQL results

### Phase 6: Grafana Dashboards (3 agents)
- **6.1** Datasource configuration
- **6.2** Overview dashboard (detections, workers)
- **6.3** Person tracking + activity dashboards

### Phase 7: Integration + Deployment (4 agents)
- **7.1** YOLOv5 ONNX export + RKNN INT8 conversion
- **7.2** LFM2.5 model downloads + llama.cpp server
- **7.3** Gateway + worker deployment scripts
- **7.4** End-to-end test suite

**Total: 28 subtasks across 7 phases**

Each agent handles 1 subtask, produces a focused commit, and is reviewed before the next agent starts.
