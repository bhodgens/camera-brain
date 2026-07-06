// Command gateway runs the Camera Brain Gateway service.
// It coordinates workers via NATS and provides HTTP APIs for camera/worker management.
package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/nats-io/nats.go"
	"rock-cluster/config"
)

// Server is the gateway HTTP server.
type Server struct {
	db          *sql.DB
	nats        *nats.Conn
	registry    *WorkerRegistry
	port        int
	mux         *http.ServeMux
	natsUnsubscribe func()
}

// Close releases resources held by the server.
func (s *Server) Close() {
	if s.natsUnsubscribe != nil {
		s.natsUnsubscribe()
	}
}

// Worker represents a connected worker node.
type Worker struct {
	ID            string     `json:"id"`
	LastHeartbeat time.Time  `json:"last_heartbeat"`
	Status        string     `json:"status"`
	CurrentCamera *uuid.UUID `json:"current_camera,omitempty"`
	AssignedAt    *time.Time `json:"assigned_at,omitempty"`
}

// Camera represents a configured camera.
type Camera struct {
	ID        uuid.UUID `json:"id"`
	Name      string    `json:"name"`
	RTSPURL   string    `json:"rtsp_url"`
	Location  string    `json:"location"`
	Active    bool      `json:"active"`
	CreatedAt time.Time `json:"created_at"`
}

// WorkerRegistry tracks connected workers.
type WorkerRegistry struct {
	mu       sync.RWMutex
	workers  map[string]*Worker
	db       *sql.DB
	natsConn *nats.Conn
}

// NewWorkerRegistry creates a new worker registry.
func NewWorkerRegistry(db *sql.DB, nc *nats.Conn) *WorkerRegistry {
	return &WorkerRegistry{
		workers:  make(map[string]*Worker),
		db:       db,
		natsConn: nc,
	}
}

// RegisterWorker records a worker heartbeat.
func (r *WorkerRegistry) RegisterWorker(id string, cameraID *uuid.UUID) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	worker := &Worker{
		ID:            id,
		LastHeartbeat: now,
		Status:        "online",
		CurrentCamera: cameraID,
		AssignedAt:    &now,
	}

	if existing, ok := r.workers[id]; ok {
		existing.LastHeartbeat = now
		existing.Status = "online"
		existing.CurrentCamera = cameraID
	} else {
		r.workers[id] = worker
	}

	// Upsert in database
	_, err := r.db.Exec(`
		INSERT INTO workers (id, last_heartbeat, status, current_camera_id, assigned_at, created_at)
		VALUES ($1, $2, $3, $4, $5, NOW())
		ON CONFLICT (id) DO UPDATE SET
			last_heartbeat = EXCLUDED.last_heartbeat,
			status = EXCLUDED.status,
			current_camera_id = EXCLUDED.current_camera_id,
			assigned_at = EXCLUDED.assigned_at
	`, id, now, "online", cameraID, &now)

	return err
}

// ListWorkers returns a deep copy of all registered workers as values.
// Returning values (not pointers) under the read lock prevents data races
// between the NATS heartbeat goroutine (which mutates the stored *Worker
// structs) and HTTP handlers that iterate and JSON-encode the result.
func (r *WorkerRegistry) ListWorkers() []Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	workers := make([]Worker, 0, len(r.workers))
	for _, w := range r.workers {
		cp := *w
		if w.CurrentCamera != nil {
			cam := *w.CurrentCamera
			cp.CurrentCamera = &cam
		}
		if w.AssignedAt != nil {
			ts := *w.AssignedAt
			cp.AssignedAt = &ts
		}
		workers = append(workers, cp)
	}
	return workers
}

// GetWorker returns a deep copy of a specific worker by ID.
func (r *WorkerRegistry) GetWorker(id string) (Worker, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	w, ok := r.workers[id]
	if !ok {
		return Worker{}, false
	}
	cp := *w
	if w.CurrentCamera != nil {
		cam := *w.CurrentCamera
		cp.CurrentCamera = &cam
	}
	if w.AssignedAt != nil {
		ts := *w.AssignedAt
		cp.AssignedAt = &ts
	}
	return cp, true
}

// AssignCamera assigns a camera to a worker via NATS.
func (r *WorkerRegistry) AssignCamera(workerID string, cameraID uuid.UUID) error {
	// Fetch camera details from database
	var rtspURL, location string
	err := r.db.QueryRow(`SELECT rtsp_url, location FROM cameras WHERE id = $1`, cameraID).Scan(&rtspURL, &location)
	if err != nil {
		return fmt.Errorf("fetch camera: %w", err)
	}

	msg := map[string]interface{}{
		"camera_id":  cameraID.String(),
		"rtsp_url":   rtspURL,
		"location":   location,
		"assigned_at": time.Now().Format(time.RFC3339),
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal assignment msg: %w", err)
	}
	return r.natsConn.Publish(fmt.Sprintf("workers.assignment.%s", workerID), data)
}

// NewServer creates a new gateway server.
func NewServer(db *sql.DB, nc *nats.Conn, registry *WorkerRegistry, port int, unsub func()) *Server {
	s := &Server{
		db:              db,
		nats:            nc,
		registry:        registry,
		port:            port,
		mux:             http.NewServeMux(),
		natsUnsubscribe: unsub,
	}
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/workers", s.handleWorkers)
	s.mux.HandleFunc("/cameras", s.handleCameras)
	s.mux.HandleFunc("/cameras/", s.handleCamera)
	return s
}

// handleHealth verifies dependencies (DB + NATS) before returning 200.
// Returns 503 if any dependency is unhealthy.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	pingCtx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := s.db.PingContext(pingCtx); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{"status": "unhealthy", "error": "db: " + err.Error()})
		return
	}
	if !s.nats.IsConnected() {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{"status": "unhealthy", "error": "nats disconnected"})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]string{"status": "ok"}); err != nil {
		slog.Warn("Failed to encode health response", "error", err)
	}
}

// handleWorkers handles GET /workers.
func (s *Server) handleWorkers(w http.ResponseWriter, r *http.Request) {
	workers := s.registry.ListWorkers()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(workers); err != nil {
		slog.Warn("Failed to encode workers response", "error", err)
	}
}

// handleCameras handles GET /cameras and POST /cameras.
func (s *Server) handleCameras(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		cameras, err := s.listCameras()
		if err != nil {
			http.Error(w, "Database error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(cameras); err != nil {
			slog.Warn("Failed to encode cameras list response", "error", err)
		}

	case http.MethodPost:
		var cam Camera
		if err := json.NewDecoder(r.Body).Decode(&cam); err != nil {
			http.Error(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
			return
		}
		if err := s.registerCamera(&cam); err != nil {
			http.Error(w, "Database error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusCreated)
		if err := json.NewEncoder(w).Encode(cam); err != nil {
			slog.Warn("Failed to encode camera create response", "error", err)
		}

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleCamera handles /cameras/{camera_id}/assign.
func (s *Server) handleCamera(w http.ResponseWriter, r *http.Request) {
	// Parse camera ID from path
	path := strings.TrimPrefix(r.URL.Path, "/cameras/")
	parts := strings.Split(path, "/")
	if len(parts) < 1 || parts[0] == "" {
		http.Error(w, "Invalid camera ID", http.StatusBadRequest)
		return
	}
	cameraIDStr := parts[0]

	// Handle assignment
	if r.Method == http.MethodPost {
		workerID := r.URL.Query().Get("worker_id")
		if workerID == "" {
			http.Error(w, "worker_id query parameter required", http.StatusBadRequest)
			return
		}

		cameraID, err := uuid.Parse(cameraIDStr)
		if err != nil {
			http.Error(w, "Invalid camera ID: "+err.Error(), http.StatusBadRequest)
			return
		}

		if err := s.registry.AssignCamera(workerID, cameraID); err != nil {
			http.Error(w, "NATS publish error: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{"status": "assigned"}); err != nil {
			slog.Warn("Failed to encode assignment response", "error", err)
		}
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

// listCameras returns all cameras from the database.
func (s *Server) listCameras() ([]Camera, error) {
	rows, err := s.db.Query(`
		SELECT id, name, rtsp_url, location, active, created_at
		FROM cameras ORDER BY name
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var cameras []Camera
	for rows.Next() {
		var cam Camera
		err := rows.Scan(&cam.ID, &cam.Name, &cam.RTSPURL, &cam.Location, &cam.Active, &cam.CreatedAt)
		if err != nil {
			return nil, err
		}
		cameras = append(cameras, cam)
	}
	return cameras, nil
}

// registerCamera inserts a new camera.
func (s *Server) registerCamera(cam *Camera) error {
	if cam.ID == uuid.Nil {
		cam.ID = uuid.New()
	}
	_, err := s.db.Exec(`
		INSERT INTO cameras (id, name, rtsp_url, location, active)
		VALUES ($1, $2, $3, $4, $5)
	`, cam.ID, cam.Name, cam.RTSPURL, cam.Location, cam.Active)
	return err
}

// Serve starts the HTTP server.
func (s *Server) Serve(ctx context.Context) error {
	server := &http.Server{
		Addr:              fmt.Sprintf(":%d", s.port),
		Handler:           s.mux,
		ReadTimeout:       30 * time.Second,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		slog.Info("Gateway starting", "port", s.port)
		errCh <- server.ListenAndServe()
	}()

	<-ctx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	return server.Shutdown(shutdownCtx)
}

// listenHeartbeats subscribes to worker heartbeats and returns an
// unsubscribe function that detaches the subscription from NATS.
func listenHeartbeats(nc *nats.Conn, registry *WorkerRegistry) (func(), error) {
	sub, err := nc.Subscribe("workers.heartbeat.>", func(msg *nats.Msg) {
		var hb struct {
			WorkerID  string `json:"worker_id"`
			CameraID  string `json:"camera_id"`
			Timestamp string `json:"timestamp"`
		}
		if err := json.Unmarshal(msg.Data, &hb); err != nil {
			slog.Warn("Invalid heartbeat", "error", err)
			return
		}

		var cameraUUID *uuid.UUID
		if hb.CameraID != "" {
			if id, err := uuid.Parse(hb.CameraID); err == nil {
				cameraUUID = &id
			}
		}

		if err := registry.RegisterWorker(hb.WorkerID, cameraUUID); err != nil {
			slog.Warn("Failed to register heartbeat", "error", err)
		}
	})
	if err != nil {
		return nil, err
	}
	return func() {
		_ = sub.Unsubscribe()
	}, nil
}

func main() {
	// Load configuration
	cfg, err := config.LoadFromEnv()
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	// Connect to database. pq.QuoteLiteral defensively escapes the password
	// (install.sh generates alphanumeric-only passwords today).
	dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		cfg.Storage.Host, cfg.Storage.Port, cfg.Storage.Username,
		pq.QuoteLiteral(cfg.Storage.Password), cfg.Storage.Database, cfg.Storage.SSLMode)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		slog.Error("Failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	// Bounded pool: prevents exhaustion of Postgres max_connections under load.
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(30 * time.Minute)

	if err := db.Ping(); err != nil {
		slog.Error("Database ping failed", "error", err)
		os.Exit(1)
	}

	// Connect to NATS. NATS_URL env var allows non-co-located deployments.
	nc, err := nats.Connect(cfg.Service.NATSURL)
	if err != nil {
		slog.Error("Failed to connect to NATS", "error", err, "url", cfg.Service.NATSURL)
		os.Exit(1)
	}
	defer nc.Close()

	// Create worker registry
	registry := NewWorkerRegistry(db, nc)

	// Start heartbeat listener
	unsub, err := listenHeartbeats(nc, registry)
	if err != nil {
		slog.Error("Failed to subscribe to heartbeats", "error", err)
		os.Exit(1)
	}

	// Create and start server
	server := NewServer(db, nc, registry, cfg.Service.Port, unsub)
	defer server.Close()

	// Setup context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		slog.Info("Shutting down...")
		cancel()
	}()

	if err := server.Serve(ctx); err != nil && err != http.ErrServerClosed {
		slog.Error("Server error", "error", err)
		os.Exit(1)
	}

	slog.Info("Gateway stopped")
}
