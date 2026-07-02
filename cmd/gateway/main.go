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
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/nats-io/nats.go"
	"rock-cluster/config"
)

// Server is the gateway HTTP server.
type Server struct {
	db       *sql.DB
	nats     *nats.Conn
	registry *WorkerRegistry
	port     int
	mux      *http.ServeMux
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

// ListWorkers returns all registered workers.
func (r *WorkerRegistry) ListWorkers() []*Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()

	workers := make([]*Worker, 0, len(r.workers))
	for _, w := range r.workers {
		workers = append(workers, w)
	}
	return workers
}

// GetWorker returns a specific worker by ID.
func (r *WorkerRegistry) GetWorker(id string) (*Worker, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	w, ok := r.workers[id]
	return w, ok
}

// AssignCamera assigns a camera to a worker via NATS.
func (r *WorkerRegistry) AssignCamera(workerID string, cameraID uuid.UUID) error {
	msg := map[string]interface{}{
		"camera_id": cameraID.String(),
		"action":    "assign",
	}
	data, _ := json.Marshal(msg)
	return r.natsConn.Publish(fmt.Sprintf("workers.assignment.%s", workerID), data)
}

// NewServer creates a new gateway server.
func NewServer(db *sql.DB, nc *nats.Conn, registry *WorkerRegistry, port int) *Server {
	s := &Server{
		db:       db,
		nats:     nc,
		registry: registry,
		port:     port,
		mux:      http.NewServeMux(),
	}
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/workers", s.handleWorkers)
	s.mux.HandleFunc("/cameras", s.handleCameras)
	s.mux.HandleFunc("/cameras/", s.handleCamera)
	return s
}

// handleHealth handles GET /health.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// handleWorkers handles GET /workers.
func (s *Server) handleWorkers(w http.ResponseWriter, r *http.Request) {
	workers := s.registry.ListWorkers()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(workers)
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
		json.NewEncoder(w).Encode(cameras)

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
		json.NewEncoder(w).Encode(cam)

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleCamera handles /cameras/{camera_id}/assign.
func (s *Server) handleCamera(w http.ResponseWriter, r *http.Request) {
	// Parse camera ID from path
	cameraIDStr := r.URL.Path[len("/cameras/"):]
	if cameraIDStr == "" || cameraIDStr == "assign" {
		http.Error(w, "Invalid camera ID", http.StatusBadRequest)
		return
	}

	// Handle assignment
	if r.URL.Path[len("/cameras/"):] != "" && r.Method == http.MethodPost {
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
		json.NewEncoder(w).Encode(map[string]string{"status": "assigned"})
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
		Addr:         fmt.Sprintf(":%d", s.port),
		Handler:      s.mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
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

// listenHeartbeats subscribes to worker heartbeats.
func listenHeartbeats(nc *nats.Conn, registry *WorkerRegistry) error {
	_, err := nc.Subscribe("workers.heartbeat.>", func(msg *nats.Msg) {
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
	return err
}

func main() {
	// Load configuration
	cfg, err := config.LoadFromEnv()
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	// Connect to database
	dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		cfg.Storage.Host, cfg.Storage.Port, cfg.Storage.Username,
		cfg.Storage.Password, cfg.Storage.Database, cfg.Storage.SSLMode)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		slog.Error("Failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	if err := db.Ping(); err != nil {
		slog.Error("Database ping failed", "error", err)
		os.Exit(1)
	}

	// Connect to NATS
	nc, err := nats.Connect(nats.DefaultURL)
	if err != nil {
		slog.Error("Failed to connect to NATS", "error", err)
		os.Exit(1)
	}
	defer nc.Close()

	// Create worker registry
	registry := NewWorkerRegistry(db, nc)

	// Start heartbeat listener
	if err := listenHeartbeats(nc, registry); err != nil {
		slog.Error("Failed to subscribe to heartbeats", "error", err)
		os.Exit(1)
	}

	// Create and start server
	server := NewServer(db, nc, registry, cfg.Service.Port)

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
