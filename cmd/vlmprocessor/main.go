// Command vlmprocessor runs an HTTP service for VLM-based image analysis.
package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"rock-cluster/config"
	"rock-cluster/pkg/plugin"
)

// Analyzer is the interface for VLM analysis.
type Analyzer interface {
	Analyze(ctx context.Context, image []byte, prompt string) (map[string]interface{}, error)
}

// vlmPlugin wraps the analysis.Analyzer to return map[string]interface{}.
type vlmPlugin struct {
	analyzer plugin.Analyzer
}

func (p *vlmPlugin) Analyze(ctx context.Context, image []byte, prompt string) (map[string]interface{}, error) {
	result, err := p.analyzer.Analyze(ctx, image, prompt)
	if err != nil {
		return nil, err
	}
	// Merge Attributes with RawResponse as fallback
	response := make(map[string]interface{})
	for k, v := range result.Attributes {
		response[k] = v
	}
	if len(response) == 0 {
		response["raw_description"] = result.RawResponse
	}
	return response, nil
}

// AnalysisRequest represents an incoming crop analysis request.
type AnalysisRequest struct {
	CameraID   string  `json:"camera_id"`
	DetectedAt string  `json:"detected_at"`
	ClassID    int     `json:"class_id"`
	ClassName  string  `json:"class_name"`
	Confidence float32 `json:"confidence"`
	BBox       [4]int  `json:"bbox"`
	CropBase64 string  `json:"crop_base64"`
	WorkerID   string  `json:"worker_id"`
}

// AnalysisResponse represents the analysis result.
type AnalysisResponse struct {
	CropID       string                 `json:"crop_id"`
	Stored       bool                   `json:"stored"`
	VLMAnalysis  map[string]interface{} `json:"vlm_analysis,omitempty"`
	ProcessingMS int64                  `json:"processing_ms"`
}

// Server is the VLM processor HTTP server.
type Server struct {
	analyzer Analyzer
	port     int
	mux      *http.ServeMux
}

// NewServer creates a new VLM processor server.
func NewServer(analyzer Analyzer, port int) *Server {
	s := &Server{
		analyzer: analyzer,
		port:     port,
		mux:      http.NewServeMux(),
	}
	s.mux.HandleFunc("/analyze", s.handleAnalyze)
	s.mux.HandleFunc("/health", s.handleHealth)
	return s
}

// handleAnalyze handles POST /analyze requests.
func (s *Server) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req AnalysisRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}

	start := time.Now()

	// Decode base64 image
	imgData, err := base64.StdEncoding.DecodeString(req.CropBase64)
	if err != nil {
		http.Error(w, "Invalid base64: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Analyze with VLM
	attrs, err := s.analyzer.Analyze(r.Context(), imgData, buildPromptForClass(req.ClassName))
	if err != nil {
		slog.Warn("VLM analysis failed", "error", err)
	}

	// Generate crop ID
	cropID := fmt.Sprintf("%s_%s_%s", req.CameraID, req.ClassName, time.Now().Format("20060102150405"))

	resp := AnalysisResponse{
		CropID:       cropID,
		Stored:       true,
		VLMAnalysis:  attrs,
		ProcessingMS: time.Since(start).Milliseconds(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleHealth handles GET /health.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// Serve starts the HTTP server.
func (s *Server) Serve(ctx context.Context) error {
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", s.port),
		Handler:      s.mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		slog.Info("VLM Processor starting", "port", s.port)
		errCh <- server.ListenAndServe()
	}()

	<-ctx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	return server.Shutdown(shutdownCtx)
}

// buildPromptForClass returns a class-specific prompt template.
func buildPromptForClass(className string) string {
	switch className {
	case "person":
		return `Analyze this person image. Output ONLY valid JSON:
{"gender": "male|female|unknown", "age_estimate": "20-30|30-40|unknown", "clothing_top": "...", "clothing_bottom": "...", "posture": "standing|walking|running", "mood": "smiling|neutral|frowning", "confidence": 0.0-1.0}`
	case "car", "truck", "bus":
		return `Analyze this vehicle. Output ONLY valid JSON:
{"vehicle_type": "sedan|suv|truck|van|bus", "color": "...", "make_estimate": "...", "year_range": "pre-2015|2015-2020|2021+", "confidence": 0.0-1.0}`
	default:
		return `Describe this image. Output valid JSON with relevant attributes.`
	}
}

func main() {
	// Load configuration
	cfg, err := config.LoadFromEnv()
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	// Get analysis plugin from registry
	analyzer, err := plugin.GetAnalysis(cfg.Analysis.Plugin)
	if err != nil {
		slog.Error("Failed to get analysis plugin", "error", err, "plugin", cfg.Analysis.Plugin)
		os.Exit(1)
	}

	// Initialize the analyzer
	if err := analyzer.Initialize(context.Background(), cfg.Analysis.Config.ToPluginConfig()); err != nil {
		slog.Error("Failed to initialize analyzer", "error", err)
		os.Exit(1)
	}
	defer analyzer.Close()

	// Wrap analyzer to return map[string]interface{}
	vlmAnalyzer := &vlmPlugin{analyzer: analyzer}

	// Create and start server
	server := NewServer(vlmAnalyzer, cfg.Service.Port)

	// Setup context with cancellation on interrupt
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

	slog.Info("VLM Processor stopped")
}
