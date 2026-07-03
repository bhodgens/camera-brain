// Command queryengine runs an HTTP service for natural language queries.
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
	"syscall"
	"time"

	"github.com/lib/pq"
	"rock-cluster/config"
	"rock-cluster/pkg/plugin"
)

// Server is the query engine HTTP server.
type Server struct {
	db           *sql.DB
	analyzer     plugin.Analyzer    // VLM for image analysis
	textAnalyzer plugin.Analyzer    // Text-only LLM for queries (optional)
	port         int
	mux          *http.ServeMux
}

// QueryRequest represents an incoming natural language query.
type QueryRequest struct {
	Query string `json:"query"`
}

// QueryResponse represents the query result.
type QueryResponse struct {
	Success      bool          `json:"success"`
	Answer       string        `json:"answer"`
	ParsedQuery  *ParsedQuery  `json:"parsed_query,omitempty"`
	ResultCount  int           `json:"result_count"`
	ProcessingMS int64         `json:"processing_ms"`
}

// ParsedQuery represents a structured query extracted from natural language.
// SQL is a parameterized template; Args are the bind values for $1, $2, ...
// The template is a compile-time constant (never interpolates user input),
// which is defense-in-depth against SQL injection even if keyword extraction
// is later extended to surface user-supplied values.
type ParsedQuery struct {
	SQL        string            `json:"sql"`
	Args       []any             `json:"args,omitempty"`
	Params     map[string]interface{} `json:"params"`
	TimeRange  TimeRange         `json:"time_range"`
	EntityType string            `json:"entity_type"`
	Filters    map[string]string `json:"filters"`
}

// TimeRange represents a time window for filtering.
type TimeRange struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

// QueryResult represents a single result from query execution.
type QueryResult struct {
	DetectedAt  time.Time `json:"detected_at"`
	CameraID    string    `json:"camera_id"`
	Type        string    `json:"type"`
	ClassName   string    `json:"class_name"`
	Attributes  map[string]interface{} `json:"attributes"`
}

// NewServer creates a new query engine server.
func NewServer(db *sql.DB, analyzer plugin.Analyzer, textAnalyzer plugin.Analyzer, port int) *Server {
	s := &Server{db: db, analyzer: analyzer, textAnalyzer: textAnalyzer, port: port, mux: http.NewServeMux()}
	s.mux.HandleFunc("/query", s.handleQuery)
	s.mux.HandleFunc("/health", s.handleHealth)
	return s
}

// handleQuery handles POST /query requests.
func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}

	start := time.Now()
	parsed := parseQuery(req.Query)

	results, err := s.executeQuery(r.Context(), parsed)
	if err != nil {
		http.Error(w, "Execute query: "+err.Error(), http.StatusInternalServerError)
		return
	}

	var answer string
	if len(results) == 0 {
		answer = "No observations found matching your query."
	} else {
		answer, err = s.generateAnswer(r.Context(), req.Query, results)
		if err != nil {
			answer = fmt.Sprintf("Found %d results (answer generation failed)", len(results))
		}
	}

	resp := QueryResponse{Success: true, Answer: answer, ParsedQuery: parsed, ResultCount: len(results), ProcessingMS: time.Since(start).Milliseconds()}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		slog.Warn("Failed to encode response", "error", err)
		return
	}
}

// handleHealth verifies DB connectivity before returning 200.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	pingCtx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := s.db.PingContext(pingCtx); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{"status": "unhealthy", "error": "db: " + err.Error()})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// Serve starts the HTTP server.
func (s *Server) Serve(ctx context.Context) error {
	server := &http.Server{
		Addr:              fmt.Sprintf(":%d", s.port),
		Handler:           s.mux,
		ReadTimeout:       30 * time.Second,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      120 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	errCh := make(chan error, 1)
	go func() {
		slog.Info("Query Engine starting", "port", s.port)
		errCh <- server.ListenAndServe()
	}()
	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	return server.Shutdown(shutdownCtx)
}

// parseQuery converts natural language to a parameterized SQL template.
// Ordering: truck is checked BEFORE the generic car/vehicle branch — without
// this, "truck" would match the car/vehicle branch and the truck-specific SQL
// would be unreachable dead code.
func parseQuery(userQuery string) *ParsedQuery {
	queryLower := strings.ToLower(userQuery)
	if strings.Contains(queryLower, "truck") {
		return &ParsedQuery{SQL: "SELECT detected_at, camera_id, class_name, attributes FROM observations WHERE class_name ILIKE $1 AND detected_at >= NOW() - INTERVAL '24 hours'", Args: []any{"%truck%"}, EntityType: "vehicle", TimeRange: TimeRange{"NOW() - INTERVAL '24 hours'", "NOW()"}, Filters: map[string]string{}}
	}
	if strings.Contains(queryLower, "car") || strings.Contains(queryLower, "vehicle") {
		return &ParsedQuery{SQL: "SELECT detected_at, camera_id, class_name, attributes FROM observations WHERE (class_name ILIKE $1 OR class_name ILIKE $2 OR class_name ILIKE $3) AND detected_at >= NOW() - INTERVAL '24 hours'", Args: []any{"%car%", "%truck%", "%bus%"}, EntityType: "vehicle", TimeRange: TimeRange{"NOW() - INTERVAL '24 hours'", "NOW()"}, Filters: map[string]string{}}
	}
	if strings.Contains(queryLower, "person") {
		return &ParsedQuery{SQL: "SELECT detected_at, camera_id, class_name, attributes FROM observations WHERE class_name ILIKE $1 AND detected_at >= NOW() - INTERVAL '24 hours'", Args: []any{"%person%"}, EntityType: "person", TimeRange: TimeRange{"NOW() - INTERVAL '24 hours'", "NOW()"}, Filters: map[string]string{}}
	}
	return &ParsedQuery{SQL: "SELECT detected_at, camera_id, class_name, attributes FROM observations WHERE detected_at >= NOW() - INTERVAL '24 hours'", EntityType: "observation", TimeRange: TimeRange{"NOW() - INTERVAL '24 hours'", "NOW()"}, Filters: map[string]string{}}
}

// executeQuery runs the parsed query against the database, honoring the
// request context for cancellation/timeout. The defer handles rows cleanup;
// no redundant explicit Close() is needed (sql.Rows.Close is idempotent).
func (s *Server) executeQuery(ctx context.Context, parsed *ParsedQuery) ([]QueryResult, error) {
	rows, err := s.db.QueryContext(ctx, parsed.SQL, parsed.Args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	cols, _ := rows.Columns()
	var results []QueryResult

	for rows.Next() {
		values := make([]interface{}, len(cols))
		for i := range values {
			values[i] = new(interface{})
		}
		if err := rows.Scan(values...); err != nil {
			return nil, err
		}
		row := make(map[string]interface{})
		for i, col := range cols {
			val := values[i].(*interface{})
			if val == nil {
				row[col] = nil
			} else {
				row[col] = *val
			}
		}
		var r QueryResult
		if v, ok := row["detected_at"].(string); ok {
			r.DetectedAt, _ = time.Parse(time.RFC3339, v)
		}
		if v, ok := row["camera_id"].(string); ok {
			r.CameraID = v
		}
		if v, ok := row["class_name"].(string); ok {
			r.ClassName = v
		}
		if v, ok := row["attributes"].([]byte); ok {
			json.Unmarshal(v, &r.Attributes)
		}
		results = append(results, r)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

// generateAnswer creates a natural language summary of query results.
func (s *Server) generateAnswer(ctx context.Context, query string, results []QueryResult) (string, error) {
	var contextBuf strings.Builder
	fmt.Fprintf(&contextBuf, "Found %d observations:\n", len(results))
	for i, r := range results {
		fmt.Fprintf(&contextBuf, "%d. [%s] %s at camera: %s\n", i+1, r.DetectedAt.Format("2006-01-02 15:04"), r.ClassName, r.CameraID)
	}
	prompt := fmt.Sprintf(`You are summarizing video observations for the user.

Original query: "%s"

%s

Generate a natural, helpful summary of the findings. If the results are sparse, acknowledge limitations. If there are clear patterns (times, locations, people), highlight them.`, query, contextBuf.String())

	// Use text analyzer if available, otherwise fall back to VLM
	analyzerToUse := s.textAnalyzer
	if analyzerToUse == nil {
		analyzerToUse = s.analyzer
	}

	result, err := analyzerToUse.Analyze(ctx, nil, prompt)
	if err != nil {
		return "", err
	}
	if raw, ok := result.Attributes["raw_description"].(string); ok {
		return raw, nil
	}
	return result.RawResponse, nil
}

func main() {
	cfg, err := config.LoadFromEnv()
	if err != nil {
		slog.Error("Failed to load config", "error", err)
		os.Exit(1)
	}

	// pq.QuoteLiteral defensively escapes the password (install.sh generates
	// alphanumeric-only passwords today).
	dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s", cfg.Storage.Host, cfg.Storage.Port, cfg.Storage.Username, pq.QuoteLiteral(cfg.Storage.Password), cfg.Storage.Database, cfg.Storage.SSLMode)
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

	analyzer, err := plugin.GetAnalysis(cfg.Analysis.Plugin)
	if err != nil {
		slog.Error("Failed to get analysis plugin", "error", err, "plugin", cfg.Analysis.Plugin)
		os.Exit(1)
	}

	if err := analyzer.Initialize(context.Background(), cfg.Analysis.Config.ToPluginConfig()); err != nil {
		slog.Error("Failed to initialize analyzer", "error", err)
		os.Exit(1)
	}
	defer analyzer.Close()

	// Optionally initialize text-only analyzer for query answer generation
	var textAnalyzer plugin.Analyzer
	if cfg.TextAnalysis.Plugin != "" && cfg.TextAnalysis.Config.Endpoint != "" {
		textAnalyzer, err = plugin.GetAnalysis(cfg.TextAnalysis.Plugin)
		if err != nil {
			slog.Warn("Text LLM plugin not available, falling back to VLM", "error", err)
		} else {
			if err := textAnalyzer.Initialize(context.Background(), cfg.TextAnalysis.Config.ToPluginConfig()); err != nil {
				slog.Warn("Text LLM initialization failed, falling back to VLM", "error", err)
				textAnalyzer = nil
			} else {
				defer textAnalyzer.Close()
				slog.Info("Text LLM initialized", "endpoint", cfg.TextAnalysis.Config.Endpoint, "model", cfg.TextAnalysis.Config.ModelPath)
			}
		}
	}

	server := NewServer(db, analyzer, textAnalyzer, cfg.Service.Port+2)

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
	slog.Info("Query Engine stopped")
}
