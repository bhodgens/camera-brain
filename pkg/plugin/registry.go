// Package plugin provides a registry for detection and analysis plugins.
package plugin

import (
	"fmt"
	"log/slog"
	"sync"

	"rock-cluster/pkg/plugin/analysis"
	"rock-cluster/pkg/plugin/detection"
)

// Type aliases for convenience
type Analyzer = analysis.Analyzer
type AnalysisResult = analysis.AnalysisResult
type AnalysisConfig = analysis.Config
type AnalysisPluginInfo = analysis.PluginInfo

// DetectionPluginFactory creates a new Detector instance.
type DetectionPluginFactory func() detection.Detector

// AnalysisPluginFactory creates a new Analyzer instance.
type AnalysisPluginFactory func() analysis.Analyzer

// Registry holds available plugins.
type Registry struct {
	mu               sync.RWMutex
	detectionPlugins map[string]DetectionPluginFactory
	analysisPlugins  map[string]AnalysisPluginFactory
}

// NewRegistry creates a new plugin registry.
func NewRegistry() *Registry {
	return &Registry{
		detectionPlugins: make(map[string]DetectionPluginFactory),
		analysisPlugins:  make(map[string]AnalysisPluginFactory),
	}
}

// Global registry instance
var globalRegistry = NewRegistry()

// DefaultRegistry returns the global plugin registry.
func DefaultRegistry() *Registry {
	return globalRegistry
}

// RegisterDetection adds a detection plugin to the registry.
// If a different factory is already registered under this name, the
// registration is logged as a warning but allowed (override). This keeps
// init.go/init_rknn.go callers working without API churn while surfacing
// accidental misconfiguration. Same-factory reregistration is silent.
func (r *Registry) RegisterDetection(name string, factory DetectionPluginFactory) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.detectionPlugins[name]; ok {
		if fmt.Sprintf("%p", existing) != fmt.Sprintf("%p", factory) {
			slog.Warn("duplicate detection plugin registration, overriding", "plugin", name, "action", "override")
		}
	}
	r.detectionPlugins[name] = factory
}

// RegisterAnalysis adds an analysis plugin to the registry.
// See RegisterDetection for the duplicate-handling rationale.
func (r *Registry) RegisterAnalysis(name string, factory AnalysisPluginFactory) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.analysisPlugins[name]; ok {
		if fmt.Sprintf("%p", existing) != fmt.Sprintf("%p", factory) {
			slog.Warn("duplicate analysis plugin registration, overriding", "plugin", name, "action", "override")
		}
	}
	r.analysisPlugins[name] = factory
}

// GetDetection creates a detection plugin by name.
func (r *Registry) GetDetection(name string) (detection.Detector, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	factory, exists := r.detectionPlugins[name]
	if !exists {
		// Inline collection: do NOT call r.AvailableDetections() here —
		// it re-acquires r.mu.RLock() and Go's RWMutex is not reentrant.
		// If a writer is waiting between this RLock and the nested one,
		// we deadlock.
		names := make([]string, 0, len(r.detectionPlugins))
		for n := range r.detectionPlugins {
			names = append(names, n)
		}
		return nil, fmt.Errorf("unknown detection plugin: %s (available: %v)", name, names)
	}
	return factory(), nil
}

// GetAnalysis creates an analysis plugin by name.
func (r *Registry) GetAnalysis(name string) (analysis.Analyzer, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	factory, exists := r.analysisPlugins[name]
	if !exists {
		// Inline collection — see GetDetection for the reentrancy rationale.
		names := make([]string, 0, len(r.analysisPlugins))
		for n := range r.analysisPlugins {
			names = append(names, n)
		}
		return nil, fmt.Errorf("unknown analysis plugin: %s (available: %v)", name, names)
	}
	return factory(), nil
}

// AvailableDetections returns a list of registered detection plugin names.
func (r *Registry) AvailableDetections() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.detectionPlugins))
	for name := range r.detectionPlugins {
		names = append(names, name)
	}
	return names
}

// AvailableAnalysis returns a list of registered analysis plugin names.
func (r *Registry) AvailableAnalysis() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.analysisPlugins))
	for name := range r.analysisPlugins {
		names = append(names, name)
	}
	return names
}

// Convenience functions for global registry

// RegisterDetection adds a detection plugin to the global registry.
func RegisterDetection(name string, factory DetectionPluginFactory) {
	globalRegistry.RegisterDetection(name, factory)
}

// RegisterAnalysis adds an analysis plugin to the global registry.
func RegisterAnalysis(name string, factory AnalysisPluginFactory) {
	globalRegistry.RegisterAnalysis(name, factory)
}

// GetDetection creates a detection plugin from the global registry.
func GetDetection(name string) (detection.Detector, error) {
	return globalRegistry.GetDetection(name)
}

// GetAnalysis creates an analysis plugin from the global registry.
func GetAnalysis(name string) (analysis.Analyzer, error) {
	return globalRegistry.GetAnalysis(name)
}
