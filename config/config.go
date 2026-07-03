// Package config provides configuration loading and validation.
package config

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"

	"rock-cluster/pkg/plugin/analysis"
	"rock-cluster/pkg/plugin/detection"
)

// Config is the root configuration structure.
type Config struct {
	Detection    DetectionConfig      `yaml:"detection"`
	Analysis     AnalysisConfig       `yaml:"analysis"`
	TextAnalysis TextAnalysisConfig   `yaml:"text_analysis"`
	Storage      StorageConfig        `yaml:"storage"`
	Service      ServiceConfig        `yaml:"service"`
}

// DetectionConfig contains detection plugin settings.
type DetectionConfig struct {
	Plugin string                `yaml:"plugin"`
	Config DetectionPluginConfig `yaml:"config"`
}

// DetectionPluginConfig holds detection plugin settings.
type DetectionPluginConfig struct {
	ModelPath           string  `yaml:"model_path"`
	ModelType           string  `yaml:"model_type"`
	InputWidth          int     `yaml:"input_width"`
	InputHeight         int     `yaml:"input_height"`
	ConfidenceThreshold float32 `yaml:"confidence_threshold"`
	NMSThreshold        float32 `yaml:"nms_threshold"`
	DeviceID            int     `yaml:"device_id,omitempty"`
	Threads             int     `yaml:"threads,omitempty"`
}

// ToPluginConfig converts to detection.Config.
func (c *DetectionPluginConfig) ToPluginConfig() detection.Config {
	return detection.Config{
		ModelPath:           c.ModelPath,
		ModelType:           c.ModelType,
		InputSize:           [2]int{c.InputWidth, c.InputHeight},
		ConfidenceThreshold: c.ConfidenceThreshold,
		NMSThreshold:        c.NMSThreshold,
		DeviceID:            c.DeviceID,
		Threads:             c.Threads,
	}
}

// AnalysisConfig contains analysis plugin settings.
type AnalysisConfig struct {
	Plugin string               `yaml:"plugin"`
	Config AnalysisPluginConfig `yaml:"config"`
}

// AnalysisPluginConfig holds analysis plugin settings.
type AnalysisPluginConfig struct {
	Endpoint    string  `yaml:"endpoint"`
	ModelPath   string  `yaml:"model_path"`
	MMProjPath  string  `yaml:"mmproj_path"`
	APIKey      string  `yaml:"api_key,omitempty"`
	MaxTokens   int     `yaml:"max_tokens"`
	Temperature float32 `yaml:"temperature"`
	TimeoutSec  int     `yaml:"timeout_sec"`
	ModelName   string  `yaml:"model_name,omitempty"`
}

// ToPluginConfig converts to analysis.Config.
func (c *AnalysisPluginConfig) ToPluginConfig() analysis.Config {
	return analysis.Config{
		Endpoint:    c.Endpoint,
		ModelPath:   c.ModelPath,
		MMProjPath:  c.MMProjPath,
		APIKey:      c.APIKey,
		MaxTokens:   c.MaxTokens,
		Temperature: c.Temperature,
		TimeoutSec:  c.TimeoutSec,
		ModelName:   c.ModelName,
	}
}

// TextAnalysisConfig contains text-only analysis plugin settings.
type TextAnalysisConfig struct {
	Plugin string                    `yaml:"plugin"`
	Config TextAnalysisPluginConfig  `yaml:"config"`
}

// TextAnalysisPluginConfig holds text-only analysis plugin settings.
type TextAnalysisPluginConfig struct {
	Endpoint    string  `yaml:"endpoint"`
	ModelPath   string  `yaml:"model_path"`
	APIKey      string  `yaml:"api_key,omitempty"`
	MaxTokens   int     `yaml:"max_tokens"`
	Temperature float32 `yaml:"temperature"`
	TimeoutSec  int     `yaml:"timeout_sec"`
	ModelName   string  `yaml:"model_name,omitempty"`
}

// ToPluginConfig converts to analysis.Config for text-only usage (MMProjPath is empty).
func (c *TextAnalysisPluginConfig) ToPluginConfig() analysis.Config {
	return analysis.Config{
		Endpoint:    c.Endpoint,
		ModelPath:   c.ModelPath,
		MMProjPath:  "",
		APIKey:      c.APIKey,
		MaxTokens:   c.MaxTokens,
		Temperature: c.Temperature,
		TimeoutSec:  c.TimeoutSec,
		ModelName:   c.ModelName,
	}
}

// StorageConfig contains database settings.
type StorageConfig struct {
	Plugin   string `yaml:"plugin"`
	Host     string `yaml:"host"`
	Port     int    `yaml:"port"`
	Database string `yaml:"database"`
	Username string `yaml:"username"`
	Password string `yaml:"password"`
	SSLMode  string `yaml:"ssl_mode"`
}

// ServiceConfig contains service-level settings.
type ServiceConfig struct {
	Port     int    `yaml:"port"`
	LogLevel string `yaml:"log_level"`
	DataDir  string `yaml:"data_dir"`
	ModelDir string `yaml:"model_dir"`
	NATSURL  string `yaml:"nats_url"`
}

// Load reads and validates configuration from a file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	cfg.setDefaults()

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	return &cfg, nil
}

func (c *Config) setDefaults() {
	if c.Detection.Config.InputWidth == 0 {
		c.Detection.Config.InputWidth = 640
	}
	if c.Detection.Config.InputHeight == 0 {
		c.Detection.Config.InputHeight = 640
	}
	if c.Detection.Config.ConfidenceThreshold == 0 {
		c.Detection.Config.ConfidenceThreshold = 0.5
	}
	if c.Detection.Config.NMSThreshold == 0 {
		c.Detection.Config.NMSThreshold = 0.45
	}
	if c.Detection.Config.Threads == 0 {
		c.Detection.Config.Threads = 4
	}
	if c.Analysis.Config.MaxTokens == 0 {
		c.Analysis.Config.MaxTokens = 256
	}
	if c.Analysis.Config.Temperature == 0 {
		c.Analysis.Config.Temperature = 0.1
	}
	if c.Analysis.Config.TimeoutSec == 0 {
		c.Analysis.Config.TimeoutSec = 120
	}
	if c.TextAnalysis.Config.MaxTokens == 0 {
		c.TextAnalysis.Config.MaxTokens = 512
	}
	if c.TextAnalysis.Config.Temperature == 0 {
		c.TextAnalysis.Config.Temperature = 0.3
	}
	if c.TextAnalysis.Config.TimeoutSec == 0 {
		c.TextAnalysis.Config.TimeoutSec = 60
	}
	if c.Storage.Plugin == "" {
		c.Storage.Plugin = "postgres"
	}
	if c.Storage.Host == "" {
		c.Storage.Host = "localhost"
	}
	if c.Storage.Port == 0 {
		c.Storage.Port = 5432
	}
	if c.Storage.SSLMode == "" {
		c.Storage.SSLMode = "disable"
	}
	if c.Service.LogLevel == "" {
		c.Service.LogLevel = "info"
	}
}

func (c *Config) Validate() error {
	if c.Detection.Plugin == "" {
		return fmt.Errorf("detection.plugin is required")
	}
	if c.Detection.Config.ModelPath == "" {
		return fmt.Errorf("detection.config.model_path is required")
	}
	if c.Detection.Plugin != "api" {
		if _, err := os.Stat(c.Detection.Config.ModelPath); os.IsNotExist(err) {
			return fmt.Errorf("model not found: %s", c.Detection.Config.ModelPath)
		}
	}
	if c.Analysis.Plugin == "" {
		return fmt.Errorf("analysis.plugin is required")
	}
	if c.Analysis.Plugin == "llamacpp" && c.Analysis.Config.ModelPath != "" {
		if _, err := os.Stat(c.Analysis.Config.ModelPath); os.IsNotExist(err) {
			return fmt.Errorf("model not found: %s", c.Analysis.Config.ModelPath)
		}
	}
	if (c.Analysis.Plugin == "anthropic" || c.Analysis.Plugin == "openai") && c.Analysis.Config.APIKey == "" {
		return fmt.Errorf("analysis.config.api_key is required for %s", c.Analysis.Plugin)
	}
	if c.Storage.Plugin == "postgres" && (c.Storage.Host == "" || c.Storage.Database == "") {
		return fmt.Errorf("storage.host and storage.database are required for postgres")
	}
	// Require password for non-local postgres (localhost trust auth is common in dev).
	if c.Storage.Plugin == "postgres" &&
		c.Storage.Host != "localhost" && c.Storage.Host != "127.0.0.1" &&
		c.Storage.Password == "" {
		return fmt.Errorf("storage.password is required for postgres when host is not localhost/127.0.0.1")
	}
	// Refuse sslmode=disable for non-local hosts unless explicitly allowed.
	if c.Storage.Plugin == "postgres" &&
		c.Storage.Host != "localhost" && c.Storage.Host != "127.0.0.1" &&
		c.Storage.SSLMode == "disable" &&
		os.Getenv("ALLOW_INSECURE_DB") != "1" {
		return fmt.Errorf("storage.ssl_mode=disable is unsafe for non-local host %q; set ssl_mode to require/verify-full or set ALLOW_INSECURE_DB=1 to opt out", c.Storage.Host)
	}
	return nil
}

// LoadFromEnv loads configuration from environment variables.
func LoadFromEnv() (*Config, error) {
	cfg := &Config{
		Detection: DetectionConfig{
			Plugin: getEnv("DETECTION_PLUGIN", "rknn"),
			Config: DetectionPluginConfig{
				ModelPath:           getEnv("WORKER_MODEL_PATH", "/models/yolov5s_int8.rknn"),
				ModelType:           "rknn",
				InputWidth:          640,
				InputHeight:         640,
				ConfidenceThreshold: getEnvFloat("CONFIDENCE_THRESHOLD", 0.5),
				NMSThreshold:        0.45,
				Threads:             4,
			},
		},
		Analysis: AnalysisConfig{
			Plugin: getEnv("ANALYSIS_PLUGIN", "llamacpp"),
			Config: AnalysisPluginConfig{
				Endpoint:    getEnv("LLAMA_SERVER_URL", "http://localhost:8888"),
				MaxTokens:   256,
				Temperature: 0.1,
				TimeoutSec:  120,
			},
		},
		TextAnalysis: TextAnalysisConfig{
			Plugin: getEnv("TEXT_ANALYSIS_PLUGIN", "llamacpp-text"),
			Config: TextAnalysisPluginConfig{
				Endpoint:    getEnv("LLAMA_TEXT_SERVER_URL", "http://localhost:8889"),
				ModelPath:   getEnv("TEXT_MODEL_PATH", "/var/lib/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf"),
				MaxTokens:   512,
				Temperature: 0.3,
				TimeoutSec:  60,
			},
		},
		Storage: StorageConfig{
			Plugin:   "postgres",
			Host:     getEnv("DB_HOST", "localhost"),
			Port:     getEnvInt("DB_PORT", 5432),
			Database: getEnv("DB_NAME", "camera_brain"),
			Username: getEnv("DB_USER", "camera_brain"),
			Password: getEnv("DB_PASSWORD", ""),
			SSLMode:  "disable",
		},
		Service: ServiceConfig{
			Port:     getEnvInt("PORT", 8080),
			LogLevel: "info",
			DataDir:  getEnv("DATA_DIR", "/var/lib/camera-brain"),
			ModelDir: getEnv("MODEL_DIR", "/var/lib/camera-brain/models"),
			NATSURL:  getEnv("NATS_URL", "nats://localhost:4222"),
		},
	}
	return cfg, cfg.Validate()
}

// getEnv returns the env var value or defaultVal if the variable is unset OR
// set to an empty string. An explicitly empty value (e.g. "DB_HOST=") is
// intentionally treated as unset: empty env values are rarely meaningful and
// treating them as unset simplifies operator workflows where variables are
// cleared rather than unset. Use os.LookupEnv directly if you need to
// distinguish "unset" from "set to empty".
func getEnv(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
	if v := os.Getenv(key); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			slog.Warn("invalid integer env var, using default", "key", key, "value", v, "default", defaultVal, "error", err)
			return defaultVal
		}
		return n
	}
	return defaultVal
}

func getEnvFloat(key string, defaultVal float32) float32 {
	if v := os.Getenv(key); v != "" {
		f, err := strconv.ParseFloat(v, 32)
		if err != nil {
			slog.Warn("invalid float env var, using default", "key", key, "value", v, "default", defaultVal, "error", err)
			return defaultVal
		}
		return float32(f)
	}
	return defaultVal
}
