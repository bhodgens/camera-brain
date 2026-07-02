// cmd/cbrain/config.go
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

type Config struct {
	DBHost     string
	DBPort     int
	DBName     string
	DBUser     string
	DBPassword string
	QueryURL   string
	GatewayURL string
}

func LoadConfig(path string) (*Config, error) {
	cfg := &Config{
		DBHost:     "localhost",
		DBPort:     5432,
		DBName:     "camera_brain",
		DBUser:     "camera_brain",
		QueryURL:   "http://localhost:8082",
		GatewayURL: "http://localhost:8080",
	}

	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Use defaults from environment or hardcoded
			loadEnvDefaults(cfg)
			return cfg, nil
		}
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key, value := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
		switch key {
		case "DB_HOST":
			cfg.DBHost = value
		case "DB_PORT":
			fmt.Sscanf(value, "%d", &cfg.DBPort)
		case "DB_NAME":
			cfg.DBName = value
		case "DB_USER":
			cfg.DBUser = value
		case "DB_PASSWORD":
			cfg.DBPassword = value
		}
	}

	loadEnvDefaults(cfg)
	return cfg, scanner.Err()
}

func loadEnvDefaults(cfg *Config) {
	if v := os.Getenv("CB_QUERY_URL"); v != "" {
		cfg.QueryURL = v
	}
	if v := os.Getenv("CB_GATEWAY_URL"); v != "" {
		cfg.GatewayURL = v
	}
}
