package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// CropUploader uploads detection crops to the gateway.
type CropUploader struct {
	gatewayURL      string
	workerID        string
	httpClient      *http.Client
	localDir        string
	genderDetector  string
	vehicleDetector string
}

// CropResponse is the response from the gateway upload endpoint.
type CropResponse struct {
	CropID    string `json:"crop_id"`
	CropPath  string `json:"crop_path"`
	Success   bool   `json:"success"`
	Message   string `json:"message,omitempty"`
}

// AttributeResult holds detected attributes for an observation.
type AttributeResult struct {
	Gender      *string  `json:"gender,omitempty"`
	GenderConf  *float64 `json:"gender_conf,omitempty"`
	Age         *int     `json:"age,omitempty"`
	VehicleType *string  `json:"vehicle_type,omitempty"`
	Color       *string  `json:"color,omitempty"`
}

// NewCropUploader creates a new crop uploader.
func NewCropUploader(gatewayURL, workerID string) *CropUploader {
	return &CropUploader{
		gatewayURL:      gatewayURL,
		workerID:        workerID,
		httpClient:      &http.Client{Timeout: 30 * time.Second},
		localDir:        "/home/camera-brain/crops",
		genderDetector:  "/home/camera-brain/cmd/gender-detector/detect.py",
		vehicleDetector: "/home/camera-brain/cmd/vehicle-detector/detect.py",
	}
}

// UploadCrop saves a crop locally and optionally uploads to gateway.
func (u *CropUploader) UploadCrop(
	img image.Image,
	bbox [4]int,
	cameraID string,
	className string,
	confidence float32,
) (string, error) {
	// Create local crop directory if needed
	if err := os.MkdirAll(u.localDir, 0755); err != nil {
		return "", fmt.Errorf("mkdir: %w", err)
	}

	// Crop the image
	cropped := cropImage(img, bbox)

	// Generate filename with timestamp
	timestamp := time.Now().Format("20060102_150405.000")
	filename := fmt.Sprintf("%s_%s_%s_%.2f_%s.jpg",
		timestamp,
		cameraID,
		className,
		confidence,
		u.workerID,
	)
	cropPath := filepath.Join(u.localDir, filename)

	// Save locally
	if err := saveJPEG(cropped, cropPath, 85); err != nil {
		return "", fmt.Errorf("save jpeg: %w", err)
	}

	// Detect attributes based on class (async, non-blocking)
	if className == "person" {
		go func() {
			attrs := u.detectPersonAttributes(cropPath)
			if attrs != nil {
				if err := u.postObservation(cameraID, className, confidence, bbox, cropPath, attrs); err != nil {
					fmt.Printf("[Attributes] post with attributes error: %v\n", err)
				}
			}
		}()
		fmt.Printf("[Attributes] person detection queued for: %s\n", cropPath)
	} else if className == "car" || className == "truck" || className == "bus" || className == "motorcycle" {
		go func() {
			attrs := u.detectVehicleAttributes(cropPath)
			if attrs != nil {
				if err := u.postObservation(cameraID, className, confidence, bbox, cropPath, attrs); err != nil {
					fmt.Printf("[Attributes] post with attributes error: %v\n", err)
				}
			}
		}()
		fmt.Printf("[Attributes] vehicle detection queued for: %s\n", cropPath)
	}

	// POST initial observation to gateway (without attributes)
	if err := u.postObservation(cameraID, className, confidence, bbox, cropPath, nil); err != nil {
		// Log but don't fail - local save is primary
		fmt.Printf("Warning: failed to post observation: %v\n", err)
	}

	return cropPath, nil
}

// postObservation sends observation data to the gateway.
func (u *CropUploader) postObservation(
	cameraID string,
	className string,
	confidence float32,
	bbox [4]int,
	cropPath string,
	attributes *AttributeResult,
) error {
	payload := map[string]interface{}{
		"camera_id":   cameraID,
		"worker_id":   u.workerID,
		"class_name":  className,
		"confidence":  confidence,
		"bbox":        bbox[:],
		"crop_path":   cropPath,
		"detected_at": time.Now().UTC(),
	}

	// Add attributes if available
	if attributes != nil {
		if attributes.Gender != nil {
			payload["gender"] = *attributes.Gender
		}
		if attributes.GenderConf != nil {
			payload["gender_conf"] = *attributes.GenderConf
		}
		if attributes.Age != nil {
			payload["age"] = *attributes.Age
		}
		if attributes.VehicleType != nil {
			payload["vehicle_type"] = *attributes.VehicleType
		}
		if attributes.Color != nil {
			payload["color"] = *attributes.Color
		}
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("%s/observations", u.gatewayURL)
	resp, err := u.httpClient.Post(url, "application/json", bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("http post: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("gateway returned %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// detectPersonAttributes calls the Python gender/age detector.
func (u *CropUploader) detectPersonAttributes(cropPath string) *AttributeResult {
	cmd := exec.Command("python3", u.genderDetector, "--crop", cropPath, "--json")
	output, err := cmd.Output()
	if err != nil {
		fmt.Printf("[Attributes] gender detector error: %v\n", err)
		return nil
	}

	var result struct {
		Success    bool    `json:"success"`
		Gender     string  `json:"gender"`
		Age        int     `json:"age"`
		GenderConf float64 `json:"gender_conf"`
	}

	if err := json.Unmarshal(output, &result); err != nil {
		fmt.Printf("[Attributes] parse gender result error: %v\n", err)
		return nil
	}

	if !result.Success || result.Gender == "" {
		return nil
	}

	return &AttributeResult{
		Gender:     &result.Gender,
		GenderConf: &result.GenderConf,
		Age:        &result.Age,
	}
}

// detectVehicleAttributes calls the Python vehicle type detector.
func (u *CropUploader) detectVehicleAttributes(cropPath string) *AttributeResult {
	cmd := exec.Command("python3", u.vehicleDetector, "--crop", cropPath, "--json")
	output, err := cmd.Output()
	if err != nil {
		fmt.Printf("[Attributes] vehicle detector error: %v\n", err)
		return nil
	}

	var result struct {
		Success     bool    `json:"success"`
		VehicleType string  `json:"vehicle_type"`
		Confidence  float64 `json:"confidence"`
		Color       string  `json:"color"`
	}

	if err := json.Unmarshal(output, &result); err != nil {
		fmt.Printf("[Attributes] parse vehicle result error: %v\n", err)
		return nil
	}

	if !result.Success || result.VehicleType == "" {
		return nil
	}

	return &AttributeResult{
		VehicleType: &result.VehicleType,
		Color:       &result.Color,
	}
}

// cropImage extracts a region from the image.
func cropImage(img image.Image, bbox [4]int) image.Image {
	x1, y1, x2, y2 := bbox[0], bbox[1], bbox[2], bbox[3]
	rect := image.Rect(x1, y1, x2, y2)
	return img.(interface {
		SubImage(image.Rectangle) image.Image
	}).SubImage(rect)
}

// saveJPEG saves an image as JPEG with the given quality (1-100).
func saveJPEG(img image.Image, path string, quality int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	opts := &jpeg.Options{Quality: quality}
	return jpeg.Encode(f, img, opts)
}
