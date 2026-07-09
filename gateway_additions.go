// handleObservations handles POST /observations to record detections.
func (s *Server) handleObservations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	type ObservationRequest struct {
		CameraID   string                 `json:"camera_id"`
		WorkerID   string                 `json:"worker_id"`
		DetectedAt string                 `json:"detected_at"`
		ClassName  string                 `json:"class_name"`
		Confidence float32                `json:"confidence"`
		BBox       [4]int                 `json:"bbox"`
		CropPath   string                 `json:"crop_path"`
		Attributes map[string]interface{} `json:"attributes,omitempty"`
	}

	var req ObservationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	if req.CameraID == "" || req.ClassName == "" {
		http.Error(w, "camera_id and class_name are required", http.StatusBadRequest)
		return
	}

	detectedAt, err := time.Parse(time.RFC3339, req.DetectedAt)
	if err != nil {
		detectedAt = time.Now().UTC()
	}

	// Get camera ID from database
	var cameraUUID string
	err = s.db.QueryRow("SELECT id FROM cameras WHERE name = $1", req.CameraID).Scan(&cameraUUID)
	if err != nil {
		http.Error(w, "camera not found: "+req.CameraID, http.StatusNotFound)
		return
	}

	bboxJSON, _ := json.Marshal(req.BBox)
	id := uuid.New().String()

	var attrs json.RawMessage
	if req.Attributes != nil {
		attrs, _ = json.Marshal(req.Attributes)
	} else {
		attrs = json.RawMessage("null")
	}
	
	_, err = s.db.Exec(
		"INSERT INTO observations (id, camera_id, detected_at, type, class_name, confidence, bbox, crop_path, attributes) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
		id, cameraUUID, detectedAt, "detection", req.ClassName, req.Confidence, bboxJSON, req.CropPath, attrs)
	
	if err != nil {
		http.Error(w, fmt.Sprintf("database error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"status": "recorded", "id": id})
}
