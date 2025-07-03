package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

func getComments(postURL string) []string {
	req, _ := http.NewRequest("GET", postURL, nil)
	req.Header.Set("User-Agent", "go:reddit.comment.fetcher:v1.0 (by /u/yourusername)")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	var jsonResponse []interface{}
	if err := json.NewDecoder(resp.Body).Decode(&jsonResponse); err != nil {
		log.Fatal(err)
	}

	// Reddit returns post info in jsonResponse[0], and comments in jsonResponse[1]
	data := jsonResponse[1].(map[string]interface{})
	comments := data["data"].(map[string]interface{})["children"].([]interface{})

	var result []string

	for _, item := range comments {
		commentData := item.(map[string]interface{})["data"].(map[string]interface{})
		body, ok := commentData["body"].(string)
		if ok {
			result = append(result, body)
		}
	}

	return result
}
func analyzeSentiment(comments []string) ([]map[string]interface{}, map[string]float64) {
	requestBody, err := json.Marshal(map[string]interface{}{
		"comments": comments,
	})
	if err != nil {
		log.Fatal(err)
	}

	var resp *http.Response
	maxRetries := 5
	retryDelay := 2 * time.Second

	// Retry loop
	for i := range maxRetries {
		resp, err = http.Post("http://localhost:5000/analyze", "application/json", bytes.NewBuffer(requestBody))
		if err == nil {
			break
		}

		log.Printf("Attempt %d: Python service not ready, retrying in %v...\n", i+1, retryDelay)
		time.Sleep(retryDelay)
	}

	if err != nil {
		log.Fatalf("Failed to reach Python service after %d attempts: %v", maxRetries, err)
	}
	defer resp.Body.Close()

	var response map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		log.Fatal("Error decoding response:", err)
	}

	results := response["results"].([]interface{})
	var commentResults []map[string]interface{}
	for _, result := range results {
		commentResults = append(commentResults, result.(map[string]interface{}))
	}

	meanScores := make(map[string]float64)
	if meanData, ok := response["meanScores"].(map[string]interface{}); ok {
		for key, value := range meanData {
			if score, ok := value.(float64); ok {
				meanScores[key] = score
			}
		}
	}

	return commentResults, meanScores
}

func analyzeHandler(w http.ResponseWriter, r *http.Request) {
	// Allow cross-origin requests
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}

	var requestData map[string]string
	if err := json.Unmarshal(body, &requestData); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	url, ok := requestData["url"]
	if !ok {
		http.Error(w, `"url" field is required`, http.StatusBadRequest)
		return
	}

	url = url + ".json"
	comments := getComments(url)
	if comments == nil {
		http.Error(w, "Failed to fetch comments", http.StatusInternalServerError)
		return
	}

	results, meanScores := analyzeSentiment(comments)
	response := map[string]interface{}{
		"comments":     results,
		"meanScores":   meanScores,
		"commentCount": len(comments),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	http.HandleFunc("/analyze-comments", analyzeHandler)
	fmt.Println("Server running on http://localhost:3001...")
	log.Fatal(http.ListenAndServe(":3001", nil))
}
