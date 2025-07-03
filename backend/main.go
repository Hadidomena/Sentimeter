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
func analyzeSentiment(comments []string) (
	[]map[string]interface{},
	map[string]int,
	map[string]float64,
	float64,
) {
	requestBody, err := json.Marshal(map[string]interface{}{
		"comments": comments,
	})
	if err != nil {
		log.Fatal(err)
	}

	var resp *http.Response
	maxRetries := 5
	retryDelay := 2 * time.Second

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

	// Extract results
	var commentResults []map[string]interface{}
	if results, ok := response["results"].([]interface{}); ok {
		for _, result := range results {
			if resMap, ok := result.(map[string]interface{}); ok {
				commentResults = append(commentResults, resMap)
			}
		}
	}

	// Extract labelCounts
	labelCounts := make(map[string]int)
	if raw, ok := response["labelCounts"].(map[string]interface{}); ok {
		for key, val := range raw {
			if countFloat, ok := val.(float64); ok {
				labelCounts[key] = int(countFloat)
			}
		}
	}

	// Extract classDistribution
	classDistribution := make(map[string]float64)
	if raw, ok := response["classDistribution"].(map[string]interface{}); ok {
		for key, val := range raw {
			if pct, ok := val.(float64); ok {
				classDistribution[key] = pct
			}
		}
	}

	// Extract weightedScore
	var weightedScore float64
	if score, ok := response["weightedScore"].(float64); ok {
		weightedScore = score
	}

	return commentResults, labelCounts, classDistribution, weightedScore
}

func analyzeHandler(w http.ResponseWriter, r *http.Request) {
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

	commentResults, labelCounts, classDistribution, weightedScore := analyzeSentiment(comments)

	response := map[string]interface{}{
		"results":           commentResults,
		"labelCounts":       labelCounts,
		"classDistribution": classDistribution,
		"weightedScore":     weightedScore,
		"commentCount":      len(comments),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	http.HandleFunc("/analyze-comments", analyzeHandler)
	fmt.Println("Server running on http://localhost:3001...")
	log.Fatal(http.ListenAndServe(":3001", nil))
}
