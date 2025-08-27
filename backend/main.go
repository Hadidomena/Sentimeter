package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

type RedditPost struct {
	Title string `json:"title"`
	Text  string `json:"selftext"`
}

func getPostAndComments(postURL string) (RedditPost, []string) {
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

	// Extract post data from jsonResponse[0]
	postData := jsonResponse[0].(map[string]interface{})
	postChildren := postData["data"].(map[string]interface{})["children"].([]interface{})
	postInfo := postChildren[0].(map[string]interface{})["data"].(map[string]interface{})

	post := RedditPost{
		Title: postInfo["title"].(string),
	}

	// Handle selftext (can be nil)
	if selftext, ok := postInfo["selftext"].(string); ok {
		post.Text = selftext
	}

	// Extract comments from jsonResponse[1]
	commentsData := jsonResponse[1].(map[string]interface{})
	comments := commentsData["data"].(map[string]interface{})["children"].([]interface{})

	var result []string
	for _, item := range comments {
		commentData := item.(map[string]interface{})["data"].(map[string]interface{})
		body, ok := commentData["body"].(string)
		if ok && body != "[deleted]" && body != "[removed]" {
			result = append(result, body)
		}
	}

	return post, result
}

func analyzeSentimentOnly(comments []string) (
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

	// Extract results (same as before)
	var commentResults []map[string]interface{}
	if results, ok := response["results"].([]interface{}); ok {
		for _, result := range results {
			if resMap, ok := result.(map[string]interface{}); ok {
				commentResults = append(commentResults, resMap)
			}
		}
	}

	labelCounts := make(map[string]int)
	if raw, ok := response["labelCounts"].(map[string]interface{}); ok {
		for key, val := range raw {
			if countFloat, ok := val.(float64); ok {
				labelCounts[key] = int(countFloat)
			}
		}
	}

	classDistribution := make(map[string]float64)
	if raw, ok := response["classDistribution"].(map[string]interface{}); ok {
		for key, val := range raw {
			if pct, ok := val.(float64); ok {
				classDistribution[key] = pct
			}
		}
	}

	var weightedScore float64
	if score, ok := response["weightedScore"].(float64); ok {
		weightedScore = score
	}

	return commentResults, labelCounts, classDistribution, weightedScore
}

func analyzeAdvancedSentiment(comments []string) map[string]interface{} {
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
		resp, err = http.Post("http://localhost:5000/analyze_advanced", "application/json", bytes.NewBuffer(requestBody))
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

	return response
}

func analyzeEmotionSentiment(comments []string) map[string]interface{} {
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
		resp, err = http.Post("http://localhost:5000/analyze_emotion", "application/json", bytes.NewBuffer(requestBody))
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

	return response
}

func analyzeWithAgreement(post RedditPost, comments []string, method string) map[string]interface{} {
	// Combine title and text for post analysis
	postText := post.Title
	if post.Text != "" {
		postText += " " + post.Text
	}

	requestBody, err := json.Marshal(map[string]interface{}{
		"post_text": postText,
		"comments":  comments,
		"method":    method,
	})
	if err != nil {
		log.Fatal(err)
	}

	var resp *http.Response
	maxRetries := 5
	retryDelay := 2 * time.Second

	for i := range maxRetries {
		resp, err = http.Post("http://localhost:5000/analyze_with_agreement", "application/json", bytes.NewBuffer(requestBody))
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

	return response
}

func detectSarcasm(comments []string) map[string]interface{} {
	requestBody, err := json.Marshal(map[string]interface{}{
		"texts": comments,
	})
	if err != nil {
		log.Fatal(err)
	}

	var resp *http.Response
	maxRetries := 5
	retryDelay := 2 * time.Second

	for i := range maxRetries {
		resp, err = http.Post("http://localhost:5000/detect_sarcasm", "application/json", bytes.NewBuffer(requestBody))
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

	return response
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

	// Check if URL already ends with .json
	if !strings.HasSuffix(url, ".json") {
		url = url + ".json"
	}

	post, comments := getPostAndComments(url)
	if comments == nil {
		http.Error(w, "Failed to fetch comments", http.StatusInternalServerError)
		return
	}

	// Get analysis method from request (default to "sentiment_only")
	analysisMethod := requestData["analysis_method"]
	if analysisMethod == "" {
		analysisMethod = "sentiment_only"
	}

	var response map[string]interface{}

	if analysisMethod == "sentiment_only" {
		// Original sentiment-only analysis
		commentResults, labelCounts, classDistribution, weightedScore := analyzeSentimentOnly(comments)

		response = map[string]interface{}{
			"post": map[string]interface{}{
				"title": post.Title,
				"text":  post.Text,
			},
			"analysis_method":   "sentiment_only",
			"results":           commentResults,
			"labelCounts":       labelCounts,
			"classDistribution": classDistribution,
			"weightedScore":     weightedScore,
			"commentCount":      len(comments),
		}
	} else if analysisMethod == "advanced_sentiment" {
		// Advanced sentiment analysis with sarcasm detection
		analysisResponse := analyzeAdvancedSentiment(comments)

		response = map[string]interface{}{
			"post": map[string]interface{}{
				"title": post.Title,
				"text":  post.Text,
			},
			"analysis_method": "advanced_sentiment",
			"commentCount":    len(comments),
		}

		// Merge the Python response
		for key, value := range analysisResponse {
			response[key] = value
		}
	} else if analysisMethod == "emotion_analysis" {
		// Comprehensive emotion and implied sentiment analysis
		analysisResponse := analyzeEmotionSentiment(comments)

		response = map[string]interface{}{
			"post": map[string]interface{}{
				"title": post.Title,
				"text":  post.Text,
			},
			"analysis_method": "emotion_analysis",
			"commentCount":    len(comments),
		}

		// Merge the Python response
		for key, value := range analysisResponse {
			response[key] = value
		}
	} else {
		// Agreement analysis (contextual or nli)
		method := "contextual"
		if analysisMethod == "nli" {
			method = "nli"
		}

		analysisResponse := analyzeWithAgreement(post, comments, method)

		response = map[string]interface{}{
			"post": map[string]interface{}{
				"title": post.Title,
				"text":  post.Text,
			},
			"analysis_method": analysisMethod,
			"commentCount":    len(comments),
		}

		// Merge the Python response
		for key, value := range analysisResponse {
			response[key] = value
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")

	// Check Python service health
	resp, err := http.Get("http://localhost:5000/health")
	if err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "unhealthy",
			"error":  "Python service unavailable",
		})
		return
	}
	defer resp.Body.Close()

	var pythonHealth map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&pythonHealth); err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "unhealthy",
			"error":  "Failed to decode Python service health",
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":         "healthy",
		"go_service":     "running",
		"python_service": pythonHealth,
	})
}

func main() {
	http.HandleFunc("/analyze-comments", analyzeHandler)
	http.HandleFunc("/health", healthHandler)
	fmt.Println("Server running on http://localhost:3001...")
	log.Fatal(http.ListenAndServe(":3001", nil))
}
