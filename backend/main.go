package main

import (
	"bytes"
	"encoding/json"
	"fmt"
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
func analyzeSentiment(comments []string) []map[string]interface{} {
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

	var results []map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		log.Fatal("Error decoding response:", err)
	}

	return results
}

func main() {
	url := "https://www.reddit.com/r/arknights/comments/1ldfb2x/the_prettiest_warfarin_drawing_ive_ever_seen_art/.json"
	comments := getComments(url)
	for i, comment := range comments {
		fmt.Println(i, comment)
	}

	analyzed := analyzeSentiment(comments)
	for comment, i := range analyzed {
		fmt.Println(comment, i)
	}
}
