from flask import Flask, request, jsonify
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    comments = data.get("comments", [])
    results = []
    sid = SentimentIntensityAnalyzer()
    sentiment_pipeline = pipeline("sentiment-analysis")

    for comment in comments:
        blob = TextBlob(comment)
        sentiment = blob.sentiment.polarity  # range: [-1.0, 1.0]
        results.append({"comment": comment, "sentimentTB": sentiment, "sentimentNLTK": sid.polarity_scores(comment), "sentimentHF": sentiment_pipeline(comment)})

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)
