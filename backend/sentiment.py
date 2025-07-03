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
    overallTB, overallNLTK, overallHF = 0, 0, 0
    for comment in comments:
        blob = TextBlob(comment)
        sentimentTB = blob.sentiment.polarity  # range: [-1.0, 1.0]
        sentimentNLTK = sid.polarity_scores(comment)['compound']
        sentimentHF = sentiment_pipeline(comment)[0]['score']
        overallTB += sentimentTB
        overallNLTK += sentimentNLTK
        overallHF += sentimentHF
        results.append({"comment": comment, "sentimentTB": sentimentTB, "sentimentNLTK": sentimentNLTK, "sentimentHF": sentimentHF})

    return jsonify({
        "results": results,
        "meanScores": {
            "meanTB": overallTB / len(comments),
            "meanNLTK": overallNLTK / len(comments),
            "meanHF": overallHF / len(comments)
        }
    })

if __name__ == '__main__':
    app.run(port=5000)
