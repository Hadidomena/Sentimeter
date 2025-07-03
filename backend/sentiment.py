from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import Counter

# Load model once at startup
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    comments = data.get("comments", [])

    results = []
    label_counts = Counter()

    for comment in comments:
        if not comment.strip():
            continue  # Skip empty comments

        try:
            analysis = sentiment_pipeline(comment)[0]  # {'label': 'positive', 'score': 0.98}
            label = analysis['label'].lower()
            score = analysis['score']

            label_counts[label] += 1
            results.append({
                "comment": comment,
                "label": label,
                "confidence": round(score, 3)
            })
        except Exception as e:
            results.append({
                "comment": comment,
                "label": "error",
                "confidence": 0,
                "error": str(e)
            })

    total = sum(label_counts.values())
    class_distribution = {
        label: round(count / total, 3) for label, count in label_counts.items()
    }

    sentiment_score = (
        label_counts.get("positive", 0) - label_counts.get("negative", 0)
    ) / total if total > 0 else 0

    return jsonify({
        "results": results,
        "labelCounts": dict(label_counts),
        "classDistribution": class_distribution,
        "weightedScore": round(sentiment_score, 3)
    })

if __name__ == '__main__':
    app.run(port=5000)
