from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np

# Load models once at startup
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load semantic similarity model
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load NLI model for agreement detection
nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli")

# Load emotion model for nuanced tone analysis
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

app = Flask(__name__)

def get_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    try:
        embeddings = similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except:
        return 0.0

def analyze_implied_sentiment(text, surface_sentiment, surface_confidence):
    """
    Attempts to correct sentiment based on nuance, emotion, and implied meaning.
    Runs after standard sentiment analysis but before sarcasm adjustment.
    """
    try:
        # Get a nuanced emotional profile of the text
        emotion_results = emotion_pipeline(text)[0]
        
        # Convert to a dictionary for easier lookup: {emotion: score}
        emotion_scores = {item['label']: item['score'] for item in emotion_results}
        
        # Logic to handle specific cases:
        
        # 1. Handle "frustrated affection" (e.g., the 3 AM example)
        # High anger + high joy often indicates playful frustration, not true negativity.
        if emotion_scores.get('anger', 0) > 0.5 and emotion_scores.get('joy', 0) > 0.3:
            # Override: the negativity is not directed at the subject
            return {
                "label": "positive",
                "confidence": surface_confidence * 0.8, # Slightly lower confidence due to override
                "implied_reason": "frustrated_affection",
                "emotion_profile": emotion_scores
            }
        
        # 2. Handle intense positive language misclassified as negative (e.g., "sick-as-hell")
        # Check for intense positive words in a text classified as negative
        intense_positive_indicators = ["sick-as-hell", "badass", "awesome", "amazing", "incredible", "stunning", "epic", "legendary"]
        if surface_sentiment == 'negative' and any(indicator in text.lower() for indicator in intense_positive_indicators):
            # Also check for high 'surprise' or 'joy' emotion from the model
            if emotion_scores.get('surprise', 0) > 0.4 or emotion_scores.get('joy', 0) > 0.4:
                return {
                    "label": "positive",
                    "confidence": surface_confidence * 0.9,
                    "implied_reason": "intense_positive_language",
                    "emotion_profile": emotion_scores
                }
        
        # 3. Handle innuendo and implied criticism (e.g., the "assets" example)
        # This is harder. We look for a mismatch between neutral emotion and negative sentiment.
        # Innuendo is often delivered with a "straight face".
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        if surface_sentiment == 'negative' and dominant_emotion in ['neutral', 'surprise'] and emotion_scores['anger'] < 0.3:
            # This might be sarcasm or innuendo that our simple detector missed.
            # We won't override sentiment yet, but will flag it for the sarcasm detector.
            return {
                "label": surface_sentiment,
                "confidence": surface_confidence,
                "implied_reason": "possible_innuendo",
                "emotion_profile": emotion_scores
            }
        
        # 4. Default case: no override needed
        return {
            "label": surface_sentiment,
            "confidence": surface_confidence,
            "implied_reason": "literal",
            "emotion_profile": emotion_scores
        }
        
    except Exception as e:
        # If anything fails, return the original analysis
        return {
            "label": surface_sentiment,
            "confidence": surface_confidence,
            "implied_reason": "error",
            "error": str(e)
        }

def detect_sarcasm_innuendo(text, confidence_threshold=0.6):
    """Detect if text contains sarcasm or innuendo using simple heuristics"""
    try:
        # Simple rule-based sarcasm detection
        sarcasm_indicators = [
            "yeah right", "sure thing", "oh great", "just perfect", "wonderful",
            "exactly what", "brilliant", "fantastic", "amazing", "terrific",
            "wow", "really", "seriously", "of course", "as if"
        ]
        
        text_lower = text.lower()
        sarcasm_score = 0
        
        # Check for sarcasm indicators
        for indicator in sarcasm_indicators:
            if indicator in text_lower:
                sarcasm_score += 0.3
        
        # Check for excessive punctuation (!!!, ???)
        if "!!!" in text or "???" in text:
            sarcasm_score += 0.2
            
        # Check for ALL CAPS words
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        if caps_words > 0:
            sarcasm_score += min(caps_words * 0.1, 0.3)
            
        # Check for quotation marks around ironic phrases
        if '"' in text:
            sarcasm_score += 0.2
            
        is_sarcastic = sarcasm_score >= confidence_threshold
        
        return {
            'is_sarcastic': is_sarcastic,
            'confidence': round(min(sarcasm_score, 1.0), 3),
            'label': 'sarcastic' if is_sarcastic else 'not_sarcastic'
        }
    except Exception as e:
        return {'is_sarcastic': False, 'confidence': 0.0, 'error': str(e)}

def analyze_sentiment_with_innuendo(text):
    """Analyze sentiment with sarcasm/innuendo awareness"""
    try:
        # 1. Get surface sentiment
        sentiment_result = sentiment_pipeline(text)[0]
        surface_sentiment = sentiment_result['label'].lower()
        surface_confidence = sentiment_result['score']
        
        # 2. Analyze for implied meaning and nuance
        implied_analysis = analyze_implied_sentiment(text, surface_sentiment, surface_confidence)
        final_sentiment = implied_analysis['label']
        final_confidence = implied_analysis['confidence']
        
        # 3. Detect sarcasm based on the final sentiment interpretation
        sarcasm_result = detect_sarcasm_innuendo(text)
        is_sarcastic = sarcasm_result['is_sarcastic']
        
        # 4. Adjust interpretation based on sarcasm detection
        # Only invert if the implied analysis wasn't already a correction
        if is_sarcastic and implied_analysis['implied_reason'] == 'literal':
            # Invert sentiment for sarcastic content
            actual_sentiment = 'positive' if final_sentiment == 'negative' else 'negative'
            confidence = min(sarcasm_result['confidence'], final_confidence)
            return {
                "label": actual_sentiment,
                "confidence": round(confidence, 3),
                "is_sarcastic": True,
                "surface_sentiment": surface_sentiment,
                "surface_confidence": round(surface_confidence, 3),
                "sarcasm_confidence": sarcasm_result['confidence'],
                "implied_analysis": implied_analysis
            }
        else:
            # Return the result of the implied analysis, which may have already corrected the sentiment
            return {
                "label": final_sentiment,
                "confidence": round(final_confidence, 3),
                "is_sarcastic": is_sarcastic,
                "surface_sentiment": surface_sentiment,
                "implied_analysis": implied_analysis,
                "sarcasm_confidence": sarcasm_result['confidence']
            }
            
    except Exception as e:
        return {"label": "error", "confidence": 0.0, "error": str(e)}

def analyze_agreement_contextual(post_text, comment_text, similarity_threshold=0.5):
    """Analyze agreement using contextual sentiment + semantic similarity"""
    try:
        # Get semantic similarity
        similarity = get_semantic_similarity(post_text, comment_text)
        
        if similarity < 0.3:
            return {"agreement": "off_topic", "similarity": similarity, "confidence": 0.0}
        
        # Get sentiments with sarcasm awareness
        post_analysis = analyze_sentiment_with_innuendo(post_text)
        comment_analysis = analyze_sentiment_with_innuendo(comment_text)
        
        post_label = post_analysis['label']
        comment_label = comment_analysis['label']
        
        # Context-aware agreement logic
        if similarity > similarity_threshold:
            if post_label == comment_label:
                agreement = "agreement"
                confidence = min(post_analysis['confidence'], comment_analysis['confidence'])
            else:
                agreement = "disagreement" 
                confidence = min(post_analysis['confidence'], comment_analysis['confidence'])
        else:
            agreement = "neutral"
            confidence = similarity
            
        return {
            "agreement": agreement,
            "similarity": round(similarity, 3),
            "confidence": round(confidence, 3),
            "post_sentiment": post_label,
            "comment_sentiment": comment_label,
            "post_is_sarcastic": post_analysis.get('is_sarcastic', False),
            "comment_is_sarcastic": comment_analysis.get('is_sarcastic', False)
        }
    except Exception as e:
        return {"agreement": "error", "error": str(e), "similarity": 0.0, "confidence": 0.0}

def analyze_agreement_nli(post_text, comment_text):
    """Analyze agreement using Natural Language Inference"""
    try:
        # Format for NLI: premise [SEP] hypothesis
        nli_input = f"{post_text} [SEP] {comment_text}"
        result = nli_pipeline(nli_input)[0]
        
        label_mapping = {
            "ENTAILMENT": "agreement",
            "CONTRADICTION": "disagreement", 
            "NEUTRAL": "neutral"
        }
        
        return {
            "agreement": label_mapping.get(result['label'], "neutral"),
            "confidence": round(result['score'], 3),
            "nli_label": result['label']
        }
    except Exception as e:
        return {"agreement": "error", "error": str(e), "confidence": 0.0}

def analyze_agreement_with_innuendo(post_text, comment_text, method="contextual"):
    """Enhanced agreement detection that handles sarcasm/innuendo"""
    try:
        # Detect sarcasm in comment
        comment_sarcasm = detect_sarcasm_innuendo(comment_text)
        is_sarcastic = comment_sarcasm['is_sarcastic']
        
        if method == "nli":
            base_agreement = analyze_agreement_nli(post_text, comment_text)
        else:
            base_agreement = analyze_agreement_contextual(post_text, comment_text)
        
        # Adjust agreement interpretation for sarcastic content
        if is_sarcastic and base_agreement['agreement'] in ['agreement', 'disagreement']:
            semantic_similarity = get_semantic_similarity(post_text, comment_text)
            
            if semantic_similarity > 0.6:
                # Sarcastic agreement - surface disagreement but underlying agreement
                return {
                    "agreement": "sarcastic_agreement",
                    "confidence": min(comment_sarcasm['confidence'], base_agreement['confidence']),
                    "is_sarcastic": True,
                    "base_agreement": base_agreement['agreement'],
                    "similarity": round(semantic_similarity, 3)
                }
            else:
                # Genuine sarcastic disagreement
                return {
                    "agreement": "sarcastic_disagreement",
                    "confidence": min(comment_sarcasm['confidence'], base_agreement['confidence']),
                    "is_sarcastic": True,
                    "base_agreement": base_agreement['agreement']
                }
        
        return {
            **base_agreement,
            "is_sarcastic": is_sarcastic,
            "sarcasm_confidence": comment_sarcasm['confidence']
        }
        
    except Exception as e:
        return {"agreement": "error", "error": str(e), "confidence": 0.0}

@app.route('/analyze', methods=['POST'])
def analyze():
    """Original sentiment analysis endpoint"""
    data = request.get_json()
    comments = data.get("comments", [])

    results = []
    label_counts = Counter()

    for comment in comments:
        if not comment.strip():
            continue

        try:
            analysis = sentiment_pipeline(comment)[0]
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

@app.route('/analyze_advanced', methods=['POST'])
def analyze_advanced():
    """Advanced analysis with sarcasm/innuendo detection"""
    data = request.get_json()
    comments = data.get("comments", [])
    
    results = []
    label_counts = Counter()
    sarcasm_counts = Counter()

    for comment in comments:
        if not comment.strip():
            continue

        try:
            analysis = analyze_sentiment_with_innuendo(comment)
            label = analysis['label']
            
            label_counts[label] += 1
            if analysis.get('is_sarcastic', False):
                sarcasm_counts['sarcastic'] += 1
            else:
                sarcasm_counts['literal'] += 1

            results.append({
                "comment": comment,
                "analysis": analysis
            })
            
        except Exception as e:
            results.append({
                "comment": comment,
                "error": str(e),
                "analysis": {"label": "error", "confidence": 0.0}
            })

    total = len([r for r in results if r.get('analysis', {}).get('label') != 'error'])
    
    sentiment_distribution = {
        label: round(count / total, 3) for label, count in label_counts.items()
    } if total > 0 else {}
    
    sarcasm_distribution = {
        label: round(count / total, 3) for label, count in sarcasm_counts.items()
    } if total > 0 else {}

    return jsonify({
        "results": results,
        "statistics": {
            "total_comments": total,
            "sentiment_distribution": sentiment_distribution,
            "sarcasm_distribution": sarcasm_distribution,
            "sarcastic_percentage": round(sarcasm_counts.get('sarcastic', 0) / total * 100, 2) if total > 0 else 0
        }
    })

@app.route('/analyze_with_agreement', methods=['POST'])
def analyze_with_agreement():
    """Enhanced analysis with agreement detection"""
    data = request.get_json()
    post_text = data.get("post_text", "")
    comments = data.get("comments", [])
    method = data.get("method", "contextual")  # "contextual" or "nli"
    
    if not post_text.strip():
        return jsonify({"error": "post_text is required"}), 400

    results = []
    label_counts = Counter()
    agreement_counts = Counter()
    sarcasm_counts = Counter()

    for comment in comments:
        if not comment.strip():
            continue

        try:
            # Get sentiment analysis with sarcasm detection
            sentiment_analysis = analyze_sentiment_with_innuendo(comment)
            label = sentiment_analysis['label']
            label_counts[label] += 1

            if sentiment_analysis.get('is_sarcastic', False):
                sarcasm_counts['sarcastic'] += 1
            else:
                sarcasm_counts['literal'] += 1

            # Get agreement analysis with sarcasm handling
            agreement_analysis = analyze_agreement_with_innuendo(post_text, comment, method)
            agreement_label = agreement_analysis['agreement']
            agreement_counts[agreement_label] += 1

            results.append({
                "comment": comment,
                "sentiment": sentiment_analysis,
                "agreement": agreement_analysis
            })
            
        except Exception as e:
            results.append({
                "comment": comment,
                "sentiment": {"label": "error", "confidence": 0},
                "agreement": {"agreement": "error", "error": str(e)},
                "error": str(e)
            })

    # Calculate statistics
    total = len([r for r in results if r.get('sentiment', {}).get('label') != 'error'])
    
    sentiment_distribution = {
        label: round(count / total, 3) for label, count in label_counts.items()
    } if total > 0 else {}
    
    agreement_distribution = {
        label: round(count / total, 3) for label, count in agreement_counts.items()
    } if total > 0 else {}
    
    sarcasm_distribution = {
        label: round(count / total, 3) for label, count in sarcasm_counts.items()
    } if total > 0 else {}

    sentiment_score = (
        label_counts.get("positive", 0) - label_counts.get("negative", 0)
    ) / total if total > 0 else 0

    agreement_score = (
        agreement_counts.get("agreement", 0) - agreement_counts.get("disagreement", 0)
    ) / total if total > 0 else 0

    return jsonify({
        "post_text": post_text,
        "method": method,
        "results": results,
        "statistics": {
            "total_comments": total,
            "sentiment": {
                "counts": dict(label_counts),
                "distribution": sentiment_distribution,
                "weighted_score": round(sentiment_score, 3)
            },
            "agreement": {
                "counts": dict(agreement_counts),
                "distribution": agreement_distribution,
                "weighted_score": round(agreement_score, 3)
            },
            "sarcasm": {
                "counts": dict(sarcasm_counts),
                "distribution": sarcasm_distribution,
                "sarcastic_percentage": round(sarcasm_counts.get('sarcastic', 0) / total * 100, 2) if total > 0 else 0
            }
        }
    })

@app.route('/detect_sarcasm', methods=['POST'])
def detect_sarcasm():
    """Endpoint specifically for sarcasm detection"""
    data = request.get_json()
    texts = data.get("texts", [])
    
    results = []
    for text in texts:
        if not text.strip():
            continue
            
        try:
            result = detect_sarcasm_innuendo(text)
            results.append({
                "text": text,
                "result": result
            })
        except Exception as e:
            results.append({
                "text": text,
                "error": str(e)
            })
    
    return jsonify({"results": results})

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """Comprehensive emotional and implied sentiment analysis"""
    data = request.get_json()
    comments = data.get("comments", [])
    
    results = []
    label_counts = Counter()
    emotion_stats = Counter()
    implied_reasons = Counter()

    for comment in comments:
        if not comment.strip():
            continue

        try:
            # Get complete sentiment analysis with emotion and implied meaning
            sentiment_result = sentiment_pipeline(comment)[0]
            surface_sentiment = sentiment_result['label'].lower()
            surface_confidence = sentiment_result['score']
            
            # Get implied sentiment analysis
            implied_analysis = analyze_implied_sentiment(comment, surface_sentiment, surface_confidence)
            
            # Get emotion analysis
            emotion_results = emotion_pipeline(comment)[0]
            emotion_scores = {item['label']: item['score'] for item in emotion_results}
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            # Get sarcasm detection
            sarcasm_result = detect_sarcasm_innuendo(comment)
            
            final_label = implied_analysis['label']
            label_counts[final_label] += 1
            emotion_stats[dominant_emotion] += 1
            implied_reasons[implied_analysis.get('implied_reason', 'literal')] += 1

            results.append({
                "comment": comment,
                "analysis": {
                    "final_sentiment": final_label,
                    "final_confidence": implied_analysis['confidence'],
                    "surface_sentiment": surface_sentiment,
                    "surface_confidence": surface_confidence,
                    "dominant_emotion": dominant_emotion,
                    "emotion_scores": emotion_scores,
                    "implied_reason": implied_analysis.get('implied_reason', 'literal'),
                    "is_sarcastic": sarcasm_result['is_sarcastic'],
                    "sarcasm_confidence": sarcasm_result['confidence']
                }
            })
            
        except Exception as e:
            results.append({
                "comment": comment,
                "error": str(e),
                "analysis": {"final_sentiment": "error", "final_confidence": 0.0}
            })

    total = len([r for r in results if r.get('analysis', {}).get('final_sentiment') != 'error'])
    
    sentiment_distribution = {
        label: round(count / total, 3) for label, count in label_counts.items()
    } if total > 0 else {}
    
    emotion_distribution = {
        emotion: round(count / total, 3) for emotion, count in emotion_stats.items()
    } if total > 0 else {}
    
    implied_distribution = {
        reason: round(count / total, 3) for reason, count in implied_reasons.items()
    } if total > 0 else {}

    return jsonify({
        "results": results,
        "statistics": {
            "total_comments": total,
            "sentiment_distribution": sentiment_distribution,
            "emotion_distribution": emotion_distribution,
            "implied_reasons_distribution": implied_distribution,
            "weighted_sentiment_score": round((label_counts.get("positive", 0) - label_counts.get("negative", 0)) / total, 3) if total > 0 else 0
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "models_loaded": True,
        "features": {
            "sentiment_analysis": True,
            "sarcasm_detection": True,
            "semantic_similarity": True,
            "agreement_detection": True,
            "nli_analysis": True,
            "emotion_analysis": True,
            "implied_sentiment": True
        }
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)