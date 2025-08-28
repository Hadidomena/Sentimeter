from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import numpy as np
import numpy as np
import re

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

# Enhanced pattern dictionaries with more comprehensive coverage
SARCASM_PATTERNS = {
    'high_confidence': [
        r'\b(oh\s+)?wow\s*,?\s*(just\s+)?(what\s+(i|we)\s+(wanted|needed))\b',
        r'\b(exactly|precisely)\s+what\s+(i|we)\s+(wanted|hoped\s+for)\b',
        r'\b(thanks\s+(a\s+lot|so\s+much))\s+for\b',
        r'\b(great|wonderful|perfect|amazing)\s*[.!]*\s*$',  # Short sarcastic praise
        r'\byeah\s+(right|sure)\b',
        r'\b(oh\s+)?sure\s*[.!]*\s*$',
        r'\b(totally|absolutely)\s+(not|never)\b',
    ],
    'medium_confidence': [
        r'\b(brilliant|genius|smart)\s+(move|idea|choice)\b',
        r'\breal(ly)?\s+(helpful|useful|great)\b',
        r'\b(just\s+)?(fantastic|wonderful|marvelous)\b',
        r'\boh\s+(joy|goody|great)\b',
        r'\bthat\'s\s+(just\s+)?(great|wonderful|perfect)\b',
    ],
    'context_dependent': [
        r'\b(great|good|nice|fine)\b',
        r'\b(awesome|cool|sweet)\b',
        r'\b(love|like)\s+(it|this|that)\b',
    ]
}

INTERNET_SLANG_POSITIVE = {
    'gaming_positive': [
        r'\b(sick|insane|fire|lit|clean|crisp|nutty|cracked)\b',
        r'\b(goated|based|chad|poggers|pog|kek)\b',
        r'\b(slaps|bangs|hits\s+different)\b',
        r'\b(no\s+cap|fr\s+fr|bussin)\b',
    ],
    'caps_memes': [
        r'\bDO\s+NOT\s+THE\s+\w+\b',
        r'\b(BONK|HORNY\s+JAIL)\b',
        r'\bNO\s+HORNY\b',
        r'\b[A-Z]{4,}\s+[A-Z]{4,}\b',  # Multiple all-caps words
    ],
    'appreciation_slang': [
        r'\b(thicc|thick)\b(?!\s+(skull|head))',  # Positive unless "thick skull"
        r'\b(blessed|gifted|stacked)\b',
        r'\b(mommy|daddy)\s*[!]*$',  # Internet appreciation
    ]
}

INNUENDO_PATTERNS = {
    'body_appreciation': [
        r'\b(assets|proportions|curves|endowments|features)\b',
        r'\binherited.*from\b',
        r'\bblessed\s+with\b',
        r'\bgifted\s+(with|in)\b',
        r'\bwell[\s-]?(endowed|equipped|built)\b',
        r'\b(big|huge|massive)\s+(.*)\s+(energy|vibes)\b',
    ],
    'suggestive_positive': [
        r'\bwaking\s+up\s+to\s+this\b',
        r'\bimagine.*morning\b',
        r'\bevery\s+morning\b.*\b(sight|view)\b',
        r'\bstep\s+on\s+me\b',
        r'\bchoke\s+me\b',
    ]
}

FRUSTRATION_AFFECTION_PATTERNS = [
    r'\b(damn|dammit|shit|fuck)\s+(you|this)\s+(are|is)\s+(so|too)\s+(cute|adorable|beautiful)\b',
    r'\bstop\s+being\s+so\s+(cute|adorable|perfect)\b',
    r'\bi\s+hate\s+how\s+(cute|pretty|beautiful|perfect)\b',
    r'\bmaking\s+me\s+(feel|act|react)\s+like\b',
    r'\b(why|how)\s+(are|is)\s+(you|this)\s+so\s+(perfect|cute|beautiful)\b',
]

def get_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    try:
        embeddings = similarity_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except:
        return 0.0

def detect_pattern_matches(text, pattern_dict):
    """Detect pattern matches and return confidence scores"""
    text_lower = text.lower()
    matches = {}
    total_confidence = 0.0
    
    for category, patterns in pattern_dict.items():
        category_matches = []
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                category_matches.append(pattern)
                if category == 'high_confidence':
                    total_confidence += 0.4
                elif category == 'medium_confidence':
                    total_confidence += 0.25
                elif category == 'context_dependent':
                    total_confidence += 0.1
                elif category in ['gaming_positive', 'caps_memes', 'appreciation_slang']:
                    total_confidence += 0.35
                elif category in ['body_appreciation', 'suggestive_positive']:
                    total_confidence += 0.3
        
        if category_matches:
            matches[category] = category_matches
    
    return matches, min(total_confidence, 1.0)

def analyze_context_clues(text, post_text=""):
    """Analyze contextual clues for better sentiment interpretation"""
    context_info = {
        'has_emojis': bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text)),
        'has_ellipsis': '...' in text,
        'has_multiple_punctuation': bool(re.search(r'[!?]{2,}', text)),
        'all_caps_ratio': len([w for w in text.split() if w.isupper() and len(w) > 2]) / max(len(text.split()), 1),
        'word_count': len(text.split()),
        'has_quotes': '"' in text or "'" in text,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?')
    }
    
    # Check for fanart/media context
    fanart_keywords = ['art', 'artist', 'fanart', 'drawing', 'illustration', 'design', 'character', 'waifu', 'husbando']
    context_info['is_fanart_context'] = any(keyword in (text + " " + post_text).lower() for keyword in fanart_keywords)
    
    # Check for gaming context
    gaming_keywords = ['game', 'gaming', 'play', 'player', 'stream', 'twitch', 'youtube', 'content']
    context_info['is_gaming_context'] = any(keyword in (text + " " + post_text).lower() for keyword in gaming_keywords)
    
    return context_info

def analyze_enhanced_sentiment(text, post_text="", confidence_threshold=0.6):
    """Enhanced sentiment analysis with comprehensive context awareness"""
    try:
        # 1. Get base sentiment and emotion
        base_sentiment = sentiment_pipeline(text)[0]
        surface_label = base_sentiment['label'].lower()
        surface_confidence = base_sentiment['score']
        
        # Get emotion analysis
        emotion_results = emotion_pipeline(text)[0]
        emotion_scores = {item['label']: item['score'] for item in emotion_results}
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # 2. Analyze context clues
        context = analyze_context_clues(text, post_text)
        
        # 3. Pattern detection
        sarcasm_matches, sarcasm_confidence = detect_pattern_matches(text, SARCASM_PATTERNS)
        slang_matches, slang_confidence = detect_pattern_matches(text, INTERNET_SLANG_POSITIVE)
        innuendo_matches, innuendo_confidence = detect_pattern_matches(text, INNUENDO_PATTERNS)
        
        # Check for frustrated affection
        frustration_matches = []
        for pattern in FRUSTRATION_AFFECTION_PATTERNS:
            if re.search(pattern, text.lower()):
                frustration_matches.append(pattern)
        
        # 4. Decision logic for sentiment adjustment
        final_label = surface_label
        final_confidence = surface_confidence
        corrections = []
        
        # Handle frustrated affection (high anger + positive words about appearance)
        if (frustration_matches or 
            (emotion_scores.get('anger', 0) > 0.4 and 
             any(word in text.lower() for word in ['cute', 'beautiful', 'perfect', 'adorable']))):
            final_label = 'positive'
            final_confidence = 0.75
            corrections.append('frustrated_affection')
        
        # Handle positive internet slang misclassified as negative
        elif slang_matches and surface_label == 'negative':
            if ('gaming_positive' in slang_matches or 
                'caps_memes' in slang_matches or 
                'appreciation_slang' in slang_matches):
                final_label = 'positive'
                final_confidence = max(0.7, slang_confidence)
                corrections.append('internet_slang_positive')
        
        # Handle innuendo in fanart context
        elif innuendo_matches and context['is_fanart_context']:
            if surface_label == 'negative':
                final_label = 'positive'
                final_confidence = max(0.65, innuendo_confidence)
                corrections.append('fanart_innuendo')
        
        # Handle sarcasm (be more conservative)
        elif sarcasm_matches and sarcasm_confidence > 0.4:
            # Only flip if we're confident it's sarcasm AND it's high-confidence positive
            if (surface_label == 'positive' and surface_confidence > 0.85 and 
                'high_confidence' in sarcasm_matches):
                final_label = 'negative'
                final_confidence = 0.6
                corrections.append('sarcastic_praise')
            elif ('caps_memes' in slang_matches):  # Don't flip caps memes
                pass  # Keep as is
            else:
                corrections.append('possible_sarcasm_detected')
        
        # Emotion-based corrections
        if not corrections:
            if (surface_label == 'negative' and 
                dominant_emotion in ['joy', 'surprise'] and 
                emotion_scores[dominant_emotion] > 0.6):
                final_label = 'positive'
                final_confidence = 0.6
                corrections.append('emotion_override_joy')
            elif (surface_label == 'positive' and 
                  dominant_emotion == 'anger' and 
                  emotion_scores['anger'] > 0.7 and
                  not corrections):
                final_label = 'negative'
                final_confidence = 0.6
                corrections.append('emotion_override_anger')
        
        return {
            "label": final_label,
            "confidence": round(final_confidence, 3),
            "original": {
                "label": surface_label,
                "confidence": round(surface_confidence, 3)
            },
            "analysis": {
                "dominant_emotion": dominant_emotion,
                "emotion_confidence": round(emotion_scores[dominant_emotion], 3),
                "corrections_applied": corrections,
                "pattern_matches": {
                    "sarcasm": sarcasm_matches,
                    "slang": slang_matches,
                    "innuendo": innuendo_matches,
                    "frustration": bool(frustration_matches)
                },
                "context_clues": context,
                "confidence_scores": {
                    "sarcasm": round(sarcasm_confidence, 3),
                    "slang": round(slang_confidence, 3),
                    "innuendo": round(innuendo_confidence, 3)
                }
            }
        }
        
    except Exception as e:
        return {
            "label": "error",
            "confidence": 0.0,
            "error": str(e)
        }

def analyze_agreement_enhanced(post_text, comment_text, similarity_threshold=0.4):
    """Enhanced agreement analysis using improved sentiment detection"""
    try:
        similarity = get_semantic_similarity(post_text, comment_text)
        
        if similarity < 0.25:
            return {
                "agreement": "off_topic", 
                "similarity": round(similarity, 3), 
                "confidence": 0.0
            }
        
        # Get enhanced sentiment analysis for both texts
        post_sentiment = analyze_enhanced_sentiment(post_text)
        comment_sentiment = analyze_enhanced_sentiment(comment_text, post_text)
        
        post_label = post_sentiment['label']
        comment_label = comment_sentiment['label']
        
        # Agreement determination logic
        if similarity > 0.6:  # High similarity - same topic
            if post_label == comment_label:
                agreement = "agreement"
                confidence = min(post_sentiment['confidence'], comment_sentiment['confidence'])
            else:
                agreement = "disagreement"
                confidence = min(post_sentiment['confidence'], comment_sentiment['confidence'])
        elif similarity > similarity_threshold:  # Medium similarity
            if post_label == comment_label:
                agreement = "neutral_agreement"
                confidence = similarity * 0.8
            else:
                agreement = "neutral"
                confidence = similarity * 0.6
        else:
            agreement = "neutral"
            confidence = similarity
        
        return {
            "agreement": agreement,
            "similarity": round(similarity, 3),
            "confidence": round(confidence, 3),
            "post_sentiment": post_label,
            "comment_sentiment": comment_label,
            "sentiment_analysis": {
                "post": post_sentiment.get('analysis', {}),
                "comment": comment_sentiment.get('analysis', {})
            }
        }
        
    except Exception as e:
        return {"agreement": "error", "error": str(e), "similarity": 0.0, "confidence": 0.0}

def analyze_agreement_nli(post_text, comment_text):
    """Analyze agreement using Natural Language Inference"""
    try:
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

@app.route('/analyze', methods=['POST'])
def analyze():
    """Original sentiment analysis endpoint with optional enhancement"""
    data = request.get_json()
    comments = data.get("comments", [])
    enhanced = data.get("enhanced", False)  # Optional enhancement flag
    post_context = data.get("post_context", "")  # Optional post context

    results = []
    label_counts = Counter()

    for comment in comments:
        if not comment.strip():
            continue

        try:
            if enhanced:
                # Use enhanced analysis
                analysis = analyze_enhanced_sentiment(comment, post_context)
                label = analysis['label']
                score = analysis['confidence']
                
                results.append({
                    "comment": comment,
                    "label": label,
                    "confidence": score,
                    "enhanced_analysis": analysis
                })
            else:
                # Original basic analysis
                analysis = sentiment_pipeline(comment)[0]
                label = analysis['label'].lower()
                score = analysis['score']
                
                results.append({
                    "comment": comment,
                    "label": label,
                    "confidence": round(score, 3)
                })
            
            label_counts[label] += 1
            
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
    } if total > 0 else {}

    sentiment_score = (
        label_counts.get("positive", 0) - label_counts.get("negative", 0)
    ) / total if total > 0 else 0

    response = {
        "results": results,
        "labelCounts": dict(label_counts),
        "classDistribution": class_distribution,
        "weightedScore": round(sentiment_score, 3)
    }
    
    if enhanced:
        response["enhancement_stats"] = {
            "corrections_made": len([r for r in results if r.get('enhanced_analysis', {}).get('analysis', {}).get('corrections_applied')]),
            "pattern_detection_active": True
        }

    return jsonify(response)

@app.route('/analyze_with_agreement', methods=['POST'])
def analyze_with_agreement():
    """Enhanced analysis with agreement detection - now uses improved sentiment by default"""
    data = request.get_json()
    post_text = data.get("post_text", "")
    comments = data.get("comments", [])
    method = data.get("method", "enhanced")  # Default to enhanced
    
    if not post_text.strip():
        return jsonify({"error": "post_text is required"}), 400

    results = []
    label_counts = Counter()
    agreement_counts = Counter()
    enhancement_stats = {
        "corrections_made": 0,
        "pattern_types_detected": Counter(),
        "emotion_overrides": 0
    }

    for comment in comments:
        if not comment.strip():
            continue

        try:
            # Always use enhanced sentiment analysis for better accuracy
            sentiment_analysis = analyze_enhanced_sentiment(comment, post_text)
            label = sentiment_analysis['label']
            label_counts[label] += 1
            
            # Track enhancement statistics
            if sentiment_analysis.get('analysis', {}).get('corrections_applied'):
                enhancement_stats["corrections_made"] += 1
                for correction in sentiment_analysis['analysis']['corrections_applied']:
                    enhancement_stats["pattern_types_detected"][correction] += 1
            
            # Get agreement analysis
            if method == "nli":
                agreement_analysis = analyze_agreement_nli(post_text, comment)
            else:  # Enhanced contextual (default)
                agreement_analysis = analyze_agreement_enhanced(post_text, comment)
            
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
            "enhancement": {
                "corrections_made": enhancement_stats["corrections_made"],
                "correction_rate": round(enhancement_stats["corrections_made"] / total * 100, 1) if total > 0 else 0,
                "pattern_types": dict(enhancement_stats["pattern_types_detected"])
            }
        }
    })

@app.route('/analyze_advanced', methods=['POST'])
def analyze_advanced():
    """Advanced sentiment analysis with sarcasm detection"""
    data = request.get_json()
    comments = data.get("comments", [])
    
    results = []
    label_counts = Counter()
    sarcasm_counts = Counter()
    
    for comment in comments:
        if not comment.strip():
            continue
            
        try:
            # Use enhanced sentiment analysis
            analysis = analyze_enhanced_sentiment(comment)
            label = analysis['label']
            label_counts[label] += 1
            
            # Check if sarcasm was detected
            corrections = analysis.get('analysis', {}).get('corrections_applied', [])
            is_sarcastic = any('sarcasm' in correction for correction in corrections)
            sarcasm_counts['sarcastic' if is_sarcastic else 'literal'] += 1
            
            results.append({
                "comment": comment,
                "analysis": {
                    "label": label,
                    "confidence": analysis['confidence'],
                    "is_sarcastic": is_sarcastic,
                    "surface_sentiment": analysis.get('original', {}).get('label', label),
                    "surface_confidence": analysis.get('original', {}).get('confidence', analysis['confidence']),
                    "corrections_applied": corrections
                }
            })
            
        except Exception as e:
            results.append({
                "comment": comment,
                "analysis": {"label": "error", "confidence": 0},
                "error": str(e)
            })
    
    total = len([r for r in results if r.get('analysis', {}).get('label') != 'error'])
    
    # Calculate distributions
    sentiment_distribution = {
        label: round(count / total, 3) for label, count in label_counts.items()
    } if total > 0 else {}
    
    sarcasm_distribution = {
        label: round(count / total, 3) for label, count in sarcasm_counts.items()
    } if total > 0 else {}
    
    sarcastic_percentage = round((sarcasm_counts.get('sarcastic', 0) / total * 100), 1) if total > 0 else 0
    
    return jsonify({
        "results": results,
        "statistics": {
            "sentiment_distribution": sentiment_distribution,
            "sarcasm_distribution": sarcasm_distribution,
            "sarcastic_percentage": sarcastic_percentage,
            "total_comments": total
        }
    })

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """Comprehensive emotion and implied sentiment analysis"""
    data = request.get_json()
    comments = data.get("comments", [])
    
    results = []
    sentiment_counts = Counter()
    emotion_counts = Counter()
    implied_reason_counts = Counter()
    
    for comment in comments:
        if not comment.strip():
            continue
            
        try:
            # Get enhanced sentiment analysis
            sentiment_analysis = analyze_enhanced_sentiment(comment)
            
            # Get detailed emotion analysis
            emotion_results = emotion_pipeline(comment)[0]
            emotion_scores = {item['label']: item['score'] for item in emotion_results}
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            # Determine implied reason based on corrections
            corrections = sentiment_analysis.get('analysis', {}).get('corrections_applied', [])
            implied_reason = 'literal'
            if corrections:
                if 'frustrated_affection' in corrections:
                    implied_reason = 'frustrated_affection'
                elif 'internet_slang_positive' in corrections:
                    implied_reason = 'internet_slang'
                elif 'fanart_innuendo' in corrections:
                    implied_reason = 'fanart_context'
                elif 'sarcastic_praise' in corrections:
                    implied_reason = 'sarcasm'
                elif 'emotion_override' in str(corrections):
                    implied_reason = 'emotion_override'
                else:
                    implied_reason = 'pattern_detected'
            
            final_sentiment = sentiment_analysis['label']
            sentiment_counts[final_sentiment] += 1
            emotion_counts[dominant_emotion] += 1
            implied_reason_counts[implied_reason] += 1
            
            results.append({
                "comment": comment,
                "analysis": {
                    "final_sentiment": final_sentiment,
                    "final_confidence": sentiment_analysis['confidence'],
                    "surface_sentiment": sentiment_analysis.get('original', {}).get('label', final_sentiment),
                    "surface_confidence": sentiment_analysis.get('original', {}).get('confidence', sentiment_analysis['confidence']),
                    "dominant_emotion": dominant_emotion,
                    "emotion_scores": emotion_scores,
                    "implied_reason": implied_reason,
                    "is_sarcastic": 'sarcasm' in str(corrections),
                    "corrections_applied": corrections
                }
            })
            
        except Exception as e:
            results.append({
                "comment": comment,
                "analysis": {"final_sentiment": "error", "final_confidence": 0},
                "error": str(e)
            })
    
    total = len([r for r in results if r.get('analysis', {}).get('final_sentiment') != 'error'])
    
    # Calculate distributions
    sentiment_distribution = {
        label: round(count / total, 3) for label, count in sentiment_counts.items()
    } if total > 0 else {}
    
    emotion_distribution = {
        emotion: round(count / total, 3) for emotion, count in emotion_counts.items()
    } if total > 0 else {}
    
    implied_reasons_distribution = {
        reason: round(count / total, 3) for reason, count in implied_reason_counts.items()
    } if total > 0 else {}
    
    # Calculate weighted sentiment score
    weighted_sentiment_score = (
        sentiment_counts.get("positive", 0) - sentiment_counts.get("negative", 0)
    ) / total if total > 0 else 0
    
    return jsonify({
        "results": results,
        "statistics": {
            "sentiment_distribution": sentiment_distribution,
            "emotion_distribution": emotion_distribution,
            "implied_reasons_distribution": implied_reasons_distribution,
            "weighted_sentiment_score": round(weighted_sentiment_score, 3),
            "total_comments": total
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "models_loaded": True,
        "features": {
            "enhanced_sentiment": True,
            "pattern_detection": True,
            "context_awareness": True,
            "internet_slang_detection": True,
            "frustration_affection_detection": True,
            "fanart_innuendo_detection": True,
            "caps_meme_detection": True,
            "emotion_analysis": True,
            "semantic_similarity": True,
            "agreement_detection": True,
            "nli_analysis": True
        }
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)