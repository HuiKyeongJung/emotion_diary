import math
import os

import joblib


MODEL_PATH = "emotion_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

EMOTION_KEYWORDS = {
    "기쁨": ["기쁘", "행복", "즐거", "웃", "좋다", "뿌듯", "신난", "즐겁"],
    "슬픔": ["슬프", "눈물", "외롭", "속상", "서럽", "허전", "울고", "울컥", "참담"],
    "분노": ["화가", "열받", "짜증", "분노", "미워", "억울", "빡치", "화난", "심통", "성질"],
    "불안": ["불안", "걱정", "초조", "두렵", "떨리", "긴장", "두려", "불편", "걱정스"],
    "우울": ["우울", "우울해", "우울하다", "다운", "무기력", "의기소침"],
    "평온": ["그냥", "보통", "평범", "무난", "아무렇지", "특별하지", "별일", "그렇다", "괜찮다"],
}

NEUTRAL_THRESHOLD = 0.45


def load_artifacts(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "학습된 모델 또는 벡터라이저가 없습니다. 먼저 train_model.py를 실행해 주세요."
        )

    return joblib.load(model_path), joblib.load(vectorizer_path)


def keyword_emotion(text: str):
    lowered = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return emotion
    return None


def _softmax(values):
    max_value = max(values)
    exp_values = [math.exp(value - max_value) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def predict_emotion(text: str, model, vectorizer):
    keyword_label = keyword_emotion(text)
    if keyword_label is not None:
        return {
            "emotion": keyword_label,
            "confidence": 1.0,
            "source": "keyword",
        }

    text_vectorized = vectorizer.transform([text])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vectorized)[0]
        classes = model.classes_
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(text_vectorized)
        scores = scores[0] if getattr(scores, "ndim", 1) > 1 else scores
        probabilities = _softmax(scores)
        classes = model.classes_
    else:
        predicted = model.predict(text_vectorized)[0]
        return {
            "emotion": predicted,
            "confidence": None,
            "source": "model",
        }

    max_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
    confidence = float(probabilities[max_index])
    predicted = classes[max_index]

    if confidence < NEUTRAL_THRESHOLD:
        return {
            "emotion": "평온",
            "confidence": confidence,
            "source": "low_confidence",
        }

    return {
        "emotion": predicted,
        "confidence": confidence,
        "source": "model",
    }
