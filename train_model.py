from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# 감정 학습 데이터
texts = [
    "오늘 너무 행복했다",
    "기분이 정말 좋다",
    "친구랑 놀아서 즐거웠다",
    "오늘은 너무 슬프다",
    "눈물이 날 것 같다",
    "아무것도 하기 싫다",
    "화가 많이 난다",
    "짜증나고 답답하다",
    "너무 열받는다",
]

labels = [
    "positive",
    "positive",
    "positive",
    "sad",
    "sad",
    "sad",
    "angry",
    "angry",
    "angry",
]

# 모델 생성
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# 학습
model.fit(texts, labels)

# 모델 저장
joblib.dump(model, "mood_model.pkl")

print("모델 학습 완료!")
print("테스트 결과:", model.predict(["오늘 너무 행복해"])[0])