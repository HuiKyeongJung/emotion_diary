import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1. 감정 데이터 불러오기
data = pd.read_csv("data/emotion_dataset.csv")

# 2. 입력 문장과 정답 감정 나누기
X = data["text"]
y = data["emotion"]

# 3. 문장을 숫자 데이터로 변환
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 4. AI 모델 만들기
model = MultinomialNB()

# 5. 모델 학습시키기
model.fit(X_vectorized, y)

# 6. 학습된 모델 저장하기
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("AI 모델 학습 완료!")
print("emotion_model.pkl 생성 완료")
print("vectorizer.pkl 생성 완료")